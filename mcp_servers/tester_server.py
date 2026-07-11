import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from fastmcp import FastMCP


def _dbg(msg: str) -> None:
    print(f"[DBG {time.time():.2f}] {msg}", file=sys.stderr, flush=True)

mcp = FastMCP("PR-Tester-Server")

TEST_TIMEOUT_SECONDS = 120
# Per-test cap enforced by pytest-timeout (--timeout flag below), separate
# from and much shorter than TEST_TIMEOUT_SECONDS: catches a genuinely hung
# or infinite-looping test (e.g. introduced by a bad patch) fast, rather than
# burning the entire outer subprocess budget on one stuck test.
TEST_INNER_TIMEOUT_SECONDS = 15

# pytest runs whatever test files/conftest.py exist on the PR's own branch,
# and Python executes module-level code in every collected file (not just
# inside test functions) -- so this subprocess is running untrusted code by
# design. The env below is the entire environment that code gets to see:
# PATH so the interpreter/pytest can be found at all, plus the handful of
# vars Windows' own interpreter startup and tempfile/cache handling need to
# function (confirmed empirically: a real pytest run against a real test
# file, from a process holding only these vars, collects/executes/passes
# normally). Every secret this worker process holds -- GROQ_API_KEY,
# GITHUB_API_TOKEN, REDIS_URL, GITHUB_WEBHOOK_SECRET, CELERY_BROKER_URL --
# is deliberately absent, so malicious test code can't read them via
# os.environ and echo them into this tool's stdout/stderr, which would
# otherwise flow straight into the LLM's context and potentially into a
# public GitHub PR review comment.
_SANDBOX_ENV_ALLOWLIST = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "PATHEXT", "COMSPEC"}


def _sanitized_subprocess_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name.upper() in _SANDBOX_ENV_ALLOWLIST}


def _sandbox_subprocess_kwargs() -> dict:
    """Extra subprocess.run kwargs that drop privileges before exec'ing
    pytest, where that's actually possible.

    subprocess's user= kwarg is POSIX-only and requires the calling process
    to already hold the privilege to change uid -- in practice, running as
    root. Two cases both fail open to "no extra kwargs" rather than raising,
    since a review shouldn't hard-fail just because privilege-dropping isn't
    available in this deployment: Windows has no equivalent at all (this
    project's primary target, per CLAUDE.md), and a worker that's already
    running unprivileged can't drop further -- which is itself the more
    common, arguably safer posture for a containerized deployment, not a
    condition to treat as an error.

    Note this is defense-in-depth on top of the env stripping above, not a
    full jail: dropping uid alone doesn't isolate the filesystem or network,
    and 'nobody' will only be able to exec pytest at all if repo checkouts
    under git_workspace_root are readable by that user -- worth confirming
    as a deployment-time permission, since PermissionError here is already
    caught and reported as a clean "Error: ..." string rather than crashing.
    """
    if os.name != "posix":
        return {}
    if os.getuid() != 0:
        _dbg("Skipping privilege drop: worker is not running as root")
        return {}
    try:
        import pwd

        pwd.getpwnam("nobody")
    except KeyError:
        _dbg("Skipping privilege drop: 'nobody' user not found on this system")
        return {}
    return {"user": "nobody"}


def _supports_timeout_flag(pytest_binary: str) -> bool:
    """Whether pytest_binary's own environment has pytest-timeout installed.

    _pytest_command may resolve to a *different* repo's own .venv pytest,
    which won't necessarily have this plugin -- unconditionally appending
    --timeout would break every validation run for such a repo with an
    "unrecognized arguments" error, the same failure mode as passing a flag
    pytest doesn't support at all. Checking --help output is a cheap,
    environment-accurate way to confirm the plugin is actually there first.
    """
    try:
        result = subprocess.run(
            [pytest_binary, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
            env=_sanitized_subprocess_env(),
            **_sandbox_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--timeout=" in result.stdout


def _pytest_command(root: Path) -> list[str]:
    """Prefer the repo's own venv pytest over whatever is on PATH."""
    scripts_dir = root / ".venv" / "Scripts"
    pytest_binary = "pytest"
    for name in ("pytest.exe", "pytest"):
        candidate = scripts_dir / name
        if candidate.is_file():
            pytest_binary = str(candidate)
            break

    command = [pytest_binary]
    if _supports_timeout_flag(pytest_binary):
        command.append(f"--timeout={TEST_INNER_TIMEOUT_SECONDS}")
    return command


def _run_pytest(command: list[str], root: Path) -> str:
    _dbg(f"_run_pytest: thread started, command={command} cwd={root}")
    try:
        _dbg("_run_pytest: calling subprocess.run")
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_SECONDS,
            # This server's own stdin is a live pipe used by the MCP stdio
            # transport to receive JSON-RPC requests from the parent process.
            # Without an explicit stdin here, pytest inherits that same pipe
            # and blocks on it for the full TEST_TIMEOUT_SECONDS on Windows,
            # regardless of what the target repo's tests actually contain
            # (confirmed by reproducing: identical command/repo completes
            # instantly as a bare script, but hangs every time when spawned
            # from within this MCP server's own event loop). Redirecting to
            # DEVNULL gives pytest an isolated, already-closed stdin instead.
            stdin=subprocess.DEVNULL,
            # Sandboxing: a stripped env (no secrets visible to PR-supplied
            # test code) plus, where possible, a privilege drop -- see the
            # helpers' own docstrings for exactly what each does and doesn't
            # cover.
            env=_sanitized_subprocess_env(),
            **_sandbox_subprocess_kwargs(),
        )
        _dbg(f"_run_pytest: subprocess.run returned, exit={result.returncode}")
    except subprocess.TimeoutExpired:
        _dbg("_run_pytest: TimeoutExpired")
        return f"Error: test suite timed out after {TEST_TIMEOUT_SECONDS}s"
    except (FileNotFoundError, PermissionError) as exc:
        _dbg(f"_run_pytest: FileNotFoundError/PermissionError {exc}")
        # PermissionError here isn't necessarily "pytest is missing" -- it's
        # also what a dropped-privilege 'nobody' hits if it can't read/exec
        # files under repo_path, which needs a deployment-side permissions
        # fix (see _sandbox_subprocess_kwargs), not a PATH fix.
        return f"Error: could not run pytest in '{root}' ({exc})"

    return (
        f"Exit code: {result.returncode}\n\n"
        f"--- STDOUT ---\n{result.stdout}\n\n"
        f"--- STDERR ---\n{result.stderr}"
    )


@mcp.tool()
async def run_validation_suite(repo_path: str) -> str:
    """Run `pytest` inside repo_path and return a combined stdout+stderr report.

    Returns an explicit "Error: ..." string if repo_path doesn't exist or isn't
    a directory, pytest isn't installed, or the run times out (120s cap)
    instead of raising, so the calling agent always gets a usable text result.

    Runs pytest via a blocking subprocess.run offloaded to a worker thread
    (asyncio.to_thread) rather than asyncio.create_subprocess_exec: nesting the
    latter inside this MCP server's own stdio_server() event loop reliably
    deadlocks on Windows (ProactorEventLoop's IOCP-based subprocess pipes
    conflict with the thread-based stdin reader anyio's stdio wrapping uses).
    A plain blocking call off the event loop thread sidesteps that entirely.

    That workaround alone isn't sufficient, though: pytest also inherits this
    server's own stdin (the live pipe anyio uses for the MCP transport)
    unless told otherwise, and blocks on it for the full TEST_TIMEOUT_SECONDS
    regardless of what the target repo's tests contain -- so _run_pytest also
    passes stdin=subprocess.DEVNULL. pytest-timeout's --timeout flag adds a
    second, much shorter (TEST_INNER_TIMEOUT_SECONDS) cap on individual tests,
    to fail fast on a genuinely hung test rather than a stdio artifact.
    """
    _dbg("run_validation_suite: tool invoked")
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return f"Error: repo_path '{repo_path}' is not an existing directory"

    command = _pytest_command(root)
    _dbg("run_validation_suite: dispatching to thread")
    result = await asyncio.to_thread(_run_pytest, command, root)
    _dbg("run_validation_suite: thread returned, returning result")
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
