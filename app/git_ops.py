import base64
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from app.config import get_settings
from app.github_client import get_installation_token

logger = logging.getLogger(__name__)

GIT_TIMEOUT_SECONDS = 120


class GitCheckoutError(RuntimeError):
    """Raised when cloning or checking out a PR branch fails."""


def _auth_extra_header_config(clone_url: str, token: str) -> str:
    """A one-off `git -c <this>` value that injects an HTTP Authorization
    header for exactly one git invocation, needed for private repos.

    Scoped to clone_url's own scheme+host (http.<scoped-url>.extraheader),
    not the global http.extraheader -- so the token is only ever sent to the
    actual remote host, never to some other host a redirect might point at.

    Passed as a transient `-c` flag rather than `git config --local` (which
    would persist it into the checkout's own .git/config): the checkout
    directory is also where an untrusted PR's own test suite executes
    (tester_server.py runs pytest with cwd=repo_path), and that test code can
    read any file under repo_path even though its *environment* is stripped
    of secrets -- a credential left sitting in .git/config would be a direct
    filesystem exfiltration path that env-stripping does nothing to prevent.
    """
    scoped_url = f"{urlsplit(clone_url).scheme}://{urlsplit(clone_url).netloc}/"
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"http.{scoped_url}.extraheader=AUTHORIZATION: basic {encoded}"


def _run_git(args: list[str], cwd: Path, config: list[str] | None = None) -> None:
    # config entries (e.g. the auth header above) are passed as their own
    # -c flags ahead of the subcommand, and deliberately excluded from the
    # "git {args}" text below -- so a credential never ends up echoed into a
    # GitCheckoutError message, which callers may log or surface upstream.
    config_flags = [flag for entry in (config or []) for flag in ("-c", entry)]
    try:
        subprocess.run(
            ["git", *config_flags, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GitCheckoutError(
            f"git {' '.join(args)} failed (exit {exc.returncode}): {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise GitCheckoutError(f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS}s") from exc


def ensure_repo_checkout(clone_url: str, full_name: str, branch: str) -> Path:
    """Ensure a local working copy of full_name exists at HEAD of branch.

    Clones into <git_workspace_root>/<full_name> if no checkout exists yet;
    otherwise fetches and resets the existing checkout to origin/<branch>.
    The initial clone is shallow (--depth=1) and single-branch: the agent
    only ever reads target_files and their local dependencies, so full
    history and every other branch would just be wasted bandwidth/disk.
    Subsequent fetches stay shallow too, for the same reason, and work fine
    for a different branch than the one originally cloned since we check out
    FETCH_HEAD directly rather than relying on the single-branch's tracked ref.
    Returns the resolved path to the working copy for MCP tools to target.
    """
    workspace_root = Path(get_settings().git_workspace_root).resolve()
    target = (workspace_root / full_name).resolve()
    if not target.is_relative_to(workspace_root):
        raise GitCheckoutError(f"repository '{full_name}' escapes workspace root '{workspace_root}'")

    # Sent on every clone/fetch, public or private repos alike -- same
    # default posture as GitHub Actions' own actions/checkout. A plain,
    # unauthenticated clone_url has no way to access a private repo at all,
    # and authenticating unconditionally avoids a public/private special case.
    # get_installation_token resolves a fresh, short-lived GitHub App
    # installation token -- git's x-access-token:<token> basic-auth format
    # accepts either a classic PAT or an installation token identically, so
    # this is a drop-in replacement for the old static PAT.
    auth_config = [_auth_extra_header_config(clone_url, get_installation_token(full_name))]

    if not (target / ".git").is_dir():
        logger.info("No local checkout for %s, cloning into %s", full_name, target)
        target.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            ["clone", "--depth=1", "--single-branch", "--branch", branch, clone_url, str(target)],
            cwd=target.parent,
            config=auth_config,
        )
    else:
        logger.info("Local checkout for %s exists at %s", full_name, target)

    logger.info("Fetching %s and checking out branch %s", full_name, branch)
    # branch is a PR's head.ref, fully attacker-controlled (any fork owner
    # names their own branch). Passed as a bare positional argument here, so
    # without "--" a value like "--upload-pack=<cmd>" is parsed by git as an
    # option rather than a literal ref name -- reproduced directly: git
    # actually invoked the injected string as a subprocess. "--" forces
    # everything after it to be read as a literal ref/pathspec, closing that
    # off. (The --branch value in the clone call above and the -B value in
    # the checkout call below don't need this: both are consumed as an
    # option's value by the immediately preceding flag, never re-parsed.)
    _run_git(["fetch", "--depth=1", "origin", "--", branch], cwd=target, config=auth_config)
    # --force: a previous review's leftover uncommitted changes to a tracked
    # file (e.g. a patch attempt) would otherwise make plain `checkout -B`
    # abort with "local changes would be overwritten" before reset --hard
    # below ever gets a chance to clean the tree -- confirmed by direct
    # reproduction, not theoretical.
    _run_git(["checkout", "--force", "-B", branch, "FETCH_HEAD"], cwd=target)
    _run_git(["reset", "--hard", "FETCH_HEAD"], cwd=target)
    # Removes untracked leftovers from a previous review (e.g. a new file a
    # patch attempt wrote) without touching anything the target repo's own
    # .gitignore excludes (no -x), so a stray venv/cache dir is left alone.
    _run_git(["clean", "-fd"], cwd=target)

    return target
