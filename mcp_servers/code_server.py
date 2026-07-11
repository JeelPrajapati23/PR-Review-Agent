import ast
from pathlib import Path

from charset_normalizer import from_bytes
from fastmcp import FastMCP

mcp = FastMCP("PR-Code-Server")

# Directories never worth listing/reading during a review: VCS internals,
# virtualenvs, and common dependency/cache dirs that dwarf actual PR content.
_EXCLUDED_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}

# Real source files reviewed here run from a few bytes to a few hundred KB.
# Without a cap, a PR that adds/modifies a huge file (accidentally or as a
# deliberate resource-exhaustion attempt) gets read fully into memory via a
# single fetch_file_contents/scan_local_dependencies call -- cheap for an
# attacker, expensive for the worker. 5MB is generous headroom over any
# genuine source file while still bounding the worst case.
_MAX_FILE_READ_BYTES = 5 * 1024 * 1024


def _resolve_within_repo(repo_path: str, file_path: str) -> Path:
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"repo_path '{repo_path}' is not an existing directory")
    target = (root / file_path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"'{file_path}' escapes repo root '{repo_path}'")
    return target


@mcp.tool()
def list_repo_files(repo_path: str) -> str:
    """Recursively list every file under repo_path, one repo-relative path per line.

    Skips VCS/dependency/cache directories (.git, .venv, node_modules, etc.).
    Call this first to discover which files exist before using
    fetch_file_contents, since the PR event carries no changed-file list.
    Returns an explicit "Error: ..." string if repo_path doesn't exist.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        return f"Error: repo_path '{repo_path}' is not an existing directory"

    paths = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not (_EXCLUDED_DIR_NAMES & {part for part in p.relative_to(root).parts})
    ]
    if not paths:
        return "No files found in repo_path."
    return "\n".join(sorted(paths))


@mcp.tool()
def fetch_file_contents(repo_path: str, file_path: str) -> str:
    """Read and return the text contents of file_path relative to repo_path,
    with each line prefixed by its 1-indexed line number (format: 'N | code').

    Detects the file's encoding (UTF-8, UTF-16, etc.) via charset_normalizer
    rather than assuming UTF-8, since PR repos may contain files saved as
    UTF-16 (common from Windows editors) which read as null-byte-interleaved
    garbage under a hardcoded UTF-8 decode.

    Line numbers are prepended specifically so a reviewing agent can quote an
    exact line for a SUGGESTION marker instead of counting lines from raw
    text by eye -- reproduced directly: without numbering, a real review
    panel mis-numbered a SUGGESTION by one line and separately extrapolated a
    second SUGGESTION 12 lines past a wrong anchor to a line past the end of
    the file, which GitHub's review-comment API then rejected outright with
    a "Line could not be resolved" 422 (the correct *relative* offset between
    two near-duplicate functions, applied to an already-wrong absolute
    starting line).

    Returns an explicit "Error: ..." string instead of raising if repo_path
    doesn't exist, file_path escapes the repo root, the file is missing, or
    it can't be read (e.g. permissions, or it's a directory) or decoded.
    """
    try:
        target = _resolve_within_repo(repo_path, file_path)
        size = target.stat().st_size
        if size > _MAX_FILE_READ_BYTES:
            return f"Error: '{file_path}' is {size} bytes, exceeding the {_MAX_FILE_READ_BYTES}-byte read limit"
        raw = target.read_bytes()
    except (NotADirectoryError, ValueError, OSError) as exc:
        return f"Error: {exc}"

    match = from_bytes(raw).best()
    if match is None:
        return f"Error: could not determine a text encoding for '{file_path}'"

    lines = str(match).splitlines()
    if not lines:
        return ""
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, start=1))


def _read_text_best_effort(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_FILE_READ_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    match = from_bytes(raw).best()
    return str(match) if match is not None else None


def _resolve_module_file(root: Path, base_dir: Path, dotted: str) -> Path | None:
    """Resolve a dotted module name, looked up under base_dir, to an actual
    .py file or package __init__.py that exists inside root. Returns None for
    stdlib/third-party modules (or anything else that doesn't resolve to a
    real file under root) so callers only see genuine local dependencies.
    """
    candidate = base_dir.joinpath(*dotted.split("."))
    for module_path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
        resolved = module_path.resolve()
        if resolved.is_file() and resolved.is_relative_to(root):
            return resolved
    return None


@mcp.tool()
def scan_local_dependencies(repo_path: str, file_path: str) -> str:
    """Parse file_path (relative to repo_path) with Python's ast module and
    resolve its local (same-project) imports to actual file paths.

    Skips stdlib/third-party imports (anything that doesn't resolve to a real
    file inside repo_path). Use this instead of scanning the whole repo when
    a target file's logic depends on another project file, to keep context
    minimal. Returns one repo-relative path per line, "No local project
    dependencies found." if there are none, or an explicit "Error: ..."
    string if repo_path/file_path is invalid or the file can't be read or
    parsed as Python.
    """
    try:
        target = _resolve_within_repo(repo_path, file_path)
    except (NotADirectoryError, ValueError) as exc:
        return f"Error: {exc}"

    root = Path(repo_path).resolve()
    source = _read_text_best_effort(target)
    if source is None:
        return f"Error: could not read '{file_path}'"

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return f"Error: could not parse '{file_path}': {exc}"

    resolved: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_path = _resolve_module_file(root, root, alias.name)
                if module_path is not None:
                    resolved.add(module_path)

        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_dir = target.parent
                for _ in range(node.level - 1):
                    base_dir = base_dir.parent
                if not base_dir.resolve().is_relative_to(root):
                    continue
                names = [node.module] if node.module else [alias.name for alias in node.names]
                for name in names:
                    module_path = _resolve_module_file(root, base_dir, name)
                    if module_path is not None:
                        resolved.add(module_path)
            elif node.module:
                module_path = _resolve_module_file(root, root, node.module)
                if module_path is not None:
                    resolved.add(module_path)

    if not resolved:
        return "No local project dependencies found."
    return "\n".join(sorted(p.relative_to(root).as_posix() for p in resolved))


@mcp.tool()
def apply_code_patch(repo_path: str, file_path: str, proposed_code: str) -> str:
    """Write proposed_code to file_path (relative to repo_path).

    Creates parent directories if needed and overwrites any existing file.
    Returns a confirmation string with the path written and byte count, or an
    explicit "Error: ..." string if repo_path is invalid or file_path escapes it.
    """
    try:
        target = _resolve_within_repo(repo_path, file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = target.write_text(proposed_code, encoding="utf-8")
        return f"Wrote {bytes_written} bytes to '{target}'"
    except (NotADirectoryError, ValueError, OSError) as exc:
        return f"Error: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
