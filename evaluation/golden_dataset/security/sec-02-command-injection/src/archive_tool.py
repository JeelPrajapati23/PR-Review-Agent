"""Zips a user-named directory for download."""
import subprocess


def compress_directory(directory_name: str) -> int:
    """Compress ``directory_name`` into ``<directory_name>.zip`` and return the exit code."""
    command = f"zip -r {directory_name}.zip {directory_name}"
    result = subprocess.run(command, shell=True, capture_output=True)
    return result.returncode
