"""Serves user-uploaded attachments from a shared base directory."""
import os


def read_user_file(base_dir: str, filename: str) -> bytes:
    """Return the bytes of ``filename`` inside ``base_dir``."""
    target_path = os.path.join(base_dir, filename)
    with open(target_path, "rb") as fh:
        return fh.read()
