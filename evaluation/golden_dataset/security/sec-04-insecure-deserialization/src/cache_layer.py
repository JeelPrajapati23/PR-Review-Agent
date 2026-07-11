"""On-disk object cache shared by report-generation jobs."""
import pickle
from pathlib import Path


def save_cached_object(cache_dir: Path, key: str, obj: object) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.cache"
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return path


def load_cached_object(cache_dir: Path, key: str):
    """Load a previously cached object. ``cache_dir`` may be a shared upload directory."""
    path = cache_dir / f"{key}.cache"
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)
