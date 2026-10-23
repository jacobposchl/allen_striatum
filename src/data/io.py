"""Repository paths and the guard for scientific computation."""
import socket
from pathlib import Path

# Relative data, checkpoint, and output paths are anchored to this repository.
ROOT = Path(__file__).resolve().parents[2]


def require_compute():
    host = socket.gethostname().lower()
    if any(word in host for word in ("login", "betty")):
        raise RuntimeError("Data processing must run in an existing allocation, not on the login node")


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path

