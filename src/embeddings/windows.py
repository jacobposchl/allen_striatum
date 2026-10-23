"""Manifest-bound recording windows, independent of export batch size."""
import hashlib
import json

import numpy as np

from src.data.io import resolve


class RecordingExport:
    def __init__(self, source, key=None):
        source = resolve(source)
        candidates = [source / "manifest.json", source / "pooled/manifest.json"]
        candidates.extend(parent / "manifest.json" for parent in source.parents)
        self.path = next((path for path in candidates if path.is_file()), None)
        if self.path is None:
            raise FileNotFoundError(f"No embedding export manifest for {source}")
        self.manifest = json.loads(self.path.read_text())
        if "recording_windows" not in self.manifest:
            raise ValueError("Re-export embeddings from Brainsets HDF5: the manifest lacks recording window schedules")
        if key is None:
            key = source.relative_to(self.path.parent).as_posix()
        self.key = str(key).strip("/")
        if self.key not in self.manifest["recording_windows"]:
            raise ValueError(f"Recording {self.key} is absent from the export schedule")
        self.schedule = self.manifest["recording_windows"][self.key]
        self.starts_source = np.asarray(self.schedule["start_source_s"], dtype=float)
        self.starts_session = np.asarray(self.schedule["start_session_s"], dtype=float)
        self.duration = float(self.schedule["duration_s"])
        if (self.starts_source.ndim != 1 or self.starts_session.shape != self.starts_source.shape
                or not np.isfinite(self.starts_source).all() or not np.isfinite(self.duration)
                or self.duration <= 0 or np.any(np.diff(self.starts_source) < self.duration - 1e-8)):
            raise ValueError("Malformed or overlapping exported recording windows")
        self.source_identity = self.schedule["source_identity"]
        listed = [name for name in self.manifest["files"] if name.startswith(self.key + "/")]
        if len(set(listed)) != len(listed):
            raise ValueError("Duplicate embedding batch in manifest")
        self.files = [(self.path.parent / name).resolve() for name in listed]
        if any(not path.is_relative_to(self.path.parent.resolve()) for path in self.files):
            raise ValueError("Embedding batch escapes the export directory")
        self.manifest_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self._view_files = None
        self._cached_path, self._cached_batch = None, None

    def _batch(self, path):
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as saved:
                self._cached_batch = {name: saved[name] for name in saved.files}
            self._cached_path = path
        return self._cached_batch

    def _window(self, batch, view):
        rows = np.flatnonzero(batch["view_index"] == view)
        if not len(rows):
            return None
        if not 0 <= view < len(self.starts_source):
            raise ValueError("Exported view index is outside its recording schedule")
        size = len(batch["view_index"])
        result = {name: values[rows] if values.ndim and len(values) == size else values
                  for name, values in batch.items()}
        for name, expected in (("view_start_source_s", self.starts_source[view]),
                               ("view_start_session_s", self.starts_session[view]),
                               ("view_duration_s", self.duration)):
            if name not in result or not np.allclose(result[name], expected, atol=1e-6, rtol=0, equal_nan=True):
                raise ValueError(f"Embedding {name} disagrees with its canonical window schedule")
        if len(np.unique(result["global_unit_id"])) != len(rows):
            raise ValueError("Duplicate unit identities within one exported window")
        return result

    def windows(self):
        seen = set()
        for path in self.files:
            batch = self._batch(path)
            for value in np.unique(batch["view_index"]):
                view = int(value)
                if view != value or view in seen:
                    raise ValueError("Noninteger or repeated exported view index")
                seen.add(view)
                yield view, self._window(batch, view)

    def window(self, view):
        """Return one view, or None for a scheduled window with no active units."""
        if self._view_files is None:
            self._view_files = {}
            for path in self.files:
                with np.load(path, allow_pickle=False) as batch:
                    views = np.unique(batch["view_index"])
                for value in views:
                    index = int(value)
                    if index != value or index in self._view_files:
                        raise ValueError("Noninteger or repeated exported view index")
                    self._view_files[index] = path
        path = self._view_files.get(int(view))
        return None if path is None else self._window(self._batch(path), int(view))
