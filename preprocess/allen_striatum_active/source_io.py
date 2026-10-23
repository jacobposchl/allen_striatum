"""Private raw Zarr I/O and lossless archive decoding for preprocessing only."""
import base64
import json
import os
from pathlib import Path

import numpy as np
import zarr


def discard_cache(fd):
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


class StreamingStore(zarr.DirectoryStore):
    def __getitem__(self, key):
        try:
            with open(Path(self.path) / key, "rb") as handle:
                value = handle.read()
                discard_cache(handle.fileno())
                return value
        except (FileNotFoundError, IsADirectoryError) as error:
            raise KeyError(key) from error


def open_source(path):
    store = StreamingStore(str(path))
    if ".zmetadata" in store:
        return zarr.open_consolidated(store, mode="r")
    return zarr.open_group(store, mode="r")


def unpack_object(value):
    kind, payload = value["type"], value["value"]
    if kind == "bytes":
        return base64.b64decode(payload)
    if kind == "dict":
        return {unpack_object(k): unpack_object(v) for k, v in payload}
    if kind in {"list", "tuple"}:
        items = [unpack_object(v) for v in payload]
        return tuple(items) if kind == "tuple" else items
    if kind == "ndarray":
        return np.asarray([unpack_object(v) for v in payload], dtype=value["dtype"]).reshape(value["shape"])
    if kind == "scalar":
        return payload
    raise ValueError(f"Unknown source object encoding: {kind}")


class SourceArray:
    """Decode the private archive during conversion and source validation."""
    def __init__(self, group):
        self.group = group
        self.dataset = group["values"]
        self.shape = tuple(json.loads(group.attrs["source_shape_json"]))
        self.ndim = len(self.shape)
        self.dtype = np.dtype(group.attrs["source_dtype"])
        self.chunks = self.dataset.chunks or self.dataset.shape
        self.attrs = json.loads(group.attrs["source_attrs_json"])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        obj = self.dataset.asstr() if self.dtype.kind in "OU" else self.dataset
        values = obj[key] if self.ndim else obj[0]
        if self.group.attrs.get("encoding") == "object_json":
            array = np.asarray(values)
            result = np.empty(array.shape, dtype=object)
            for index in np.ndindex(array.shape):
                result[index] = unpack_object(json.loads(array[index]))
            return result.item() if array.ndim == 0 else result
        return values


class SourceGroup:
    def __init__(self, group):
        self.group = group
        self.attrs = json.loads(group.attrs.get("source_attrs_json", "{}"))

    def __contains__(self, key):
        return key in self.group

    def __iter__(self):
        return iter(self.group)

    def __len__(self):
        return len(self.group)

    def keys(self):
        return self.group.keys()

    def items(self):
        return ((key, self[key]) for key in self)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __getitem__(self, key):
        group = self.group[key]
        return SourceArray(group) if "source_shape_json" in group.attrs else SourceGroup(group)
