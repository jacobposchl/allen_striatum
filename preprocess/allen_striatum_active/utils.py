"""Recording discovery, source identity, and lossless NWB array storage."""
import base64
import hashlib
import json
import math
from pathlib import Path
import re

import h5py
import numpy as np


def discover(root):
    root = Path(root).expanduser().resolve()
    found = {}
    for asset in sorted(root.iterdir()):
        if not asset.is_dir():
            continue
        match = re.match(r"ecephys_(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", asset.name)
        if not match:
            continue
        animal, session = match.groups()
        for path in sorted([*asset.glob("*.nwb"), *(asset / "nwb").glob("*.nwb")]):
            rec = re.search(r"experiment(\d+)_recording(\d+)", path.name)
            if not rec or not path.is_dir():
                continue
            recording = f"experiment{rec[1]}_recording{rec[2]}"
            key = (animal, session, recording)
            item = dict(animal=animal, session=session, recording=recording,
                        asset=str(asset), source=str(path),
                        id=f"{animal}_{session}_{recording}")
            if key in found and not asset.name.endswith("_nwb"):
                continue
            found[key] = item
    if not found:
        raise ValueError(f"No active NWB-Zarr recordings found in {root}")
    return [found[k] for k in sorted(found)]


def sidecars(item):
    result = {}
    for name in ("session.json", "data_description.json", "subject.json", "processing.json"):
        path = Path(item["asset"]) / name
        if path.is_file():
            result[name] = json.loads(path.read_text())
    return result


def source_identity(item):
    """Content identity of source metadata plus every chunk's size/mtime.

    Array values receive independent SHA256 digests during conversion. This
    cheaper signature detects normal source changes before an existing output
    is reused, without rereading the entire dataset on each invocation.
    """
    root = Path(item["source"])
    digest = hashlib.sha256(json.dumps(sidecars(item), sort_keys=True).encode())
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        if path.name.startswith(".z"):
            digest.update(path.read_bytes())
    # Recording placement also depends on acquisition sample-clock metadata.
    # Include every acquisition in this asset because a later recording uses
    # its primary recording's clock as well as its own.
    asset = Path(item["asset"])
    clocks = [*asset.glob("curated/*/numpysorting_info.json"),
              *asset.glob("curated/*/spikes.npy"),
              *asset.glob("postprocessed/*/recording_info/.zattrs")]
    for path in sorted(clocks):
        stat = path.stat()
        digest.update(f"{path.relative_to(asset)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        if path.suffix != ".npy":
            digest.update(path.read_bytes())
    return digest.hexdigest()


def data_group(parent, name):
    group = parent.create_group(name, track_order=True)
    group.attrs.update(object="Data", absolute_start=0.0)
    return group


def blocks(array, target_bytes=16 * 1024**2):
    if array.ndim == 0:
        yield ()
        return
    # Zarr source chunks keep large waveform/LFP reads bounded, even when one
    # complete first-axis row exceeds the target buffer size.
    chunks = array.chunks or array.shape
    sizes = [max(1, min(int(c), max(1, int(n)))) for c, n in zip(chunks, array.shape)]
    while math.prod(sizes) * max(array.dtype.itemsize, 8) > target_bytes:
        axis = int(np.argmax(sizes))
        sizes[axis] = max(1, sizes[axis] // 2)
    for index in np.ndindex(*(math.ceil(n / c) for n, c in zip(array.shape, sizes))):
        yield tuple(slice(i*c, min((i+1)*c, n)) for i, c, n in zip(index, sizes, array.shape))


def normalized(value):
    value = np.asarray(value)
    if value.dtype.kind in "OU":
        def text(x):
            if isinstance(x, bytes):
                return x.decode("utf-8")
            if isinstance(x, (str, np.str_)):
                return str(x)
            raise TypeError(f"Unsupported non-text object in source array: {type(x)}")
        return np.asarray([text(x) for x in value.flat], dtype=object).reshape(value.shape)
    return value


def pack_object(value):
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "dtype": value.dtype.str, "shape": value.shape,
                "value": [pack_object(v) for v in value.flat]}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {"type": "dict", "value": [[pack_object(k), pack_object(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "value": [pack_object(v) for v in value]}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"type": "scalar", "value": value}
    raise TypeError(f"Unsupported source object: {type(value)}")


def hash_update(digest, value):
    value = normalized(value)
    if value.dtype.kind == "O":
        for entry in value.flat:
            encoded = entry.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        digest.update(np.ascontiguousarray(value).tobytes())


def copy_tree(source, destination):
    """Return array checksums, also validating every just-written block."""
    checks = []
    destination.attrs["source_attrs_json"] = json.dumps(dict(source.attrs), sort_keys=True)
    for name, obj in source.items():
        if not hasattr(obj, "shape"):
            child = data_group(destination, name)
            checks.extend({**c, "path": name + "/" + c["path"]} for c in copy_tree(obj, child))
            continue
        group = destination.create_group(name)
        codecs = [codec.get_config().get("id") for codec in (getattr(obj, "filters", None) or [])]
        encoding = "object_json" if obj.dtype.kind == "O" and "vlen-utf8" not in codecs else "native"
        group.attrs.update(object="ArrayDict", source_shape_json=json.dumps(obj.shape),
                           source_dtype=obj.dtype.str,
                           encoding=encoding,
                           source_attrs_json=json.dumps(dict(obj.attrs), sort_keys=True))
        # temporaldata 0.1.6 decodes Unicode as ASCII. Keep archive strings as
        # HDF5 UTF-8 bytes there; SourceArray restores actual Unicode/objects.
        group.attrs["_unicode_keys"] = np.asarray([], dtype="S")
        shape = obj.shape if obj.ndim else (1,)
        dtype = h5py.string_dtype("utf-8") if obj.dtype.kind in "OU" else obj.dtype
        chunks = tuple(max(1, min(c, max(1, n))) for c, n in zip(obj.chunks, shape)) if obj.ndim else None
        # HDF5 cannot use a positive chunk dimension for a fixed empty axis.
        chunks = chunks if all(shape) else None
        target = group.create_dataset("values", shape=shape, dtype=dtype, chunks=chunks)
        source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        for selection in blocks(obj):
            original = np.asarray(obj[selection])
            if encoding == "object_json":
                values = np.asarray([json.dumps(pack_object(v), ensure_ascii=True) for v in original.flat],
                                    dtype=object).reshape(original.shape)
            else:
                values = normalized(original)
            target_selection = selection if obj.ndim else 0
            target[target_selection] = values
            reread = target.asstr()[target_selection] if obj.dtype.kind in "OU" else target[target_selection]
            hash_update(source_hash, values)
            hash_update(output_hash, reread)
            equal = (np.array_equal(values, reread, equal_nan=True) if values.dtype.kind in "fc"
                     else np.array_equal(values, reread))
            if not equal:
                raise ValueError(f"Source/output mismatch for {target.name}, {selection}")
        if source_hash.digest() != output_hash.digest():
            raise ValueError(f"Checksum mismatch: {target.name}")
        group.attrs["sha256"] = source_hash.hexdigest()
        checks.append(dict(path=name, shape=list(obj.shape), dtype=obj.dtype.str,
                           sha256=source_hash.hexdigest()))
    return checks
