"""Validated storage for the experimental, approximate affinity reuse path.

Cropping a full-complex trunk state does not reproduce a trunk evaluation on
the affinity crop. This cache is an explicit alternative inference procedure.

Format (version 2) is a *stored* (uncompressed) ``.npz`` holding the identity
fields plus ``s``, ``z`` and ``coords`` for the whole complex, unpadded. Two
things make it cheap enough that writing it does not eat the time the affinity
pass saves:

* **No zlib.** ``z`` is ``[N, N, 192]``; deflating it costs seconds of CPU on a
  few-hundred-token complex and compresses trunk activations barely at all.
  Uncompressed members are also the precondition for the next point.
* **Windowed reads.** The affinity pass keeps at most 256 tokens. Since each
  member is stored verbatim, the loader seeks to the rows the crop names and
  reads only those, instead of pulling the whole ``N x N x 192`` array through
  the page cache to throw most of it away.

Under the ``cache_fp16`` lever ``s`` and ``z`` are stored as float16, halving
both again; coordinates always stay float32 because the affinity head reads
them as geometry. Everything is returned as float32.
"""

import hashlib
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from boltz.opt import enabled

CACHE_VERSION = 2

#: Written by save_structure_cache; every one must be present on load.
REQUIRED_FIELDS = frozenset({"version", "record_id", "structure_sha256", "s", "z", "coords"})

#: float16 holds ~3 decimal digits; a trunk state outside this range would lose
#: far more than that, so such a cache falls back to float32 instead.
_FLOAT16_MAX = float(np.finfo(np.float16).max)


def _structure_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranked_sample_index(record_index: int, samples_per_record: int, ranks: dict) -> int:
    """Return the flattened sample belonging to this record's rank-zero pose."""
    if samples_per_record < 1 or set(ranks) != set(range(samples_per_record)):
        raise ValueError("Invalid structure sample ranks")
    if sorted(ranks.values()) != list(range(samples_per_record)):
        raise ValueError("Structure sample ranks must form a permutation")
    return record_index * samples_per_record + next(i for i, rank in ranks.items() if rank == 0)


def cache_array(tensor) -> np.ndarray:
    """Detach without retaining GPU storage; NumPy cannot serialize bfloat16."""
    return tensor.detach().float().cpu().numpy()


def _storage_dtype(array: np.ndarray) -> np.dtype:
    """float16 when the lever is on and the values survive the narrowing."""
    if not enabled("cache_fp16"):
        return np.dtype(np.float32)
    if array.size and float(np.max(np.abs(array))) > _FLOAT16_MAX:
        return np.dtype(np.float32)
    return np.dtype(np.float16)


def save_structure_cache(
    path: Path,
    *,
    record_id: str,
    structure_path: Path,
    s: np.ndarray,
    z: np.ndarray,
    coords: np.ndarray,
) -> None:
    """Write an unpadded cache tied to the precise pre-affinity structure file."""
    _validate_arrays(s, z, coords, s.shape[0], coords.shape[0])
    # The digest invalidates a cache if a rerun replaces the selected pose.
    write = np.savez if enabled("cache_io") else np.savez_compressed
    write(
        path,
        version=np.array(CACHE_VERSION),
        record_id=np.array(record_id),
        structure_sha256=np.array(_structure_digest(structure_path)),
        s=np.asarray(s, dtype=_storage_dtype(s)),
        z=np.asarray(z, dtype=_storage_dtype(z)),
        coords=np.asarray(coords, dtype=np.float32),
    )


def _validate_arrays(s, z, coords, token_count, atom_count) -> None:
    if s.ndim != 2 or s.shape[0] != token_count:
        raise ValueError("Cached single embeddings do not match the structure token axis")
    if z.ndim != 3 or z.shape[:2] != (token_count, token_count):
        raise ValueError("Cached pair embeddings do not match both structure token axes")
    if coords.shape != (atom_count, 3):
        raise ValueError("Cached coordinates do not match the structure atom axis")
    for array in (s, z, coords):
        if array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError("Structure cache must contain finite floating-point arrays")


def _validate_shapes(shapes, token_count, atom_count) -> None:
    """The same axis checks as _validate_arrays, from headers alone."""
    s, z, coords = shapes["s"], shapes["z"], shapes["coords"]
    if len(s) != 2 or s[0] != token_count:
        raise ValueError("Cached single embeddings do not match the structure token axis")
    if len(z) != 3 or z[:2] != (token_count, token_count):
        raise ValueError("Cached pair embeddings do not match both structure token axes")
    if coords != (atom_count, 3):
        raise ValueError("Cached coordinates do not match the structure atom axis")


def _validate_indices(indices, count: int, name: str) -> np.ndarray:
    indices = np.asarray(indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or not indices.size
        or np.any(indices < 0)
        or np.any(indices >= count)
        or np.unique(indices).size != indices.size
    ):
        raise ValueError(f"Invalid cached structure {name} mapping")
    return indices


def _read_header(handle):
    """Return (shape, dtype, fortran_order), leaving the handle at the data.

    ``read_array_header_*`` are the documented readers; if a NumPy release stops
    exporting the pair, the caller treats the archive as not row-addressable and
    reads whole arrays instead.
    """
    version = npy_format.read_magic(handle)
    reader = {
        (1, 0): npy_format.read_array_header_1_0,
        (2, 0): npy_format.read_array_header_2_0,
    }.get(version)
    if reader is None:
        raise _NotRowAddressable(f"unsupported .npy version {version}")
    shape, fortran_order, dtype = reader(handle)
    return shape, dtype, fortran_order


class _NotRowAddressable(Exception):
    """This archive cannot be read a row at a time; read whole arrays."""


class _StoredNpz:
    """Row-addressable view over an uncompressed .npz.

    Every member is a plain ``.npy`` stored verbatim, so a member's rows sit at
    a fixed stride from the end of its header and can be read individually. A
    member that is compressed, or an archive that cannot be opened this way,
    makes ``rows`` fall back to reading the array whole.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.archive = zipfile.ZipFile(self.path)
        self.names = {}
        for info in self.archive.infolist():
            name = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            self.names[name] = info
        self.stored = enabled("cache_io") and all(
            info.compress_type == zipfile.ZIP_STORED for info in self.names.values()
        )
        self.shapes = {}
        self.dtypes = {}
        self.contiguous = {}
        for name, info in self.names.items():
            try:
                with self.archive.open(info) as handle:
                    shape, dtype, fortran_order = _read_header(handle)
            except _NotRowAddressable:
                self.stored = False
                continue
            self.shapes[name] = shape
            self.dtypes[name] = dtype
            # A one-dimensional or scalar array is trivially C-contiguous; the
            # flag only ever matters for the multi-axis members.
            self.contiguous[name] = not fortran_order or len(shape) < 2

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.archive.close()
        return False

    @property
    def files(self):
        return set(self.names)

    def scalar(self, name):
        with self.archive.open(self.names[name]) as handle:
            return np.load(handle, allow_pickle=False).item()

    def whole(self, name) -> np.ndarray:
        with self.archive.open(self.names[name]) as handle:
            return np.load(handle, allow_pickle=False)

    def rows(self, name, indices: np.ndarray) -> np.ndarray:
        """``array[indices]`` reading only the selected rows where possible."""
        if not self.stored or not self.contiguous.get(name, False):
            return self.whole(name)[indices]
        shape, dtype = self.shapes[name], self.dtypes[name]
        row_items = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        row_bytes = row_items * dtype.itemsize
        if row_bytes == 0:
            return self.whole(name)[indices]
        out = np.empty((len(indices), *shape[1:]), dtype=dtype)
        # Ascending file order keeps the reads sequential even when the crop
        # hands us its tokens in distance order.
        order = np.argsort(indices, kind="stable")
        with self.archive.open(self.names[name]) as handle:
            start = self._data_start(handle)
            for position in order:
                handle.seek(start + int(indices[position]) * row_bytes)
                buffer = handle.read(row_bytes)
                if len(buffer) != row_bytes:
                    raise ValueError(f"Truncated structure cache member {name!r}")
                out[position] = np.frombuffer(buffer, dtype=dtype).reshape(shape[1:])
        return out

    @staticmethod
    def _data_start(handle) -> int:
        _read_header(handle)
        return handle.tell()

    def is_windowed(self) -> bool:
        """Whether reads select rows rather than pulling whole arrays."""
        return self.stored


def load_structure_cache(
    path: Path,
    *,
    record_id: str,
    structure_path: Path,
    token_count: int,
    atom_count: int,
    token_ids: np.ndarray,
    atom_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Validate identity and full axes before applying the affinity crop.

    Legacy caches intentionally fail validation: they have no reliable pose or
    record identity and may have been written with incorrect batch indexing.

    Only the rows the crop names are read from ``s`` and ``z``; the full-axis
    checks come from the stored array headers, so a cache built for a different
    complex is still rejected before any of its data is touched.
    """
    with _StoredNpz(path) as cache:
        # Header shapes are only trustworthy for members we could parse.
        if not REQUIRED_FIELDS.issubset(cache.files):
            raise ValueError("Legacy or incomplete structure cache; rerun with --override")
        if cache.scalar("version") != CACHE_VERSION:
            raise ValueError("Unsupported structure cache version; rerun with --override")
        if cache.scalar("record_id") != record_id:
            raise ValueError("Structure cache belongs to a different record")
        if cache.scalar("structure_sha256") != _structure_digest(structure_path):
            raise ValueError("Structure cache does not match the selected pose; rerun with --override")
        if not {"s", "z", "coords"}.issubset(cache.shapes):
            raise ValueError("Unreadable structure cache arrays; rerun with --override")
        _validate_shapes(cache.shapes, token_count, atom_count)
        token_ids = _validate_indices(token_ids, token_count, "token")
        atom_ids = _validate_indices(atom_ids, atom_count, "atom")
        cropped = {
            "s": cache.rows("s", token_ids).astype(np.float32),
            # Rows first, then the same selection on the second token axis.
            "z": cache.rows("z", token_ids)[:, token_ids].astype(np.float32),
            "coords": cache.rows("coords", atom_ids).astype(np.float32),
        }
    for name, array in cropped.items():
        if not np.isfinite(array).all():
            raise ValueError(f"Structure cache {name!r} is not finite over the affinity crop")
    return cropped
