"""The v2 cache layout: uncompressed members, windowed reads, float16 storage."""

import zipfile

import numpy as np
import pytest

from boltz import opt
from boltz.data.structure_cache import (
    CACHE_VERSION,
    _StoredNpz,
    load_structure_cache,
    save_structure_cache,
)


@pytest.fixture
def complex_cache(tmp_path):
    """A cache large enough that a crop reads a minority of its rows."""
    structure = tmp_path / "pre_affinity_rec.npz"
    structure.write_bytes(b"selected pose")
    rng = np.random.default_rng(11)
    tokens, atoms, width = 40, 90, 6
    arrays = {
        "s": rng.standard_normal((tokens, width), dtype=np.float32),
        "z": rng.standard_normal((tokens, tokens, width), dtype=np.float32),
        "coords": rng.standard_normal((atoms, 3), dtype=np.float32),
    }

    def write(profile):
        opt.configure(profile)
        path = tmp_path / f"cache_{profile}.npz"
        save_structure_cache(path, record_id="rec", structure_path=structure, **arrays)
        return path

    def read(path, token_ids, atom_ids):
        return load_structure_cache(
            path, record_id="rec", structure_path=structure,
            token_count=tokens, atom_count=atoms,
            token_ids=token_ids, atom_ids=atom_ids,
        )

    return arrays, write, read


def test_members_are_stored_so_rows_are_addressable(complex_cache):
    _, write, _ = complex_cache
    path = write("exact")
    with zipfile.ZipFile(path) as archive:
        kinds = {info.filename: info.compress_type for info in archive.infolist()}
    assert set(kinds.values()) == {zipfile.ZIP_STORED}, kinds
    with _StoredNpz(path) as cache:
        assert cache.is_windowed()
        assert cache.scalar("version") == CACHE_VERSION


def test_windowed_read_matches_a_whole_array_read(complex_cache):
    arrays, write, read = complex_cache
    path = write("exact")
    # Distance-ordered, non-contiguous, non-monotonic: what the cropper hands us.
    token_ids = np.array([17, 3, 39, 4, 0, 22])
    atom_ids = np.array([88, 1, 2, 3, 40])
    got = read(path, token_ids, atom_ids)
    np.testing.assert_array_equal(got["s"], arrays["s"][token_ids])
    np.testing.assert_array_equal(got["z"], arrays["z"][np.ix_(token_ids, token_ids)])
    np.testing.assert_array_equal(got["coords"], arrays["coords"][atom_ids])


def test_windowed_and_whole_paths_agree(complex_cache):
    """The fallback used for an unreadable layout must return the same values."""
    arrays, write, _ = complex_cache
    path = write("exact")
    token_ids = np.array([9, 2, 31])
    with _StoredNpz(path) as cache:
        windowed = cache.rows("z", token_ids)
        cache.stored = False
        whole = cache.rows("z", token_ids)
    np.testing.assert_array_equal(windowed, whole)
    np.testing.assert_array_equal(whole, arrays["z"][token_ids])


def test_fp16_lever_halves_storage_and_keeps_coordinates_exact(complex_cache):
    arrays, write, read = complex_cache
    exact, fast = write("exact"), write("fast")
    assert fast.stat().st_size < exact.stat().st_size * 0.6
    with _StoredNpz(fast) as cache:
        assert cache.dtypes["z"] == np.float16
        assert cache.dtypes["coords"] == np.float32
    token_ids, atom_ids = np.array([1, 5, 7]), np.array([0, 4])
    got = read(fast, token_ids, atom_ids)
    assert got["z"].dtype == np.float32
    np.testing.assert_array_equal(got["coords"], arrays["coords"][atom_ids])
    # float16 keeps ~3 decimal digits of a unit-scale trunk state.
    np.testing.assert_allclose(
        got["z"], arrays["z"][np.ix_(token_ids, token_ids)], rtol=1e-3, atol=1e-3
    )


def test_compressed_cache_still_loads(complex_cache, tmp_path):
    """A deflated archive is not row-addressable but stays readable."""
    arrays, write, read = complex_cache
    source = write("exact")
    deflated = tmp_path / "cache_deflated.npz"
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(
        deflated, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for info in src.infolist():
            dst.writestr(info.filename, src.read(info.filename))
    with _StoredNpz(deflated) as cache:
        assert not cache.is_windowed()
    token_ids = np.array([6, 1])
    got = read(deflated, token_ids, np.array([3]))
    np.testing.assert_array_equal(got["s"], arrays["s"][token_ids])
