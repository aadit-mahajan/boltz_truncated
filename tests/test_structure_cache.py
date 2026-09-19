"""Regression checks for pose selection, crop mapping and cache identity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from boltz.data.structure_cache import (
    cache_array,
    load_structure_cache,
    ranked_sample_index,
    save_structure_cache,
)


class StructureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.structure = self.directory / "pre_affinity_record.npz"
        self.structure.write_bytes(b"selected structure and topology")
        self.path = self.directory / "cache.npz"
        self.s = np.arange(20, dtype=np.float32).reshape(5, 4)
        self.z = np.arange(75, dtype=np.float32).reshape(5, 5, 3)
        self.coords = np.arange(27, dtype=np.float32).reshape(9, 3)
        self.write()

    def write(self):
        save_structure_cache(
            self.path, record_id="record", structure_path=self.structure,
            s=self.s, z=self.z, coords=self.coords,
        )

    def read(self, **overrides):
        arguments = dict(
            record_id="record", structure_path=self.structure,
            token_count=5, atom_count=9,
            token_ids=np.array([1, 4]), atom_ids=np.array([2, 3, 8]),
        )
        arguments.update(overrides)
        return load_structure_cache(self.path, **arguments)

    def test_rank_zero_pose_uses_record_and_sample_axes(self):
        # Two records with three samples each; their best poses are #2 and #1.
        self.assertEqual(ranked_sample_index(0, 3, {0: 2, 1: 1, 2: 0}), 2)
        self.assertEqual(ranked_sample_index(1, 3, {0: 1, 1: 0, 2: 2}), 4)
        self.assertEqual(ranked_sample_index(2, 1, {0: 0}), 2)

    def test_noncontiguous_crop_indexes_both_pair_axes_and_atoms(self):
        cache = self.read()
        np.testing.assert_array_equal(cache["s"], self.s[[1, 4]])
        np.testing.assert_array_equal(cache["z"], self.z[[1, 4]][:, [1, 4]])
        np.testing.assert_array_equal(cache["coords"], self.coords[[2, 3, 8]])

    def test_source_dtype_serializes_as_float32(self):
        self.s, self.z, self.coords = (v.astype(np.float16) for v in (self.s, self.z, self.coords))
        self.write()
        self.assertTrue(all(v.dtype == np.float32 for v in self.read().values()))

    def test_legacy_cache_rejected(self):
        np.savez(self.path, s=self.s, z=self.z, coords=self.coords)
        with self.assertRaisesRegex(ValueError, "Legacy or incomplete"):
            self.read()

    def test_wrong_record_rejected_even_for_equal_shapes(self):
        with self.assertRaisesRegex(ValueError, "different record"):
            self.read(record_id="another-record")

    def test_changed_pose_rejected(self):
        self.structure.write_bytes(b"replacement selected structure")
        with self.assertRaisesRegex(ValueError, "selected pose"):
            self.read()

    def test_rectangular_pair_axis_rejected(self):
        self.z = self.z[:, :4]
        with self.assertRaisesRegex(ValueError, "both structure token axes"):
            self.write()

    def test_full_structure_axis_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "structure token axis"):
            self.read(token_count=6)
        with self.assertRaisesRegex(ValueError, "structure atom axis"):
            self.read(atom_count=10)

    def test_bad_crop_mappings_rejected(self):
        for token_ids in (np.array([1, 5]), np.array([1, 1]), np.array([-1]), np.array([1.0])):
            with self.subTest(token_ids=token_ids):
                with self.assertRaisesRegex(ValueError, "token mapping"):
                    self.read(token_ids=token_ids)
        with self.assertRaisesRegex(ValueError, "atom mapping"):
            self.read(atom_ids=np.array([9]))

    def test_nonfinite_cache_rejected(self):
        self.coords[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite floating-point"):
            self.write()

    def test_bfloat16_torch_serialization(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in the lightweight test environment")
        array = cache_array(torch.tensor([1.25, 2.5], dtype=torch.bfloat16))
        self.assertEqual(array.dtype, np.float32)
        np.testing.assert_array_equal(array, [1.25, 2.5])


if __name__ == "__main__":
    unittest.main()
