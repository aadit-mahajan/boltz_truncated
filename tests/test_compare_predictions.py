"""CPU-only artifact-validation regressions. Run with unittest; no Boltz imports."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_predictions.py"
SPEC = importlib.util.spec_from_file_location("compare_predictions", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class PredictionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"
        self.reference.mkdir()
        self.candidate.mkdir()

    def write_unit(self, root, *, record="complex", seed=None, model=0, score=0.85):
        directory = (root / "boltz_results_inputs" / "predictions" / record if seed is None
                     else root / "by_seed" / record / f"s{seed}")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{record}_model_{model}.cif").write_text("data_complex\n#\n")
        (directory / f"confidence_{record}_model_{model}.json").write_text(json.dumps({
            "confidence_score": score, "pair_chains_iptm": {"0": {"1": 0.6}}}))
        np.savez_compressed(directory / f"plddt_{record}_model_{model}.npz", plddt=np.array([0.7, 0.8], dtype=np.float32))
        return directory

    def compare(self, **kwargs):
        return compare.compare_predictions(self.reference, self.candidate, **kwargs)

    def test_stock_and_kit_pair_with_explicit_seed(self):
        self.write_unit(self.reference, record="record_model_7", model=2)
        self.write_unit(self.candidate, record="record_model_7", seed=42, model=2)
        report = self.compare(reference_seed=42)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["summary"]["paired_files"], 3)
        self.assertEqual({row["model"] for row in report["comparisons"]}, {2})
        self.assertEqual({row["seed"] for row in report["comparisons"]}, {42})
        self.assertFalse(self.compare()["passed"], "Must not silently infer stock seed")

    def test_seed_model_and_record_are_not_cross_paired(self):
        for root in (self.reference, self.candidate):
            self.write_unit(root, seed=0, model=0)
            self.write_unit(root, seed=1, model=1)
            self.write_unit(root, record="other", seed=0)
        changed = self.candidate / "by_seed" / "complex" / "s1" / "confidence_complex_model_1.json"
        changed.write_text('{"confidence_score": 0.1, "pair_chains_iptm": {"0": {"1": 0.6}}}')
        report = self.compare()
        bad = [row for row in report["comparisons"] if not row["passed"]]
        self.assertFalse(report["passed"])
        self.assertEqual(len(bad), 1)
        self.assertEqual((bad[0]["record"], bad[0]["seed"], bad[0]["model"]), ("complex", 1, 1))

    def test_numeric_tolerance_and_exact_mode(self):
        self.write_unit(self.reference)
        self.write_unit(self.candidate, score=0.850001)
        self.assertTrue(self.compare(atol=1e-5, rtol=0)["passed"])
        self.assertFalse(self.compare(atol=0, rtol=0)["passed"])
        self.assertFalse(self.compare(exact=True)["passed"])

    def test_numeric_npz_compares_content_not_zip_compression(self):
        self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        np.savez(right / "plddt_complex_model_0.npz", plddt=np.array([0.7, 0.8], dtype=np.float32))
        report = self.compare()
        self.assertTrue(report["passed"])
        row = next(row for row in report["comparisons"] if row["comparison"] == "npz")
        self.assertFalse(row["byte_identical"])
        self.assertFalse(self.compare(exact=True)["passed"])

    def test_missing_and_extra_files_and_units_fail(self):
        self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        (right / "confidence_complex_model_0.json").unlink()
        (right / "unexpected.txt").write_text("unexpected output")
        self.write_unit(self.candidate, record="other")
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["missing"]), 1)
        self.assertEqual(len(report["extra"]), 4)

    def test_empty_trees_and_confidence_only_units_fail(self):
        self.assertFalse(self.compare()["passed"])
        for root in (self.reference, self.candidate):
            directory = self.write_unit(root)
            (directory / "complex_model_0.cif").unlink()
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertTrue(any("Missing structure" in issue for issue in report["reference_issues"]))

    def test_ambiguous_duplicate_runs_fail_instead_of_overwrite(self):
        original = self.write_unit(self.reference)
        self.write_unit(self.candidate)
        duplicate = self.reference / "second_run" / "predictions" / "complex"
        duplicate.mkdir(parents=True)
        (duplicate / "complex_model_0.cif").write_bytes((original / "complex_model_0.cif").read_bytes())
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertTrue(any("Ambiguous duplicate" in issue for issue in report["reference_issues"]))

    def test_expected_roster_catches_shared_missing_affinity_and_record(self):
        self.write_unit(self.reference)
        self.write_unit(self.candidate)
        expected = [{"record": "complex", "seed": None, "affinity": True},
                    {"record": "never_completed", "seed": None}]
        report = self.compare(expected=expected)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["reference_issues"]), 2)
        self.assertEqual(len(report["candidate_issues"]), 2)

    def test_affinity_values_are_compared(self):
        left = self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        (left / "affinity_complex.json").write_text('{"affinity_pred_value": -4.2, "affinity_probability_binary": 0.8}')
        (right / "affinity_complex.json").write_text('{"affinity_pred_value": -3.2, "affinity_probability_binary": 0.8}')
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"]["failed_files"], 1)

    def test_structures_are_exact_even_with_loose_tolerance(self):
        self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        (right / "complex_model_0.cif").write_text("data_changed\n#\n")
        self.assertFalse(self.compare(atol=100, rtol=100)["passed"])

    def test_nan_infinity_and_pickle_are_never_valid_even_if_identical(self):
        left = self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        for value in (float("nan"), float("inf")):
            for directory in (left, right):
                np.savez(directory / "plddt_complex_model_0.npz", plddt=np.array([value]))
            self.assertFalse(self.compare()["passed"])
        for directory in (left, right):
            np.savez(directory / "plddt_complex_model_0.npz", plddt=np.array([{"untrusted": "object"}], dtype=object))
        report = self.compare()
        self.assertFalse(report["passed"])
        self.assertTrue(any("allow_pickle=False" in row.get("error", "") for row in report["comparisons"]))

    def test_structured_arrays_compare_numeric_fields_and_exact_metadata(self):
        left = self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        dtype = np.dtype([("name", "U4"), ("coords", "f4", (3,)), ("index", "i8")])
        ref = np.array([("CA", [1, 2, 3], 2**60)], dtype=dtype)
        cand = ref.copy()
        cand["coords"] += 1e-6
        np.savez(left / "pre_affinity_complex.npz", atoms=ref)
        np.savez(right / "pre_affinity_complex.npz", atoms=cand)
        self.assertTrue(self.compare()["passed"])
        cand["index"] += 1
        np.savez(right / "pre_affinity_complex.npz", atoms=cand)
        self.assertFalse(self.compare()["passed"], "Large integers must not be rounded through float64")

    def test_dtype_shape_keys_json_types_and_corruption_fail(self):
        self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        for arrays in ({"plddt": np.array([0.7, 0.8], dtype=np.float64)},
                       {"plddt": np.array([[0.7, 0.8]], dtype=np.float32)},
                       {"wrong_name": np.array([0.7, 0.8], dtype=np.float32)}):
            np.savez(right / "plddt_complex_model_0.npz", **arrays)
            self.assertFalse(self.compare()["passed"])
        (right / "plddt_complex_model_0.npz").write_bytes(b"PK\x03\x04bad zip")
        self.assertFalse(self.compare()["passed"])
        right = self.write_unit(self.candidate)
        for content in ('{"confidence_score": true}', '{"confidence_score": NaN}',
                        '{"confidence_score": 0.85, "confidence_score": 0.85}', 'not JSON'):
            (right / "confidence_complex_model_0.json").write_text(content)
            self.assertFalse(self.compare()["passed"])

    def test_failure_logs_fail_even_with_identical_artifacts(self):
        self.write_unit(self.reference)
        self.write_unit(self.candidate)
        log = self.root / "run.log"
        for line in ("Number of failed examples: 1", "[boltz2-opt] FAILED item=complex seed=0 reason=oom",
                     "[boltz2-opt] EXIT mode=fast predictions=1 ok=1 failed=0 rc=5"):
            log.write_text(line)
            self.assertFalse(self.compare(candidate_log=log)["passed"])
        log.write_text("Number of failed examples: 0\n[boltz2-opt] EXIT mode=fast predictions=1 ok=1 failed=0 rc=0\n")
        self.assertTrue(self.compare(candidate_log=log)["passed"])

    def test_explicit_exclusions_are_recorded(self):
        self.write_unit(self.reference)
        right = self.write_unit(self.candidate)
        np.savez(right / "structure_cache_complex.npz", array=np.array([0]))
        self.assertFalse(self.compare()["passed"])
        report = self.compare(ignore=["structure_cache_*.npz"])
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["ignored"]["candidate"]), 1)

    def test_cli_machine_readable_report_and_exit_codes(self):
        self.write_unit(self.reference)
        self.write_unit(self.candidate)
        report_path = self.root / "report.json"
        command = [sys.executable, str(SCRIPT), str(self.reference), str(self.candidate), "--report", str(report_path)]
        success = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertTrue(json.loads(report_path.read_text())["passed"])
        self.write_unit(self.candidate, score=0.1)
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
        self.assertFalse(json.loads(report_path.read_text())["passed"])
        self.assertEqual(subprocess.run(command + ["--atol", "nan"], capture_output=True).returncode, 2)

    def test_expected_manifest_validation(self):
        path = self.root / "expected.json"
        valid = [{"record": "complex", "seed": 7, "models": [0, 1], "affinity": True}]
        path.write_text(json.dumps(valid))
        self.assertEqual(compare._read_expected(path), valid)
        for bad in ([], {"record": "complex"}, [{"record": "complex"}],
                    [{"record": "complex", "seed": True}],
                    [{"record": "complex", "seed": 7, "models": []}], valid + valid):
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                compare._read_expected(path)


if __name__ == "__main__":
    unittest.main()
