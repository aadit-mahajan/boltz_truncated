#!/usr/bin/env python3
"""Compare Boltz prediction artifacts without importing Boltz, Torch, or pickle.

Examples::

    python scripts/compare_predictions.py stock/ optimized/ \
        --reference-seed 42 --atol 1e-5 --rtol 1e-4 --report comparison.json
    python scripts/compare_predictions.py original/ changed/ --exact

Recognizes ``predictions/<record>/`` (including boltz_results_* parents) and
``by_seed/<record>/s<seed>/``. Stock output does not encode its random seed: supply
--reference-seed/--candidate-seed when comparing it to seeded kit output. Two
unlabelled stock runs can be compared, but their seeds cannot be verified.

JSON and NPZ are compared numerically, with exact shape, key, dtype, integer,
boolean, and string checks. Non-finite values fail, even if equal on both sides.
CIF/PDB default to exact hashes. With --structure-rmsd-atol ANGSTROM, gemmi is
required and atom identities must match in order. The gate checks all-atom aligned
RMSD and, when present, non-polymer atom RMSD after fitting polymer atoms. This
detects ligand movement relative to the receptor, unlike an all-atom fit alone.
Water, ions and other non-polymers enter this second metric too. These metrics do
not establish scientific equivalence or account for atom/residue symmetry.
Models are paired by output rank, not by diffusion sample identity.

For completeness beyond the files present, pass --expected-units with a JSON list
like [{"record": "complex", "seed": 42, "models": [0, 1], "affinity": true}].
Each listed model must have a structure; affinity=true also requires affinity JSON.
Use --reference-log/--candidate-log to detect recorded stock/kit prediction failures.
Logs and an expected input roster are necessary to detect failures absent on BOTH
sides. Keep checkpoints, inputs, MSA, seeds, sampling settings and hardware fixed;
this tool neither runs predictions nor measures speed or biological accuracy.
Exit codes: 0 pass, 1 mismatch/invalid artifact, 2 invalid command/configuration.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np


def _identity(record, seed, model, artifact):
    return {"record": record, "seed": seed, "model": model, "artifact": artifact}


def _artifact(record, name):
    match = re.fullmatch(rf"(?:(.*?)_)?{re.escape(record)}_model_(\d+)(\.[^.]+)", name)
    if match:
        return int(match[2]), (match[1] or "structure") + match[3]
    # Per-record affinity and optional cache/embedding files have no model rank.
    return None, name


def discover(root, seed=None, ignore=()):
    """Return artifacts keyed by (record, seed, rank, kind); never guess a seed."""
    root = Path(root).resolve()
    files, issues, ignored = {}, [], []
    if not root.is_dir():
        return files, [f"Not a directory: {root}"], ignored
    dirs = [root, *(p for p in root.rglob("*") if p.is_dir())]
    for directory in sorted(dirs):
        if directory.parent.name == "predictions":
            record, unit_seed = directory.name, seed
        elif directory.parent.parent.name == "by_seed":
            match = re.fullmatch(r"s(-?\d+)", directory.name)
            if not match:
                issues.append(f"Invalid seed directory: {directory}")
                continue
            record, unit_seed = directory.parent.name, int(match[1])
        else:
            continue
        children = sorted(p for p in directory.iterdir() if p.is_file())
        if not children:
            issues.append(f"Empty prediction directory: {directory}")
        for path in children:
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in ignore):
                ignored.append(str(path))
                continue
            model, artifact = _artifact(record, path.name)
            key = (record, unit_seed, model, artifact)
            if key in files:
                issues.append(f"Ambiguous duplicate artifact {key}: {files[key]} and {path}")
            else:
                files[key] = path
    if not files:
        issues.append(f"No prediction artifacts found in {root}")
    # Confidence/PAE-only leftovers are not successful predictions.
    units = {(record, unit_seed) for record, unit_seed, _, _ in files}
    for record, unit_seed in sorted(units, key=str):
        models = {m for r, s, m, _ in files if (r, s) == (record, unit_seed) and m is not None}
        if not models:
            issues.append(f"No structure models for record={record!r} seed={unit_seed}")
        for model in models:
            if not any((record, unit_seed, model, "structure" + suffix) in files for suffix in (".cif", ".pdb", ".npz")):
                issues.append(f"Missing structure for record={record!r} seed={unit_seed} model={model}")
    return files, issues, ignored


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_array(reference, candidate, path, atol, rtol):
    if reference.shape != candidate.shape:
        return [{"path": path, "error": "shape mismatch", "reference": list(reference.shape), "candidate": list(candidate.shape)}]
    if reference.dtype != candidate.dtype:
        return [{"path": path, "error": "dtype mismatch", "reference": str(reference.dtype), "candidate": str(candidate.dtype)}]
    if reference.dtype.hasobject:
        return [{"path": path, "error": "object arrays are forbidden"}]
    if reference.dtype.names:
        return [issue for name in reference.dtype.names for issue in _numeric_array(reference[name], candidate[name], path + "." + name, atol, rtol)]
    if reference.dtype.kind not in "fc":
        return [] if np.array_equal(reference, candidate) else [{"path": path, "error": "non-floating values differ"}]
    return _floating(reference, candidate, path, atol, rtol)


def _floating(reference, candidate, path, atol, rtol):
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        return [{"path": path, "error": "non-finite numeric values"}]
    # Cast before subtraction to avoid float16 overflow. isclose's reference is
    # its second argument, so relative tolerance is based on the baseline.
    dtype = np.complex128 if np.iscomplexobj(reference) else np.float64
    ref, cand = np.asarray(reference, dtype=dtype), np.asarray(candidate, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(cand - ref)
        close = np.isclose(cand, ref, atol=atol, rtol=rtol, equal_nan=False)
    maximum = float(np.max(difference)) if difference.size else 0.0
    return [{"path": path, "numeric": True, "passed": bool(close.all()),
             "count": int(difference.size), "mismatched": int(np.count_nonzero(~close)),
             "max_abs_error": maximum if math.isfinite(maximum) else "overflow"}]


def _json_compare(reference, candidate, path, atol, rtol):
    # bool is an int subclass; treat it as a separate, exact JSON type.
    if isinstance(reference, bool) or isinstance(candidate, bool):
        return [] if type(reference) is type(candidate) and reference == candidate else [{"path": path, "error": "boolean/type mismatch"}]
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        if isinstance(reference, int) and isinstance(candidate, int):
            return [] if reference == candidate else [{"path": path, "error": "integer mismatch"}]
        return _floating(reference, candidate, path, atol, rtol)
    if type(reference) is not type(candidate):
        return [{"path": path, "error": "JSON type mismatch"}]
    if isinstance(reference, dict):
        if reference.keys() != candidate.keys():
            return [{"path": path, "error": "JSON keys differ", "missing": sorted(reference.keys() - candidate.keys()), "extra": sorted(candidate.keys() - reference.keys())}]
        return [issue for key in sorted(reference) for issue in _json_compare(reference[key], candidate[key], path + "." + key, atol, rtol)]
    if isinstance(reference, list):
        if len(reference) != len(candidate):
            return [{"path": path, "error": "JSON list lengths differ"}]
        return [issue for index, (ref, cand) in enumerate(zip(reference, candidate)) for issue in _json_compare(ref, cand, f"{path}[{index}]", atol, rtol)]
    return [] if reference == candidate else [{"path": path, "error": "JSON values differ"}]


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _structure_atoms(path):
    try:
        import gemmi
    except ImportError as error:
        raise ValueError("Coordinate comparison requires gemmi; install gemmi or omit --structure-rmsd-atol") from error
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    identities, coordinates, polymer = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    identities.append((model.num, chain.name, residue.seqid.num, residue.seqid.icode,
                                       residue.name, residue.het_flag, atom.name, atom.altloc, atom.element.name))
                    coordinates.append((atom.pos.x, atom.pos.y, atom.pos.z))
                    polymer.append(residue.entity_type == gemmi.EntityType.Polymer)
    if not coordinates:
        raise ValueError(f"No atoms in structure: {path}")
    if len(set(identities)) != len(identities):
        raise ValueError(f"Duplicate atom identities in structure: {path}")
    coords = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError(f"Non-finite structure coordinates: {path}")
    return identities, coords, np.asarray(polymer, dtype=bool)


def _align(reference, candidate, mask):
    """Kabsch fit, avoiding reflection; transform all candidate atoms."""
    ref_center, cand_center = reference[mask].mean(axis=0), candidate[mask].mean(axis=0)
    ref_fit, cand_fit = reference[mask] - ref_center, candidate[mask] - cand_center
    if len(ref_fit) < 3 or np.linalg.matrix_rank(ref_fit) < 2 or np.linalg.matrix_rank(cand_fit) < 2:
        raise ValueError("At least three non-collinear fit atoms are required for a stable structure alignment")
    u, _, vt = np.linalg.svd(cand_fit.T @ ref_fit)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return (candidate - cand_center) @ (u @ correction @ vt) + ref_center


def _rmsd(reference, candidate):
    displacement = np.linalg.norm(reference - candidate, axis=1)
    return float(np.sqrt(np.mean(displacement ** 2))), float(displacement.max())


def _structure_compare(reference, candidate, rmsd_atol, max_atom_error):
    left_ids, left, left_polymer = _structure_atoms(reference)
    right_ids, right, right_polymer = _structure_atoms(candidate)
    if left_ids != right_ids or not np.array_equal(left_polymer, right_polymer):
        return [{"error": "Atom identities/order or polymer classifications differ", "reference_atoms": len(left_ids), "candidate_atoms": len(right_ids)}]
    all_atoms = np.ones(len(left), dtype=bool)
    aligned = _align(left, right, all_atoms)
    rmsd, maximum = _rmsd(left, aligned)
    raw_rmsd, _ = _rmsd(left, right)
    metrics = {"path": "structure", "coordinate_metrics": True, "units": "angstrom",
               "atom_count": len(left), "unaligned_rmsd": raw_rmsd,
               "aligned_all_atom_rmsd": rmsd, "max_aligned_atom_displacement": maximum,
               "rmsd_atol": rmsd_atol, "max_atom_error": max_atom_error,
               "passed": rmsd <= rmsd_atol and (max_atom_error is None or maximum <= max_atom_error)}
    if left_polymer.any() and (~left_polymer).any():
        receptor_aligned = _align(left, right, left_polymer)
        ligand_rmsd, ligand_max = _rmsd(left[~left_polymer], receptor_aligned[~left_polymer])
        metrics.update(polymer_atom_count=int(left_polymer.sum()), nonpolymer_atom_count=int((~left_polymer).sum()),
                       nonpolymer_rmsd_after_polymer_alignment=ligand_rmsd,
                       max_nonpolymer_displacement_after_polymer_alignment=ligand_max)
        metrics["passed"] &= ligand_rmsd <= rmsd_atol and (max_atom_error is None or ligand_max <= max_atom_error)
    return [metrics]


def compare_file(reference, candidate, *, atol, rtol, exact, structure_rmsd_atol=None, structure_max_atom_error=None):
    result = {"reference": str(reference), "candidate": str(candidate)}
    try:
        left_hash, right_hash = _hash(reference), _hash(candidate)
        result.update(reference_sha256=left_hash, candidate_sha256=right_hash, byte_identical=left_hash == right_hash)
        # Even identical bytes must be inspected: corrupt archives, pickled object
        # arrays, invalid JSON, and NaNs cannot pass the validation gate.
        if reference.suffix == ".json":
            left = json.loads(reference.read_text(), object_pairs_hook=_no_duplicate_keys)
            right = json.loads(candidate.read_text(), object_pairs_hook=_no_duplicate_keys)
            details = _json_compare(left, right, "$", atol, rtol)
            result["comparison"] = "json"
        elif reference.suffix == ".npz":
            # Own the file handles even when malformed ZIP construction fails.
            with reference.open("rb") as left_stream, candidate.open("rb") as right_stream, \
                    np.load(left_stream, allow_pickle=False) as left, np.load(right_stream, allow_pickle=False) as right:
                if len(set(left.files)) != len(left.files) or len(set(right.files)) != len(right.files):
                    raise ValueError("Duplicate NPZ array names")
                if set(left.files) != set(right.files):
                    details = [{"error": "NPZ keys differ", "missing": sorted(set(left.files) - set(right.files)), "extra": sorted(set(right.files) - set(left.files))}]
                else:
                    details = [issue for key in sorted(left.files) for issue in _numeric_array(left[key], right[key], key, atol, rtol)]
            result["comparison"] = "npz"
        elif reference.suffix in (".cif", ".pdb") and structure_rmsd_atol is not None:
            details = _structure_compare(reference, candidate, structure_rmsd_atol, structure_max_atom_error)
            result["comparison"] = "coordinates"
        else:
            details = [] if left_hash == right_hash else [{"error": "file bytes differ; no tolerant structural comparison implemented"}]
            result["comparison"] = "sha256"
        result["details"] = details
        result["passed"] = all(d.get("passed", "error" not in d) for d in details) and (not exact or left_hash == right_hash)
    except (OSError, ValueError, TypeError, OverflowError, EOFError, RuntimeError, zipfile.BadZipFile) as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    return result


def _expected_issues(files, expected):
    issues = []
    for unit in expected:
        record, seed = unit["record"], unit["seed"]
        for model in unit.get("models", [0]):
            if not any((record, seed, model, "structure" + suffix) in files for suffix in (".cif", ".pdb", ".npz")):
                issues.append(f"Expected structure missing: record={record!r} seed={seed} model={model}")
        if unit.get("affinity", False) and (record, seed, None, f"affinity_{record}.json") not in files:
            issues.append(f"Expected affinity missing: record={record!r} seed={seed}")
    return issues


def _log_issues(path):
    if path is None:
        return []
    issues = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        stock = re.search(r"Number of failed examples:\s*(\d+)", line)
        kit = "[boltz2-opt]" in line and ("FAILED item=" in line or "SKIPPED item=" in line)
        failed = re.search(r"\bfailed=(\d+)\b", line) if "[boltz2-opt] EXIT " in line else None
        rc = re.search(r"\brc=(-?\d+)\b", line) if "[boltz2-opt] EXIT " in line else None
        if kit or (stock and int(stock[1])) or (failed and int(failed[1])) or (rc and int(rc[1])):
            issues.append(f"Failure in {path}:{line_number}: {line.strip()}")
    return issues


def compare_predictions(reference, candidate, *, reference_seed=None, candidate_seed=None,
                        atol=1e-5, rtol=1e-4, exact=False, ignore=(), expected=(),
                        reference_log=None, candidate_log=None, structure_rmsd_atol=None,
                        structure_max_atom_error=None):
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol, structure_rmsd_atol, structure_max_atom_error) if v is not None):
        raise ValueError("Tolerances must be finite and nonnegative")
    if structure_max_atom_error is not None and structure_rmsd_atol is None:
        raise ValueError("--structure-max-atom-error requires --structure-rmsd-atol")
    left, left_issues, left_ignored = discover(reference, reference_seed, ignore)
    right, right_issues, right_ignored = discover(candidate, candidate_seed, ignore)
    left_issues += _expected_issues(left, expected) + _log_issues(reference_log)
    right_issues += _expected_issues(right, expected) + _log_issues(candidate_log)
    missing = [_identity(*key) for key in sorted(left.keys() - right.keys(), key=str)]
    extra = [_identity(*key) for key in sorted(right.keys() - left.keys(), key=str)]
    comparisons = []
    for key in sorted(left.keys() & right.keys(), key=str):
        comparisons.append({**_identity(*key), **compare_file(left[key], right[key], atol=atol, rtol=rtol, exact=exact,
                             structure_rmsd_atol=structure_rmsd_atol, structure_max_atom_error=structure_max_atom_error)})
    report = {
        "schema_version": 1,
        "passed": not (left_issues or right_issues or missing or extra) and bool(comparisons) and all(row["passed"] for row in comparisons),
        "mode": "exact" if exact else ("tolerant_numeric_coordinates" if structure_rmsd_atol is not None else "tolerant_numeric_exact_structure"),
        "tolerances": {"atol": atol, "rtol": rtol, "structure_rmsd_atol_angstrom": structure_rmsd_atol,
                       "structure_max_atom_error_angstrom": structure_max_atom_error},
        "reference_root": str(Path(reference).resolve()), "candidate_root": str(Path(candidate).resolve()),
        "unlabelled_seed_assignments": {"reference": reference_seed, "candidate": candidate_seed},
        "reference_issues": left_issues, "candidate_issues": right_issues,
        "missing": missing, "extra": extra, "comparisons": comparisons,
        "ignored": {"patterns": list(ignore), "reference": left_ignored, "candidate": right_ignored},
        "coverage": {"expected_units_supplied": bool(expected), "reference_log_supplied": reference_log is not None, "candidate_log_supplied": candidate_log is not None},
        "limitations": ["CIF/PDB use exact hashes unless coordinate tolerance is explicitly enabled; atom/residue symmetry is not resolved.",
                        "Coordinate mode checks atom identity/order and coordinates, not every mmCIF metadata field; confidence NPZ/JSON remain separate checks.",
                        "Non-polymer displacement after polymer fitting includes ligands, ions and water; individual ligand identity is not inferred.",
                        "Model numbers are output confidence ranks, not diffusion sample identities.",
                        "Failures absent from both output trees require an expected input roster or failure logs.",
                        "Artifact agreement does not establish biological accuracy, speedup, or matching run settings."],
    }
    report["summary"] = {"paired_files": len(comparisons), "byte_identical_files": sum(row.get("byte_identical", False) for row in comparisons),
                         "failed_files": sum(not row["passed"] for row in comparisons), "missing_files": len(missing), "extra_files": len(extra)}
    return report


def _read_expected(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=_no_duplicate_keys)
    if not isinstance(value, list) or not value:
        raise ValueError("Expected units must be a nonempty JSON list")
    seen = set()
    for unit in value:
        if not isinstance(unit, dict) or not isinstance(unit.get("record"), str) or not unit["record"] or "seed" not in unit:
            raise ValueError("Each expected unit needs a nonempty record and a seed (integer or null)")
        if unit["seed"] is not None and type(unit["seed"]) is not int:
            raise ValueError("Expected seed must be an integer or null")
        models = unit.get("models", [0])
        if not isinstance(models, list) or not models or any(type(m) is not int or m < 0 for m in models):
            raise ValueError("Expected models must be a nonempty list of nonnegative integer ranks")
        if type(unit.get("affinity", False)) is not bool:
            raise ValueError("Expected affinity must be boolean")
        key = (unit["record"], unit["seed"])
        if key in seen:
            raise ValueError(f"Duplicate expected unit: {key}")
        seen.add(key)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--candidate-seed", type=int)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--structure-rmsd-atol", type=float, help="Enable gemmi coordinate comparison; maximum aligned RMSD in angstroms (global and non-polymer after polymer fitting)")
    parser.add_argument("--structure-max-atom-error", type=float, help="Optional maximum aligned atom displacement in angstroms; requires --structure-rmsd-atol")
    parser.add_argument("--exact", action="store_true", help="Require matching SHA-256 for every artifact, in addition to validity checks")
    parser.add_argument("--ignore-glob", action="append", default=[], help="Explicit filename exclusion (repeatable); recorded in report")
    parser.add_argument("--expected-units", type=Path)
    parser.add_argument("--reference-log", type=Path)
    parser.add_argument("--candidate-log", type=Path)
    parser.add_argument("--report", type=Path, help="Write JSON here (default: stdout)")
    args = parser.parse_args(argv)
    try:
        expected = _read_expected(args.expected_units) if args.expected_units else ()
        report = compare_predictions(args.reference, args.candidate, reference_seed=args.reference_seed,
                                     candidate_seed=args.candidate_seed, atol=args.atol, rtol=args.rtol,
                                     exact=args.exact, ignore=args.ignore_glob, expected=expected,
                                     reference_log=args.reference_log, candidate_log=args.candidate_log,
                                     structure_rmsd_atol=args.structure_rmsd_atol, structure_max_atom_error=args.structure_max_atom_error)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report:
            args.report.write_text(encoded)
        else:
            sys.stdout.write(encoded)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
