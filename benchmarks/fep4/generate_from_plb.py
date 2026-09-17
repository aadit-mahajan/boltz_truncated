"""Build the public 87-complex FEP+4 Boltz input set from PLB v0.2.1.

The four targets are CDK2, TYK2, JNK1 and p38.  This is the public source
cited by the Boltz-2 paper; it is not an official Boltz release split.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

TARGETS = {
    "cdk2": "2019-12-13_cdk2",
    "tyk2": "2020-02-07_tyk2",
    "jnk1": "2019-09-23_jnk1",
    "p38": "2019-12-09_p38",
}
AA = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def protein_sequence(pdb: Path) -> str:
    seen, residues = set(), []
    for line in pdb.read_text().splitlines():
        if not line.startswith("ATOM") or line[21] != "A":
            continue
        residue = line[17:20].strip()
        key = (line[22:26], line[26])
        if key not in seen and residue in AA:
            seen.add(key)
            residues.append(AA[residue])
    return "".join(residues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plb_data", type=Path, help="PLB v0.2.1 data directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "inputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, source_name in TARGETS.items():
        source = args.plb_data / source_name
        labels = yaml.safe_load((source / "00_data/ligands.yml").read_text())
        sequence = protein_sequence(source / "01_protein/crd/protein.pdb")
        target_dir = args.out / target
        target_dir.mkdir(exist_ok=True)
        for ligand_id, ligand in labels.items():
            doc = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": sequence}},
                    {"ligand": {"id": "B", "smiles": ligand["smiles"]}},
                ],
                "properties": [{"affinity": {"binder": "B"}}],
            }
            (target_dir / f"{ligand_id}.yaml").write_text(
                yaml.safe_dump(doc, sort_keys=False)
            )
            measurement = ligand["measurement"]
            rows.append({
                "target": target,
                "input_id": f"{target}__{ligand_id}",
                "ligand_id": ligand_id,
                "value": measurement["value"],
                "unit": measurement["unit"],
                "measurement_type": measurement["type"],
            })
    with (args.out.parent / "experimental_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} inputs to {args.out}")


if __name__ == "__main__":
    main()
