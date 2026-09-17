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
    "cdk2": ("2019-12-13_cdk2", "GPLGSMENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL"),
    "tyk2": ("2020-02-07_tyk2", "MGSPASDPTVFHKRYLKKIRDLGEGHFGKVSLYCYDPTNDGTGEMVAVKALKADAGPQHRSGWKQEIDILRTLYHEHIIKYKGCCEDAGAASLQLVMEYVPLGSLRDYLPRHSIGLAQLLLFAQQICEGMAYLHAQHYIHRNLAARNVLLDNDRLVKIGDFGLAKAVPEGHEYYRVREDGDSPVFWYAPECLKEYKFYYASDVWSFGVTLYELLTHCDSSQSPPTKFLELIGIAQGQMTVLRLTELLERGERLPRPDKCPAEVYHLMKNCWETEASFRPTFENLIPILKTVHEKYRHHHHHH"),
    "jnk1": ("2019-09-23_jnk1", "MSRSKRDNNFYSVEIGDSTFTVLKRYQNLKPIGSGAQGIVCAAYDAILERNVAIKKLSRPFQNQTHAKRAYRELVLMKCVNHKNIIGLLNVFTPQKSLEEFQDVYIVMELMDANLCQVIQMELDHERMSYLLYQMLCGIKHLHSAGIIHRDLKPSNIVVKSDCTLKILDFGLARTAGTSFMMEPEVVTRYYRAPEVILGMGYKENVDLWSVGCIMGEMVCHKILFPGRDYIDQWNKVIEQLGTPCPEFMKKLQPTVRTYVENRPKYAGYSFEKLFPDVLFPADSEHNKLKASQARDLLSKMLVIDASKRISVDEALQHPYINVWYDPSEAEAPPPKIPDKQLDEREHTIEEWKELIYKEVMDLEHHHHHH"),
    "p38": ("2019-12-09_p38", "MRGSHHHHHHGSMSQERPTFYRQELNKTIWEVPERYQNLSPVGSGAYGSVCAAFDTKTGLRVAVKKLSRPFQSIIHAKRTYRELRLLKHMKHENVIGLLDVFTPARSLEEFNDVYLVTHLMGADLNNIVKCQKLTDDHVQFLIYQILRGLKYIHSADIIHRDLKPSNLAVNEDCELKILDFGLARHTDDEMTGYVATRWYRAPEIMLNWMHYNQTVDIWSVGCIMAELLTGRTLFPGTDHIDQLKLILRLVGTPGAELLKKISSESARNYIQSLTQMPKMNFANVFIGANPLAVDLLEKMLVLDSDKRITAAQALAHAYFAQYHDPDDEPVADPYDQSFESRDLLIDEWKSLTYDEVISFVPPPLDQEEMES"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plb_data", type=Path, help="PLB v0.2.1 data directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "inputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, (source_name, sequence) in TARGETS.items():
        source = args.plb_data / source_name
        labels = yaml.safe_load((source / "00_data/ligands.yml").read_text())
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
