# Public FEP+4 benchmark inputs

This directory contains the 87 protein-ligand inputs across CDK2 (16), TYK2
(16), JNK1 (21), and p38 (34) used as the public four-target source referred
to in the Boltz-2 paper. Inputs are derived from v0.2.1 of the
[Open Force Field protein-ligand benchmark](https://github.com/openforcefield/protein-ligand-benchmark/tree/0.2.1).

`experimental_labels.csv` retains the published measurement value, unit, and
type. The Boltz-2 paper standardizes these measurements into log micromolar
units, but individual assay values should be compared within target/assay.

This is a reproducible public reconstruction, not an official Boltz-held-out
archive: the official exact split was not released with the model.
