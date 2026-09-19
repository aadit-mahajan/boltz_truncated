# Boltz-2 fork: optimized inference and latent reuse

A fork of [Boltz-2](https://github.com/jwohlwend/boltz) 2.2.1 that makes inference faster in
two separable ways. The upstream pin is in `optimizations.lock.json`.

## 1. Optimization profiles (`--opt_profile`)

Levers applied inside this fork's model, on any CUDA GPU of compute capability 8.0 or newer:

```bash
boltz predict input.yaml --out_dir out --opt_profile exact   # default; identical to stock
boltz predict input.yaml --out_dir out --opt_profile fast    # SDPA, fused projections, TF32
boltz predict input.yaml --out_dir out --opt_profile off     # stock Boltz-2, for comparison
```

`exact` skips work whose result is already determined — eval dropout masks that are all ones,
an all-dummy template stack that contributes exactly zero, the parts of the diffusion
conditioning and the atom/token glue that the sampler rebuilds identically at every step, and
weight initialization that the strict checkpoint load immediately overwrites. `fast` adds SDPA attention through the
diffusion sampler (which stock runs as a float32 `einsum` 24 times per sampling step), folds
the conditioning's 30 LayerNorm→Linear projections into 3, and turns on TF32.

Every lever, what it changes and why it is safe: **`OPTIMIZATIONS.md`**.

## 2. Structure-cache affinity (`--experimental_structure_cache`)

Off by default. Reuses the structure pass's trunk state and rank-zero pose for the affinity
pass, so that pass skips its own trunk, diffusion and confidence work:

```bash
boltz predict complex.yaml --out_dir out --opt_profile fast --experimental_structure_cache
```

This is an alternative inference procedure, not an accelerated identical one — cropping a
full-complex trunk state is not a trunk evaluation on the crop. Validate affinity numbers
against the standard path before relying on them. Every cache is bound by digest to the pose
it was written for, and affinity outputs record which procedure produced them.

## Testing

```bash
pip install -e '.[test]'
pytest                                     # CPU only; no GPU or weights needed
```

`tests/test_opt_levers.py` and `tests/test_diffusion_sampler_opt.py` assert that `exact` is
bitwise `off` — including a complete diffusion roll-out with uneven sample chunks — and bound
each fast lever against the path it replaces. `tests/test_structure_cache*.py` cover the
cache's identity checks and its windowed reads.

`notebooks/boltz2_optimized_colab.ipynb` runs the GPU side end to end in Colab: profile
benchmarks against `off`, artifact comparison through `scripts/compare_predictions.py`, and
the structure-cache affinity validation.

## Comparing runs

```bash
python scripts/compare_predictions.py out_off/ out_fast/ --structure-rmsd-atol 0.5
```

Compares predictions without importing Boltz, Torch or pickle; `--exact` requires matching
SHA-256 for every artifact.
