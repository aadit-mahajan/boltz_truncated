# Inference optimizations in this fork

Two independent things make this fork faster than stock Boltz-2, and they compose:

1. **Optimization levers inside the model** — selected with `--opt_profile`, applied to
   every prediction. Documented below.
2. **The experimental structure cache** — `--experimental_structure_cache`, which replaces
   the affinity pass's trunk, diffusion and confidence work with the structure pass's own
   trunk state and selected pose. This is an *alternative inference procedure*, not an
   accelerated identical one; see "Structure cache" below.

## Profiles

```
boltz predict input.yaml --out_dir out --opt_profile fast
```

| profile | levers | outputs |
|---|---|---|
| `off` | none | stock Boltz-2 |
| `exact` (default) | `resid` `templ_skip` `dit_hoist` `atom_hoist` `ctorskip` `cache_io` | identical arithmetic to `off` |
| `fast` | `exact` plus `flash_attn` `condproj` `tf32` `cache_fp16` | within Boltz-2's own seed-to-seed variation |

`--disable_opt name[,name...]` switches individual levers off inside the chosen profile, so a
single change can be A/B'd against the rest. An unknown name is a usage error rather than a
silently ignored one. `BOLTZ_OPT_PROFILE` and `BOLTZ_OPT_DISABLE` carry the selection to
dataloader and prediction workers; the CLI writes them, so they rarely need setting by hand.

Every run prints its selection once:

```
[boltz-opt] profile=fast levers=atom_hoist cache_fp16 cache_io condproj ctorskip dit_hoist flash_attn resid templ_skip tf32
```

## Exact levers

These skip or hoist work whose result is already determined. `tests/test_opt_levers.py` and
`tests/test_diffusion_sampler_opt.py` assert bitwise equality against `--opt_profile off`,
including a whole diffusion roll-out with uneven sample chunks.

### `resid` — eval dropout is the identity
`get_dropout_mask` multiplies by `training`, so in eval it builds a `[B, N, N, 1]` tensor of
exact `1.0`s and broadcasts it over the pair representation. Seventeen call sites use it only
as `mask * update`. The lever returns a sentinel whose `__mul__` is the identity, removing
four allocations and four broadcast multiplies over `[B, N, N, 128]` per Pairformer layer,
across 48 trunk layers, every template and MSA layer, and every recycling pass.

### `templ_skip` — an all-dummy template stack contributes +0.0
With no template supplied, Boltz-2 still featurizes one dummy slot. Both template modules
weight `v` by the template mask, sum over the template axis, and project through a
bias-free `u_proj`, so the update is exactly zero. The lever skips the module — its template
Pairformer blocks over `[B, T, N, N, 64]` — once per recycling pass. `feats["template_mask"]`
is checked per prediction, so an input that *does* carry templates runs the module normally.

### `dit_hoist` — the noise-independent half of the single conditioning
`SingleConditioning` computes `single_embed(norm_single(cat(s_trunk, s_inputs)))` before
adding the Fourier embedding of the noise level. That half does not depend on the diffusion
step, but stock recomputes it on every one of the sampling steps, for every sample chunk.
The sampler now evaluates it once per distinct chunk width and reuses it across the
roll-out. Replicating the inputs *before* the projection keeps the GEMM shape, and therefore
its reduction order, exactly what the per-step call used — which is what makes this bitwise.

### `atom_hoist` — the atom/token glue the sampler rebuilds every step
The atom encoder and decoder replicate their conditioning to the sample axis on every
diffusion step: `q`, `c`, the atom pad mask, and `atom_to_token` — a dense `[atoms, tokens]`
matrix — plus the row-normalized `atom_to_token_mean` the encoder divides out. None of those
depend on the noise level. A cache created per roll-out, and destroyed with it, holds them
instead, so the sampler builds each once rather than once per step. Consumers get the same
tensor object, which is what makes this bitwise; nothing writes to them in place.

A tensor larger than `MEMO_MAX_BYTES` (256 MiB) is rebuilt per step instead of held, so a
very large complex cannot be pushed out of memory by the cache — declining costs only the
work stock already does.

### `ctorskip` — weights the checkpoint is about to overwrite are not sampled
Constructing Boltz-2 initializes every weight in the model — `nn.Linear`'s Kaiming uniform for
each of a few thousand layers, plus four truncated normals sampled through scipy — and the
`strict=True` checkpoint load then writes over all of it. Inside
`boltz.model.layers.initialize.skip_parameter_init()` those writes do not happen; the
parameters come out of `torch.empty` and the load fills them.

Measured on the 48-block Pairformer stack alone (147 M parameters): **4.28 s to 0.02 s**. A
prediction run pays this twice, once for the structure model and once for the affinity model,
so it is one of the larger end-to-end savings here even though it changes no prediction time.

Safety rests entirely on the load being strict: strict loading fails by name on any parameter
the checkpoint does not supply, so an uninitialized tensor cannot reach a forward pass. The
context manager is therefore used at exactly the two `load_from_checkpoint(strict=True)` call
sites in `boltz/main.py` and nowhere else. It must never wrap a model that will be trained
from scratch.

### `cache_io` — the structure cache is stored, not deflated
`np.savez_compressed` on `z` (`[N, N, 192]` float32) spends seconds of single-threaded zlib
per record and compresses trunk activations barely at all. The cache is written with
`np.savez` instead, which also makes every member byte-addressable: the loader seeks to the
rows the affinity crop names and reads only those, rather than pulling the whole array
through the page cache to keep at most 256 tokens of it. A cache whose layout is not
row-addressable (an older compressed one, say) still loads, by reading whole arrays.

## Fast levers

Small, documented numeric differences. `tests/test_opt_levers.py` bounds each against the
stock path it replaces.

### `flash_attn` — SDPA for every attention, including the sampler
The fork already routed `AttentionPairBias` through `torch.nn.functional.scaled_dot_product_attention`
in the trunk. The diffusion transformer did not accept the switch, so its 24 layers ran the
float32 `einsum` path — materializing a `[B·samples, heads, N, N]` float32 score tensor and
its softmax — once per layer per sampling step. That is the single largest tensor traffic in
a structure prediction. `use_flash_attn` now threads through `DiffusionModule`,
`DiffusionTransformer`, `AtomTransformer` and both atom encoder/decoder stacks. The pair bias
becomes SDPA's additive `attn_mask`, so PyTorch serves it from the memory-efficient backend.
Numerics: SDPA reduces in its own order and in the module's dtype.

`--flash_attn` remains an independent flag, so SDPA can be used under `exact` without the
other fast levers. Boltz-2 disables it below compute capability 8.0, where PyTorch would fall
back to the math backend and lose to the existing float32 path.

### `condproj` — the diffusion conditioning's 30 projections as 3
`DiffusionConditioning` builds the sampler's attention biases with 24 + 3 + 3 independent
`LayerNorm -> Linear(bias=False)` pairs, each reading the whole pair (or atom-pair) tensor and
each contributing a few channels to a concatenation. A LayerNorm's affine part is a
per-channel scale, so it folds into the Linear that follows:

```
Linear_i(LayerNorm_i(x)) = x_hat @ (W_i * w_i).T + (W_i @ b_i)
```

`x_hat`, the affine-free normalization, is the same tensor for every layer in a stack. One
normalization pass and one GEMM against the stacked weights therefore reproduce the
concatenated output — 30 passes over the largest tensor in the model become 3. The folded
weights are derived from checkpoint parameters, so they are built on first use and cached per
(device, dtype); they are deliberately not registered as buffers, which would add keys the
strict checkpoint load rejects. Numerics: `W_i * w_i` is rounded once at build time instead of
being applied to activations, and the GEMM sums in its own order.

### `tf32` — tensor cores for the remaining float32 GEMMs
`torch.backends.cuda.matmul.allow_tf32`, `cudnn.allow_tf32` and
`set_float32_matmul_precision("high")`, plus the expandable-segments CUDA allocator, which
has to be requested before CUDA initializes and so is set before `torch` is imported. TF32
keeps float32 range with 10 mantissa bits on the multiply inputs.

### `cache_fp16` — float16 trunk state in the cache
`s` and `z` are stored as float16, halving the cache again on top of `cache_io`. Coordinates
stay float32: the affinity head reads them as geometry. A trunk state that would overflow
float16 falls back to float32 for that record. Everything is returned as float32.

## Structure cache

`--experimental_structure_cache` is a change of inference procedure, not an optimization of
an existing one, and it is off by default.

Stock Boltz-2 runs the affinity pass as a second, independent prediction: it re-crops the
complex around the ligand, runs the trunk again with 5 recycling steps on the crop, runs the
diffusion sampler again, and only then the affinity head. With the cache, the structure pass
writes its own trunk state and its rank-zero pose; the affinity pass crops those and feeds
them straight to the affinity head. The trunk, diffusion and confidence work of the affinity
pass disappears.

What it is *not*: cropping a full-complex trunk state does not reproduce a trunk evaluation
on the crop, and the cached state comes from the structure checkpoint while the affinity head
belongs to the affinity checkpoint. Affinity numbers from this path need their own validation
against the standard path before being trusted — `notebooks/boltz2_optimized_colab.ipynb`
runs exactly that comparison.

The cache is bound to the pose it was written for. Every load checks the format version, the
record id, and a SHA-256 of the `pre_affinity_<record>.npz` the pose was written to; a rerun
that produces a different pose invalidates it by digest. Full token and atom axes are checked
against the freshly tokenized structure from the stored array headers, before any array data
is read. Affinity outputs record which procedure produced them
(`"affinity_inference_mode"`), and a rerun will not reuse an output produced by the other one.

Caches written by an older format are rejected by name; regenerate them with `--override`.

## What is deliberately not here

These are the remaining ideas from Anthropic's published Boltz-2 optimization kit
(`anthropics/uplifting-biomolecular-modeling`), and why each is absent:

* **Fused triangle attention, triangle multiplication, transition, outer-product-mean,
  pair-weighted-averaging and atom encoder/decoder kernels.** The kit ships these as Triton and
  CUDA-native (sm_90a / sm_80) kernels, selected from a per-card lookup table and numerically
  self-checked on first use. They are the kit's largest single gain and they cannot be written
  in PyTorch. This fork instead routes the attentions it can through
  `scaled_dot_product_attention` (`flash_attn`), including the triangle attentions and, now,
  the diffusion stack.
* **CUDA-graph capture** of the Pairformer stacks and of the diffusion roll-out. Achievable in
  PyTorch, but capture requires static shapes and pinned buffers throughout, and neither the
  capture nor its headroom gating can be validated without a GPU. Shipping it untested would be
  worse than not shipping it.
* **A single resident bf16 pair tensor** (`pairfuse`) threaded through every C=128 pair stack.
  This is a restructuring of the whole pair track, not a lever.
* **Row-chunked evaluation and lazy frees for very large inputs** (`xl_trans`, `xl_cond`,
  `xl_free`, `relpos_lazy`) and **row-sharded multi-GPU** (`--n_gpu`). Memory levers for inputs
  beyond one GPU; nothing here addresses that case yet.
* **Background writer and featurizer prefetch** (`writer_overlap`, `prefetch`). Pure Python, and
  a real gain for multi-input runs, but they add a process hand-over with its own transport and
  deadlines. Worth doing next if you batch many ligands.
* **The `mask2` shortcut** — skipping the all-zero padding-mask term. Inside the kit's fused
  kernels this costs nothing to omit; in PyTorch, deciding it is safe needs a device sync per
  call, and the term it saves is one broadcast add next to an N x N x d attention. Not worth it
  here.

The kit reaches those through byte-pinned stock upstream in its own interpreter with a pinned
CUDA 13 stack and a 580-series driver, which also means it cannot carry this fork's structure
cache. That trade is why this repository ports the portable levers rather than wrapping the kit.
