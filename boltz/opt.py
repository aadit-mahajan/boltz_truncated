"""Inference optimization levers for this Boltz-2 fork.

A *lever* is one independently switchable change to the inference path. Levers
are grouped into profiles so a run selects them with a single flag, and every
active lever is printed once so a result can be traced back to the code that
produced it.

Two classes of lever exist:

``exact``
    Provably identical arithmetic to stock Boltz-2 on the same inputs: work
    that is skipped because its result is the identity (an all-ones eval
    dropout mask, an all-dummy template update of ``+0.0``) or hoisted out of a
    loop that recomputed it unchanged.

``fast``
    Documented numeric differences: fused projections re-associate a sum,
    SDPA reduces in its own order, TF32 truncates GEMM operands, and the
    structure cache stores float16. Differences are of the same order as
    Boltz-2's own seed-to-seed variation, but outputs are not bitwise equal.

The profile is process-global because several levers sit inside leaf modules
(``get_dropout_mask``) that have no route to the model object. It is resolved
once, at CLI parse time, and never mutated mid-run.
"""

from __future__ import annotations

import os

#: Levers whose arithmetic is identical to stock Boltz-2.
EXACT_LEVERS = (
    # get_dropout_mask returns None in eval: the mask is exactly all-ones, so
    # both the [B,N,N,1] allocation and the multiply are skipped.
    "resid",
    # An all-dummy template stack contributes u_proj(relu(0)) == 0 (no bias).
    "templ_skip",
    # SingleConditioning's trunk half does not depend on the diffusion step;
    # it is computed once per prediction instead of once per step per sample.
    "dit_hoist",
    # The atom<->token glue the atom encoder and decoder rebuild every step is
    # the same tensor at every noise level; it is built once per roll-out.
    "atom_hoist",
    # Structure cache written uncompressed and read back through a window, so
    # only the affinity crop's rows leave the disk.
    "cache_io",
    # Weight initialization a strict checkpoint load immediately overwrites is
    # not performed at all.
    "ctorskip",
)

#: Levers with small, documented numeric differences.
FAST_LEVERS = (
    # torch SDPA for every AttentionPairBias, including the diffusion stack.
    "flash_attn",
    # DiffusionConditioning's 24 + 3 + 3 LayerNorm->Linear pairs as one
    # normalization and one GEMM each.
    "condproj",
    # TF32 tensor cores for the remaining fp32 GEMMs, and the expandable
    # CUDA allocator.
    "tf32",
    # float16 storage for the cached trunk state (coordinates stay float32).
    "cache_fp16",
)

PROFILES: dict[str, frozenset[str]] = {
    "off": frozenset(),
    "exact": frozenset(EXACT_LEVERS),
    "fast": frozenset(EXACT_LEVERS + FAST_LEVERS),
}

DEFAULT_PROFILE = "exact"

_ALL_LEVERS = frozenset(EXACT_LEVERS + FAST_LEVERS)

_state: dict[str, object] = {
    "profile": DEFAULT_PROFILE,
    "levers": PROFILES[DEFAULT_PROFILE],
}


class ProfileError(ValueError):
    """An unknown profile or lever name was requested."""


def resolve(profile: str, disable: tuple[str, ...] = ()) -> frozenset[str]:
    """Return the lever set for ``profile`` minus ``disable``, validating names."""
    if profile not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise ProfileError(f"Unknown optimization profile {profile!r}; choose from {known}")
    unknown = sorted(set(disable) - _ALL_LEVERS)
    if unknown:
        known = ", ".join(sorted(_ALL_LEVERS))
        raise ProfileError(f"Unknown lever(s) {', '.join(unknown)}; known levers: {known}")
    return PROFILES[profile] - set(disable)


def configure(
    profile: str | None = None, disable: tuple[str, ...] | None = None
) -> frozenset[str]:
    """Select the active profile for this process.

    ``BOLTZ_OPT_PROFILE`` and ``BOLTZ_OPT_DISABLE`` (comma separated) supply
    whichever argument the caller leaves out, so a dataloader or prediction
    worker started without the CLI runs what its parent chose. Both are written
    back exactly as given, never as the derived lever set: re-deriving from the
    profile keeps a later call to this function idempotent.
    """
    if profile is None:
        profile = os.environ.get("BOLTZ_OPT_PROFILE", DEFAULT_PROFILE)
    if disable is None:
        raw = os.environ.get("BOLTZ_OPT_DISABLE", "")
        disable = tuple(name.strip() for name in raw.split(",") if name.strip())
    levers = resolve(profile, disable)
    _state["profile"] = profile
    _state["levers"] = levers
    os.environ["BOLTZ_OPT_PROFILE"] = profile
    os.environ["BOLTZ_OPT_DISABLE"] = ",".join(disable)
    return levers


def enabled(lever: str) -> bool:
    """Whether ``lever`` is active. Unknown names raise rather than read False."""
    if lever not in _ALL_LEVERS:
        raise ProfileError(f"Unknown lever {lever!r}")
    return lever in _state["levers"]


def active_levers() -> frozenset[str]:
    return _state["levers"]  # type: ignore[return-value]


def active_profile() -> str:
    return _state["profile"]  # type: ignore[return-value]


def describe() -> str:
    """One line naming the profile and every lever that is on."""
    levers = sorted(active_levers())
    names = " ".join(levers) if levers else "none"
    return f"[boltz-opt] profile={active_profile()} levers={names}"


#: What the CLI's --accelerator calls an NVIDIA GPU, plus torch's own spelling.
_CUDA_NAMES = frozenset({"gpu", "cuda"})


def apply_runtime_knobs(device_type: str = "cuda") -> list[str]:
    """Apply the process-wide `tf32` lever. Returns the knobs that were set.

    Called once, after the device is known. Everything here is a global torch
    setting rather than a module change, so it is kept out of the model code.
    """
    applied: list[str] = []
    if not enabled("tf32") or device_type not in _CUDA_NAMES:
        return applied

    import torch

    if not torch.cuda.is_available():
        return applied
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    applied += ["cuda.matmul.allow_tf32", "cudnn.allow_tf32"]
    # set_float32_matmul_precision covers the ops that read the newer knob.
    torch.set_float32_matmul_precision("high")
    applied.append("float32_matmul_precision=high")
    return applied


def prepare_allocator() -> bool:
    """Request the expandable-segments CUDA allocator, if `tf32` is on.

    PyTorch parses this variable when its caching allocator first runs, so it
    must be set before any CUDA allocation and cannot be changed afterwards.
    An explicit ``PYTORCH_CUDA_ALLOC_CONF`` from the caller always wins.
    """
    if not enabled("tf32") or "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return True
configure()
