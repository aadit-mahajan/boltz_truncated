"""Each lever against the stock path it replaces, on real tensors.

The exact levers must be bitwise; the fast levers must stay within a tolerance
far tighter than Boltz-2's seed-to-seed variation.
"""

import os

import numpy as np
import pytest
import torch

from boltz import opt
from boltz.model.layers.dropout import IDENTITY_MASK, get_dropout_mask
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
from boltz.model.modules.encodersv2 import SingleConditioning
from boltz.model.modules.transformersv2 import DiffusionTransformer


# --------------------------------------------------------------------------
# profile plumbing


def test_profiles_are_nested_and_names_validated():
    assert opt.PROFILES["off"] < opt.PROFILES["exact"] < opt.PROFILES["fast"]
    with pytest.raises(opt.ProfileError):
        opt.resolve("turbo")
    with pytest.raises(opt.ProfileError):
        opt.resolve("fast", ("not_a_lever",))
    assert "condproj" not in opt.resolve("fast", ("condproj",))


def test_disabling_one_lever_leaves_the_rest(monkeypatch):
    monkeypatch.setenv("BOLTZ_OPT_DISABLE", "")
    levers = opt.configure("fast", ("flash_attn",))
    assert not opt.enabled("flash_attn")
    assert opt.enabled("condproj") and opt.enabled("resid")
    # Re-resolving from the environment must reproduce the same selection.
    assert opt.configure() == levers


# --------------------------------------------------------------------------
# resid


def test_eval_dropout_mask_is_the_identity_and_allocates_nothing():
    z = torch.randn(2, 5, 5, 3)
    opt.configure("exact")
    mask = get_dropout_mask(0.25, z, training=False)
    assert mask is IDENTITY_MASK
    update = torch.randn_like(z)
    assert mask * update is update
    assert update * mask is update

    opt.configure("off")
    stock = get_dropout_mask(0.25, z, training=False)
    assert torch.is_tensor(stock)
    torch.testing.assert_close(stock, torch.ones_like(stock), rtol=0, atol=0)
    # The stock mask is exactly ones, so both residuals are bitwise equal.
    assert torch.equal(stock * update, update)


def test_training_dropout_is_untouched_by_the_lever():
    z = torch.randn(1, 4, 4, 2)
    for profile in ("off", "exact", "fast"):
        opt.configure(profile)
        torch.manual_seed(5)
        mask = get_dropout_mask(0.25, z, training=True)
        assert torch.is_tensor(mask) and mask.shape == (1, 4, 1, 1)


# --------------------------------------------------------------------------
# dit_hoist


def test_hoisted_single_conditioning_is_bitwise():
    torch.manual_seed(19)
    module = SingleConditioning(sigma_data=16.0, token_s=12, dim_fourier=8).eval()
    s_trunk = torch.randn(2, 6, 12)
    s_inputs = torch.randn(2, 6, 12)
    times = torch.randn(6)
    multiplicity = 3

    with torch.no_grad():
        stock, _ = module(
            times,
            s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs.repeat_interleave(multiplicity, 0),
        )
        # The sampler replicates before embedding, keeping the GEMM shape and
        # therefore the reduction order the per-step call would have used.
        hoisted, _ = module(
            times,
            None,
            None,
            module.embed_trunk(
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            ),
        )
    assert stock.abs().max() > 0.01
    assert torch.equal(stock, hoisted)


# --------------------------------------------------------------------------
# condproj


def _conditioning():
    torch.manual_seed(3)
    module = DiffusionConditioning(
        token_s=16, token_z=24, atom_s=8, atom_z=12,
        atom_encoder_depth=2, atom_encoder_heads=2,
        token_transformer_depth=4, token_transformer_heads=2,
        atom_decoder_depth=2, atom_decoder_heads=2,
    ).eval()
    # The checkpoint's LayerNorm affines are not the identity; defaults would
    # let a fusion that drops the scale or bias pass.
    with torch.no_grad():
        for stack in (module.token_trans_proj_z, module.atom_enc_proj_z, module.atom_dec_proj_z):
            for layer in stack:
                layer[0].weight.normal_(1.0, 0.3)
                layer[0].bias.normal_(0.0, 0.3)
                layer[1].weight.normal_(0.0, 0.5)
    return module


@pytest.mark.parametrize(
    "stack_name,shape",
    [("token_trans", (1, 7, 7, 24)), ("atom_enc", (1, 5, 9, 12)), ("atom_dec", (1, 5, 9, 12))],
)
def test_fused_projection_stack_matches_the_loop(stack_name, shape):
    module = _conditioning()
    layers = getattr(module, f"{stack_name}_proj_z")
    x = torch.randn(*shape)
    with torch.no_grad():
        opt.configure("off")
        stock = module._project(layers, stack_name, x)
        opt.configure("fast")
        fused = module._project(layers, stack_name, x)
    assert stock.shape == fused.shape == (*shape[:-1], len(layers) * layers[0][1].out_features)
    assert stock.abs().max() > 0.01
    torch.testing.assert_close(fused, stock, rtol=1e-5, atol=1e-6)


def test_fusion_cache_survives_a_dtype_change():
    module = _conditioning()
    opt.configure("fast")
    x = torch.randn(1, 4, 4, 24)
    with torch.no_grad():
        first = module._project(module.token_trans_proj_z, "token_trans", x)
        module.double()
        second = module._project(module.token_trans_proj_z, "token_trans", x.double())
    torch.testing.assert_close(second.float(), first, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# flash_attn through the diffusion stack


@pytest.mark.parametrize("multiplicity", [1, 3])
def test_diffusion_transformer_sdpa_matches_eager(multiplicity):
    torch.manual_seed(23)
    depth, heads, dim, batch, length = 2, 4, 32, 2, 7
    transformer = DiffusionTransformer(depth=depth, heads=heads, dim=dim).eval()
    # Every layer's output projection is zero-initialized in production, which
    # would hide a wrong attention result behind an all-zero residual.
    with torch.no_grad():
        for layer in transformer.layers:
            layer.output_projection_linear.weight.normal_(std=0.1)
            layer.pair_bias_attn.proj_o.weight.normal_(std=0.1)

    a = torch.randn(batch * multiplicity, length, dim)
    s = torch.randn(batch * multiplicity, length, dim)
    bias = torch.randn(batch, length, length, depth * heads)
    mask = torch.ones(batch * multiplicity, length)
    mask[:, -2:] = 0

    with torch.no_grad():
        eager = transformer(a, s, bias=bias, mask=mask, multiplicity=multiplicity)
        sdpa = transformer(
            a, s, bias=bias, mask=mask, multiplicity=multiplicity, use_flash_attn=True
        )
    assert (eager - a).abs().max() > 0.01
    torch.testing.assert_close(sdpa, eager, rtol=2e-5, atol=2e-6)


# --------------------------------------------------------------------------
# templ_skip


@pytest.mark.parametrize("module_name", ["TemplateModule", "TemplateV2Module"])
def test_all_dummy_template_update_is_exactly_zero(module_name):
    """What the templ_skip lever relies on: no real template, no contribution.

    Both template modules average over the template axis weighted by the
    template mask and then project through a bias-free Linear, so an all-dummy
    input yields exactly +0.0 and skipping the module changes nothing.
    """
    import boltz.model.modules.trunkv2 as trunkv2
    from boltz.data import const

    torch.manual_seed(41)
    tokens, templates, token_z, template_dim = 5, 2, 8, 6
    module = getattr(trunkv2, module_name)(
        token_z=token_z, template_dim=template_dim, template_blocks=1,
        pairwise_head_width=2, pairwise_num_heads=2,
    ).eval()
    # A zero-initialized projection would make any input give zero.
    with torch.no_grad():
        module.u_proj.weight.normal_(std=0.5)

    feats = {
        "asym_id": torch.zeros(1, tokens, dtype=torch.long),
        "template_restype": torch.zeros(1, templates, tokens, const.num_tokens),
        "template_frame_rot": torch.eye(3).expand(1, templates, tokens, 3, 3).contiguous(),
        "template_frame_t": torch.randn(1, templates, tokens, 3),
        "template_mask_frame": torch.ones(1, templates, tokens),
        "template_cb": torch.randn(1, templates, tokens, 3),
        "template_ca": torch.randn(1, templates, tokens, 3),
        "template_mask_cb": torch.ones(1, templates, tokens),
        # No template is real: this is what a no-template input featurizes to.
        "template_mask": torch.zeros(1, templates, tokens),
    }
    if module_name == "TemplateV2Module":
        feats["visibility_ids"] = torch.zeros(1, templates, tokens)

    z = torch.randn(1, tokens, tokens, token_z)
    pair_mask = torch.ones(1, tokens, tokens)
    with torch.no_grad():
        update = module(z, feats, pair_mask)
    assert torch.equal(update, torch.zeros_like(update))


# --------------------------------------------------------------------------
# atom_hoist


def test_memo_returns_the_same_object_and_declines_oversized_tensors():
    from boltz.model.modules import utils

    cache = {}
    calls = []

    def build():
        calls.append(1)
        return torch.arange(6.0)

    first = utils.memo(cache, "k", build)
    second = utils.memo(cache, "k", build)
    assert first is second and len(calls) == 1
    # A consumer must see the identical tensor, not an equal one.
    assert torch.equal(second, torch.arange(6.0))

    # No cache means no memoization at all.
    assert utils.memo(None, "k", build) is not first
    assert len(calls) == 2

    big = torch.empty(utils.MEMO_MAX_BYTES // 4 + 8, dtype=torch.float32)
    assert utils.memo(cache, "big", lambda: big) is big
    assert "big" not in cache, "an oversized tensor must not be held for the roll-out"


def test_atom_hoist_is_a_switch():
    assert "atom_hoist" in opt.PROFILES["exact"]
    opt.configure("exact", ("atom_hoist",))
    assert not opt.enabled("atom_hoist")
    # An omitted argument inherits from the environment, which is how workers
    # pick the selection up; passing it explicitly is what clears it.
    opt.configure("exact")
    assert not opt.enabled("atom_hoist")
    opt.configure("exact", ())
    assert opt.enabled("atom_hoist")


# --------------------------------------------------------------------------
# cache_io and tf32


def test_cache_io_switch_selects_the_storage_form(tmp_path):
    from boltz.data.structure_cache import _StoredNpz, save_structure_cache

    structure = tmp_path / "pose.npz"
    structure.write_bytes(b"pose")
    arrays = {
        "s": np.zeros((4, 3), dtype=np.float32),
        "z": np.zeros((4, 4, 3), dtype=np.float32),
        "coords": np.zeros((5, 3), dtype=np.float32),
    }
    for profile, windowed in (("off", False), ("exact", True)):
        opt.configure(profile)
        path = tmp_path / f"{profile}.npz"
        save_structure_cache(path, record_id="r", structure_path=structure, **arrays)
        with _StoredNpz(path) as cache:
            assert cache.is_windowed() is windowed


def test_runtime_knobs_only_fire_for_cuda_under_fast():
    opt.configure("exact")
    assert opt.apply_runtime_knobs("gpu") == []
    opt.configure("fast")
    assert opt.apply_runtime_knobs("cpu") == []
    assert opt.apply_runtime_knobs("mps") == []
    # 'gpu' is the CLI's spelling of what torch calls 'cuda'; without a CUDA
    # device both return nothing rather than raising.
    if not torch.cuda.is_available():
        assert opt.apply_runtime_knobs("gpu") == []


def test_allocator_request_respects_an_explicit_setting(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    opt.configure("fast")
    assert opt.prepare_allocator() is True
    assert "expandable_segments" in os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    assert opt.prepare_allocator() is False
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    opt.configure("exact")
    assert opt.prepare_allocator() is False


# --------------------------------------------------------------------------
# ctorskip


def test_skip_parameter_init_restores_every_patched_initializer():
    from boltz.model.layers import initialize

    originals = {name: getattr(initialize, name) for name in initialize._OWN_INITIALIZERS}
    resets = {cls: cls.reset_parameters for cls in initialize._RESET_PARAMETER_TYPES}

    opt.configure("exact")
    with initialize.skip_parameter_init():
        assert all(
            getattr(initialize, name) is not originals[name]
            for name in initialize._OWN_INITIALIZERS
        )
    for name, function in originals.items():
        assert getattr(initialize, name) is function
    for cls, reset in resets.items():
        assert cls.reset_parameters is reset

    # An exception inside the block must not leave torch patched either.
    with pytest.raises(RuntimeError):
        with initialize.skip_parameter_init():
            raise RuntimeError("boom")
    assert torch.nn.Linear.reset_parameters is resets[torch.nn.Linear]


def test_skip_parameter_init_is_a_no_op_when_the_lever_is_off():
    from boltz.model.layers import initialize

    opt.configure("off")
    with initialize.skip_parameter_init():
        layer = torch.nn.Linear(8, 8)
    # Off means stock: the layer is initialized as torch would.
    assert layer.weight.abs().max() > 0
    assert torch.isfinite(layer.weight).all()


def test_a_strict_load_fills_everything_ctorskip_left_alone():
    """The premise of the lever: strict loading defines every parameter."""
    from boltz.model.layers import initialize

    opt.configure("off")
    reference = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.LayerNorm(4))
    state = reference.state_dict()

    opt.configure("exact")
    with initialize.skip_parameter_init():
        built = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.LayerNorm(4))
    built.load_state_dict(state, strict=True)

    for name, parameter in built.named_parameters():
        assert torch.equal(parameter, state[name]), name
    x = torch.randn(3, 6)
    torch.testing.assert_close(built(x), reference(x), rtol=0, atol=0)


def test_strict_load_still_refuses_a_missing_parameter():
    """Uninitialized memory can only escape if strict loading stops catching it."""
    from boltz.model.layers import initialize

    opt.configure("exact")
    with initialize.skip_parameter_init():
        built = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.LayerNorm(4))
    incomplete = {k: v for k, v in built.state_dict().items() if "1." not in k}
    with pytest.raises(RuntimeError, match="Missing key"):
        built.load_state_dict(incomplete, strict=True)
