"""The diffusion sampler end to end, one profile against another.

These run the real ``AtomDiffusion.sample`` roll-out — chunking, the hoisted
conditioning, the SDPA switch and the alignment step — on a small but genuine
``DiffusionModule``. The conditioning tensors are supplied directly, so the
test exercises the sampler without building a whole Boltz-2.
"""

from functools import partial

import pytest
import torch

from boltz import opt
from boltz.model.modules.diffusionv2 import AtomDiffusion, _get_sample_id_chunks
from boltz.model.modules.encodersv2 import get_indexing_matrix, single_to_keys

TOKENS, ATOMS, TOKEN_S, ATOM_S = 3, 8, 8, 8
QUERIES, KEYS = 4, 8
ENC_DEPTH = DEC_DEPTH = 2
TOKEN_DEPTH, TOKEN_HEADS, ATOM_HEADS = 2, 2, 2


def _sampler():
    return AtomDiffusion(
        score_model_args=dict(
            token_s=TOKEN_S,
            atom_s=ATOM_S,
            atoms_per_window_queries=QUERIES,
            atoms_per_window_keys=KEYS,
            dim_fourier=16,
            atom_encoder_depth=ENC_DEPTH,
            atom_encoder_heads=ATOM_HEADS,
            token_transformer_depth=TOKEN_DEPTH,
            token_transformer_heads=TOKEN_HEADS,
            atom_decoder_depth=DEC_DEPTH,
            atom_decoder_heads=ATOM_HEADS,
        ),
        num_sampling_steps=4,
        alignment_reverse_diff=True,
        coordinate_augmentation_inference=False,
    ).eval()


def _inputs(batch=1):
    torch.manual_seed(101)
    windows = ATOMS // QUERIES
    feats = {
        "ref_pos": torch.randn(batch, ATOMS, 3),
        "atom_pad_mask": torch.ones(batch, ATOMS),
        "token_pad_mask": torch.ones(batch, TOKENS),
        "atom_to_token": torch.zeros(batch, ATOMS, TOKENS),
    }
    for atom in range(ATOMS):
        feats["atom_to_token"][:, atom, atom % TOKENS] = 1.0
    indexing = get_indexing_matrix(windows, QUERIES, KEYS, torch.device("cpu"))
    conditioning = {
        "q": torch.randn(batch, ATOMS, ATOM_S),
        "c": torch.randn(batch, ATOMS, ATOM_S),
        "to_keys": partial(single_to_keys, indexing_matrix=indexing, W=QUERIES, H=KEYS),
        "atom_enc_bias": torch.randn(batch, windows, QUERIES, KEYS, ENC_DEPTH * ATOM_HEADS),
        "atom_dec_bias": torch.randn(batch, windows, QUERIES, KEYS, DEC_DEPTH * ATOM_HEADS),
        "token_trans_bias": torch.randn(batch, TOKENS, TOKENS, TOKEN_DEPTH * TOKEN_HEADS),
    }
    return feats, conditioning


def _roll_out(sampler, profile, *, multiplicity, max_parallel_samples=None, seed=7):
    opt.configure(profile)
    feats, conditioning = _inputs()
    torch.manual_seed(seed)
    with torch.no_grad():
        out = sampler.sample(
            atom_mask=feats["atom_pad_mask"],
            multiplicity=multiplicity,
            max_parallel_samples=max_parallel_samples,
            s_inputs=torch.zeros(1, TOKENS, TOKEN_S),
            s_trunk=torch.zeros(1, TOKENS, TOKEN_S) + 0.5,
            feats=feats,
            diffusion_conditioning=conditioning,
        )
    return out["sample_atom_coords"]


@pytest.mark.parametrize(
    "multiplicity,max_parallel_samples", [(1, None), (4, None), (5, 2)]
)
def test_exact_profile_reproduces_stock_coordinates(multiplicity, max_parallel_samples):
    sampler = _sampler()
    stock = _roll_out(
        sampler, "off", multiplicity=multiplicity,
        max_parallel_samples=max_parallel_samples,
    )
    exact = _roll_out(
        sampler, "exact", multiplicity=multiplicity,
        max_parallel_samples=max_parallel_samples,
    )
    assert stock.shape == (multiplicity, ATOMS, 3)
    assert stock.abs().max() > 0.01
    assert torch.equal(stock, exact)


def test_uneven_chunks_keep_every_sample_hoisted_correctly():
    """A trailing chunk narrower than the rest gets its own cached embedding."""
    chunks = _get_sample_id_chunks(1, 5, 2, torch.device("cpu"))
    assert [chunk.numel() for chunk in chunks] == [2, 2, 1]
    sampler = _sampler()
    stock = _roll_out(sampler, "off", multiplicity=5, max_parallel_samples=2)
    exact = _roll_out(sampler, "exact", multiplicity=5, max_parallel_samples=2)
    assert torch.equal(stock, exact)


def test_fast_profile_stays_close_to_stock():
    sampler = _sampler()
    stock = _roll_out(sampler, "off", multiplicity=2)
    fast = _roll_out(sampler, "fast", multiplicity=2)
    torch.testing.assert_close(fast, stock, rtol=1e-4, atol=1e-4)
