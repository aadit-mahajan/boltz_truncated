"""Real tensor regression tests for the fork's SDPA and sample batching changes."""
import pytest
import torch

from boltz.model.layers.attentionv2 import AttentionPairBias
from boltz.model.modules.diffusionv2 import _get_sample_id_chunks


@pytest.mark.parametrize("batch,multiplicity", [(1, 1), (2, 1), (2, 3)])
@pytest.mark.parametrize("precomputed_bias", [False, True])
def test_sdpa_matches_eager_fp32_with_padding_and_pair_bias(batch, multiplicity, precomputed_bias):
    torch.manual_seed(317)
    heads, width, length = 4, 32, 9
    module = AttentionPairBias(width, 8, heads, compute_pair_bias=not precomputed_bias).eval()
    # The production initialization zeros this projection. Use nonzero weights
    # so a broken attention implementation cannot pass with all-zero outputs.
    with torch.no_grad():
        module.proj_o.weight.normal_(std=0.1)
    s = torch.randn(batch * multiplicity, length, width)
    keys = torch.randn_like(s)
    z = torch.randn(batch, length, length, heads if precomputed_bias else 8)
    mask = torch.ones(batch * multiplicity, length)
    mask[:, -2:] = 0
    with torch.no_grad():
        expected = module(s, z, mask, keys, multiplicity=multiplicity)
        result = module(s, z, mask, keys, multiplicity=multiplicity, use_flash_attn=True)
    assert expected.abs().max() > 0.01
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("batch,multiplicity,parallel", [(1, 5, 2), (2, 5, 2), (3, 4, 3), (2, 5, 5)])
def test_diffusion_chunks_cover_each_record_sample_once(batch, multiplicity, parallel):
    chunks = _get_sample_id_chunks(batch, multiplicity, parallel, torch.device("cpu"))
    ids = torch.cat(chunks)
    assert sorted(ids.tolist()) == list(range(batch * multiplicity))
    for chunk in chunks:
        assert len(chunk) <= batch * parallel
        records = (chunk // multiplicity).reshape(batch, -1)
        assert torch.equal(records, torch.arange(batch)[:, None].expand_as(records))
