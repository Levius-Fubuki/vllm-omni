# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-shape checks for MammothModa2's fused QK norm/RoPE and shared CUDA
attention backend against the pre-change bf16 arithmetic."""

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
import vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model as mammoth_dit
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import TransformerBlock

from .test_dit_attention import _reference_attention

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.advanced_model,
    pytest.mark.cuda,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

# MammothModa2-Preview gen_dit_config: hidden 2520, 21 heads, 7 KV heads, head_dim 120.
DIM, HEADS, KV_HEADS = 2520, 21, 7


def _production_rotary(batch, seq, head_dim, dtype):
    """Mammoth's real RoPE repeats one angle across each adjacent pair."""
    angles = torch.rand(batch, seq, head_dim // 2, device="cuda") * 6.283
    return (
        angles.cos().repeat_interleave(2, dim=-1).to(dtype),
        angles.sin().repeat_interleave(2, dim=-1).to(dtype),
    )


def test_production_rope_uses_fused_qk_norm_rope(monkeypatch):
    torch.manual_seed(0)
    seq = 512
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(2, seq, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, seq, dtype=torch.bool, device="cuda")
    mask[0, -37:] = False
    rotary = _production_rotary(1, seq, block.head_dim, torch.bfloat16)
    calls = []
    fused_qk_norm_rope = mammoth_dit.fused_qk_norm_rope

    def record_fused(q, k, q_weight, k_weight, rope_table, eps, **kwargs):
        calls.append((q.shape, k.shape, rope_table.shape, kwargs))
        return fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, **kwargs)

    monkeypatch.setenv("VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS", "0")
    monkeypatch.setattr(mammoth_dit, "fused_qk_norm_rope", record_fused)
    with torch.no_grad():
        got = block.attn(hidden, hidden, attention_mask=mask, image_rotary_emb=rotary)
        want = _reference_attention(block.attn, hidden, mask, rotary)

    assert calls == [
        (
            torch.Size((2 * seq, HEADS, block.head_dim)),
            torch.Size((2 * seq, KV_HEADS, block.head_dim)),
            torch.Size((2 * seq, block.head_dim)),
            {"head_dim": block.head_dim, "rotary_dim": block.head_dim, "interleaved": True},
        )
    ]
    diff = (got.float() - want.float()).abs()[mask]
    assert diff.max().item() < 2e-2, diff.max().item()
    assert diff.mean().item() < 1e-3, diff.mean().item()


@pytest.mark.parametrize("seq", [77 + 4096, 512], ids=["t2i_1024", "short"])
def test_real_shape_matches_previous_arithmetic_bf16(seq):
    torch.manual_seed(0)
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(2, seq, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, seq, dtype=torch.bool, device="cuda")
    mask[0, seq - 300 :] = False
    mask[1, seq - 17 :] = False
    rotary = _production_rotary(1, seq, block.head_dim, torch.bfloat16)
    with torch.no_grad():
        got = block.attn(
            hidden_states=hidden, encoder_hidden_states=hidden, attention_mask=mask, image_rotary_emb=rotary
        )
        want = _reference_attention(block.attn, hidden, mask, rotary)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[~mask]) == 0
    diff = (got.float() - want.float()).abs()[mask]
    # bf16 kernels differ in accumulation order; measured on A800: max 1.0e-3, mean 5e-5.
    assert diff.max().item() < 2e-2, diff.max().item()
    assert diff.mean().item() < 1e-3, diff.mean().item()


def test_empty_text_stream_on_the_default_backend():
    """The recipe's text-to-image request with text_guidance_scale > 1 runs the
    context refiner on a zero-token unconditional prompt. Before the guard this
    raised ``RuntimeError: step must be nonzero`` from the FA varlen fallback."""
    torch.manual_seed(0)
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5, modulation=False)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    hidden = torch.randn(1, 0, DIM, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(1, 0, dtype=torch.bool, device="cuda")
    angles = torch.rand(1, 0, block.head_dim, device="cuda")
    with torch.no_grad():
        out = block.attn(
            hidden_states=hidden,
            encoder_hidden_states=hidden,
            attention_mask=mask,
            image_rotary_emb=(angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16)),
        )
    assert out.shape == (1, 0, DIM)


@pytest.mark.parametrize(
    ("seq", "force_gqa_fallback"),
    [(512, False), (77 + 4096, False), (32, True)],
    ids=["short", "t2i_1024", "unsupported_native_gqa"],
)
def test_fp32_falls_back_to_sdpa_and_matches_reference(monkeypatch, seq, force_gqa_fallback):
    """FP32 keeps its dtype and reaches shared SDPA's GQA compatibility check.

    Cover the long masked shape that otherwise falls back to quadratic math
    attention, and force the unsupported-GQA branch independently of hardware.
    """
    torch.manual_seed(0)
    dt = torch.float32
    block = (
        TransformerBlock(DIM, HEADS, KV_HEADS, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5)
        .cuda()
        .to(dt)
        .eval()
    )
    hidden = torch.randn(1, seq, DIM, device="cuda", dtype=dt)
    mask = torch.ones(1, seq, dtype=torch.bool, device="cuda")
    mask[:, -13:] = False
    angles = torch.rand(1, seq, block.head_dim, device="cuda")
    rotary = (angles.cos().to(dt), angles.sin().to(dt))
    capabilities, calls, phases = [], [], []
    can_use_fused_gqa = sdpa_backend.can_sdpa_use_fused_gqa
    sdpa = torch.nn.functional.scaled_dot_product_attention
    strategy = block.attn.omni_attn._get_active_parallel_strategy()
    pre_attention, post_attention = strategy.pre_attention, strategy.post_attention

    def prepare(*args):
        phases.append("pre")
        return pre_attention(*args)

    def restore(*args):
        phases.append("post")
        return post_attention(*args)

    def check_gqa(*args):
        supported = False if force_gqa_fallback else can_use_fused_gqa(*args)
        capabilities.append(supported)
        return supported

    def record_sdpa(query, key, value, **kwargs):
        phases.append("sdpa")
        calls.append((query.shape, key.shape, value.shape, query.dtype, kwargs))
        return sdpa(query, key, value, **kwargs)

    def reject_fused(*args, **kwargs):
        raise AssertionError("FP32 must keep Mammoth's native QK norm/RoPE path")

    with torch.no_grad(), monkeypatch.context() as patch:
        patch.setattr(mammoth_dit, "fused_qk_norm_rope", reject_fused)
        patch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", check_gqa)
        patch.setattr(torch.nn.functional, "scaled_dot_product_attention", record_sdpa)
        patch.setattr(strategy, "pre_attention", prepare)
        patch.setattr(strategy, "post_attention", restore)
        got = block.attn(hidden, hidden, attention_mask=mask, image_rotary_emb=rotary)

    assert phases == ["pre", "sdpa", "post"], "FP32 must preserve the shared parallel dispatch"
    assert len(capabilities) == len(calls) == 1, "FP32 must use shared SDPA's runtime GQA check"
    q_shape, k_shape, v_shape, dtype, kwargs = calls[0]
    expected_kv_heads = KV_HEADS if capabilities[0] else HEADS
    assert q_shape == (1, HEADS, seq, block.head_dim)
    assert k_shape == v_shape == (1, expected_kv_heads, seq, block.head_dim)
    assert dtype == got.dtype == torch.float32
    assert kwargs["enable_gqa"] == capabilities[0]
    assert kwargs["scale"] == block.attn.scale
    assert kwargs["is_causal"] is False
    assert torch.equal(kwargs["attn_mask"], mask[:, None, None, :])

    with torch.no_grad():
        want = _reference_attention(block.attn, hidden, mask, rotary)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[~mask]) == 0
    diff = (got.float() - want.float()).abs()[mask]
    assert diff.max().item() < 1e-3, diff.max().item()
