# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark MammothModa2's Q/K RMSNorm + adjacent-pair real RoPE.

Example:

    CUDA_VISIBLE_DEVICES=0 python \
      benchmarks/diffusion/benchmark_mammoth_moda2_qk_norm_rope.py \
      --tokens 512 4173 8346 --warmup 50 --iters 200
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from collections.abc import Callable

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _fused_cuda_supported,
    fused_qk_norm_rope,
)
from vllm_omni.diffusion.models.mammoth_moda2.rope_real import (
    apply_real_rotary_emb,
)

_HEAD_DIM = 120
_Q_HEADS = 21
_KV_HEADS = 7
_EPS = 1e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 4173, 8346])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _measure(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    torch.accelerator.reset_peak_memory_stats()
    baseline_bytes = torch.accelerator.memory_allocated()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "extra_peak_mib": (torch.accelerator.max_memory_allocated() - baseline_bytes) / 2**20,
    }


def _run_shape(tokens: int, warmup: int, iters: int) -> dict[str, object]:
    dtype = torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(42)
    query = torch.randn(tokens, _Q_HEADS, _HEAD_DIM, device=device, dtype=dtype)
    key = torch.randn(tokens, _KV_HEADS, _HEAD_DIM, device=device, dtype=dtype)
    norm_q = Qwen2RMSNorm(_HEAD_DIM, eps=_EPS).to(device=device, dtype=dtype)
    norm_k = Qwen2RMSNorm(_HEAD_DIM, eps=_EPS).to(device=device, dtype=dtype)
    angles = torch.rand(tokens, _HEAD_DIM // 2, device=device) * (2 * torch.pi)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)

    def native() -> tuple[torch.Tensor, torch.Tensor]:
        q_out = apply_real_rotary_emb(norm_q(query).unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0)).squeeze(0)
        k_out = apply_real_rotary_emb(norm_k(key).unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0)).squeeze(0)
        return q_out, k_out

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        rope_table = torch.cat((cos[..., 0::2], sin[..., 0::2]), dim=-1)
        return fused_qk_norm_rope(
            query,
            key,
            norm_q.weight,
            norm_k.weight,
            rope_table,
            _EPS,
            head_dim=_HEAD_DIM,
            rotary_dim=_HEAD_DIM,
            interleaved=True,
        )

    expected_q, expected_k = native()
    actual_q, actual_k = fused()
    errors = {
        "q_max_abs": (actual_q.float() - expected_q.float()).abs().max().item(),
        "q_mean_abs": (actual_q.float() - expected_q.float()).abs().mean().item(),
        "k_max_abs": (actual_k.float() - expected_k.float()).abs().max().item(),
        "k_mean_abs": (actual_k.float() - expected_k.float()).abs().mean().item(),
    }
    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)

    native_stats = _measure(native, warmup, iters)
    fused_stats = _measure(fused, warmup, iters)
    return {
        "tokens": tokens,
        "native": native_stats,
        "fused_including_rope_pack": fused_stats,
        "median_speedup": native_stats["median_ms"] / fused_stats["median_ms"],
        "errors": errors,
    }


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    probe_q = torch.empty(1, _Q_HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    probe_k = torch.empty(1, _KV_HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    if not _fused_cuda_supported(probe_q, probe_k, _HEAD_DIM, _HEAD_DIM, interleaved=True):
        raise RuntimeError("The fused CUDA QK norm/RoPE path is unavailable")

    result = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": _version("triton"),
            "vllm": _version("vllm"),
            "vllm_omni": _version("vllm-omni"),
        },
        "config": {
            "dtype": "bfloat16",
            "q_heads": _Q_HEADS,
            "kv_heads": _KV_HEADS,
            "head_dim": _HEAD_DIM,
            "warmup": args.warmup,
            "iters": args.iters,
        },
        "results": [_run_shape(tokens, args.warmup, args.iters) for tokens in args.tokens],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
