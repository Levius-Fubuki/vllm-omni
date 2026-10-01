# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

from benchmarks.diffusion.compare_helios_attention import compare_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_exact_outputs_have_json_safe_psnr():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    metrics = compare_metrics(video, video.copy())
    assert metrics == {
        "exact": True,
        "mae": 0.0,
        "max_abs": 0.0,
        "rmse": 0.0,
        "psnr_db": None,
        "temporal_delta_mae": 0.0,
    }


def test_constant_offset_has_known_metrics():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    metrics = compare_metrics(video, video + 0.25)
    assert not metrics["exact"]
    assert metrics["mae"] == metrics["rmse"] == metrics["max_abs"] == 0.25
    assert metrics["psnr_db"] == pytest.approx(12.041199826559248)
    assert metrics["temporal_delta_mae"] == 0.0


def test_temporal_difference():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    changed = video.copy()
    changed[1] = 0.5
    assert compare_metrics(video, changed)["temporal_delta_mae"] == 0.5


def test_rejects_invalid_outputs():
    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="matching FHWC"):
        compare_metrics(video, video[:, :, :, :1])
    invalid = video.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        compare_metrics(video, invalid)


def sample_runs():
    from benchmarks.diffusion.compare_helios_attention import BACKENDS

    return {
        name: {
            "metadata": {
                "frames": [33],
                "seeds": [42],
                "repeats": 1,
                "backend": name,
                "attention_implementations": {"provider": name},
            },
            "records": [{"warmup": False, "num_frames": 33, "seed": 42, "repeat": 0, "prompt": "train"}],
        }
        for name in BACKENDS
    }


def test_contract_allows_backend_specific_bindings():
    from benchmarks.diffusion.compare_helios_attention import validate_runs

    assert len(validate_runs(sample_runs())) == 3


@pytest.mark.parametrize("change", ["duplicate", "missing", "environment", "prompt"])
def test_contract_rejects_incomparable_runs(change):
    from benchmarks.diffusion.compare_helios_attention import validate_runs

    runs = sample_runs()
    run = runs["FLASH_ATTN"]
    if change == "duplicate":
        run["records"] *= 2
    elif change == "missing":
        run["records"] = []
    elif change == "environment":
        run["metadata"]["torch"] = "different"
    else:
        run["records"][0]["prompt"] = "beach"
    with pytest.raises(ValueError):
        validate_runs(runs)


def test_export_excludes_warmup_and_raw_timelines():
    from benchmarks.diffusion.export_helios_attention import PROVENANCE_KEYS, RECORD_KEYS, build_artifact

    runs = sample_runs()
    for run in runs.values():
        run["metadata"].update(model="/local/model", engine_startup_ms=1)
        run["summary"] = {"33": {"count": 1}}
        row = run["records"][0]
        row.update({key: 1 for key in RECORD_KEYS if key not in row})
        row.update(transformer_timings=[{"gpu_ms": 1}], stage_durations_ms={"decode": 1})
        run["records"].append({"warmup": True})
    provenance = {key: "historical" for key in PROVENANCE_KEYS}
    artifact = build_artifact(runs, {"alignment": "example"}, provenance)
    assert artifact["benchmark_script_sha256"] == "historical"
    assert "model" not in artifact["metadata"]
    assert artifact["prompts_by_seed"] == {"42": "train"}
    for backend in artifact["backends"].values():
        assert len(backend["measurements"]) == 1
        assert set(backend["measurements"][0]) == set(RECORD_KEYS)


def test_ssim_matches_exact_frames_when_available():
    pytest.importorskip("skimage.metrics")
    from benchmarks.diffusion.compare_helios_attention import compare

    video = np.zeros((2, 8, 8, 3), dtype=np.float32)
    result = compare(video, video.copy())
    assert result["ssim_mean"] == 1.0
    assert result["psnr_db"] is None
