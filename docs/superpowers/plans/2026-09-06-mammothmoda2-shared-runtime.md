# MammothModa2 Shared Diffusion Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the MammothModa2 DiT stage through vLLM-Omni's shared diffusion request-mode runtime while preserving the existing single-request AR-to-DiT conditioning and image output behavior.

**Architecture:** Keep stage 0 on the existing `LLM_AR` engine and change stage 1 to `DIFFUSION`, with a model-specific bridge that packages AR hidden states and token metadata into one diffusion prompt. The native diffusion pipeline is constructed from `OmniDiffusionConfig`, loads only the root checkpoint's `gen_*` weights, accepts `DiffusionRequestBatch`, and returns `DiffusionOutput`; shared runtime guards keep request batching and step execution disabled. No AR KV cache is transported because the current DiT consumes hidden states and token IDs only.

**Tech Stack:** Python 3.10+, PyTorch, vLLM, vLLM-Omni shared diffusion runtime, Diffusers VAE utilities, pytest, CUDA on NVIDIA A800.

---

## File map

- Modify `vllm_omni/model_executor/models/mammoth_moda2/pipeline.py`: declare stage 1 as the native diffusion stage and explicitly disable AR KV receipt.
- Modify `vllm_omni/diffusion/registry.py`: register `MammothModa2DiTPipeline` with the shared diffusion loader.
- Modify `vllm_omni/model_executor/models/registry.py`: remove the legacy generation-runner registration so the topology cannot silently select it.
- Modify `vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py`: replace the list-shaped `ar2dit` adapter with a single-request diffusion prompt adapter.
- Modify `vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py`: adopt native construction, request parsing, deterministic RNG, shared output, warmup, and root checkpoint loading.
- Keep `vllm_omni/model_executor/models/mammoth_moda2/pipeline_mammothmoda2_dit.py`: import-only compatibility shim; do not add runtime logic.
- Create `tests/config/test_mammoth_moda2_shared_runtime.py`: topology and registry regression coverage.
- Modify `tests/config/test_omni_config.py`: prove the structured stage config projects the root model path and request-mode limits into the diffusion backend.
- Create `tests/model_executor/stage_input_processors/test_mammoth_moda2.py`: AR-to-diffusion payload and failure coverage.
- Create `tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py`: constructor, request, RNG, warmup, output, and unsupported-mode coverage.
- Modify `tests/worker/test_omni_connector_mixin.py`: update the MammothModa2 processor symbol used by the connector compatibility assertion.
- Modify `tests/e2e/offline_inference/test_mammoth_moda2_expansion.py`: exercise native diffusion sampling params and unskip real-weight E2E.
- Modify `recipes/MammothModa2/MammothModa2.md`: document the shared runtime, standard sampling fields, A800 commands, and unsupported features.

### Task 1: Lock the topology and registry boundary

**Files:**
- Create: `tests/config/test_mammoth_moda2_shared_runtime.py`
- Modify: `tests/config/test_omni_config.py`
- Modify: `vllm_omni/model_executor/models/mammoth_moda2/pipeline.py:17-49`
- Modify: `vllm_omni/diffusion/registry.py:20-45`
- Modify: `vllm_omni/model_executor/models/registry.py:140-144`

- [ ] **Step 1: Write the failing topology and registry tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
from vllm_omni.model_executor.models.mammoth_moda2.pipeline import (
    MAMMOTH_MODA2_AR_PIPELINE,
    MAMMOTH_MODA2_PIPELINE,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS


def test_mammothmoda2_generation_stage_uses_shared_diffusion_runtime() -> None:
    ar_stage, dit_stage = MAMMOTH_MODA2_PIPELINE.stages

    assert ar_stage.execution_type is StageExecutionType.LLM_AR
    assert dit_stage.execution_type is StageExecutionType.DIFFUSION
    assert dit_stage.model_arch == "MammothModa2DiTPipeline"
    assert dit_stage.custom_process_input_func.endswith(".ar2diffusion")
    assert dit_stage.omni_kv_config == {"need_recv_cache": False}
    assert dit_stage.input_sources == (0,)
    assert dit_stage.final_output is True
    assert dit_stage.final_output_type == "image"


def test_mammothmoda2_dit_is_registered_only_with_diffusion_runtime() -> None:
    assert _DIFFUSION_MODELS["MammothModa2DiTPipeline"] == (
        "mammoth_moda2",
        "pipeline_mammothmoda2_dit",
        "MammothModa2DiTPipeline",
    )
    assert "MammothModa2DiTPipeline" not in _OMNI_MODELS


def test_mammothmoda2_ar_only_topology_is_unchanged() -> None:
    assert len(MAMMOTH_MODA2_AR_PIPELINE.stages) == 1
    stage = MAMMOTH_MODA2_AR_PIPELINE.stages[0]
    assert stage.execution_type is StageExecutionType.LLM_AR
    assert stage.final_output_type == "text"
```

Append this model-specific structured-config regression to
`tests/config/test_omni_config.py`; its existing imports already include
`VllmOmniDiffusionStageConfig` and the `_from_pipeline_key` helper:

```python
def test_mammothmoda2_diffusion_stage_projects_native_backend_config() -> None:
    config = _from_pipeline_key(
        "mammoth_moda2",
        cli_overrides={"model": "/models/MammothModa2-Preview"},
    )

    stage = config.stage_by_id(1)
    assert isinstance(stage, VllmOmniDiffusionStageConfig)
    assert stage.diffusion_config.model_class_name == "MammothModa2DiTPipeline"
    assert stage.diffusion_config.model == "/models/MammothModa2-Preview"
    assert stage.diffusion_config.step_execution is False
    assert stage.scheduler_config.max_num_seqs == 1
    assert stage.connector_config.omni_kv_config == {"need_recv_cache": False}
```

- [ ] **Step 2: Run the tests and verify they fail for the legacy stage**

Run:

```bash
python -m pytest \
  tests/config/test_mammoth_moda2_shared_runtime.py \
  tests/config/test_omni_config.py::test_mammothmoda2_diffusion_stage_projects_native_backend_config \
  -q
```

Expected: three migration assertions fail: stage 1 is still
`LLM_GENERATION`, the diffusion registry has no MammothModa2 entry, and the
structured stage is not `VllmOmniDiffusionStageConfig`. The unchanged AR-only
test passes.

- [ ] **Step 3: Change the stage-1 topology**

Replace the stage-1 declaration in `vllm_omni/model_executor/models/mammoth_moda2/pipeline.py` with:

```python
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            owns_tokenizer=False,
            requires_multimodal_data=False,
            model_arch="MammothModa2DiTPipeline",
            custom_process_input_func=f"{_PROC}.ar2diffusion",
            omni_kv_config={"need_recv_cache": False},
        ),
```

- [ ] **Step 4: Register the native diffusion class and remove legacy runner selection**

Add this item to `_DIFFUSION_MODELS` in `vllm_omni/diffusion/registry.py`:

```python
    "MammothModa2DiTPipeline": (
        "mammoth_moda2",
        "pipeline_mammothmoda2_dit",
        "MammothModa2DiTPipeline",
    ),
```

Delete this item from `_OMNI_MODELS` in `vllm_omni/model_executor/models/registry.py`:

```python
    "MammothModa2DiTPipeline": (
        "mammoth_moda2",
        "pipeline_mammothmoda2_dit",
        "MammothModa2DiTPipeline",
    ),
```

Do not modify the compatibility shim at `vllm_omni/model_executor/models/mammoth_moda2/pipeline_mammothmoda2_dit.py`.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
python -m pytest \
  tests/config/test_mammoth_moda2_shared_runtime.py \
  tests/config/test_omni_config.py::test_mammothmoda2_diffusion_stage_projects_native_backend_config \
  -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit the topology boundary**

```bash
git add tests/config/test_mammoth_moda2_shared_runtime.py \
  tests/config/test_omni_config.py \
  vllm_omni/model_executor/models/mammoth_moda2/pipeline.py \
  vllm_omni/diffusion/registry.py \
  vllm_omni/model_executor/models/registry.py
git commit -m "feat: route MammothModa2 DiT through diffusion runtime"
```

### Task 2: Convert AR output into one native diffusion prompt

**Files:**
- Create: `tests/model_executor/stage_input_processors/test_mammoth_moda2.py`
- Modify: `vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py:1-116`
- Modify: `tests/worker/test_omni_connector_mixin.py:376-388`

- [ ] **Step 1: Write the failing stage adapter tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import (
    ar2diffusion,
)


def _source_output(*, include_latent: bool = True):
    multimodal_output = {"latent": torch.arange(32, dtype=torch.float32).reshape(4, 8)} if include_latent else {}
    completion = SimpleNamespace(
        cumulative_token_ids=[100, 101, 102],
        multimodal_output=multimodal_output,
    )
    return SimpleNamespace(
        request_id="req-7",
        prompt_token_ids=[10, 11],
        outputs=[completion],
    )


def test_ar2diffusion_builds_one_prompt_with_raw_ar_conditions() -> None:
    result = ar2diffusion(
        [_source_output()],
        {
            "prompt": "a cat",
            "mm_processor_kwargs": {"target_h": 512, "target_w": 768},
        },
    )

    assert not isinstance(result, list)
    assert result["prompt"] == ""
    assert result["height"] == 512
    assert result["width"] == 768
    info = result["additional_information"]
    assert info["full_token_ids"] == [10, 11, 100, 101]
    assert info["answer_start_index"] == 2
    torch.testing.assert_close(
        info["full_hidden_states"],
        torch.arange(32, dtype=torch.float32).reshape(4, 8),
    )
    assert info["full_hidden_states"].is_contiguous()


def test_ar2diffusion_uses_prompt_dimension_fallbacks() -> None:
    result = ar2diffusion(
        [_source_output()],
        {
            "additional_information": {
                "image_height": [256],
                "image_width": [384],
            }
        },
    )

    assert (result["height"], result["width"]) == (256, 384)


def test_ar2diffusion_unwraps_the_orchestrator_prompt_list() -> None:
    result = ar2diffusion(
        [_source_output()],
        [{"mm_processor_kwargs": {"target_h": 640, "target_w": 960}}],
    )

    assert (result["height"], result["width"]) == (640, 960)


def test_ar2diffusion_rejects_multiple_source_requests() -> None:
    with pytest.raises(ValueError, match="exactly one AR output"):
        ar2diffusion([_source_output(), _source_output()], {})


def test_ar2diffusion_reports_missing_latent_with_request_id() -> None:
    with pytest.raises(ValueError, match="req-7"):
        ar2diffusion([_source_output(include_latent=False)], {})


def test_ar2diffusion_rejects_hidden_state_length_mismatch() -> None:
    source = _source_output()
    source.outputs[0].multimodal_output["latent"] = torch.zeros(3, 8)

    with pytest.raises(ValueError, match="Hidden states length mismatch"):
        ar2diffusion([source], {})
```

- [ ] **Step 2: Run the adapter tests and verify the native symbol is missing**

Run:

```bash
python -m pytest tests/model_executor/stage_input_processors/test_mammoth_moda2.py -q
```

Expected: collection fails because `ar2diffusion` is not defined.

- [ ] **Step 3: Replace `ar2dit` with the single-request adapter**

Replace the imports and adapter body in `vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py` with:

```python
"""Stage input processor for MammothModa2 (AR -> diffusion)."""

from collections.abc import Mapping
from typing import Any


def _as_dict(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _coerce_dim(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return default
    return resolved if resolved > 0 else default


def ar2diffusion(
    source_outputs: list[Any],
    prompt: Any | None = None,
    requires_multimodal_data: bool = False,
) -> dict[str, Any]:
    del requires_multimodal_data
    if len(source_outputs) != 1:
        raise ValueError(
            "MammothModa2 request-mode diffusion expects exactly one AR output, "
            f"got {len(source_outputs)}"
        )

    ar_output = source_outputs[0]
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    prompt_dict = _as_dict(prompt)
    additional = prompt_dict.get("additional_information") or {}
    mm_kwargs = prompt_dict.get("mm_processor_kwargs") or {}
    height = _coerce_dim(
        mm_kwargs.get("target_h"),
        _coerce_dim((additional.get("image_height") or [None])[0], 1024),
    )
    width = _coerce_dim(
        mm_kwargs.get("target_w"),
        _coerce_dim((additional.get("image_width") or [None])[0], 1024),
    )

    completion = ar_output.outputs[0]
    generated_token_ids = list(completion.cumulative_token_ids[:-1])
    prompt_token_ids = list(ar_output.prompt_token_ids)
    full_token_ids = prompt_token_ids + generated_token_ids
    multimodal_output = getattr(completion, "multimodal_output", None)
    if not isinstance(multimodal_output, Mapping) or "latent" not in multimodal_output:
        raise ValueError(
            "MammothModa2 AR stage output is missing latent multimodal output; "
            f"request_id={getattr(ar_output, 'request_id', None)}"
        )

    full_hidden_states = multimodal_output["latent"]
    hidden_total = int(full_hidden_states.shape[0])
    if hidden_total != len(full_token_ids):
        raise ValueError(
            "Hidden states length mismatch: "
            f"expected {len(full_token_ids)}, got {hidden_total}; "
            f"request_id={getattr(ar_output, 'request_id', None)}"
        )

    return {
        "prompt": "",
        "height": height,
        "width": width,
        "additional_information": {
            "full_hidden_states": full_hidden_states.float().contiguous(),
            "full_token_ids": full_token_ids,
            "answer_start_index": len(prompt_token_ids),
        },
    }
```

- [ ] **Step 4: Update the connector compatibility fixture**

In `tests/worker/test_omni_connector_mixin.py`, replace:

```python
"vllm_omni.model_executor.stage_input_processors.mammoth_moda2.ar2dit",
```

with:

```python
"vllm_omni.model_executor.stage_input_processors.mammoth_moda2.ar2diffusion",
```

- [ ] **Step 5: Run focused bridge and connector tests**

Run:

```bash
python -m pytest \
  tests/model_executor/stage_input_processors/test_mammoth_moda2.py \
  tests/worker/test_omni_connector_mixin.py::TestLoadCustomFuncSelection \
  -q
```

Expected: `7 passed`.

- [ ] **Step 6: Commit the stage contract**

```bash
git add vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py \
  tests/model_executor/stage_input_processors/test_mammoth_moda2.py \
  tests/worker/test_omni_connector_mixin.py
git commit -m "refactor: emit MammothModa2 diffusion requests"
```

### Task 3: Adopt native diffusion construction and weight loading

**Files:**
- Create: `tests/diffusion/models/mammoth_moda2/__init__.py`
- Create: `tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py`
- Modify: `vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py:1-125`

- [ ] **Step 1: Write failing configuration and checkpoint source tests**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _build_mammoth_config,
    _root_weight_source,
)


def _raw_config() -> dict:
    return {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {
                "model_type": "mammothmoda2_qwen2_5_vl_text",
                "hidden_size": 8,
                "gen_vocab_start_index": 100,
            },
        },
        "gen_vae_config": {"block_out_channels": [8, 8]},
        "gen_dit_config": {"hidden_size": 8, "in_channels": 4},
    }


def _od_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(_raw_config()),
    )


def test_build_mammoth_config_uses_shared_transformer_projection() -> None:
    config = _build_mammoth_config(_od_config())

    assert config.model_type == "mammothmoda2"
    assert config.get_text_config().hidden_size == 8
    assert config.gen_dit_config == {"hidden_size": 8, "in_channels": 4}


def test_root_weight_source_loads_combined_checkpoint_once() -> None:
    source = _root_weight_source(_od_config())

    assert source.model_or_path == "/models/MammothModa2-Preview"
    assert source.subfolder is None
    assert source.prefix == ""
    assert source.fall_back_to_pt is True


def test_pipeline_declares_single_request_mode_only() -> None:
    assert MammothModa2DiTPipeline.supports_request_batch is False
    assert MammothModa2DiTPipeline.supports_step_execution is False


def test_root_weight_source_rejects_missing_model_path() -> None:
    config = SimpleNamespace(model=None, revision=None)

    try:
        _root_weight_source(config)
    except ValueError as exc:
        assert "model path" in str(exc)
    else:
        raise AssertionError("missing model path must fail")
```

Create an empty `tests/diffusion/models/mammoth_moda2/__init__.py` so pytest treats the directory as a package.

- [ ] **Step 2: Run the tests and verify the native helpers are missing**

Run:

```bash
python -m pytest \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  -q
```

Expected: collection fails because `_build_mammoth_config` and `_root_weight_source` are not defined.

- [ ] **Step 3: Replace legacy config imports and add native helpers**

At the top of `pipeline_mammothmoda2_dit.py`, remove `VllmConfig` and add the
native construction imports below. Keep `Any` and `OmniOutput` until Task 4 so
the legacy `forward` remains internally valid throughout this intermediate
commit.

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
```

Add these helpers above the pipeline class:

```python
def _build_mammoth_config(od_config: OmniDiffusionConfig) -> Mammothmoda2Config:
    raw_config = od_config.tf_model_config.to_dict()
    if not raw_config:
        raise ValueError("MammothModa2 diffusion stage requires the root checkpoint config")
    return Mammothmoda2Config(**raw_config)


def _root_weight_source(
    od_config: OmniDiffusionConfig,
) -> DiffusersPipelineLoader.ComponentSource:
    if not od_config.model:
        raise ValueError("MammothModa2 diffusion stage requires a model path")
    return DiffusersPipelineLoader.ComponentSource(
        model_or_path=od_config.model,
        subfolder=None,
        revision=od_config.revision,
        prefix="",
        fall_back_to_pt=True,
    )
```

- [ ] **Step 4: Change the class and constructor to the shared runtime contract**

Change the class boundary and constructor setup to:

```python
class MammothModa2DiTPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = ["gen_transformer"]
    _encoder_modules: ClassVar[list[str]] = ["gen_image_condition_refiner"]
    _vae_modules: ClassVar[list[str]] = ["gen_vae"]

    supports_request_batch = False
    supports_step_execution = False

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm_model.": None,
            "gen_tokenizer.": None,
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix

        self.od_config = od_config
        self.device = get_local_device()
        self.config = _build_mammoth_config(od_config)
        self.weights_sources = [_root_weight_source(od_config)]
```

Keep the existing module construction, caption embedder setup, optional refiner
selection, RoPE construction, legacy `forward`, and generation-runner
compatibility members through this task. Task 4 replaces the runtime contract
and removes those members in the same commit, so every intermediate revision
remains importable and passes static checks.

- [ ] **Step 5: Run configuration tests**

Run:

```bash
python -m pytest \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  tests/model_executor/models/mammoth_moda2/test_mammoth_moda2_config.py \
  -q
```

Expected: all tests pass, with the new file reporting `4 passed`.

- [ ] **Step 6: Commit native construction**

```bash
git add vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py \
  tests/diffusion/models/mammoth_moda2/__init__.py \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py
git commit -m "refactor: construct MammothModa2 from diffusion config"
```

### Task 4: Implement the request-mode forward and output contract

**Files:**
- Modify: `tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py`
- Modify: `vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py:126-405`

- [ ] **Step 1: Add failing request parsing and sampling tests**

Append to `test_pipeline_mammothmoda2_dit.py`:

```python
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _pipeline_shell() -> MammothModa2DiTPipeline:
    pipeline = object.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.config = SimpleNamespace(
        llm_config=SimpleNamespace(gen_vocab_start_index=100),
        image_token_id=900,
        video_token_id=901,
        vision_start_token_id=902,
        vision_end_token_id=903,
    )
    pipeline._llm_hidden_size = 8
    return pipeline


def _batch(
    *,
    request_id: str = "req-7",
    prompt: object | None = None,
    sampling: OmniDiffusionSamplingParams | None = None,
) -> DiffusionRequestBatch:
    if prompt is None:
        prompt = {
            "prompt": "",
            "height": 32,
            "width": 48,
            "additional_information": {
                "full_hidden_states": torch.arange(32, dtype=torch.float32).reshape(4, 8),
                "full_token_ids": [10, 11, 100, 101],
                "answer_start_index": 2,
            },
        }
    if sampling is None:
        sampling = OmniDiffusionSamplingParams(
            height=32,
            width=48,
            seed=42,
            guidance_scale=4.0,
            num_inference_steps=7,
            extra_args={"cfg_range": [0.2, 0.8]},
        )
    return DiffusionRequestBatch(
        [OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=request_id)]
    )


def test_parse_request_uses_ar_payload_and_standard_sampling_fields() -> None:
    parsed = _pipeline_shell()._parse_request(_batch())

    assert parsed.request_id == "req-7"
    assert parsed.height == 32
    assert parsed.width == 48
    assert parsed.num_inference_steps == 7
    assert parsed.text_guidance_scale == 4.0
    assert parsed.cfg_range == (0.2, 0.8)
    assert parsed.seed == 42
    assert parsed.answer_start_index == 2


def test_parse_request_preserves_legacy_extra_arg_precedence() -> None:
    sampling = OmniDiffusionSamplingParams(
        height=32,
        width=48,
        seed=9,
        guidance_scale=3.0,
        num_inference_steps=5,
        extra_args={
            "text_guidance_scale": 6.0,
            "num_inference_steps": 11,
            "cfg_range": [0.0, 0.5],
        },
    )

    parsed = _pipeline_shell()._parse_request(_batch(sampling=sampling))

    assert parsed.text_guidance_scale == 6.0
    assert parsed.num_inference_steps == 11
    assert parsed.cfg_range == (0.0, 0.5)


def test_parse_request_rejects_batching_and_multiple_images() -> None:
    batch = _batch()
    batch.requests.append(batch.requests[0])
    with pytest.raises(ValueError, match="exactly one request"):
        _pipeline_shell()._parse_request(batch)

    sampling = OmniDiffusionSamplingParams(num_outputs_per_prompt=2)
    with pytest.raises(ValueError, match="num_outputs_per_prompt=1"):
        _pipeline_shell()._parse_request(_batch(sampling=sampling))


@pytest.mark.parametrize(
    ("cfg_range", "message"),
    [
        ([0.5], "two values"),
        ([-0.1, 0.5], "0 <= start <= end <= 1"),
        ([0.7, 0.2], "0 <= start <= end <= 1"),
        ([0.2, 1.1], "0 <= start <= end <= 1"),
    ],
)
def test_parse_request_rejects_invalid_cfg_range(cfg_range, message) -> None:
    sampling = OmniDiffusionSamplingParams(extra_args={"cfg_range": cfg_range})
    with pytest.raises(ValueError, match=message):
        _pipeline_shell()._parse_request(_batch(sampling=sampling))


def test_parse_request_requires_ar_conditions_for_real_request() -> None:
    with pytest.raises(ValueError, match="req-missing"):
        _pipeline_shell()._parse_request(
            _batch(request_id="req-missing", prompt={"prompt": "", "height": 32, "width": 32})
        )


def test_parse_request_rejects_hidden_state_token_mismatch() -> None:
    prompt = {
        "prompt": "",
        "height": 32,
        "width": 32,
        "additional_information": {
            "full_hidden_states": torch.zeros(3, 8),
            "full_token_ids": [10, 11, 100, 101],
            "answer_start_index": 2,
        },
    }
    with pytest.raises(ValueError, match="hidden-state/token-count mismatch"):
        _pipeline_shell()._parse_request(_batch(prompt=prompt))


def test_dummy_request_synthesizes_conditions_without_ar_output() -> None:
    parsed = _pipeline_shell()._parse_request(
        _batch(
            request_id="dummy_req_id",
            prompt={"prompt": "dummy run"},
            sampling=OmniDiffusionSamplingParams(
                height=512,
                width=512,
                seed=1,
                guidance_scale=0.0,
                num_inference_steps=2,
            ),
        )
    )

    assert parsed.full_hidden_states.shape == (2, 8)
    assert parsed.full_token_ids == [0, 100]
    assert parsed.answer_start_index == 1
```

- [ ] **Step 2: Run the parsing tests and verify `_parse_request` is missing**

Run:

```bash
python -m pytest \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  -q
```

Expected: the new tests fail with `AttributeError: ... has no attribute '_parse_request'`.

- [ ] **Step 3: Add the native request imports and a typed request boundary**

Add the `dataclass` import, import the shared request/output types, and define:

```python
from dataclasses import dataclass

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch


@dataclass(frozen=True)
class _MammothRequest:
    request_id: str
    full_hidden_states: torch.Tensor
    full_token_ids: list[int]
    answer_start_index: int
    height: int
    width: int
    text_guidance_scale: float
    cfg_range: tuple[float, float]
    num_inference_steps: int
    seed: int | None
    generator: torch.Generator | list[torch.Generator] | None
```

Add this method to `MammothModa2DiTPipeline`:

```python
    def _parse_request(self, req: DiffusionRequestBatch) -> _MammothRequest:
        if req.num_reqs != 1:
            raise ValueError(
                "MammothModa2 request mode supports exactly one request, "
                f"got {req.num_reqs}"
            )
        sampling = req.sampling_params
        if sampling.num_outputs_per_prompt != 1:
            raise ValueError("MammothModa2 request mode requires num_outputs_per_prompt=1")

        prompt = req.prompts[0]
        prompt_dict = prompt if isinstance(prompt, dict) else {}
        info = prompt_dict.get("additional_information")
        if req.is_dummy_run():
            full_hidden_states = torch.zeros((2, self._llm_hidden_size), dtype=torch.float32)
            full_token_ids = [0, int(self.config.llm_config.gen_vocab_start_index)]
            answer_start_index = 1
        else:
            if not isinstance(info, dict):
                raise ValueError(
                    "MammothModa2 diffusion request is missing AR conditions; "
                    f"request_id={req.request_id}"
                )
            full_hidden_states = info.get("full_hidden_states")
            full_token_ids = info.get("full_token_ids")
            answer_start_index = info.get("answer_start_index")
            if not isinstance(full_hidden_states, torch.Tensor) or not isinstance(full_token_ids, list):
                raise ValueError(
                    "MammothModa2 diffusion request has invalid AR conditions; "
                    f"request_id={req.request_id}"
                )
            try:
                answer_start_index = int(answer_start_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "MammothModa2 diffusion request has an invalid answer_start_index; "
                    f"request_id={req.request_id}"
                ) from exc
            if full_hidden_states.ndim != 2:
                raise ValueError(
                    "MammothModa2 full_hidden_states must be 2D; "
                    f"request_id={req.request_id}"
                )
            if full_hidden_states.shape[0] != len(full_token_ids):
                raise ValueError(
                    "MammothModa2 hidden-state/token-count mismatch; "
                    f"hidden_states={full_hidden_states.shape[0]}, tokens={len(full_token_ids)}; "
                    f"request_id={req.request_id}"
                )
            if not 0 <= answer_start_index <= len(full_token_ids):
                raise ValueError(
                    "MammothModa2 answer_start_index is outside the token sequence; "
                    f"request_id={req.request_id}"
                )

        height = DiffusionRequestBatch.get_prompt_field(prompt, "height") or sampling.height or 1024
        width = DiffusionRequestBatch.get_prompt_field(prompt, "width") or sampling.width or 1024
        extra_args = sampling.extra_args or {}
        if "text_guidance_scale" in extra_args:
            text_guidance_scale = float(extra_args["text_guidance_scale"])
        elif sampling.guidance_scale_provided:
            text_guidance_scale = float(sampling.guidance_scale)
        else:
            text_guidance_scale = 9.0
        num_inference_steps = int(extra_args.get("num_inference_steps", sampling.num_inference_steps or 50))
        raw_cfg_range = extra_args.get("cfg_range", [0.0, 1.0])
        if not isinstance(raw_cfg_range, (list, tuple)) or len(raw_cfg_range) != 2:
            raise ValueError("MammothModa2 cfg_range must contain two values")
        cfg_range = float(raw_cfg_range[0]), float(raw_cfg_range[1])
        if not 0.0 <= cfg_range[0] <= cfg_range[1] <= 1.0:
            raise ValueError("MammothModa2 cfg_range must satisfy 0 <= start <= end <= 1")
        if num_inference_steps <= 0:
            raise ValueError("MammothModa2 num_inference_steps must be positive")

        return _MammothRequest(
            request_id=req.request_id,
            full_hidden_states=full_hidden_states,
            full_token_ids=[int(token_id) for token_id in full_token_ids],
            answer_start_index=answer_start_index,
            height=int(height),
            width=int(width),
            text_guidance_scale=text_guidance_scale,
            cfg_range=cfg_range,
            num_inference_steps=num_inference_steps,
            seed=sampling.seed,
            generator=sampling.generator,
        )
```

- [ ] **Step 4: Run parsing tests**

Run:

```bash
python -m pytest \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  -q
```

Expected: all constructor and parsing tests pass.

- [ ] **Step 5: Add a failing end-to-end forward-contract test with lightweight modules**

Append this test and helper classes to `test_pipeline_mammothmoda2_dit.py`:

```python
from unittest.mock import patch

from vllm_omni.diffusion.data import DiffusionOutput


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(in_channels=4)
        self.time_caption_embed = SimpleNamespace(image_embedder=None)
        self.calls = 0

    def forward(self, *, hidden_states, **kwargs):
        self.calls += 1
        return torch.zeros_like(hidden_states)


class _FakeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(scaling_factor=None, shift_factor=None)

    def decode(self, latents, return_dict=False):
        assert return_dict is False
        return (torch.zeros(1, 3, 32, 48, dtype=latents.dtype),)


class _FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([])
        self.requested_steps = None

    def set_timesteps(self, *, num_inference_steps, device, num_tokens):
        self.requested_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.float32)

    def step(self, model_pred, timestep, latents, return_dict=False):
        assert return_dict is False
        return (latents - model_pred,)


def test_forward_returns_diffusion_output_and_uses_request_seed() -> None:
    pipeline = _pipeline_shell()
    pipeline.gen_transformer = _FakeTransformer()
    pipeline.gen_image_condition_refiner = None
    pipeline.gen_vae = _FakeVae()
    pipeline.gen_freqs_cis = torch.zeros(1)
    scheduler = _FakeScheduler()
    seen = {}

    def fake_randn(shape, *, generator, device, dtype):
        seen["shape"] = shape
        seen["seed"] = generator.initial_seed()
        return torch.zeros(shape, device=device, dtype=dtype)

    with (
        patch(
            "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.FlowMatchEulerDiscreteScheduler",
            return_value=scheduler,
        ),
        patch(
            "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.randn_tensor",
            side_effect=fake_randn,
        ),
    ):
        output = pipeline.forward(
            _batch(
                sampling=OmniDiffusionSamplingParams(
                    height=32,
                    width=48,
                    seed=42,
                    guidance_scale=1.0,
                    num_inference_steps=2,
                )
            )
        )

    assert isinstance(output, DiffusionOutput)
    assert output.output.shape == (1, 3, 32, 48)
    assert seen == {"shape": (1, 4, 4, 6), "seed": 42}
    assert scheduler.requested_steps == 2
    assert pipeline.gen_transformer.calls == 2


def test_forward_rejects_request_without_visual_condition_tokens() -> None:
    pipeline = _pipeline_shell()
    prompt = {
        "prompt": "",
        "height": 32,
        "width": 32,
        "additional_information": {
            "full_hidden_states": torch.zeros(3, 8),
            "full_token_ids": [10, 11, 12],
            "answer_start_index": 2,
        },
    }

    with pytest.raises(ValueError, match="no visual-token hidden states.*req-empty"):
        pipeline.forward(_batch(request_id="req-empty", prompt=prompt))
```

- [ ] **Step 6: Run the forward-contract test and verify the legacy signature/output fail**

Run:

```bash
python -m pytest \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py::test_forward_returns_diffusion_output_and_uses_request_seed \
  -q
```

Expected: failure because the current `forward` expects keyword inputs and returns `OmniOutput`.

- [ ] **Step 7: Rewrite `forward` around `_MammothRequest`**

Replace the legacy `forward` method with:

```python
    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        request = self._parse_request(req)
        text_cond, image_cond = self._split_ar_conditions(
            full_hidden_states=request.full_hidden_states,
            full_token_ids=request.full_token_ids,
            answer_start_index=request.answer_start_index,
        )

        model_device = next(self.parameters()).device
        if self.gen_image_condition_refiner is not None:
            target_dtype = next(self.gen_image_condition_refiner.parameters()).dtype
        else:
            target_dtype = next(self.gen_transformer.parameters()).dtype

        if image_cond.shape[0] == 0:
            answer_token_ids = request.full_token_ids[request.answer_start_index :]
            raise ValueError(
                "MammothModa2 AR stage produced no visual-token hidden states; "
                "the DiT stage requires at least one generated visual token; "
                f"request_id={request.request_id}; generated_token_ids={answer_token_ids[:32]}"
            )
        text_cond = text_cond.to(
            device=model_device,
            dtype=target_dtype,
            non_blocking=True,
        ).contiguous()
        image_cond = image_cond.to(
            device=model_device,
            dtype=target_dtype,
            non_blocking=True,
        ).contiguous()

        text_embeds = text_cond.unsqueeze(0)
        text_attention_mask = torch.ones(
            (1, text_embeds.shape[1]),
            dtype=torch.bool,
            device=text_embeds.device,
        )
        image_embeds = image_cond.unsqueeze(0)
        image_attention_mask = torch.ones(
            (1, image_embeds.shape[1]),
            dtype=torch.bool,
            device=image_embeds.device,
        )

        if self.gen_image_condition_refiner is not None:
            image_embeds = self.gen_image_condition_refiner(
                image_embeds,
                ~image_attention_mask,
            )
            image_attention_mask = torch.ones(
                image_embeds.shape[:2],
                dtype=torch.bool,
                device=image_embeds.device,
            )

        nested_image_embedder = getattr(
            self.gen_transformer.time_caption_embed,
            "image_embedder",
            None,
        )
        if nested_image_embedder is None:
            prompt_embeds = torch.cat([text_embeds, image_embeds], dim=1)
            prompt_attention_mask = torch.cat(
                [text_attention_mask, image_attention_mask],
                dim=1,
            )
            ar_image_embeds = None
            ar_image_attention_mask = None
        else:
            prompt_embeds = text_embeds
            prompt_attention_mask = text_attention_mask
            ar_image_embeds = image_embeds
            ar_image_attention_mask = image_attention_mask

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if request.text_guidance_scale > 1.0:
            negative_prompt_embeds = torch.zeros(
                (1, 0, prompt_embeds.shape[-1]),
                dtype=target_dtype,
                device=prompt_embeds.device,
            )
            negative_prompt_attention_mask = torch.zeros(
                (1, 0),
                dtype=torch.bool,
                device=prompt_embeds.device,
            )

        generator = request.generator
        if generator is None and request.seed is not None:
            generator = torch.Generator(device=prompt_embeds.device).manual_seed(request.seed)

        height, width = request.height, request.width
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image size: {height}x{width}; request_id={request.request_id}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "Image size must be multiples of 16, "
                f"got {height}x{width}; request_id={request.request_id}"
            )
        vae_scale_factor = 16
        latent_channels = int(self.gen_transformer.config.in_channels)
        shape = (1, latent_channels, 2 * height // vae_scale_factor, 2 * width // vae_scale_factor)
        latents = randn_tensor(
            shape,
            generator=generator,
            device=prompt_embeds.device,
            dtype=prompt_embeds.dtype,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(
            num_inference_steps=request.num_inference_steps,
            device=prompt_embeds.device,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )

        total_steps = max(1, len(scheduler.timesteps))
        for index, timestep_value in enumerate(scheduler.timesteps):
            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
            model_pred = self.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=prompt_embeds,
                text_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=None,
                ar_image_hidden_states=ar_image_embeds,
                ar_image_attention_mask=ar_image_attention_mask,
                freqs_cis=self.gen_freqs_cis,
            )
            progress = index / total_steps
            guidance_scale = (
                request.text_guidance_scale
                if request.cfg_range[0] <= progress <= request.cfg_range[1]
                else 1.0
            )
            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                model_pred_uncond = self.gen_transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=negative_prompt_embeds,
                    text_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                    freqs_cis=self.gen_freqs_cis,
                )
                model_pred = model_pred_uncond + guidance_scale * (
                    model_pred - model_pred_uncond
                )
            latents = scheduler.step(
                model_pred,
                timestep_value,
                latents,
                return_dict=False,
            )[0]
            latents = latents.to(dtype=prompt_embeds.dtype)

        if self.gen_vae.config.scaling_factor is not None:
            latents = latents / self.gen_vae.config.scaling_factor
        if self.gen_vae.config.shift_factor is not None:
            latents = latents + self.gen_vae.config.shift_factor
        image = self.gen_vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)
```

Keep `load_weights` unchanged: the native loader now feeds it the root checkpoint source added in Task 3, and the existing mapper filters `llm_model.*` and `gen_tokenizer.*`.

After replacing `forward`, change the typing import to
`from typing import ClassVar`, remove the `OmniOutput` import, and delete the
legacy `have_multimodal_outputs` flag,
`get_dummy_runtime_additional_information`, `make_empty_intermediate_tensors`,
`embed_input_ids`, and `compute_logits` members. Native dummy requests and
`DiffusionOutput` now own those contracts, and `Any` is no longer used.

- [ ] **Step 8: Run all pipeline contract tests**

Run:

```bash
python -m pytest \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  -q
```

Expected: all tests pass, including deterministic seed, dummy warmup parsing, invalid CFG ranges, and unsupported batch sizes.

- [ ] **Step 9: Commit the request-mode pipeline**

```bash
git add vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py
git commit -m "feat: add MammothModa2 diffusion request mode"
```

### Task 5: Update real-weight E2E and user documentation

**Files:**
- Modify: `tests/e2e/offline_inference/test_mammoth_moda2_expansion.py:18-170`
- Modify: `recipes/MammothModa2/MammothModa2.md:12-120`

- [ ] **Step 1: Change the E2E test to native diffusion sampling params**

In `tests/e2e/offline_inference/test_mammoth_moda2_expansion.py`, import `OmniDiffusionSamplingParams`:

```python
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
```

Delete the `@pytest.mark.skip` decorator. Replace `dit_sampling` with:

```python
    dit_sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        seed=42,
        guidance_scale=1.0,
        num_inference_steps=2,
        extra_args={"cfg_range": [0.0, 1.0]},
    )
```

Remove `num_inference_steps`, `text_guidance_scale`, and `cfg_range` from the prompt's `additional_information`; those values now belong to the diffusion request. Keep AR grid/token metadata and image dimensions unchanged.

- [ ] **Step 2: Collect the E2E test without starting a GPU run**

Run:

```bash
python -m pytest tests/e2e/offline_inference/test_mammoth_moda2_expansion.py --collect-only -q
```

Expected: one `test_mammothmoda2_t2i_e2e` item is collected and it is no longer skipped by issue #3201.

- [ ] **Step 3: Rewrite the recipe's runtime and parameter guidance**

Replace the legacy-interface paragraph under “When to use this recipe” with:

```markdown
MammothModa2's DiT stage runs in the shared diffusion runtime in request mode.
The first integration intentionally supports one request and one image per
forward only (`max_num_seqs: 1`, `num_outputs_per_prompt: 1`). Request-level
batching, step execution, continuous batching, cache acceleration,
compilation, quantization, parallelism, and offload are not enabled by this
recipe.

Image size, seed, guidance, and denoising steps use the standard diffusion
request fields. `cfg_range` remains a MammothModa2-specific `extra_body`
parameter. For compatibility, the runtime also accepts the former
`text_guidance_scale` and `num_inference_steps` keys in `extra_body`; when
present, those keys take precedence over the standard fields.
```

Update the offline command to:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --guidance-scale 4.0 \
  --num-inference-steps 50 \
  --extra-body '{"cfg_range": [0.0, 1.0]}' \
  --output mammoth_t2i.png
```

Replace the existing “Hardware Support” introduction through the NVIDIA
environment list with this exact A800-only text. Keep the existing MI300X
section, but rename its heading to
`### 1x AMD MI300X, MammothModa2 Preview (pre-migration baseline)`.

```markdown
## Hardware Support

The default deploy config runs both the AR and DiT stages on one NVIDIA A800
80 GB (`devices: "0"`). Its committed `gpu_memory_utilization` split is 0.5
for stage 0 and 0.3 for stage 1. The validation section below also shows a
two-A800 placement with one stage per GPU for attributable timing and memory.

## GPU

### 1x NVIDIA A800 80GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with one A800 80 GB
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from
```

- [ ] **Step 4: Add the two-A800 serving validation command to the recipe**

Add this subsection after the single-A800 smoke command:

```markdown
### 2x NVIDIA A800 80GB validation

Use one A800 per stage so AR and DiT memory and timing are attributable. The
per-stage override changes placement only; both stages remain single-rank.

```bash
vllm serve ./MammothModa2-Preview --omni \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --stage-overrides '{"0":{"devices":"0"},"1":{"devices":"1"}}' \
  --port 8099 \
  --log-stats
```

The server log for stage 1 must name `StageDiffusionClient`,
`DiffusionEngine`, and `MammothModa2DiTPipeline`. Seeing the legacy generation
model runner for stage 1 is a failed migration.
```

- [ ] **Step 5: Run formatting and focused documentation checks**

Run:

```bash
pre-commit run --files \
  tests/e2e/offline_inference/test_mammoth_moda2_expansion.py \
  recipes/MammothModa2/MammothModa2.md
```

Expected: all hooks pass.

- [ ] **Step 6: Commit E2E and documentation**

```bash
git add tests/e2e/offline_inference/test_mammoth_moda2_expansion.py \
  recipes/MammothModa2/MammothModa2.md
git commit -m "test: validate MammothModa2 shared runtime path"
```

### Task 6: Run the CPU regression gate

**Files:**
- Test only; no file changes expected.

- [ ] **Step 1: Run the complete focused CPU suite**

Run:

```bash
python -m pytest \
  tests/config/test_mammoth_moda2_shared_runtime.py \
  tests/config/test_omni_config.py::test_mammothmoda2_diffusion_stage_projects_native_backend_config \
  tests/model_executor/models/mammoth_moda2/test_mammoth_moda2_config.py \
  tests/model_executor/stage_input_processors/test_mammoth_moda2.py \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  tests/worker/test_omni_connector_mixin.py::TestLoadCustomFuncSelection \
  -q
python -m pytest \
  tests/config/test_config_factory.py \
  tests/model_extras/test_model_extras.py \
  -k mammoth \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run static checks on every changed Python file**

Run:

```bash
pre-commit run --files \
  vllm_omni/model_executor/models/mammoth_moda2/pipeline.py \
  vllm_omni/diffusion/registry.py \
  vllm_omni/model_executor/models/registry.py \
  vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py \
  vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py \
  tests/config/test_mammoth_moda2_shared_runtime.py \
  tests/config/test_omni_config.py \
  tests/model_executor/stage_input_processors/test_mammoth_moda2.py \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  tests/e2e/offline_inference/test_mammoth_moda2_expansion.py \
  tests/worker/test_omni_connector_mixin.py \
  recipes/MammothModa2/MammothModa2.md
```

Expected: all hooks pass without modifying files. If a hook formats a file, inspect the diff, rerun the focused tests, and commit only the formatting change.

- [ ] **Step 3: Verify scope exclusions mechanically**

Run:

```bash
git diff 32370968..HEAD -- \
  vllm_omni/diffusion/cache \
  vllm_omni/diffusion/sched \
  vllm_omni/diffusion/offloader \
  vllm_omni/diffusion/attention
```

Expected: no output. The migration must not include TeaCache, scheduler, offload, quantization, compile, CUDA graph, or attention changes.

- [ ] **Step 4: Commit any check-driven corrections**

If checks changed files, stage only the reviewed corrections and commit:

```bash
git add vllm_omni/model_executor/models/mammoth_moda2/pipeline.py \
  vllm_omni/diffusion/registry.py \
  vllm_omni/model_executor/models/registry.py \
  vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py \
  vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py \
  tests/config/test_mammoth_moda2_shared_runtime.py \
  tests/config/test_omni_config.py \
  tests/model_executor/stage_input_processors/test_mammoth_moda2.py \
  tests/diffusion/models/mammoth_moda2/test_pipeline_mammothmoda2_dit.py \
  tests/e2e/offline_inference/test_mammoth_moda2_expansion.py \
  tests/worker/test_omni_connector_mixin.py \
  recipes/MammothModa2/MammothModa2.md
git commit -m "chore: satisfy MammothModa2 migration checks"
```

If checks made no changes, do not create an empty commit.

### Task 7: Validate real weights and measure baseline/head on A800

**Files:**
- Runtime artifacts only: `/root/mammoth-7086-results/`
- Test: `tests/e2e/offline_inference/test_mammoth_moda2_expansion.py`

- [ ] **Step 1: Provision the AutoDL environment on one A800**

First publish the reviewed implementation branch from the local worktree. The
fork command creates `Levius-Fubuki/vllm-omni` if it is still absent and adds
it as the `fork` remote:

```bash
cd /Users/levius/Desktop/Idea/projects/vllm-omni-7086
gh repo fork vllm-project/vllm-omni --remote --remote-name fork
git push -u fork codex/issue-7086-mammoth-runtime
```

Then run on an AutoDL NVIDIA A800 80 GB instance:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python --version
git clone --branch codex/issue-7086-mammoth-runtime \
  https://github.com/Levius-Fubuki/vllm-omni.git \
  /root/vllm-omni-7086
cd /root/vllm-omni-7086
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -v -e .
hf download bytedance-research/MammothModa2-Preview \
  --local-dir /root/models/MammothModa2-Preview
mkdir -p /root/mammoth-7086-results
```

Expected: `nvidia-smi` reports A800 with about 80 GB, installation succeeds, and the model directory contains `config.json` plus checkpoint shards.

- [ ] **Step 2: Run the real-weight E2E test on one A800**

Run:

```bash
cd /root/vllm-omni-7086
source .venv/bin/activate
HF_HOME=/root/.cache/huggingface \
python -m pytest \
  tests/e2e/offline_inference/test_mammoth_moda2_expansion.py \
  -q -s
```

Expected: the test loads both stages, stage 1 logs the shared diffusion runtime, produces a 256×256 image tensor, and passes.

- [ ] **Step 3: Run the 1024×1024 single-request smoke test on one A800**

Run:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /root/models/MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --guidance-scale 4.0 \
  --num-inference-steps 50 \
  --extra-body '{"cfg_range":[0.0,1.0]}' \
  --enable-diffusion-pipeline-profiler \
  --log-stats \
  --output /root/mammoth-7086-results/head-smoke.png \
  2>&1 | tee /root/mammoth-7086-results/head-smoke.log
```

Expected: a valid 1024×1024 RGB PNG and log entries naming `DiffusionEngine` and `MammothModa2DiTPipeline` for stage 1.

- [ ] **Step 4: Smoke-test the unchanged AR-only understanding path**

Run text-to-text with the already downloaded Preview checkpoint:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model /root/models/MammothModa2-Preview \
  --prompt "Explain multimodal generation in three sentences." \
  2>&1 | tee /root/mammoth-7086-results/preview-t2t.log
```

Create a deterministic local image with Pillow and run image-to-text:

```bash
python -c "from PIL import Image; Image.new('RGB',(256,256),(32,96,160)).save('/root/mammoth-7086-results/input.png')"
python examples/offline_inference/x_to_text/x_to_text.py \
  --model /root/models/MammothModa2-Preview \
  --image /root/mammoth-7086-results/input.png \
  --prompt "Describe the dominant color in this image." \
  2>&1 | tee /root/mammoth-7086-results/preview-i2t.log
```

Expected: both commands select `mammoth_moda2_ar.yaml`, return non-empty text, and never initialize the DiT stage.

- [ ] **Step 5: Verify image output and capture one-GPU peak memory**

Run:

```bash
python -c "from PIL import Image; image=Image.open('/root/mammoth-7086-results/head-smoke.png'); print(image.mode, image.size)"
grep -E "DiffusionEngine|MammothModa2DiTPipeline|peak.*memory|Model loading took" \
  /root/mammoth-7086-results/head-smoke.log
```

Expected: `RGB (1024, 1024)` and stage-1 shared-runtime/load-memory evidence.

- [ ] **Step 6: Provision the paired two-A800 baseline checkout**

On a two-A800 AutoDL instance, keep the head checkout and add a worktree at the pre-migration commit:

```bash
cd /root/vllm-omni-7086
git worktree add /root/vllm-omni-7086-baseline 32370968d00e34d704688c0b56792d279aac8aae
```

Use the same virtual environment and model directory for both checkouts.

- [ ] **Step 7: Start the baseline server with one A800 per stage**

Run in terminal A:

```bash
cd /root/vllm-omni-7086-baseline
source /root/vllm-omni-7086/.venv/bin/activate
vllm serve /root/models/MammothModa2-Preview --omni \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --stage-overrides '{"0":{"devices":"0"},"1":{"devices":"1"}}' \
  --port 8099 \
  --log-stats \
  2>&1 | tee /root/mammoth-7086-results/baseline-server.log
```

Expected: the health endpoint responds before benchmarking:

```bash
curl --fail http://127.0.0.1:8099/health
```

- [ ] **Step 8: Benchmark ten serial baseline requests**

Run in terminal B:

```bash
cd /root/vllm-omni-7086-baseline
source /root/vllm-omni-7086/.venv/bin/activate
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8099 \
  --endpoint /v1/images/generations \
  --model /root/models/MammothModa2-Preview \
  --task t2i \
  --dataset random \
  --num-prompts 10 \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --warmup-num-inference-steps 2 \
  --width 1024 \
  --height 1024 \
  --num-inference-steps 50 \
  --seed 42 \
  --extra-body '{"guidance_scale":4.0,"cfg_range":[0.0,1.0]}' \
  --return-stage-metrics \
  --save-dir /root/mammoth-7086-results/baseline-images \
  --output-file /root/mammoth-7086-results/baseline.json
```

Expected: ten successful requests and JSON fields for latency p50/p95, stage durations, throughput, and peak memory when exposed by the baseline path.

- [ ] **Step 9: Stop the baseline server and start the head server**

Stop terminal A with `Ctrl-C`, verify port 8099 is free, then run:

```bash
cd /root/vllm-omni-7086
source .venv/bin/activate
vllm serve /root/models/MammothModa2-Preview --omni \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --stage-overrides '{"0":{"devices":"0"},"1":{"devices":"1"}}' \
  --port 8099 \
  --log-stats \
  2>&1 | tee /root/mammoth-7086-results/head-server.log
```

Expected: stage 1 logs `StageDiffusionClient`, `DiffusionEngine`, and `MammothModa2DiTPipeline`; it must not log the legacy generation model runner as stage 1's executor.

- [ ] **Step 10: Benchmark ten serial head requests with the identical workload**

Run the same benchmark command from Step 8, changing only:

```text
working directory: /root/vllm-omni-7086
save directory:    /root/mammoth-7086-results/head-images
output file:       /root/mammoth-7086-results/head.json
```

Expected: ten successful requests with the same prompt source, seed, image size, steps, guidance, concurrency, and GPU placement as baseline.

- [ ] **Step 11: Compare outputs and performance evidence**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
from PIL import Image, ImageChops, ImageStat

root = Path('/root/mammoth-7086-results')
baseline = json.loads((root / 'baseline.json').read_text())
head = json.loads((root / 'head.json').read_text())
keys = ['latency_p50', 'latency_p95', 'request_throughput', 'peak_memory_mb_max', 'stage_durations_p50']
print({key: {'baseline': baseline.get(key), 'head': head.get(key)} for key in keys})

baseline_images = sorted((root / 'baseline-images').glob('*.png'))
head_images = sorted((root / 'head-images').glob('*.png'))
assert len(baseline_images) == len(head_images) == 10
for old_path, new_path in zip(baseline_images, head_images):
    old = Image.open(old_path).convert('RGB')
    new = Image.open(new_path).convert('RGB')
    assert old.size == new.size == (1024, 1024)
    mean_abs_diff = sum(ImageStat.Stat(ImageChops.difference(old, new)).mean) / 3.0
    print(old_path.name, new_path.name, f'mean_abs_pixel_diff={mean_abs_diff:.6f}')
PY
```

Expected: both runs contain ten 1024×1024 RGB images, metrics are printed side by side, and image differences are recorded rather than hidden. Exact pixel parity is the target because the head now honors seed 42; if the legacy baseline ignored that seed, report the observed quality comparison and use the head's repeated-seed determinism as the correctness claim.

- [ ] **Step 12: Preserve the complete validation artifacts**

Keep these files together under `/root/mammoth-7086-results/` for the final handoff and PR comment:

```text
baseline.json
head.json
baseline-server.log
head-server.log
head-smoke.log
head-smoke.png
preview-t2t.log
preview-i2t.log
input.png
baseline-images/
head-images/
```

Report the exact baseline and head values for `latency_p50`, `latency_p95`, `request_throughput`, `peak_memory_mb_max`, and `stage_durations_p50`. Also report the head commit from `git rev-parse HEAD`, the ten per-image mean absolute differences printed in Step 11, the AR-only smoke results, and the stage-1 runtime proof from `head-server.log`. Do not claim a speedup when the measurements show only parity or overhead.

### Task 8: Final verification and PR handoff

**Files:**
- Review all files changed from `32370968`.

- [ ] **Step 1: Run the focused CPU suite again after A800-driven corrections**

Run the complete command from Task 6 Step 1.

Expected: all tests pass.

- [ ] **Step 2: Run all pre-commit hooks against the branch diff**

Run:

```bash
pre-commit run --from-ref 32370968d00e34d704688c0b56792d279aac8aae --to-ref HEAD
```

Expected: all hooks pass.

- [ ] **Step 3: Verify the branch is clean and inspect the final diff**

Run:

```bash
git status --short
git diff --stat 32370968d00e34d704688c0b56792d279aac8aae..HEAD
git log --oneline 32370968d00e34d704688c0b56792d279aac8aae..HEAD
```

Expected: `git status --short` has no output; the diff is confined to the files in this plan; commits are small and ordered by topology, bridge, pipeline, tests/docs, and measured evidence.

- [ ] **Step 4: Prepare the PR description from verified artifacts**

The PR description must contain these exact sections and facts:

- `Summary`: migration from `LLM_GENERATION` to shared diffusion request mode; preserved AR conditioning, identity, parameters, and image output; explicit single-request/no-step boundary.
- `Validation`: the complete CPU pytest command and pass count; 2× A800 80 GB topology; Preview 1024×1024/50-step/guidance-4.0/seed-42 workload; baseline and head E2E p50/p95; baseline and head DiT p50; baseline and head peak memory; valid-image, determinism, and pixel-comparison results; stage-1 `DiffusionEngine` runtime proof.
- `Scope exclusions`: request batching, step execution, continuous batching, TeaCache/Cache-DiT, compilation, CUDA graphs, quantization, parallelism, and offload.
- Footer: `Refs #7075` and `Closes #7086`.

Copy numeric values directly from `/root/mammoth-7086-results/baseline.json` and `head.json`, and copy runtime evidence from `head-server.log`; do not estimate missing measurements.

- [ ] **Step 5: Request review without merging**

Create the PR only after the user approves the final diff and evidence. Request `@hsliuustc0106` and one diffusion-runtime maintainer such as `@wtomin`; mention PR #5357 only as a coordination boundary and do not include TeaCache code.
