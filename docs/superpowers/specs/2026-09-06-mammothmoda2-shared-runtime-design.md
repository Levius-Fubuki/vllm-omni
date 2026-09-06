# MammothModa2 Request-Mode Shared Diffusion Runtime Design

- **Issue:** https://github.com/vllm-project/vllm-omni/issues/7086
- **Parent:** https://github.com/vllm-project/vllm-omni/issues/7075
- **Baseline:** `32370968d00e34d704688c0b56792d279aac8aae`
- **Owner:** `Levius-Fubuki`

## Objective

Migrate the MammothModa2 image-generation DiT stage from the generation-model
runner (`StageExecutionType.LLM_GENERATION`) to the shared diffusion runtime
(`StageExecutionType.DIFFUSION`). The first change supports request-mode,
single-request execution only and preserves the current MammothModa2-Preview
text-to-image result and the existing AR-only understanding pipelines.

This is a runtime-foundation change. It does not attempt to improve kernels,
introduce cache algorithms, or add batching. Its value is to make later
diffusion features consume the same configuration, request lifecycle, and
output contracts as other native diffusion pipelines.

## Chosen Approach

Use a native shared-runtime pipeline rather than a dual-mode implementation or
an outer wrapper around the legacy generation runner.

`MammothModa2DiTPipeline` will adopt the native diffusion construction and
request contracts:

```text
OmniDiffusionConfig
    -> MammothModa2DiTPipeline
DiffusionRequestBatch (exactly one request)
    -> request-mode denoising and VAE decode
DiffusionOutput
```

The module-path compatibility shim under `vllm_omni.model_executor` remains an
import alias only. It must not preserve or dispatch to the old runtime.

### Rejected alternatives

1. **Dual-mode pipeline:** Accepting both `VllmConfig`/`OmniOutput` and
   `OmniDiffusionConfig`/`DiffusionOutput` would reduce the initial cut-over but
   leave two configuration and output lifecycles in the same class. That makes
   silent fallback possible and complicates all later scheduling work.
2. **Wrapper around the generation runner:** A thin diffusion wrapper would
   minimize edits, but ownership of model loading, profiling, cache hooks, and
   output metadata would remain ambiguous. The wrapper would become technical
   debt as soon as request batching or step execution is added.

## Architecture and Data Flow

### Stage topology

The image-generation topology keeps stage 0 as `LLM_AR` and changes stage 1 to
`DIFFUSION`. The AR-only topology remains unchanged.

The deployment contract continues to set `max_num_seqs: 1` for stage 1. The
pipeline also validates the single-request condition at runtime so a future
configuration change cannot accidentally claim batching support.

### AR-to-DiT bridge

The existing `ar2dit` processor remains responsible for reconstructing the
ordered AR payload:

- prompt token IDs plus generated visual token IDs;
- token-aligned hidden states;
- the question/answer boundary;
- target image height and width;
- guidance scale, CFG interval, denoising-step count, and seed-bearing sampling
  parameters;
- the logical request identity supplied by the shared request lifecycle.

The bridge will emit the prompt/additional-information shape expected by an
`OmniDiffusionRequest`. It will not split text and image conditioning itself;
that model-aware split remains inside `MammothModa2DiTPipeline`, where the
checkpoint configuration supplies visual and generation token IDs.

No KV cache crosses the AR-to-DiT boundary in this design. If implementation
tracing discovers an actual KV dependency, work stops and the dependency is
recorded on #7086 before adopting any HYImage KV-manager pattern.

### Pipeline construction

The pipeline constructor receives `OmniDiffusionConfig`. Model metadata and
checkpoint configuration are resolved through the same shared-runtime loading
path used by native diffusion models. Component discovery continues to expose
the DiT, condition refiner, and VAE modules for framework integrations, but
this change does not enable offload or parallel execution.

The pipeline is registered as a built-in diffusion model architecture. Registry
resolution must instantiate this class directly through the shared diffusion
loader; the model-executor registry must not be required for stage 1.

### Request-mode forward path

`forward` accepts one `DiffusionRequestBatch` and performs the current complete
request in one call:

1. Validate that the batch contains exactly one request.
2. Extract AR conditioning and image dimensions from the request prompt.
3. Extract MammothModa2 generation controls from request-local sampling
   parameters.
4. Split and normalize AR hidden states into text and image conditioning.
5. Run the existing denoising loop and VAE decode without changing numerical
   behavior.
6. Return a `DiffusionOutput` containing the generated image payload and shared
   output metadata.

The implementation may extract small private helpers from the current forward
method to isolate input validation and output construction. It must not rewrite
the transformer, scheduler mathematics, or VAE path.

### Configuration ownership

Stage configuration is converted once into `OmniDiffusionConfig` by the shared
runtime and is the pipeline's only runtime configuration source. Tests will
prove that relevant stage fields reach the executing diffusion backend.

PR #5357 currently modifies MammothModa2 and diffusion-cache configuration.
This change will not import its TeaCache behavior. Cache fields remain owned by
the shared runtime, and the MammothModa2 pipeline will neither install cache
hooks nor introduce model-specific cache configuration.

## Error Handling

The migration must fail explicitly for unsupported or malformed requests:

- a request batch whose size is not exactly one;
- missing AR hidden states, token IDs, or answer boundary;
- hidden-state/token-count mismatch;
- no generated visual-token hidden states;
- invalid image dimensions or denoising parameters;
- incompatible MammothModa2 checkpoint configuration.

Exceptions propagate through the shared diffusion request lifecycle as failed
request output. The code must not retry through `LLM_GENERATION`, synthesize
conditioning, or return a nominally successful empty image.

## Testing Strategy

Implementation follows red-green-refactor. Each behavior is first represented
by a focused failing test.

### CPU tests

1. Pipeline topology selects `DIFFUSION` for stage 1 and leaves the AR-only
   topology unchanged.
2. Diffusion registry resolves `MammothModa2DiTPipeline` through its native
   module path.
3. Pipeline construction consumes `OmniDiffusionConfig` and preserves component
   discovery.
4. `ar2dit` preserves token-aligned hidden states, dimensions, sampling values,
   and request-scoped prompt data.
5. Single-request forward input is decoded from `DiffusionRequestBatch` and the
   result uses `DiffusionOutput`.
6. Multi-request input and each malformed AR payload fail with a stable,
   descriptive exception.
7. Existing MammothModa2 configuration, stage-input, and AR-only tests remain
   green.

Heavy model modules will be constructed with small test doubles only at external
weight-loading boundaries. Tests should exercise the real request parsing,
validation, and output-building code rather than asserting mock call counts.

### A800 validation

GPU work begins only after CPU tests and static checks pass.

1. **Single A800 80GB:** establish the legacy baseline and shared-runtime smoke
   test using MammothModa2-Preview, 1024x1024, 50 steps, guidance 4.0, seed 42.
2. **Two A800 80GB in one host:** place AR on GPU 0 and DiT on GPU 1; run the
   same baseline and candidate revision on identical hardware.
3. Warm each revision three times, then collect 30 measured requests for p50 and
   p95 DiT latency, end-to-end latency, and peak GPU memory.
4. Save the generated image, exact command, commit SHA, environment report, and
   profiler/log output. Confirm through logs that the shared diffusion backend
   executed stage 1 and no legacy fallback occurred.
5. Run Preview text-to-text and image-to-text smoke checks through the AR-only
   configuration to detect topology regressions.

Baseline and candidate runs must use the same host, driver, software
environment, prompt, seed, and generation parameters. Performance is supporting
evidence; numerical and behavioral correctness is the merge gate.

## Delivery Boundaries

### Included

- stage-1 topology migration;
- native diffusion registry entry;
- shared configuration, request, and output contracts;
- preservation of AR-to-DiT conditioning and request identity;
- focused CPU tests and A800 correctness/performance evidence;
- documentation necessary to run the migrated request-mode path.

### Excluded

- request-level batching;
- step execution, cancellation checkpoints, and continuous batching;
- TeaCache, Cache-DiT, or AR/DiT KV-cache transport;
- compilation and CUDA graphs;
- quantization, tensor/sequence/pipeline parallelism, and CPU offload;
- transfer-copy optimization and AR/DiT pipeline overlap;
- transformer, attention, FFN, or VAE kernel changes.

## Completion Criteria

The change is complete when all of the following are true:

- MammothModa2 stage 1 is declared and executed as `DIFFUSION`;
- registry loading and stage configuration use only shared diffusion contracts;
- a valid single request produces a `DiffusionOutput` image with behavior
  matching the legacy baseline;
- unsupported batching and malformed payloads fail explicitly;
- Preview/Dev AR-only understanding behavior remains intact;
- targeted CPU tests and repository lint checks pass;
- the dual-A800 run records baseline/candidate correctness, p50/p95 latency,
  peak memory, environment, and proof of shared-backend execution;
- the PR uses `Refs #7075` and links #7086 without closing the umbrella issue.
