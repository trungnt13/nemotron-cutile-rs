---
status: implemented-with-open-gaps
goal: Validate the cutile-backed GPU wrapper path, extend local validation and benchmarking, and record the remaining GPU-server blocker
prompt: "double check and validate the cutile kernel implementation, make sure the tests run and pass on GPU server, the outputs match, and check which precision used in cutile, run the full integration test for benchmark and comparing outputs of Nemotron model, make sure all run with cutile and GPU"
created: 2026-03-19T09:39:03Z
finished: 2026-03-19T10:03:48Z
---

# Executed Plan — Validate Cutile GPU Wrapper Path

## Summary

Validated and tightened the current `cutile` integration as it exists today: a Linux-only GPU tensor/device scaffold with async wrapper paths that still delegate kernel math to host implementations. The session improved wrapper safety, expanded GPU-wrapper validation from `5/5` to `7/7` across all bundled kernel fixtures, added a benchmark mode that compares host vs GPU-wrapper timing and output parity, and recorded the active precision policy.

The main open gap is unchanged from the research phase: this repository still does **not** contain real `#[cutile::module]` compute kernels, and no Linux+CUDA GPU server access was available in this session to rerun the full suite remotely.

## What Was Done

### 1. Hardened GPU tensor/device scaffolding

- Tightened `GpuTensor` shape validation:
  - reject zero-sized dimensions
  - detect shape-multiplication overflow
  - fix `numel()` consistency
- Propagated Linux sync transfer/allocation failures through `TensorError` instead of panicking in the fallible paths.
- Added `try_to_host()` so new synchronous callers can avoid panic-based reads.
- Made repeated same-thread `GpuDevice::new()` creation safe for the same ordinal under cutile's thread-local initialization model.

### 2. Audited and tightened kernel wrapper behavior

- Basic GPU wrapper fixes:
  - `gemm_into(...)` now validates the destination tensor length instead of silently replacing incompatible outputs
  - GEMM and RMSNorm wrappers validate inputs before transfer
  - async `sigmoid_in_place(...)` now exists alongside the other in-place activation wrappers
- Advanced GPU wrapper coverage:
  - added/strengthened async tests for `attention`, `conv1d`, `embedding`, `ssm`, `quantize`, and `moe_routing`
  - locked in current shape reconstruction and error-propagation behavior for the D2H -> host -> H2D path

### 3. Audited NN/model wrapper surfaces

- `nemotron-nn` GPU wrapper paths were tightened to stop silently reinterpreting flattened tensors when `GpuTensor` shape metadata disagreed with the logical layer shape.
- `nemotron-model` GPU forward path now preserves the host-visible empty-input behavior by short-circuiting through the host path before attempting a zero-length GPU tensor upload.
- Existing and added tests keep host/GPU-wrapper parity explicit at the NN/model levels.

### 4. Extended validation coverage in `nemotron-validate`

- `run_gpu_kernel_validation(...)` now covers the two remaining bundled kernel fixtures that already existed in `data/reference_kernels/fixtures.json`:
  - `gpu/causal_conv1d`
  - `gpu/moe_routing`
- The validator now prints an explicit note that `attention`, `ssm`, `embedding`, and `quantize` are **not** GPU-validated there because bundled fixtures do not exist for them.
- Binary integration coverage was updated to assert the new GPU validation lines and summary.

### 5. Added benchmark/comparison mode

- Added `benchmark` / `--benchmark` mode to `nemotron-validate`.
- Added `nemotron-validate/src/benchmark.rs` to compare host vs GPU-wrapper timing and `max_abs_diff` for:
  - GEMM
  - RMSNorm
  - softmax
  - SiLU
  - ReLU²
  - synthetic model `forward_tokens` vs `forward_tokens_gpu`
- The benchmark output explicitly states that the current GPU-wrapper path measures transfer/wrapper overhead plus parity, not real GPU compute speed.

### 6. Documented precision behavior

- Recorded the current precision matrix in `ai-docs/executed-plan-audit-precision-matrix.md`.
- Current policy:
  - storage/API surfaces are `f32`
  - numerically sensitive reductions commonly accumulate in `f64`
  - INT4 support exists as affine packed-weight infrastructure, but not as a real active cutile execution path

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Scope GPU validation by real fixtures | Validate only kernels with bundled fixtures | Avoids pretending `attention`, `ssm`, `embedding`, or `quantize` have reference-backed parity coverage when they do not |
| Benchmark existing wrapper path as-is | Measure host vs GPU-wrapper timing and parity now | Gives a useful baseline without waiting for real cutile kernels |
| Preserve host-visible behavior | Keep wrapper errors/parity aligned with host code | Makes future real-kernel swaps easier to validate against a stable contract |
| Treat server verification as blocked | Record the blocker instead of fabricating a result | No GPU server access was available in this session |

## Verification

Local verification completed successfully:

```bash
cargo test --workspace --quiet
cargo run -q -p nemotron-validate -- data/reference_kernels data/reference_outputs
cargo run -q -p nemotron-validate -- benchmark data/reference_kernels data/reference_outputs
```

Observed results:

- workspace tests: **204 passed**, **1 ignored**
- host validation: **9/9 passed**
- GPU-wrapper validation: **7/7 passed**
- benchmark comparisons: **6/6 passed**

Representative benchmark output confirms the current wrapper path is slower than host, as expected for a host-delegating implementation:

- `benchmark/gemm`: wrapper ~`1.12x` host
- `benchmark/rms_norm`: wrapper ~`2.89x` host
- `benchmark/model/constant_world_runtime/forward_tokens`: wrapper ~`1.31x` host

All observed `max_abs_diff` values stayed at `0.000000` in the local synthetic/bundled coverage.

## Blockers / Open Gaps

1. **No GPU server access in this session**
   - `gpu-server-build-test` is blocked
   - `gpu-benchmark-run` is blocked
   - Linux+CUDA reruns still need to be performed on the real target machine

2. **Real cutile compute kernels do not exist yet**
   - current async GPU paths still delegate to host kernels
   - benchmark numbers are wrapper/transfer overhead measurements, not true GPU compute performance

3. **Fixture coverage is still incomplete**
   - bundled GPU fixture validation still does not cover `attention`, `ssm`, `embedding`, or `quantize`
   - that is a fixture limitation, not an implementation claim

## Next Steps

1. Run the verified command set on the actual Linux+CUDA GPU server once access details are available:
   - `cargo build --workspace`
   - `cargo test --workspace`
   - `cargo run -p nemotron-validate -- data/reference_kernels data/reference_outputs`
   - `cargo run -p nemotron-validate -- benchmark data/reference_kernels data/reference_outputs`
2. If remote fixture data is missing, copy or regenerate `data/reference_kernels/fixtures.json` and `data/reference_outputs/fixtures.json` first.
3. When real cutile kernels are implemented, re-run the same benchmark/validation path with tolerance-oriented parity expectations instead of exact wrapper parity.
