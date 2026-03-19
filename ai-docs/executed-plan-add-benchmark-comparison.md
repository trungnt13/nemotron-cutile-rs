---
status: complete
goal: Add a benchmark/comparison entry point under nemotron-validate for host vs GPU-wrapper timing and parity
prompt: >-
  You are implementing benchmark/comparison infrastructure in
  /Users/trungnt13/codes/nemotron-cutile-rs.

  Scope:
  - Primary target: nemotron-validate crate.
  - Goal: add a benchmark/comparison entry point that times host vs GPU wrapper
    paths and reports output parity, without adding new dependencies unless truly
    necessary.
  - Prefer adding a new binary/module under nemotron-validate rather than invasive
    refactors.
  - Use existing fixture data if possible; synthetic E2E runtime is acceptable for
    model-level comparison.
  - Avoid files currently being edited by other agents: do not touch
    nemotron-kernels/src/tensor.rs or device.rs.
  - Run focused tests/build for the touched crate(s).
created: 2026-03-19T09:54:59Z
finished: 2026-03-19T09:54:59Z
---

# Executed Plan — Add Benchmark Comparison

## Research and Plan

- Inspected `nemotron-validate/src/main.rs` and confirmed the crate already has host validation plus async GPU-wrapper validation for GEMM, RMSNorm, softmax, SiLU, and ReLU².
- Inspected `nemotron-model/src/model.rs` and confirmed `forward_tokens_gpu` exists as the model-level wrapper path, with explicit host embedding / transfer / wrapper execution / transfer-back behavior.
- Chose a surgical CLI extension over a new shared library: add a `benchmark` mode to the existing `nemotron-validate` binary and place the implementation in a new `nemotron-validate/src/benchmark.rs` module.
- Reused bundled kernel fixtures from `data/reference_kernels/fixtures.json` and the synthetic e2e fixture from `data/reference_outputs/fixtures.json` for the model comparison path.

## What Was Changed

### `nemotron-validate/src/benchmark.rs`
- Added a benchmark harness that measures average per-iteration host time vs GPU-wrapper time using `std::time::Instant` and no new dependencies.
- Added comparisons for GEMM, RMSNorm, softmax, SiLU, ReLU², and a model-level `forward_tokens` vs `forward_tokens_gpu` run based on the bundled synthetic runtime.
- Added explicit notes clarifying that current GPU wrappers still delegate to host kernels, so the benchmark measures wrapper/transfer overhead plus parity rather than real GPU compute speed.
- Reported `max_abs_diff` for each comparison and marked each result PASS/FAIL using the same tolerance profile already used by validation.
- Added an async unit test that exercises the bundled benchmark fixtures.

### `nemotron-validate/src/main.rs`
- Added CLI parsing for a `benchmark` / `--benchmark` entry point while preserving the existing validation invocation format.
- Refactored validation printing into a helper and added benchmark-mode output that prints host timing, GPU-wrapper timing, wrapper-vs-host ratio, max absolute diff, and iteration count.
- Added focused CLI parsing unit coverage for the new benchmark entry point.

### `nemotron-validate/tests/validator_integration.rs`
- Added a binary-level integration test that runs `nemotron-validate benchmark ...` and asserts the benchmark note, timing/parity output, and model comparison line.

## Alignment Notes

- The implementation stayed inside `nemotron-validate` and did not touch `nemotron-kernels/src/tensor.rs` or `nemotron-kernels/src/device.rs`.
- No new dependencies were added; the benchmark uses only standard-library timing plus existing crate APIs.
- The benchmark output is explicit that current results are not real GPU compute benchmarks.

## Verification

- `cargo fmt --all` ✅
- `cargo test -p nemotron-validate --quiet` ✅
- `cargo build -p nemotron-validate --quiet` ✅
- `cargo run -p nemotron-validate --quiet -- benchmark data/reference_kernels data/reference_outputs` ✅
  - Observed 6/6 PASS with `max_abs_diff=0.000000` and wrapper timings slower than host, matching the expected host-fallback overhead profile.

## Next Steps

- When real cutile kernels land, the benchmark can keep the same interface but should be reinterpreted as actual GPU compute timing rather than transport-overhead timing.
- If desired later, benchmark mode could be extended to cover additional wrappers such as Conv1D or MoE once async device paths exist for them.
