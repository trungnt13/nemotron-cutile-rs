---
status: complete
goal: Extend nemotron-validate GPU fixture validation to cover the remaining bundled kernel fixtures
prompt: >-
  Please extend your current nemotron-validate work to also complete todo
  `add-missing-gpu-validations` while you are already editing `src/main.rs`.

  Important scope refinement based on the actual fixture inventory:
  - `data/reference_kernels/fixtures.json` currently contains: `causal_conv1d`,
    `gemm`, `moe_routing`, `relu2`, `rms_norm`, `silu`, `softmax`
  - So in `run_gpu_kernel_validation(...)`, please add GPU wrapper validation for
    the kernels with existing fixtures that are still missing: `conv1d` and
    `moe_routing`
  - Do NOT fake fixture-based validation for `attention`, `ssm`, `embedding`, or
    `quantize` because the reference fixtures are not present. If useful, add a
    clear note/report entry stating those fixture gaps.
created: 2026-03-19T09:57:13Z
finished: 2026-03-19T09:57:13Z
---

# Executed Plan — Add Missing GPU Validations

## Research and Plan

- Confirmed `data/reference_kernels/fixtures.json` currently provides bundled fixtures only for `causal_conv1d`, `gemm`, `moe_routing`, `relu2`, `rms_norm`, `silu`, and `softmax`.
- Confirmed `nemotron-kernels/src/conv1d.rs` already exposes the async wrapper `depthwise_causal_conv1d(...)` and `nemotron-kernels/src/moe_routing.rs` already exposes the async wrapper `moe_route_softmax(...)`.
- Chose a surgical extension inside `nemotron-validate/src/main.rs` so the new coverage stays aligned with the existing GPU validation flow and the benchmark entry point remains unchanged.

## What Was Changed

### `nemotron-validate/src/main.rs`
- Extended `run_gpu_kernel_validation(...)` to validate `gpu/causal_conv1d` against the existing bundled fixture by running the async wrapper per batch, applying the fixture bias on host just like the host validation path, and comparing against the expected output with the existing tolerance.
- Extended `run_gpu_kernel_validation(...)` to validate `gpu/moe_routing` against the existing bundled `moe_routing` fixture via `moe_route_softmax(...)`, checking both top-k indices and weights.
- Added a GPU-validation note that explicitly calls out fixture gaps for `attention`, `ssm`, `embedding`, and `quantize`, rather than inventing unsupported validations.
- Added an async unit test asserting the bundled GPU validation report now includes Conv1D and MoE routing.

### `nemotron-validate/tests/validator_integration.rs`
- Tightened the binary-level validation test to assert the new `gpu/causal_conv1d` and `gpu/moe_routing` PASS lines, the updated `gpu summary: 7/7`, and the fixture-gap note.

## Alignment Notes

- The implementation only validated kernels with actual bundled reference fixtures, matching the explicit scope refinement.
- No fake fixture-based validation was added for `attention`, `ssm`, `embedding`, or `quantize`.
- The benchmark mode added earlier remained compatible and was re-run after the `main.rs` change.

## Verification

- `cargo fmt --all` ✅
- `cargo test -p nemotron-validate --quiet` ✅
- `cargo run -p nemotron-validate --quiet -- data/reference_kernels data/reference_outputs` ✅
  - Observed `gpu summary: 7/7 gpu validations passed`, including `gpu/causal_conv1d` and `gpu/moe_routing`.
- `cargo run -p nemotron-validate --quiet -- benchmark data/reference_kernels data/reference_outputs` ✅
  - Observed benchmark mode still passing `6/6 comparisons passed`.

## Next Steps

- If new bundled kernel fixtures are added later for `attention`, `ssm`, `embedding`, or `quantize`, they can be wired into the same GPU validation report pattern without changing the CLI surface.
- Once real GPU kernels replace host-fallback wrappers, these validations should continue to check parity but may eventually need wider tolerances for numerically different implementations.
