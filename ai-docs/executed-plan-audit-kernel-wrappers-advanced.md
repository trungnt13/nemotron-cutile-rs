---
status: complete
goal: Audit the advanced async GPU kernel wrappers for shape safety, transfer behavior, error propagation, and host-contract parity.
prompt: >-
  You are auditing the complex GPU kernel wrappers in /Users/trungnt13/codes/nemotron-cutile-rs.
  Scope: nemotron-kernels/src/attention.rs, conv1d.rs, ssm.rs, embedding.rs, quantize.rs,
  and moe_routing.rs. Verify each async GPU wrapper correctly preserves shapes, transfers
  data safely, propagates errors, and matches the host kernel contract; identify fixture
  coverage gaps that matter for adding missing GPU validations; make precise fixes if needed,
  run focused validation if code changes, and update the audit todo.
created: 2026-03-19T09:45:08Z
finished: 2026-03-19T09:47:54Z
---

# Executed Plan — Audit Advanced Kernel Wrappers

## Research

- Reviewed the host and async wrapper implementations in:
  - `nemotron-kernels/src/attention.rs`
  - `nemotron-kernels/src/conv1d.rs`
  - `nemotron-kernels/src/ssm.rs`
  - `nemotron-kernels/src/embedding.rs`
  - `nemotron-kernels/src/quantize.rs`
  - `nemotron-kernels/src/moe_routing.rs`
- Cross-checked the validation surface in `nemotron-validate/src/main.rs`.
- Ran the scoped crate baseline before changes with `cargo test -p nemotron-kernels --quiet`.

## Findings

1. All scoped async wrappers still intentionally implement the current bridge design: device-to-host copy, host kernel execution, then host-to-device copy (except MoE routing, which returns host vectors by design, and INT4 dequantize, which accepts packed host bytes because `GpuTensor` is `f32`-only).
2. For the current contracts, shape and error handling are mostly correct:
   - `attention` reconstructs `[batch, query_seq, query_heads, head_dim]` from `AttentionShape`.
   - `conv1d` preserves the input tensor shape metadata on the way back.
   - `ssm` reconstructs `[sequence_len, channel_count]` output and `[channel_count, state_size]` final state.
   - `embedding` reconstructs `[token_count, hidden_size]`.
   - `quantize` dequantize returns `[value_count]`.
   - `moe_routing` returns host `indices`/`weights`, matching its current API contract.
3. The wrappers rely on host-kernel length validation after D2H transfer; they do not currently reject semantically different `GpuTensor` metadata when element counts match. That is consistent with the present flat-slice host contracts, but it is an important future constraint once real GPU kernels consume tensor metadata or raw cutile tensors directly.
4. The highest-value validation gap is not a logic bug in the wrappers themselves, but missing async wrapper coverage:
   - `attention`, `ssm`, and `quantize` had no async wrapper tests covering shape reconstruction.
   - `moe_route_softmax` had no async wrapper coverage.
   - `nemotron-validate` includes host fixture validation for `causal_conv1d` and `moe_routing`, but `run_gpu_kernel_validation(...)` still skips every wrapper in this audit scope.
   - `data/reference_kernels/fixtures.json` currently has fixture keys for `causal_conv1d` and `moe_routing`, but not for `attention`, `ssm`, `embedding`, or `quantize`.

## What Was Changed

### `nemotron-kernels/src/attention.rs`
- Strengthened the existing async wrapper test to assert the reconstructed output shape.

### `nemotron-kernels/src/conv1d.rs`
- Strengthened the existing async wrapper test to assert that the wrapper preserves the input tensor shape metadata.

### `nemotron-kernels/src/embedding.rs`
- Strengthened the existing async wrapper test to assert the `[token_count, hidden_size]` output shape.

### `nemotron-kernels/src/ssm.rs`
- Added async wrapper coverage for:
  - parity with the host fallback when `initial_state` is provided
  - correct reconstruction of output and final-state shapes
  - propagation of `InvalidDeltaT` from the host kernel

### `nemotron-kernels/src/quantize.rs`
- Added async wrapper coverage for:
  - INT4 dequantize parity on an odd element count
  - preservation of the `[value_count]` output shape
  - propagation of invalid quantization parameter errors

### `nemotron-kernels/src/moe_routing.rs`
- Added async wrapper coverage for `moe_route_softmax` to match existing coverage for `moe_route`.

## Alignment Notes

- No functional wrapper bug required a behavioral code fix inside the audited implementations.
- The added tests were the most precise way to lock in the current contract without inventing new wrapper semantics.
- I did not change `quantize` to accept device-resident packed INT4 data or change `moe_routing` to return `GpuTensor`s, because those would be API/architecture changes rather than clear correctness fixes for the current host-fallback design.

## Verification

- Baseline before changes: `cargo test -p nemotron-kernels --quiet` ✅
- After changes: `cargo test -p nemotron-kernels --quiet` ✅ (125 passed, 1 ignored)

## Follow-up Notes

1. Add reference fixtures and async GPU validation harness coverage for `attention`, `ssm`, `embedding`, and `quantize`.
2. Extend `run_gpu_kernel_validation(...)` in `nemotron-validate/src/main.rs` to exercise `conv1d` and `moe_routing` wrappers too, since those kernels already have host fixtures.
3. Before introducing real cutile kernels, decide whether wrapper contracts should remain flat-slice-plus-shape or start validating `GpuTensor` metadata explicitly.
