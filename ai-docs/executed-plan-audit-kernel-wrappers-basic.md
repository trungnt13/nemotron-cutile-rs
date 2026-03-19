---
status: complete
goal: Audit the simple async GPU kernel wrappers for shape safety, error propagation, and host-contract parity
prompt: >-
  You are auditing the simple GPU kernel wrappers in /Users/trungnt13/codes/nemotron-cutile-rs.
  Scope: nemotron-kernels/src/gemm.rs, nemotron-kernels/src/rms_norm.rs,
  nemotron-kernels/src/softmax.rs, and nemotron-kernels/src/activations.rs.
  Verify each async GPU wrapper preserves shapes, transfers data safely,
  propagates errors, and matches the host kernel contract; make precise fixes if
  needed, run focused validation if code changes, and note anything relevant for
  downstream GPU validation or benchmarking work.
created: 2026-03-19T09:45:10Z
finished: 2026-03-19T09:51:23Z
---

# Executed Plan — Audit Basic Kernel Wrappers

## Summary

Audited the four simple async GPU wrapper modules and then synced the audit with actual code changes after a follow-up check showed only the audit doc had landed. The wrappers still intentionally use the D2H -> CPU kernel -> H2D path, so they remain correctness bridges rather than performance paths. The main contract issue found was `gemm_into`, which previously discarded the caller-provided output tensor instead of honoring `_into` semantics.

## Research and Findings

- `gemm`, `rms_norm`, `gated_rms_norm`, `softmax`, `silu`, `relu2`, and `sigmoid` all delegate to host kernels after async transfer, so numerical behavior currently matches host execution exactly.
- `softmax` and `gemm` preserve host precision characteristics because they reuse the host kernels' f64 accumulation before writing f32 outputs.
- `rms_norm` and `gated_rms_norm` preserve the input tensor shape on the way back to the device.
- `softmax`, `silu`, `relu2`, and `sigmoid` also preserve input shapes, which matters for downstream model code that expects tensor metadata to survive the wrapper hop.
- `gemm_into` was the only clear host-contract mismatch in scope: it allocated a fresh tensor and replaced `output` without validating the caller's output buffer size.

## What Was Changed

### `nemotron-kernels/src/gemm.rs`
- Added pre-transfer GPU tensor length validation for `lhs` and `rhs` against `GemmShape`.
- Fixed `gemm_into` to validate `output.numel()` before any transfer.
- Changed `gemm_into` to preserve the caller-provided output shape metadata while filling it with the computed GEMM result, rather than silently reshaping by replacing it with a newly shaped `[m, n]` tensor.
- Added GPU tests covering shape preservation and output-length mismatch handling for `gemm_into`.

### `nemotron-kernels/src/rms_norm.rs`
- Added pre-transfer wrapper validation for empty input, negative epsilon, and mismatched `weight` / `gate` lengths.
- Added GPU tests for gated RMSNorm parity and mismatched weight rejection.

### `nemotron-kernels/src/softmax.rs`
- Simplified async error propagation to use the existing `From<TensorError> for SoftmaxError` conversion consistently.
- Added a GPU test confirming that softmax preserves higher-rank shape metadata.

### `nemotron-kernels/src/activations.rs`
- Made in-place async wrappers preserve shape metadata explicitly via a cloned shape.
- Added the missing async `sigmoid_in_place` wrapper to match the host activation surface.
- Added GPU tests for sigmoid parity and in-place sigmoid behavior.

## Verification

- Baseline before changes: `cargo test -p nemotron-kernels --quiet` ✅ (116 passed, 1 ignored doc/integration placeholder)
- After landing the wrapper fixes: `cargo test -p nemotron-kernels --quiet` ✅ (132 passed, 1 ignored doc/integration placeholder)

## Downstream Notes

- GPU validation should continue to expect exact equality for these wrappers today, because the device path is still just transport around the host kernel.
- Once real cutile kernels land, validation should switch these audited paths to tolerance-based comparisons instead of exact equality, especially for GEMM and softmax.
- Benchmarking should still expect these wrappers to underperform the pure CPU path on current host-fallback implementations because every call performs D2H + CPU compute + H2D.
