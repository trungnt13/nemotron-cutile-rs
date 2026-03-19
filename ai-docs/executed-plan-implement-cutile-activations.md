---
status: complete
goal: Implement real cutile-backed activation kernels in nemotron-kernels while preserving host fallback semantics and parity
prompt: >-
  You are implementing todo `implement-cutile-activations` in this repository.
  Implement real cutile-backed activations for the supported GPU activation kernels in
  `nemotron-kernels/src/activations.rs`, preserving the current public API, host fallback semantics,
  and parity guarantees. Reuse Wave 1 patterns from `nemotron-kernels/src/{rms_norm,softmax,gemm}.rs`
  and shared helpers in `nemotron-kernels/src/tensor.rs` where appropriate. Update any benchmark/validator
  reporting only if needed and keep changes surgical.
created: 2026-03-19T13:33:51Z
finished: 2026-03-19T14:35:10Z
---

# Executed Plan — Implement cutile Activations

## Summary

Implemented Linux-only cutile-backed async activation kernels for SiLU, ReLU², and sigmoid in `nemotron-kernels/src/activations.rs` while preserving the existing host APIs and non-Linux host fallback behavior. The real device path now activates when the flattened tensor width is a supported power of two up to 4096; unsupported Linux shapes and all non-Linux targets still route through the host bridge.

## Research and Decisions

- Wave 1 established the pattern of exposing platform-primary kernel metadata while keeping runtime fallback heuristics inside the async wrapper. The activation module now follows that same split.
- cutile’s current 1D tile compilation requires power-of-two tile extents, so the activation dispatch heuristic intentionally matches that constraint instead of attempting broader shapes and risking runtime compilation failures.
- Flattening through `GpuTensor::cutile_tensor_for_shape(&[numel])` preserves the public tensor API while allowing the cutile kernels to operate over a single supported 1D tile.
- The benchmark note needed a narrow wording update because activations are no longer always host-bridged on Linux.

## What Changed

### `nemotron-kernels/src/activations.rs`

- Added Linux-only cutile registry metadata for `SILU`, `RELU2`, and `SIGMOID`, while keeping non-Linux metadata on `HostFallback`.
- Added a shared Linux dispatch heuristic that enables cutile only for flattened widths that are powers of two up to 4096 and fit `i32` launch bounds.
- Added Linux-only `#[cutile::module]` kernels for SiLU, ReLU², and sigmoid using `load_tile_like_1d`, `exp`, `negf`, `max_tile`, and `constant`.
- Reworked the async GPU wrappers to choose between the cutile path and the existing host bridge without changing the public function signatures.
- Simplified the async in-place wrappers to delegate through the out-of-place async API and then write the result back, preserving tensor shapes.
- Updated focused tests to cover platform-reported backends, the cutile safety envelope, Linux cutile dispatch, parity, and shape preservation.

### `nemotron-validate/src/benchmark.rs`

- Updated the benchmark note so it accurately reports that Linux can now run elementwise activations on cutile for supported power-of-two flattened widths.

## Verification

### Local

- Baseline before changes: `cargo test -p nemotron-kernels activations --quiet` ✅ (10 passed)
- Focused activation tests after changes: `cargo test -p nemotron-kernels activations --quiet` ✅ (11 passed)
- Relevant crate tests after changes: `cargo test -p nemotron-kernels --quiet` ✅ (139 passed, 1 ignored integration placeholder)
- Validate binary compile/smoke without bundled fixtures: `cargo test -p nemotron-validate --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs` ✅
- Full workspace check: `cargo test --workspace --quiet` ⚠️ expected fixture-loading failures in `nemotron-validate` because `data/reference_kernels/fixtures.json` is absent in this worktree; all other workspace crates passed before that runtime data dependency failed.

### Linux / `tn`

After rsyncing the worktree to `tn:~/codes/nemotron-cutile-rs--cutile-activations` and exporting the required CUDA / LLVM environment variables:

- `cargo test -p nemotron-kernels activations --quiet` ✅ (12 passed)
- `cargo test -p nemotron-kernels --quiet` ✅ (142 passed, 1 ignored integration placeholder)

## Alignment Notes

- The public Rust activation API remains unchanged: the host scalar / host slice functions, async wrappers, and in-place async wrappers keep their existing signatures.
- Unsupported Linux activation widths intentionally stay on the host bridge instead of failing cutile compilation, matching the requested fallback semantics.
- Benchmark / validator code was only touched where the previous reporting text had become inaccurate.

## Next Steps

- If later waves broaden cutile’s supported 1D tile shapes, the activation heuristic can be relaxed in one place (`select_cutile_block_size`) without changing the public API.
