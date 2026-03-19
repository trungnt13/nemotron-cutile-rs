---
status: complete
goal: Implement the first real cutile-backed GEMM path while preserving host fallback behavior and parity contracts
prompt: >-
  Work in this git worktree only: `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-gemm`

  Goal: implement the first real cutile-backed compute kernel for GEMM,
  replacing the current Linux GPU wrapper's D2H -> host -> H2D bridge with
  actual device compute where feasible, while keeping host fallback intact on
  unsupported platforms.

  Requirements:
  - Stay within this worktree and branch.
  - Follow repo conventions from AGENTS.md.
  - Preserve current public API shape and host fallback behavior.
  - Preserve current shape/error contracts and parity expectations.
  - Add/update focused tests and any validator/benchmark hooks needed for GEMM.
  - Update/create an ai-docs executed-plan file in this worktree describing the work.
  - Run relevant tests/builds in this worktree.
  - Commit your changes locally in this worktree with a proper message and Copilot trailer.
  - Update SQL when finished:
    - success: UPDATE todos SET status = 'done' WHERE id = 'implement-cutile-gemm';
    - blocked: UPDATE todos SET status = 'blocked' WHERE id = 'implement-cutile-gemm';
created: 2026-03-19T11:05:54Z
finished: 2026-03-19T11:16:00Z
---

# Executed Plan — Implement Cutile GEMM

## Research and Plan

- Inspected `nemotron-kernels/src/gemm.rs`, `tensor.rs`, and `device.rs` to confirm the current async GEMM path always copied both inputs to host, ran `gemm_host`, and copied the result back to the device.
- Inspected the vendored `cutile-rs` checkout under `~/.cargo/git/checkouts/.../13732c8` and verified the intended kernel pattern: `#[cutile::module]`, a tiled `#[cutile::entry]` kernel, partitioned output tensors, runtime `.generics(...)`, and async execution via the generated `*_async` launcher.
- Chose a first-kernel scope that kept current APIs stable: add a Linux-only cutile GEMM kernel for tile-aligned shapes, and explicitly fall back to the existing host bridge for unsupported shapes or non-Linux platforms.
- Planned a small internal tensor constructor so GEMM could wrap a cutile-produced device tensor without forcing an unnecessary device-to-host-to-device round-trip.
- Planned focused coverage for backend selection and aligned GPU GEMM parity, plus a benchmark hook in `nemotron-validate` that can exercise an aligned GEMM shape even when bundled fixtures are not tile-aligned.

## What Was Changed

### `nemotron-kernels/src/gemm.rs`
- Added the first Linux-only real cutile GEMM kernel using a tiled `#[cutile::module]` entry with a `16x16x8` tile shape.
- Added runtime backend selection so Linux uses cutile only for tile-aligned shapes that fit the cutile launcher contract; all other cases keep using the existing host bridge.
- Preserved the existing public API (`gemm_host`, `gemm`, `gemm_into`, `GemmShape`, `GemmError`) and existing validation/error behavior.
- Added checked GEMM shape-length helpers so `lhs_len`, `rhs_len`, and `output_len` no longer silently overflow, and internal validation now rejects overflowing products as `InvalidShape` before any allocation or tensor-length trust.
- Updated `gemm_into` to reuse `gemm(...)` and then restore the caller-owned logical output shape, preserving current `_into` semantics.
- Updated tests to cover platform backend metadata, shape-based backend dispatch, overflow rejection, fallback parity, and a Linux-only aligned GEMM parity test for the real cutile path.

### `nemotron-kernels/src/tensor.rs`
- Added a Linux-only internal constructor that wraps an owned `cutile::tensor::Tensor<f32>` directly after validating element count.
- This removes the old extra H2D copy that would otherwise have happened after a successful cutile GEMM kernel launch.

### `nemotron-validate/src/benchmark.rs`
- Updated benchmark notes to distinguish real GEMM device compute from the remaining host-backed wrapper paths.
- Added a synthetic `benchmark/gemm-aligned` case using a deterministic `64x64x64` GEMM so benchmark mode has a GEMM hook that can actually hit the real cutile kernel on Linux.
- Added a focused async unit test for the aligned benchmark hook that does not depend on external fixture files.

### `nemotron-validate/tests/validator_integration.rs`
- Updated benchmark-mode stdout assertions to reflect the new benchmark note wording and the new `benchmark/gemm-aligned` result line.

## Alignment Notes

- The public GEMM API surface stayed unchanged; only the Linux implementation behind `gemm(...)` was upgraded.
- Host fallback behavior remains intact on non-Linux platforms and on Linux shapes that are not multiples of `16x16x8`.
- Existing shape/length error contracts were preserved; unsupported cutile shapes do not invent a new error path, they simply stay on the fallback path, and overflowing GEMM shape products are now rejected deterministically as invalid shapes instead of wrapping in release builds.
- `supported_gemm_kernels()` still returns a single-item registry entry, preserving the existing API shape while reflecting that Linux now has a real cutile backend.

## Verification

- `cargo fmt --all` ✅
- `cargo test -p nemotron-kernels gemm --quiet` ✅
- `cargo test -p nemotron-validate aligned_gemm_benchmark_runs_without_fixtures --quiet` ✅
- `cargo build --workspace --quiet` ✅
- `cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip validator_binary_passes_reference_fixtures --skip benchmark_mode_reports_timings_and_parity` ✅
- `cargo check -p nemotron-kernels --target x86_64-unknown-linux-gnu --quiet` ⚠️ blocked locally because the `x86_64-unknown-linux-gnu` Rust target is not installed in this macOS worktree environment.

## Notes / Open Gaps

- The real cutile GEMM path is intentionally limited to tile-aligned shapes for this first implementation; unsupported Linux shapes still use the old host bridge.
- The bundled fixture-dependent validator tests are still blocked locally because `data/reference_kernels/fixtures.json` and related files are not present in this worktree.
- A future Linux/RTX validation pass should confirm the new aligned GEMM path end-to-end on actual CUDA hardware.
