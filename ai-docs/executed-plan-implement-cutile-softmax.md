---
status: complete
goal: Implement the first real cutile-backed softmax compute kernel while preserving host fallback and validator parity
prompt: "Work in this git worktree only: `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-softmax`

Goal: implement the first real cutile-backed compute kernel for softmax, replacing the current Linux GPU wrapper's D2H -> host -> H2D bridge with actual device compute where feasible, while keeping host fallback intact on unsupported platforms.

Requirements:
- Stay within this worktree and branch.
- Follow repo conventions from AGENTS.md.
- Preserve current public API shape and host fallback behavior.
- Preserve current numerical stability behavior and parity expectations.
- Add/update focused tests and any validator/benchmark hooks needed for softmax.
- Update/create an ai-docs executed-plan file in this worktree describing the work.
- Run relevant tests/builds in this worktree.
- Commit your changes locally in this worktree with a proper message and Copilot trailer.
- Update SQL when finished:
  - success: UPDATE todos SET status = 'done' WHERE id = 'implement-cutile-softmax';
  - blocked: UPDATE todos SET status = 'blocked' WHERE id = 'implement-cutile-softmax';

Return a concise summary with what changed, what tests ran, the commit hash, and whether the todo is fully complete."
created: 2026-03-19T11:06:36Z
finished: 2026-03-19T11:16:46Z
---

# Executed Plan — Implement Cutile Softmax

## Summary

Implemented the first real cutile-backed compute kernel in `nemotron-kernels`: Linux softmax now runs device-side cutile math for supported flattened widths instead of performing the previous D2H -> host -> H2D bridge. Non-Linux platforms still use the existing host fallback path, and Linux keeps a host bridge fallback for widths outside the current cutile safety envelope.

## What Was Done

### 1. Added the Linux cutile softmax kernel

- Added a Linux-only `#[cutile::module]` softmax kernel in `nemotron-kernels/src/softmax.rs`.
- The cutile kernel preserves the host algorithm's stability pattern:
  - compute row max first
  - subtract the max before exponentiation
  - normalize by the summed exponentials
- The runtime flattens the logical tensor to a single `[1, numel]` row for kernel execution so the public async API still behaves like the existing flat softmax wrapper.

### 2. Preserved public API and host fallback behavior

- Kept `pub async fn softmax(input: &GpuTensor) -> Result<GpuTensor, SoftmaxError>` unchanged.
- Kept `softmax_host`, `softmax_in_place_host`, and `softmax_into_host` unchanged.
- Non-Linux platforms still route through the exact host bridge path.
- Linux uses cutile only when the flattened width is within the current supported range (`<= 4096` and `i32`-representable); otherwise it falls back to the pre-existing host bridge for correctness.

### 3. Added zero-copy wrapping for cutile outputs

- Extended `GpuTensor` with a Linux-only internal constructor that wraps an existing cutile tensor without copying it back through host memory.
- This lets the cutile softmax path return a `GpuTensor` directly from device output while preserving the caller-visible shape metadata.

### 4. Tightened softmax-focused tests

- Updated the softmax kernel registry test to assert the current platform's primary backend (`Cutile` on Linux, `HostFallback` elsewhere).
- Added a heuristic boundary test for cutile eligibility.
- Switched GPU softmax parity tests to tolerance-based slice comparison so Linux device compute can stay numerically aligned without requiring bitwise identity.
- Added a GPU numerical-stability test for large shifted inputs.
- Added a Linux-only oversized-row fallback test.

### 5. Refreshed validator and benchmark messaging

- Updated `nemotron-validate` benchmark notes so they no longer claim every GPU path is still host-delegating.
- Updated GPU validation notes to call out the softmax cutile exception explicitly.
- Added focused note tests so the softmax messaging stays accurate even when fixture-backed validator tests are skipped locally.
- Fixed the GPU softmax validator path so it always records a comparison result instead of silently skipping reporting when output lengths disagree.

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Preserve API shape | Keep the public softmax signatures unchanged | Prevents rippling changes through kernels, NN layers, model code, and validation tools |
| Flatten before device softmax | Run cutile softmax over `[1, numel]` | Matches the existing wrapper semantics that treat the input tensor as a flat buffer while still preserving returned shape metadata |
| Limit current cutile execution width | Use cutile for flattened widths up to 4096 and fall back otherwise | Keeps the first real kernel on a bounded, example-backed execution strategy instead of over-claiming unsupported tile shapes |
| Keep host fallback on unsupported platforms | Use the old bridge on macOS/other non-Linux targets | Preserves existing behavior and keeps development/test workflows working outside Linux/CUDA |

## Alignment Notes

- The request asked for the first real cutile-backed compute kernel for softmax. That is implemented for Linux in the async `softmax(...)` path.
- The request also asked to keep host fallback intact. That remains true for non-Linux targets and for Linux widths outside the current supported cutile range.
- Numerical stability behavior is preserved conceptually by matching the host kernel's max-subtraction pattern before exponentiation.
- Full runtime execution of the Linux/CUDA path was not possible from this macOS worktree, so local verification focused on non-Linux behavior, compile/build safety, and the surrounding tests/documentation. The implementation remains cfg-gated so unsupported platforms keep compiling cleanly.

## Verification

### Baseline observation

Before changing code, `cargo test --workspace` failed in this worktree because the gitignored fixture files under `data/reference_kernels/` and `data/reference_outputs/` are not present locally. The failing tests were the bundled validator/benchmark tests that require those fixture JSON files.

### Commands run

```bash
cargo test --workspace
cargo fmt --all
cargo build --workspace
cargo test -p nemotron-kernels softmax
cargo test -p nemotron-validate -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip benchmark_mode_reports_timings_and_parity --skip validator_binary_passes_reference_fixtures
cargo test --workspace -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip benchmark_mode_reports_timings_and_parity --skip validator_binary_passes_reference_fixtures
```

### Observed results

- Baseline `cargo test --workspace`: **failed only in fixture-dependent validator tests** because local `data/reference_kernels/fixtures.json` and `data/reference_outputs/fixtures.json` are absent in this worktree.
- `cargo build --workspace`: **passed**.
- `cargo test -p nemotron-kernels softmax`: **13 passed**.
- `cargo test -p nemotron-validate ...skips...`: **4 passed, 0 failed, 5 filtered out**.
- `cargo test --workspace ...skips...`: **all executed tests passed**; the fixture-dependent validator tests were filtered out intentionally because the required gitignored fixture files are not available locally.

## Next Steps

1. Re-run the new Linux softmax path on a Linux/CUDA machine to validate real cutile execution and benchmark behavior directly.
2. If larger flattened widths are needed, extend the cutile launch strategy beyond the initial `<= 4096` envelope and keep the same validator parity checks.
