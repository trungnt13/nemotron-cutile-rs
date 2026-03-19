---
status: complete
goal: Implement a real cutile-backed Conv1D path while preserving host fallback and bundled fixture parity
prompt: >-
  You are implementing todo `implement-cutile-conv1d`.

  Working tree:
  - Local repo: `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-conv1d`
  - Branch: `cutile-conv1d`
  - Base: `a804940`
  - Remote GPU test path: `tn:~/codes/nemotron-cutile-rs--cutile-conv1d`

  Task:
  Implement a real cutile-backed Conv1D path in `nemotron-kernels/src/conv1d.rs` (and tightly related files only as needed), preserving the current async API, host fallback behavior, and bundled fixture parity. Reuse the established Wave 1 cutile patterns on main. Keep unsupported shapes/platforms on the host bridge if necessary rather than forcing risky device dispatch.

  Requirements:
  - Preserve existing shape/error semantics.
  - Add focused tests and keep required `///` test doc comments.
  - Update validator/benchmark coverage if the implementation changes dispatch/reporting.
  - Add/update the ai-doc executed plan for this task.
  - Commit completed work with Co-authored-by trailer.
  - Do not push.
created: 2026-03-19T15:20:00Z
finished: 2026-03-19T16:40:22Z
---

# Executed Plan — Implement Cutile Conv1D

## Research and Plan

- Inspected `nemotron-kernels/src/conv1d.rs`, `gemm.rs`, `softmax.rs`, `rms_norm.rs`, and `tensor.rs` to confirm the current async Conv1D path always copied device inputs to host, ran the host kernel, and copied the result back.
- Reused the established Wave 1 approach of exposing a Linux Cutile backend while keeping explicit host-bridge fallback for unsupported shapes.
- Prototyped a direct Cutile tile kernel first, but remote CUDA validation showed that the straightforward partitioned formulation was either unsupported or produced zeroed outputs for the targeted shapes.
- Chose a safer real-compute implementation for this task: lower depthwise causal Conv1D into an internal im2col-style matrix multiply and dispatch that through the existing Linux Cutile GEMM path only when the transformed GEMM shape is 16x16x8-aligned. All other shapes keep the previous host bridge.

## What Was Changed

### `nemotron-kernels/src/conv1d.rs`

- Kept the public host API unchanged: `depthwise_causal_conv1d_host`, `depthwise_causal_conv1d_into_host`, `Conv1dShape`, and `Conv1dError` still behave the same for shape and length validation.
- Added Linux backend metadata so `supported_conv1d_kernels()` reports a Cutile-backed primary kernel on Linux while non-Linux platforms continue to report host fallback.
- Added runtime tensor-length validation for the async API without introducing stricter logical-shape requirements than the existing host implementation.
- Added a Linux Cutile dispatch heuristic that only enables the real device path when the internal GEMM lowering is safe and aligned:
  - `m = sequence_len` must be a multiple of 16
  - `n = channel_count` must be a multiple of 16
  - `k = channel_count * kernel_size` must be a multiple of 8
  - `kernel_size` must stay within the current bounded envelope (`<= 32`)
- Implemented a real Cutile-backed Conv1D path by:
  1. building an im2col-style left-hand matrix from causal per-channel windows,
  2. building a block-diagonal right-hand weight matrix, and
  3. calling the existing async `gemm(...)` kernel path so aligned Linux shapes execute on Cutile device compute.
- Preserved the async API contract by reshaping the GEMM output back to the caller-visible input shape metadata before returning.
- Kept unsupported shapes on the old host bridge instead of forcing risky device dispatch.

### `nemotron-validate/src/benchmark.rs`

- Added a synthetic `benchmark/causal_conv1d` case using a deterministic `16x16x4` depthwise shape whose internal `16x64x16` GEMM can hit the real Cutile Conv1D path on Linux.
- Updated benchmark notes so they accurately describe that Conv1D now has a real Cutile-backed path only when the internal GEMM lowering is aligned.
- Added a focused unit test so the synthetic Conv1D benchmark hook runs without fixture files.

### `nemotron-validate/tests/validator_integration.rs`

- Updated benchmark-mode stdout assertions so the benchmark report expects the new Conv1D note and the new `benchmark/causal_conv1d` result line.

## Key Decisions

- **Real compute via GEMM lowering instead of a bespoke tile kernel:** this still gives Conv1D a real Linux Cutile execution path, but it reuses the already-validated Wave 1 GEMM backend rather than landing a more fragile custom kernel.
- **Fallback for bundled fixture shape:** the bundled Conv1D fixture is `12x16x4`, which lowers to an internal GEMM with `m = 12`; because that is not 16-aligned, it intentionally stays on the host bridge while still preserving bundled parity.
- **No public API changes:** the async signature and host behavior stayed intact, and unsupported shapes still succeed through the host bridge rather than surfacing a new user-visible error.

## Alignment Notes

- The task asked for a real Cutile-backed Conv1D path while preserving host fallback and bundled parity. That is now true: Linux has a real Cutile-backed Conv1D implementation for aligned lowered shapes, while fixture-shaped and unsupported cases continue to use the existing host bridge.
- Shape and length error semantics were preserved; the async path still validates only against element counts implied by `Conv1dShape` and does not require the caller's tensor metadata to already be `[sequence_len, channel_count]` or `[channel_count, kernel_size]`.
- Benchmark and reporting coverage were updated to make the real Conv1D path visible even though the bundled reference fixture shape intentionally remains on the fallback path.

## Verification

### Local

- `cargo test -p nemotron-kernels conv1d --quiet` ✅
- `cargo test -p nemotron-validate conv1d_benchmark_runs_without_fixtures --quiet` ✅
- `cargo build --workspace --quiet` ✅
- `cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip validator_binary_passes_reference_fixtures --skip benchmark_mode_reports_timings_and_parity` ✅
- Local bundled-fixture tests still require skipping because this worktree does not contain the gitignored `data/reference_kernels/*.json` and `data/reference_outputs/*.json` files.

### Remote (tn)

The worktree was synced with:

```bash
rsync -az --exclude .git --exclude target /Users/trungnt13/codes/nemotron-cutile-rs--cutile-conv1d/ tn:~/codes/nemotron-cutile-rs--cutile-conv1d/
```

Remote validation used:

```bash
ssh tn 'set -euxo pipefail; export PATH="$HOME/.cargo/bin:/usr/local/cuda-13/bin:/usr/lib/llvm-21/bin:$PATH"; export CUDA_TOOLKIT_PATH=/usr/local/cuda-13; export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21; cd ~/codes/nemotron-cutile-rs--cutile-conv1d; cargo test -p nemotron-kernels conv1d --quiet'
```

Result: **13 passed, 0 failed**.

```bash
ssh tn 'set -euxo pipefail; export PATH="$HOME/.cargo/bin:/usr/local/cuda-13/bin:/usr/lib/llvm-21/bin:$PATH"; export CUDA_TOOLKIT_PATH=/usr/local/cuda-13; export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21; cd ~/codes/nemotron-cutile-rs--cutile-conv1d; cargo test -p nemotron-validate conv1d_benchmark_runs_without_fixtures --quiet; cargo build --workspace --quiet; cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip validator_binary_passes_reference_fixtures --skip benchmark_mode_reports_timings_and_parity; cargo run -p nemotron-validate --quiet -- ~/codes/nemotron-cutile-rs/data/reference_kernels ~/codes/nemotron-cutile-rs/data/reference_outputs; cargo run -p nemotron-validate --quiet -- benchmark ~/codes/nemotron-cutile-rs/data/reference_kernels ~/codes/nemotron-cutile-rs/data/reference_outputs'
```

Observed results:

- `cargo test -p nemotron-validate conv1d_benchmark_runs_without_fixtures --quiet` ✅
- `cargo build --workspace --quiet` ✅
- `cargo test --workspace ...skips...` ✅
- `cargo run -p nemotron-validate --quiet -- ~/codes/nemotron-cutile-rs/data/reference_kernels ~/codes/nemotron-cutile-rs/data/reference_outputs` ✅
  - `summary: 9/9 validations passed`
  - `gpu summary: 7/7 gpu validations passed`
  - `gpu/causal_conv1d: PASS (max_abs_diff=0.000000)`
- `cargo run -p nemotron-validate --quiet -- benchmark ~/codes/nemotron-cutile-rs/data/reference_kernels ~/codes/nemotron-cutile-rs/data/reference_outputs` ✅
  - `benchmark/causal_conv1d: PASS (... max_abs_diff=0.000000, iterations=25)`
  - `benchmark summary: 8/8 comparisons passed`

## Next Steps

- If future work needs the bundled `12x16x4` reference fixture itself to execute on-device, extend the Conv1D lowering or add a bespoke Cutile kernel for non-16-aligned `m` rather than forcing the current path beyond the already-validated GEMM envelope.
