---
status: complete
goal: Implement a real cutile-backed embedding lookup path while preserving host fallback behavior and honest fixture coverage
prompt: >-
  Implement todo `implement-cutile-embedding` in `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-embedding` by adding a real cutile-backed embedding lookup path in `nemotron-kernels/src/embedding.rs`, preserving API/error semantics, keeping host fallback for unsupported cases, reusing existing tensor/device helpers and launch conventions, adding focused coverage if bundled embedding fixtures are still missing, updating ai-docs, and verifying both locally and on `tn` without pushing.
created: 2026-03-19T17:27:18Z
finished: 2026-03-19T17:27:18Z
---

# Executed Plan — Implement Cutile Embedding

## Research and Plan

- Inspect `nemotron-kernels/src/embedding.rs`, `tensor.rs`, and the existing cutile-backed kernels (`gemm`, `softmax`, `rms_norm`) to mirror the current Linux-only dispatch and zero-copy output wrapping conventions.
- Inspect local `cutile-rs` examples to confirm the safest gather strategy for embedding rows. The likely approach is a small Linux-only kernel that uses token-id device input plus raw table pointers to load contiguous row tiles directly into a partitioned output tensor.
- Preserve the current public API and error ordering by validating shape/table/token constraints before backend dispatch, and by falling back to the existing host bridge whenever the cutile launcher contract cannot safely represent the requested shape.
- Keep fixture coverage honest: if `data/reference_kernels/fixtures.json` still lacks embedding fixtures, add focused unit coverage for backend selection, fallback behavior, and Linux GPU parity instead of pretending validator fixture coverage exists.
- Verify locally with focused kernel tests/builds, then rsync this worktree to `tn` and rerun the relevant Linux/CUDA tests there with the repo's CUDA/LLVM environment.


## What Was Done

### `nemotron-kernels/src/embedding.rs`
- Replaced the Linux async embedding bridge with a real cutile-backed gather kernel that reads embedding rows directly from device memory and writes into a device output buffer.
- Kept the public host APIs unchanged and preserved error ordering by validating the table length and token bounds before backend dispatch.
- Added Linux backend selection so supported shapes use cutile while unsupported cases still route through the old host bridge.
- Preserved the existing empty-token GPU error by intentionally falling back to the host bridge for zero-length requests.

### Focused coverage
- Updated backend-registry tests to assert the current platform backend (`Cutile` on Linux, `HostFallback` elsewhere).
- Added dispatcher and tiling tests for the cutile heuristic/block-size selector.
- Added async GPU parity tests that exercise both a simple lookup and a multi-block hidden-size lookup.
- Added an explicit async test documenting that empty-token GPU lookup still surfaces the current device-shape error.

## Fixture / Validator Scope

- Bundled embedding fixtures are still not present in `data/reference_kernels/fixtures.json` in this worktree.
- I did **not** fake validator coverage. Instead, the new focused unit tests cover backend dispatch, multi-block gather parity, and the preserved empty-input error contract.
- The existing validator note that embedding lacks bundled fixture-backed GPU validation remains accurate and did not need code changes.

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Linux cutile scope | Use a raw-pointer gather kernel over a flat output buffer | Reuses the existing Wave 1 tensor helpers while avoiding extra D2H/H2D copies |
| Fallback behavior | Keep the host bridge for unsupported/empty cases | Preserves the established API and current empty-token GPU error semantics |
| Coverage strategy | Add focused kernel tests instead of validator fixtures | The repository still does not ship bundled embedding fixtures, so pretending otherwise would be misleading |
| Remote PATH | Add the cutile Python wheel's `tileiras` bin directory on `tn` | The cutile compiler runtime requires `tileiras` to compile the generated kernel on Linux/CUDA |

## Verification

### Local commands

```bash
cargo fmt --all
cargo test -p nemotron-kernels embedding --quiet
cargo build --workspace --quiet
cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip validator_binary_passes_reference_fixtures --skip benchmark_mode_reports_timings_and_parity
```

### Local results

- `cargo test -p nemotron-kernels embedding --quiet`: **16 passed**
- `cargo build --workspace --quiet`: **passed**
- `cargo test --workspace --quiet -- --skip ...`: **all executed tests passed**
  - `nemotron-kernels`: **142 passed**
  - `nemotron-nn`: **17 passed**
  - `nemotron-model`: **39 passed**
  - `nemotron-cli`: **3 passed**
  - `nemotron-validate`: **3 passed**, **3 filtered out**
  - one existing ignored test remained ignored

### Remote sync command

```bash
rsync -az --delete --exclude .git --exclude target /Users/trungnt13/codes/nemotron-cutile-rs--cutile-embedding/ tn:~/codes/nemotron-cutile-rs--cutile-embedding/
```

### Remote commands

```bash
export PATH="$HOME/.cargo/bin:$HOME/codes/cutile/.venv/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2
export CUDA_PATH=/usr/local/cuda-13.2
export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21
export LLVM_SYS_210_PREFIX=/usr/lib/llvm-21
export LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config
cargo build --workspace --quiet
cargo test -p nemotron-kernels embedding --quiet
cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip validator_binary_passes_reference_fixtures --skip benchmark_mode_reports_timings_and_parity
```

### Remote results (`tn`, RTX 3090)

- `cargo build --workspace --quiet`: **passed**
- `cargo test -p nemotron-kernels embedding --quiet`: **16 passed**
- `cargo test --workspace --quiet -- --skip ...`: **all executed tests passed**
  - `nemotron-kernels`: **144 passed**
  - `nemotron-nn`: **17 passed**
  - `nemotron-model`: **39 passed**
  - `nemotron-cli`: **3 passed**
  - `nemotron-validate`: **3 passed**, **3 filtered out**
  - one existing ignored test remained ignored
