---
status: complete
goal: Implement a real cutile-backed selective-scan path in nemotron-kernels while preserving host fallback and current SSM semantics
prompt: "You are implementing todo `implement-cutile-ssm`.

Working tree:
- Local repo: `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-ssm`
- Branch: `cutile-ssm`
- Base: `a804940`
- Remote GPU test path: `tn:~/codes/nemotron-cutile-rs--cutile-ssm`

Task:
Implement as much of a real cutile-backed SSM/selective-scan path as is safely feasible in `nemotron-kernels/src/ssm.rs`, preserving current semantics and host fallback. If the kernel is too complex to finish cleanly, prefer a documented, minimal supported-shape device path or a clearly-blocked outcome over speculative code.

Requirements:
- Preserve the current public API and numeric contracts.
- Reuse integrated cutile helper patterns from Wave 1.
- Add focused tests and required doc comments.
- Update ai-doc executed plan with decisions, open gaps, and verification.
- Commit if complete; if blocked, document and mark blocked in SQL.
- Do not push.

Testing:
- Run relevant local tests.
- Mandatory remote validation on `tn` after rsync to `~/codes/nemotron-cutile-rs--cutile-ssm`, with the required PATH/CUDA/LLVM env.
- Include exact commands/results.

SQL status update (required):
- Success: `UPDATE todos SET status = 'done' WHERE id = 'implement-cutile-ssm'`
- Blocked: `UPDATE todos SET status = 'blocked' WHERE id = 'implement-cutile-ssm'`

Always return a summary with completed work, done/not-done status, blockers/questions, commit hash if any, and local/tn validation results."
created: 2026-03-19T14:04:12Z
finished: 2026-03-19T16:50:49Z
---

# Executed Plan — Implement Cutile SSM

## Summary

Implemented the first real cutile-backed selective-scan path in `nemotron-kernels/src/ssm.rs` for a documented minimal support envelope on Linux/CUDA: `state_size == 1` with bounded sequence length. Unsupported shapes and all non-Linux targets keep the existing host bridge fallback, so the public API and host semantics remain intact.

## Research and Plan

- Inspected `nemotron-kernels/src/ssm.rs`; the current async GPU path still performed a full D2H -> host selective scan -> H2D bridge.
- Reviewed Wave 1 cutile patterns in `tensor.rs`, `softmax.rs`, `gemm.rs`, and `rms_norm.rs`: Linux-only `#[cutile::module]` kernels, `cutile_tensor_for_shape(...)`, `GpuTensor::from_cutile_tensor(...)`, bounded shape heuristics, and host fallback for unsupported shapes.
- Chose a bounded first device path instead of a speculative full general kernel: a Linux-only cutile selective-scan kernel for supported shapes, with the existing host bridge retained for unsupported shapes and all non-Linux platforms.
- Preserved host-visible error contracts by validating `delta_t` semantics before device compute.
- Kept the output API unchanged, while allowing a small host-side transpose/repack step after device compute for the minimal implementation.

## What Was Done

### `nemotron-kernels/src/ssm.rs`

- Switched the Linux primary SSM backend metadata from `HostFallback` to `Cutile`, while preserving the same single-entry public registry shape.
- Refactored host-side dt validation into `validated_delta_t(...)` so the host and GPU paths share the same runtime contract.
- Added GPU tensor validation mirroring the existing host length/shape checks.
- Added a Linux-only helper that materializes a validated dt tensor on host and uploads it back to device so the cutile kernel only receives already bias-adjusted / softplus-adjusted legal timesteps.
- Added a Linux-only `#[cutile::module]` selective-scan kernel for the documented minimal support envelope:
  - one device block per channel
  - recurrent scan over timesteps on device
  - real device-side state update and output computation
  - support limited to `state_size == 1`
- Reused Wave 1 device helpers:
  - `GpuTensor::cutile_tensor_for_shape(...)`
  - `GpuTensor::from_cutile_tensor(...)`
  - cutile async launch + partition patterns
- Preserved the host bridge path for:
  - all non-Linux targets
  - Linux shapes outside the current support envelope
- Added a host-side transpose helper so the minimal kernel can emit per-channel rows and still return the existing `[sequence_len, channel_count]` output shape.

### Tests

- Updated the backend registry test to assert `Cutile` on Linux and `HostFallback` elsewhere.
- Updated the support-heuristic coverage to reflect the documented minimal path (`state_size == 1`, bounded sequence length).
- Added/updated GPU parity tests so Linux actually exercises the real cutile path on `state_size == 1` shapes.
- Added an explicit fallback test confirming unsupported `state_size > 1` shapes keep host parity.

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Initial cutile scope | Support only `state_size == 1` | This kept the first real kernel small enough to validate on the RTX host instead of shipping speculative tile/reduction logic for the general recurrent case |
| dt contract preservation | Prevalidate / materialize dt on host before launch | The existing API returns `InvalidDeltaT` with timestep/channel detail; preserving that contract is more important than forcing every check into the first kernel |
| Output layout handling | Device compute + host transpose/repack | This avoided a second speculative transpose kernel while still moving the core recurrent computation onto cutile |
| Unsupported Linux shapes | Fall back to the old bridge | Safer than pretending the minimal kernel is general-purpose |

## Alignment Notes

- The public API stayed unchanged:
  - `SelectiveScanShape`, `SelectiveScanParams`, `SelectiveScanOutput`
  - `GpuSelectiveScanParams`, `GpuSelectiveScanOutput`
  - `pub async fn selective_scan(...)`
- The host kernel semantics and error contracts were preserved.
- The real cutile path is intentionally minimal, not universal. That is an intentional scope boundary, not an accident.
- The Linux kernel now performs real device-side SSM math for supported shapes, so this task is complete without over-claiming unsupported cases.

## Open Gaps

1. **Only `state_size == 1` is cutile-backed today**
   - `state_size > 1` shapes still fall back to the host bridge on Linux.
   - Extending beyond this will need a more capable state-tiled recurrent kernel.

2. **Output layout still does one host repack**
   - The cutile kernel writes a channel-major intermediate which is copied back and transposed before returning the public `[sequence_len, channel_count]` tensor.
   - A future fully device-native transpose/output layout path could remove that last round trip.

3. **dt preparation still touches host memory**
   - This is deliberate to preserve the existing `InvalidDeltaT` contract and avoid speculative error-flag plumbing in the first kernel.

## Verification

### Local commands and results

```bash
cargo test -p nemotron-kernels ssm --quiet
cargo test -p nemotron-kernels --quiet
cargo build --workspace --quiet
```

Observed local results:

- `cargo test -p nemotron-kernels ssm --quiet` ✅ **13 passed**
- `cargo test -p nemotron-kernels --quiet` ✅ **141 passed**, plus **1 ignored** doctest target run
- `cargo build --workspace --quiet` ✅ passed

### Remote `tn` commands and results

Rsync command:

```bash
rsync -az --exclude target --exclude .git ./ tn:~/codes/nemotron-cutile-rs--cutile-ssm/
```

Known-good environment used on `tn` for the successful remote rerun:

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13/bin:/usr/lib/llvm-21/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13
export CUDA_PATH=/usr/local/cuda-13
export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21
export LLVM_SYS_210_PREFIX=/usr/lib/llvm-21
export LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config
```

Remote validation commands:

```bash
cargo test -p nemotron-kernels ssm --quiet
cargo test --workspace --quiet
cargo build --workspace --quiet &&   cargo test --workspace --quiet --     --skip bundled_reference_fixtures_validate     --skip bundled_gpu_reference_fixtures_validate     --skip bundled_benchmark_comparison_runs     --skip validator_binary_passes_reference_fixtures     --skip benchmark_mode_reports_timings_and_parity
```

Observed remote results on `tn`:

- `cargo test -p nemotron-kernels ssm --quiet` ✅ **13 passed** (this exercised the Linux cutile path)
- retrying the earlier failing `/usr/local/cuda-13.2` shell with the known-good `/usr/local/cuda-13` environment fixed the remote SSM validation failure
- `cargo test --workspace --quiet` ⚠️ **failed only in fixture-dependent `nemotron-validate` tests** because `/home/trungnt13/codes/nemotron-cutile-rs--cutile-ssm/data/reference_kernels/fixtures.json` was not present on the remote path
- `cargo build --workspace --quiet` ✅ passed
- skipped workspace command ✅ all executed tests passed:
  - `nemotron-kernels`: **143 passed**
  - `nemotron-model`: **17 passed**
  - `nemotron-cli`: **2 passed**
  - `nemotron-nn`: **39 passed**
  - `nemotron-validate`: **3 passed**, **3 filtered out**

## Next Steps

1. Generalize the cutile SSM kernel beyond `state_size == 1` once a safe state-tiling/reduction strategy is proven on the GPU host.
2. Replace the remaining host-side output transpose with a device-native layout path when the minimal kernel is stable.
3. Re-run full remote fixture validation once the reference JSON files are present on `tn`.
