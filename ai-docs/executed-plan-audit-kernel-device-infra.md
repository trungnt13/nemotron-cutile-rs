---
status: complete
goal: Audit the GPU tensor/device scaffolding for shape validation, transfer safety, and platform gating.
prompt: "You are auditing the GPU tensor/device infrastructure in /Users/trungnt13/codes/nemotron-cutile-rs."
created: 2026-03-19T09:42:56Z
finished: 2026-03-19T09:45:12Z
---

# Executed Plan — Audit Kernel Device Infrastructure

## Research

- Reviewed `nemotron-kernels/src/tensor.rs` and `nemotron-kernels/src/device.rs`.
- Confirmed the intended design: Linux uses cutile-backed tensors/devices, while non-Linux uses host stubs.
- Checked downstream usage of `GpuTensor`/`GpuDevice` across kernels, NN layers, model, and validation code.
- Inspected local cutile sources to verify:
  - `copy_host_vec_to_device(...).sync_on(...)` and `to_host_vec().sync_on(...)` return `Result`
  - `init_device_contexts(...)` is thread-local and errors if called more than once on the same thread
  - current cutile host-vector copies only preserve a flat 1D device-side shape

## Findings

1. `GpuTensor::numel()` used `.max(1)`, which makes zero-sized shapes report `1` element and can break reshape validation.
2. `shape_numel()` did not guard against multiplication overflow.
3. Linux sync tensor constructors used `expect(...)`, so some device failures panicked instead of propagating through `TensorError`.
4. `GpuDevice::new()` re-ran cutile async context initialization on every construction; on Linux that can fail on the second creation in the same thread.
5. The cutile tensor created from a host vec is flat on device; current wrappers rely on wrapper-side shape metadata only. That is okay for host-delegating kernels, but it is a downstream risk once real GPU kernels start consuming raw cutile tensors directly.

## Plan

1. Tighten tensor shape validation and fix `numel()` consistency.
2. Propagate Linux sync transfer/allocation errors where the public API already returns `Result`.
3. Add a fallible sync host-read path and keep the existing convenience API for compatibility.
4. Make `GpuDevice::new()` tolerate repeated same-device initialization while rejecting thread-local device mismatches clearly.
5. Add focused tests for the new safety behavior, then run the touched test subsets.

## What Was Done

- Added `ZeroSizedDimension` and `ShapeOverflow` validation errors in `TensorError`.
- Reworked `shape_numel()` to reject zero-sized dimensions and checked-multiply shape extents.
- Fixed `GpuTensor::numel()` so it now matches the validated shape product exactly.
- Changed Linux sync tensor allocation/copy helpers to propagate `TensorError::DeviceError` instead of panicking during context creation or host/device copies.
- Added `GpuTensor::try_to_host()` for fallible sync reads while preserving the existing `to_host()` convenience wrapper.
- Updated Linux `GpuDevice::new()` so repeated same-thread creation of the same ordinal no longer fails due to cutile's thread-local async-context initialization behavior.
- Added focused unit tests for zero-sized shapes, shape overflow, fallible sync read parity, and repeated device construction.

## Key Decisions

| Decision | Reason |
|----------|--------|
| Reject zero-sized dimensions | Safer for the current cutile-backed scaffolding, avoids zero-byte device allocations and fixes the previous `numel()` inconsistency. |
| Add `try_to_host()` instead of breaking `to_host()` | Keeps downstream callers working while providing an explicit non-panicking path for new code. |
| Treat repeated same-device context init as benign | cutile async contexts are thread-local and single-init; recreating the same device handle should not fail, but switching ordinals on the same thread should still surface clearly. |

## Verification

- `cargo test -p nemotron-kernels tensor::tests::`
- `cargo test -p nemotron-kernels device::tests::`
- `cargo test -p nemotron-kernels`

Result: **127/127 crate tests passed** for `nemotron-kernels` after the scoped changes.

## Downstream Notes

1. `GpuTensor::cutile_tensor()` still exposes a flat 1D cutile tensor because `copy_host_vec_to_device` only preserves the linear buffer. That is acceptable today because all compute paths still hop through host memory, but real cutile kernels will need shaped device tensors (or a reshape-capable wrapper) before consuming raw cutile tensors directly.
2. Sync transfer safety is improved, but the legacy `to_host()` API still panics on failure by design for compatibility. New sync call sites that need robust error handling should prefer `try_to_host()`.
3. The current platform gate remains `target_os = "linux"`, which matches the intentional Linux+CUDA scaffolding. macOS continues to use the host fallback without pulling in cutile.
