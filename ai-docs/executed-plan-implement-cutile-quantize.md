---
status: complete
goal: Implement the safest feasible real cutile-backed affine INT4 quantize/dequantize path in quantize.rs while preserving host fallback semantics
prompt: |
  You are implementing todo `implement-cutile-quantize`.

  Working tree:
  - Local repo: `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-quantize`
  - Branch: `cutile-quantize`
  - Base: `a804940`
  - Remote GPU test path: `tn:~/codes/nemotron-cutile-rs--cutile-quantize`

  Task:
  Implement as much of a real cutile-backed quantize/dequantize path as is safely feasible in `nemotron-kernels/src/quantize.rs`, preserving existing semantics and host fallback behavior. Be especially careful about the project’s current distinction between generic affine INT4 infrastructure and any AWQ-specific behavior. If fixture or contract gaps prevent a clean implementation, document them explicitly rather than guessing.

  Requirements:
  - Preserve current API/error semantics.
  - Reuse Wave 1 cutile patterns where possible.
  - Add focused tests with required doc comments.
  - Update ai-doc executed plan with exact scope, decisions, and blockers/gaps.
  - Commit if complete; if blocked, document and mark blocked in SQL.
  - Do not push.

  Testing:
  - Run relevant local tests.
  - Mandatory remote validation on `tn` after rsync to `~/codes/nemotron-cutile-rs--cutile-quantize`, with the required PATH/CUDA/LLVM env.
  - Include exact commands/results.

  SQL status update (required):
  - Success: `UPDATE todos SET status = 'done' WHERE id = 'implement-cutile-quantize'`
  - Blocked: `UPDATE todos SET status = 'blocked' WHERE id = 'implement-cutile-quantize'`

  Always return a summary with completed work, whether fully done or still needing work, blockers/questions, commit hash if any, and local/tn validation results.
created: 2026-03-19T13:55:40Z
finished: 2026-03-19T15:04:41Z
---

# Executed Plan — Implement Cutile Quantize

## Summary

Implemented the safest real Linux-only cutile path currently supported in `nemotron-kernels/src/quantize.rs`: async `dequantize_int4(...)` now dispatches to real device-side affine dequantization for supported power-of-two lengths on Linux, while preserving the existing host fallback on unsupported platforms and unsupported shapes. The generic affine INT4 host helpers and their API/error semantics remain intact, and no AWQ-specific behavior was introduced.

## What Was Done

### 1. Added a real cutile-backed async dequantize path

- Added Linux-only backend selection for `dequantize_int4(...)`.
- Kept the public async API unchanged.
- Implemented a Linux-only `#[cutile::module]` kernel that performs the affine dequantization arithmetic on device:
  - input code values as `f32`
  - subtract affine `zero_point`
  - multiply by affine `scale`
- Returned the result as a device-backed `GpuTensor` without a host round-trip.

### 2. Preserved existing generic affine INT4 semantics

- Left `pack_int4_*`, `unpack_int4_*`, `quantize_int4_*_host`, `dequantize_int4_*_host`, and `QuantizeError` semantics unchanged.
- Kept the implementation explicitly generic affine INT4 (`round(v / scale + zero_point).clamp(0, 15)` and `(code - zero_point) * scale`), not AWQ-specific runtime behavior.
- Preserved host fallback behavior for:
  - non-Linux platforms
  - Linux value counts outside the current safe cutile envelope

### 3. Reused Wave 1 backend/reporting patterns

- Made `supported_quantize_kernels()` platform-aware, reporting `Cutile` on Linux and `HostFallback` elsewhere.
- Added a bounded Linux dispatch heuristic instead of pretending all lengths are supported.
- Kept a dedicated host-bridge helper for the fallback path.

### 4. Added focused tests

- Added/updated tests for:
  - platform backend reporting
  - the Linux cutile support heuristic
  - odd-length async fallback parity and shape preservation
  - Linux-only power-of-two cutile parity against the host formula
- Kept required test doc comments throughout.

### 5. Updated related documentation

- Updated `AGENTS.md` so repository state now records affine INT4 dequantize as part of the Wave 1 real cutile compute set.

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Preserve public surface | Did not add a new async GPU quantize API | The existing contract only exposes async GPU dequantize; adding more surface without clear downstream use would be speculative |
| Keep host helpers truly host-side | Did not hide GPU work inside `*_host` quantize/dequantize helpers | Preserves caller expectations and avoids surprising Linux-only CUDA requirements in host-named functions |
| Bound Linux cutile dispatch | Support only positive power-of-two `value_count`s within `i32` bounds | Real cutile execution proved shape-sensitive; non-power-of-two lengths compiled reliably through the host bridge, not the current device kernel |
| Stay generic affine, not AWQ-specific | No per-group/per-channel AWQ logic was added | The module’s current contract is generic affine INT4 infrastructure, and guessing AWQ semantics here would be incorrect |
| Materialize unpacked codes as `f32` on host before upload | Do host nibble unpack + `u8 -> f32` conversion, then do affine math on device | The attempted cutile-side `u8 -> f32` conversion was not supported by the current cutile kernel path, so the safe real-device scope is the affine transform itself |

## Alignment Notes

- The request asked for as much real cutile-backed quantize/dequantize behavior as is safely feasible. The implemented scope is a real Linux cutile dequantize compute path, while quantize remains host-only because there is still no public device-resident packed-INT4 API and no existing async quantize contract to preserve.
- Existing host fallback behavior is preserved exactly on non-Linux and for unsupported Linux lengths.
- The implementation remains intentionally **generic affine INT4 infrastructure**, not an AWQ-specific execution path.

## Blockers / Open Gaps

1. **No public packed-device INT4 tensor abstraction yet.** The async API still accepts packed host bytes, so nibble unpacking remains on the host boundary.
2. **No async GPU quantize API exists today.** Adding one now would expand the module contract rather than completing the existing one.
3. **Current cutile path is shape-limited.** Power-of-two value counts are supported; other lengths still use the host bridge.
4. **Current cutile kernel path did not safely support `u8 -> f32` conversion in-kernel.** The final implementation therefore uploads unpacked affine codes as `f32` and performs the affine arithmetic on device.
5. **No bundled quantize reference fixtures exist.** Validation therefore relied on focused tests rather than fixture-backed validator parity.

## Verification

### Local baseline before changes

```bash
cargo test -p nemotron-kernels --quiet
```

Result: **138 passed, 1 ignored**.

### Local verification after changes

```bash
cargo fmt --all
cargo test -p nemotron-kernels quantize --quiet
cargo test -p nemotron-kernels --quiet
```

Observed results:

- `cargo test -p nemotron-kernels quantize --quiet`: **16 passed**
- `cargo test -p nemotron-kernels --quiet`: **139 passed, 1 ignored**

### Remote `tn` validation after rsync

Sync command:

```bash
rsync -az --delete --exclude .git --exclude target ./ tn:~/codes/nemotron-cutile-rs--cutile-quantize/
```

Remote commands:

```bash
cd ~/codes/nemotron-cutile-rs--cutile-quantize
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:/usr/lib/llvm-21/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2
export CUDA_PATH=/usr/local/cuda-13.2
export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21
export LLVM_SYS_210_PREFIX=/usr/lib/llvm-21
export LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config
cargo test -p nemotron-kernels quantize --quiet
cargo test -p nemotron-kernels --quiet
```

Observed remote results on `tn` (`NVIDIA GeForce RTX 3090`, driver `595.45.04`):

- `cargo test -p nemotron-kernels quantize --quiet`: **17 passed**
  - includes the Linux-only cutile power-of-two dequantize parity test
- `cargo test -p nemotron-kernels --quiet`: **142 passed, 1 ignored**

### Validation notes

- An initial remote run exposed that the cutile kernel path requires the CUDA/LLVM toolchain binaries on `PATH`; the final validation command includes those explicitly.
- Another Linux-only iteration showed that non-power-of-two cutile lengths and in-kernel `u8 -> f32` conversion were not safe contracts to assume, so the final implementation narrowed dispatch and kept those steps on the host boundary.
