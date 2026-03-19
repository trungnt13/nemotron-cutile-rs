---
status: complete
goal: Implement a real cutile-backed grouped-query attention path while preserving host fallback semantics
prompt: >-
  You are implementing todo `implement-cutile-attention`.

  Working tree:
  - Local repo: `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-attention`
  - Branch: `cutile-attention`
  - Base: `a804940`
  - Remote GPU test path: `tn:~/codes/nemotron-cutile-rs--cutile-attention`

  Task:
  Implement as much of a real cutile-backed attention path as is safely feasible in
  `nemotron-kernels/src/attention.rs` while preserving current API/host fallback behavior.
  If a full implementation is not feasible cleanly in one pass, prefer a correct supported-shape
  Linux device path with explicit fallback/constraints over a risky partial rewrite. Do not change
  attention semantics.

  Requirements:
  - Preserve shape contracts, masking semantics, and parity expectations.
  - Reuse Wave 1 cutile patterns and shared tensor helpers.
  - Add focused tests; document any unsupported constraints clearly.
  - Update ai-doc executed plan with exact scope, decisions, and gaps.
  - Commit if complete; if genuinely blocked, document why and mark blocked in SQL.
  - Do not push.

  Testing:
  - Run relevant local tests.
  - Mandatory remote validation on `tn` after rsync to `~/codes/nemotron-cutile-rs--cutile-attention`,
    with the required PATH/CUDA/LLVM env.
  - Include exact commands/results.

  SQL status update (required):
  - Success: `UPDATE todos SET status = 'done' WHERE id = 'implement-cutile-attention'`
  - Blocked: `UPDATE todos SET status = 'blocked' WHERE id = 'implement-cutile-attention'`
created: 2026-03-19T13:38:15Z
finished: 2026-03-19T17:07:29Z
---

# Executed Plan — Implement Cutile Attention

## Research and Plan

- Inspected `nemotron-kernels/src/attention.rs` to confirm the current async GPU path is still a pure D2H -> host compute -> H2D bridge, while the host implementation already defines the parity contracts, masking behavior, grouped-query head mapping, and error conditions that must remain unchanged.
- Reviewed the Wave 1 cutile implementations in `gemm.rs`, `softmax.rs`, `rms_norm.rs`, and `tensor.rs` to reuse the established dispatch structure: Linux-only cutile execution for explicitly supported shapes, host fallback everywhere else, zero-copy wrapping for cutile outputs, and focused backend-selection tests.
- Inspected the local `cutile-rs` checkout to confirm the kernel primitives needed for a safe first attention path: 3D tensor partitions, `partition_permuted` for transposed views, `mma` for tiled score products, and elementwise compare/select helpers for causal masking.
- Chose a bounded first scope instead of a risky full rewrite: implement a Linux-only cutile path when query length, KV length, and head dimension are tile-aligned and the softmax row width stays inside the existing cutile softmax envelope; otherwise preserve the current bridge behavior.

## Planned Implementation Scope

1. Add internal backend selection in `attention.rs` so Linux can advertise/use a real cutile backend for supported shapes while non-Linux and unsupported Linux shapes continue to use host fallback.
2. Implement the device path in three cutile-backed stages that keep tensors resident on device:
   - tiled Q·Kᵀ score computation for grouped-query attention
   - row-wise cutile softmax, with a causal+offset specialization for masked rows
   - weighted value aggregation on device
3. Keep the public async attention API and host functions unchanged, including existing scale validation, head-group validation, and error surface.
4. Add focused tests for backend selection, shape constraints, fallback behavior, and GPU parity for supported cutile shapes.

## Constraints Chosen Up Front

- Linux-only real cutile execution.
- KV sequence length must be a multiple of 16.
- Head dimension must be a multiple of 8.
- KV sequence length must also satisfy the current cutile softmax safety envelope (power-of-two width, `<= 4096`, `i32`-representable).
- Unsupported shapes must silently use the existing host bridge rather than inventing new semantic error modes.

## Risks / Watchpoints

- The cutile path must preserve causal masking and `query_position_offset` exactly; if that is not demonstrably safe on device, the code should fall back rather than approximating.
- Any shape/partition assumptions must be explicit and test-covered so Linux never launches unsupported kernels accidentally.
- Remote Linux validation on `tn` is mandatory before claiming the task is complete because this macOS worktree cannot execute the real CUDA path locally.

## What Was Changed

### `nemotron-kernels/src/attention.rs`

- Added Linux-aware backend metadata so `supported_attention_kernels()` now advertises `Cutile` on Linux while keeping `HostFallback` elsewhere.
- Added GPU tensor length validation for the async path so device calls reject the same shape/length mismatches before dispatch.
- Added an explicit `backend_for_shape(...)` / `supports_cutile_attention(...)` gate that keeps the real device path limited to:
  - Linux only
  - `key_value_sequence_len` power-of-two, `<= 4096`, and divisible by 16
  - `head_dim` divisible by 8
  - all relevant shape dimensions and `query_position_offset` representable as `i32`
- Implemented a real cutile-backed attention path in three stages:
  1. `attention_scores` kernel computes grouped-query `Q·Kᵀ` scores directly on device using reshaped query rows plus `partition_permuted` key views.
  2. `row_softmax` and `masked_row_softmax` kernels normalize attention rows on device, with the masked variant preserving causal masking and `query_position_offset`.
  3. `attention_weighted_values` kernel computes the weighted value reduction on device using the softmax weights and a permuted value view.
- Preserved the existing host implementation and host bridge. Unsupported Linux shapes still fall back to the old D2H -> host compute -> H2D path.
- Kept the public API unchanged: host helpers, async signature, error enum, and shape contracts all remain intact.

### Tests

- Replaced the old backend test with a platform-aware backend metadata test.
- Added a shape-heuristic test that documents the supported device envelope and verifies fallback outside it.
- Kept the existing host-parity and fallback tests intact.
- Added a supported-shape GPU parity test that exercises both:
  - non-causal cutile attention
  - causal decode-style attention with `query_position_offset`

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Device scope | Supported-shape Linux path only | Safer than over-claiming unsupported cutile coverage |
| Score kernel layout | Flatten query rows to `[batch * q_len * q_heads, head_dim]` | Matches the existing row-major memory layout without changing semantics |
| K/V access pattern | Use `partition_permuted` views instead of materialized transposes | Avoids host transposes and keeps tensors resident on device |
| Masking strategy | Special masked softmax kernel for causal rows | Preserves causal/offset semantics exactly without a separate host-side masking step |
| Fallback behavior | Silent host bridge for unsupported shapes | Matches Wave 1 dispatch behavior and avoids new semantic error paths |
| Remote env | Include `/usr/local/cuda-13.2/bin` in `PATH` | Required so `tileiras` is discoverable during cutile runtime compilation |

## Alignment Notes

- Attention semantics were preserved: grouped-query head sharing, scale resolution, causal masking, and decode-style `query_position_offset` all still match the host reference path.
- The new cutile path is intentionally narrower than the host path. Query length itself does **not** need to be tile-aligned, but KV length and head dimension do because the current kernels rely on full-row softmax and fixed reduction tiles.
- Unsupported shapes still use the old bridge instead of erroring, which keeps current callers working and preserves the existing API behavior on macOS and for non-aligned Linux shapes.
- The implementation reuses the Wave 1 patterns rather than introducing a new abstraction layer: internal backend selection, `GpuTensor::cutile_tensor_for_shape`, zero-copy `from_cutile_tensor`, and focused parity tests.

## Verification

### Local commands

```bash
cargo fmt --all
cargo test -p nemotron-kernels attention --quiet
cargo test -p nemotron-kernels --quiet
cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip validator_binary_passes_reference_fixtures --skip benchmark_mode_reports_timings_and_parity
```

### Local results

- `cargo test -p nemotron-kernels attention --quiet` ✅ **12 passed**
- `cargo test -p nemotron-kernels --quiet` ✅ **140 passed**
- `cargo test --workspace ...skips...` ✅ all executed tests passed locally

### Remote sync + validation on `tn`

```bash
rsync -az --exclude target --exclude .git /Users/trungnt13/codes/nemotron-cutile-rs--cutile-attention/ tn:~/codes/nemotron-cutile-rs--cutile-attention/

ssh tn 'cd ~/codes/nemotron-cutile-rs--cutile-attention && \
  export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH" && \
  export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2 && \
  export CUDA_PATH=/usr/local/cuda-13.2 && \
  export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21 && \
  export LLVM_SYS_210_PREFIX=/usr/lib/llvm-21 && \
  export LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config && \
  cargo test -p nemotron-kernels attention -- --nocapture'

ssh tn 'cd ~/codes/nemotron-cutile-rs--cutile-attention && \
  export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH" && \
  export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2 && \
  export CUDA_PATH=/usr/local/cuda-13.2 && \
  export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21 && \
  export LLVM_SYS_210_PREFIX=/usr/lib/llvm-21 && \
  export LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config && \
  cargo test -p nemotron-kernels --quiet && \
  cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip validator_binary_passes_reference_fixtures --skip benchmark_mode_reports_timings_and_parity'
```

### Remote results

- Attention-focused remote test run ✅ **12 passed**
- Remote `cargo test -p nemotron-kernels --quiet` ✅ **142 passed**
- Remote `cargo test --workspace ...skips...` ✅ all executed tests passed
- During bring-up, remote validation revealed two implementation issues that were fixed before final verification:
  - `tileiras` had to be added to `PATH`
  - several cutile kernel typing/shape issues were corrected through iterative remote reruns until parity passed

## Open Gaps

1. The real cutile path currently covers only the bounded supported-shape envelope described above.
2. Query reshaping still relies on the shared `cutile_tensor_for_shape(...)` helper, which may perform a device-to-device copy when the logical shape changes.
3. No new validator fixture was added because the repository still lacks dedicated bundled attention GPU fixtures; coverage is currently via the focused unit tests and full remote crate/workspace test runs.
