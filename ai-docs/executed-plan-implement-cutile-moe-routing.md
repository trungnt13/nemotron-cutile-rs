---
status: complete
goal: Implement a real cutile-backed sigmoid MoE routing path with host fallback intact
prompt: >-
  You are implementing todo `implement-cutile-moe-routing` in
  `/Users/trungnt13/codes/nemotron-cutile-rs--cutile-moe-routing`.
  Implement a real cutile-backed MoE routing path in
  `nemotron-kernels/src/moe_routing.rs`, preserve the current
  sigmoid-routing-with-bias-correction semantics and top-k contracts,
  keep host fallback for unsupported shapes/platforms, add focused tests,
  update validator/benchmark/reporting if needed, run local plus mandatory
  remote `tn` validation, commit locally, and do not push.
created: 2026-03-19T16:09:57Z
finished: 2026-03-19T16:09:57Z
---

# Executed Plan — Implement Cutile MoE Routing

## Research and Plan

- Inspected `nemotron-kernels/src/moe_routing.rs` to confirm the existing async GPU APIs still performed a pure `to_host_async()` bridge for both sigmoid routing and softmax routing.
- Inspected the existing Wave 1 cutile integrations in `gemm.rs`, `rms_norm.rs`, `softmax.rs`, plus `tensor.rs`, to mirror the established Linux-only pattern: kernel registry metadata, shape heuristics, cutile launch helpers, and host fallback bridges.
- Inspected the local cutile checkout under `~/.cargo/git/checkouts/cutile-rs-*/13732c8` to confirm available tile operations (`iota`, `reduce_max`, `reduce_min`, `select`, `broadcast_scalar`, scalar kernel arguments) and the constraints that surfaced during Linux validation.
- Planned a first real MoE kernel scope that preserved the public API exactly, implemented only the sigmoid top-k route on Linux, and kept host fallback for unsupported expert-count/top-k/platform combinations.
- Deviation: the user handed over an execution task directly, so implementation started before this ai-doc existed. This file records the research/plan retroactively and captures the final aligned state.

## What Was Changed

### `nemotron-kernels/src/moe_routing.rs`

- Added Linux platform backend metadata so `MOE_SIGMOID_TOPK` reports `Cutile` on Linux while other platforms keep `HostFallback`.
- Added GPU-side shape validation and a cutile support heuristic for the first supported routing envelope:
  - expert counts: `1, 2, 4, 8, 16, 32, 64, 128`
  - `top_k <= 8`
  - `token_count`, `expert_count`, and `top_k` must fit the cutile/i32 launch bounds
- Implemented a real Linux-only `#[cutile::module]` kernel for sigmoid top-k routing.
  - The kernel keeps the existing sigmoid semantics rather than switching to softmax.
  - It preserves tie-breaking toward lower expert indices via `reduce_max` + `reduce_min` over tied candidates.
  - It pads `top_k` and `token_count` to power-of-two cutile-friendly launch/output shapes, then compacts back to the original logical `[token_count, top_k]` contract on the host side.
- Kept `moe_route_softmax(...)` on the existing host bridge path because this task was specifically about the sigmoid routing contract and the validator’s softmax fixture path still needs to remain correct without inventing new GPU semantics.
- Added explicit host bridge helpers and used backend dispatch in the async GPU path so unsupported Linux shapes still fall back cleanly.

### Tests in `nemotron-kernels/src/moe_routing.rs`

- Replaced the old backend test with a platform-aware backend-registry assertion.
- Added a focused cutile support-heuristic test.
- Updated the GPU parity test wording so it remains correct on both Linux cutile and non-Linux host-fallback platforms.
- Added a Linux-only model-like routing test for a `128`-expert, `top_k=6` shape to prove the padded cutile path works for the actual Nemotron-style routing envelope.

### `nemotron-validate/src/benchmark.rs`

- Updated the benchmark/report note so it no longer implies MoE routing is still always host-delegating on Linux.
- No fixture-backed validator logic changed because this worktree does not contain the gitignored reference fixture directories locally or on `tn`, and the existing validator’s bundled MoE route coverage is for the softmax path, not the new sigmoid cutile kernel.

## Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Preserve routing semantics | Keep sigmoid top-k logic and lower-index tie-breaking exactly | The user explicitly warned against accidental softmax semantics drift |
| Scope of first cutile path | Only the sigmoid `moe_route(...)` Linux path is real cutile compute | That is the routing path used by the model contract this todo targeted |
| Unsupported Linux shapes | Fall back to host bridge instead of erroring | Matches the Wave 1 pattern and preserves existing caller behavior |
| Handle non-power-of-two `top_k=6` | Pad token/output shapes to power-of-two cutile shapes, then compact | Required to support Nemotron’s model-like `top_k=6` contract under cutile’s current tile constraints |
| Softmax GPU path | Leave `moe_route_softmax(...)` on host fallback | Avoids introducing a second semantics change under a task scoped to sigmoid routing |

## Alignment Notes

- The implementation adds a real cutile-backed MoE route in `moe_routing.rs` for Linux rather than another D2H → host → H2D bridge.
- Sigmoid routing semantics were preserved exactly at the contract level: sigmoid scoring, top-k selection, and lower-index tie-breaking remain unchanged.
- Softmax routing remains host-backed intentionally; this is consistent with the task’s scope and avoids conflating the two routing semantics.
- Validator/benchmark reporting was updated only where it had become misleading. No fixture-backed validator path was expanded because the required `data/` fixtures are absent in this worktree and on the target `tn` path.

## Verification

### Baseline before changes

```bash
cargo test -p nemotron-kernels moe_routing -- --nocapture
```

Observed result:

- `nemotron-kernels` MoE routing tests: **15 passed**, **0 failed**

### Final local verification

Commands run:

```bash
cargo fmt --all
cargo test -p nemotron-kernels moe_routing --quiet
cargo build --workspace --quiet
cargo test --workspace --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip benchmark_mode_reports_timings_and_parity --skip validator_binary_passes_reference_fixtures
```

Observed results:

- `cargo test -p nemotron-kernels moe_routing --quiet`: **16 passed**, **0 failed**
- `cargo build --workspace --quiet`: **passed**
- `cargo test --workspace --quiet -- --skip ...`: **207 passed**, **0 failed**, **1 ignored**; fixture-dependent validator/benchmark integration tests were intentionally skipped because `data/reference_kernels` and `data/reference_outputs` are absent in this worktree

### Final remote validation on `tn`

Repository sync command:

```bash
ssh tn 'mkdir -p ~/codes/nemotron-cutile-rs--cutile-moe-routing'
rsync -az --delete --exclude target --exclude .git /Users/trungnt13/codes/nemotron-cutile-rs--cutile-moe-routing/ tn:~/codes/nemotron-cutile-rs--cutile-moe-routing/
```

Validation command run on `tn`:

```bash
cd ~/codes/nemotron-cutile-rs--cutile-moe-routing
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.2/bin:/usr/lib/llvm-21/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.2
export CUDA_PATH=/usr/local/cuda-13.2
export CUDA_TILE_USE_LLVM_INSTALL_DIR=/usr/lib/llvm-21
export LLVM_SYS_210_PREFIX=/usr/lib/llvm-21
export LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config
export LD_LIBRARY_PATH="/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"
cargo build --workspace --quiet
cargo test -p nemotron-kernels moe_routing --quiet
cargo test -p nemotron-validate --quiet -- --skip bundled_reference_fixtures_validate --skip bundled_gpu_reference_fixtures_validate --skip bundled_benchmark_comparison_runs --skip benchmark_mode_reports_timings_and_parity --skip validator_binary_passes_reference_fixtures
```

Observed results:

- `cargo build --workspace --quiet`: **passed**
- `cargo test -p nemotron-kernels moe_routing --quiet`: **17 passed**, **0 failed**
  - this includes the Linux-only cutile test for the model-like `128`-expert, `top_k=6` shape
- `cargo test -p nemotron-validate --quiet -- --skip ...`: **3 passed**, **0 failed**, **3 filtered out** in the library test target; the binary/integration targets that depend on fixture files remained filtered/skipped as intended

## Blockers / Open Gaps

1. The real cutile MoE path currently supports a bounded first-wave shape set instead of every possible expert-count/top-k combination.
   - unsupported Linux shapes still fall back to host correctly
2. Fixture-backed validator coverage for the new sigmoid cutile path was not added because this worktree and the target `tn` checkout do not contain the gitignored `data/` reference fixtures.

## Next Steps

1. If broader Linux routing shapes are needed, extend the cutile dispatch table/shape heuristic while keeping the same host-fallback contract.
2. If fixture data becomes available in this worktree, add a bundled validator or benchmark case that directly exercises the sigmoid cutile MoE route.
