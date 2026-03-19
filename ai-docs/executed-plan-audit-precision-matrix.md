---
status: complete
goal: Audit precision types and numeric behavior across GPU-related kernel, nn, and model paths
prompt: >-
  You are auditing precision and numeric behavior in /Users/trungnt13/codes/nemotron-cutile-rs.
  Scope: review kernel, nn, and model GPU-related code to document precision types in use:
  tensor storage, kernel inputs/outputs, accumulators, tolerances, and AWQ/int4 handling.
  Produce a concise precision matrix and identify whether any code change is needed right now.
created: 2026-03-19T09:50:00Z
finished: 2026-03-19T09:50:00Z
---

# Executed Plan — Audit Precision Matrix

## Summary

Audited the precision-sensitive GPU-facing code paths in `nemotron-kernels`, `nemotron-nn`, `nemotron-model`, and `nemotron-validate` without touching the concurrently edited `tensor.rs` and `device.rs` files. The current stack is consistently **f32 storage / f32 API / f64 reduction accumulation** for numerically sensitive host kernels, with **INT4 affine packing** available in the quantization helpers and NN linear weight types, but not yet wired into the active projection execution path.

No correctness fix is required immediately for the audited precision behavior. The main near-term gap is documentation and future-proofing for real cutile kernels, not a current numeric bug.

## Research

Inspected:

- `nemotron-kernels/src/{gemm,softmax,rms_norm,attention,conv1d,ssm,activations,embedding,quantize,moe_routing}.rs`
- `nemotron-nn/src/{linear,attention,mlp,moe,block,mamba2}.rs`
- `nemotron-model/src/{model,weights}.rs`
- `nemotron-validate/src/{main,e2e}.rs`

Also checked repo-wide `cutile` and GPU wrapper references to confirm that the current async GPU API still round-trips through host kernels.

## Precision Matrix

| Area | Storage | Inputs / outputs | Accumulator / internal math | Tolerance / comparison | Notes |
|---|---|---|---|---|---|
| `GpuTensor` interface usage | `f32` buffers | `f32` host/device transfers | n/a | exact in wrapper parity tests today | Call sites and existing docs indicate `GpuTensor` wraps `cutile::Tensor<f32>` on Linux and `Vec<f32>` elsewhere. |
| `gemm.rs` | `f32` | `&[f32] -> [f32]` | inner product in `f64`, cast to `f32` | unit tests `1e-6`; validator observed `1e-6` max diff | Current async GPU wrapper delegates to host GEMM. |
| `softmax.rs` | `f32` | `&[f32] -> [f32]` | max-subtracted exponentials in `f32`, partition sum in `f64` | unit tests `1e-6`; validator `0.0` observed | Numerically stable max-subtraction already present. |
| `rms_norm.rs` | `f32` | `&[f32] -> [f32]`, `epsilon: f32` | mean-square reduction in `f64`, denominator/output in `f32` | unit tests `1e-6`; validator `0.0` observed | Final model norm uses `epsilon = 1e-5`. |
| `attention.rs` | `f32` Q/K/V/outputs | `f32` scores, weights, outputs | Q·K reduction in `f64`; softmax partition in `f64`; weighted V reduction in `f64` | kernel tests `1e-5` | Default scale is `(head_dim as f32).sqrt().recip()`. |
| `conv1d.rs` | `f32` | `&[f32] -> [f32]` | tap accumulation in `f64` | unit tests `1e-6`; validator `0.0` observed | Depthwise causal conv path only. |
| `ssm.rs` | `f32` state and outputs | `f32` params / outputs | output projection accumulation in `f64`; state transition math in `f32` exp/ops | tests use `1e-5` | Important for Mamba-2 numeric drift budget. |
| `activations.rs` | `f32` | `f32 -> f32` | pure elementwise `f32` | unit tests `1e-6` | Sigmoid uses split branches for stability. |
| `embedding.rs` | `f32` table | token ids -> `f32` rows | no reduction | exact / direct copy | Explicit token bounds checks exist. |
| `moe_routing.rs` | `f32` scores/weights | `f32` scores -> `usize` indices + `f32` weights | sigmoid/softmax routing in `f32` | unit tests `1e-6` | Default routing is sigmoid top-k, matching project guidance. |
| `nemotron-nn/src/linear.rs` active path | dense weights stored as `Vec<f32>` | `f32` input/output | inherits GEMM `f64` accumulation | unit tests `1e-6` | Active projection path only supports `DenseF32`; INT4 is represented but rejected for host projection. |
| `nemotron-nn` GPU wrappers | `GpuTensor` + host `Vec<f32>` | `f32` end-to-end | preserves host precision exactly today | mostly exact host parity tests | All audited `forward_gpu` / `project_gpu` paths transfer to host, compute, then re-upload. |
| `nemotron-model` runtime | embeddings, norms, logits in `Vec<f32>` | `f32` hidden states / logits | inherits kernel math above | E2E tolerance `1e-5` | `forward_tokens_gpu` still performs final norm + LM head on host. |
| `weights.rs` manifest / loader | raw bytes `Vec<u8>` plus `dtype: String` metadata | no numeric decode yet | n/a | metadata only | Loader records `dtype` strings such as `F32` / `U8`; it does not yet materialize typed tensors. |
| INT4 / AWQ-related handling | packed `u8` nibbles + `f32` scale + `u8` zero_point | quantize/dequantize uses `f32` values | scalar affine math in `f32` | unit tests `1e-6` | Formula is `round(v / scale + zero_point).clamp(0, 15)` and dequant `(code - zero_point) * scale`. This is generic affine INT4, not a full AWQ runtime path. |

## Key Findings

1. **Current GPU precision is effectively host precision.** Every audited async GPU wrapper still performs D2H -> host kernel -> H2D, so there is no separate cutile numeric behavior yet.
2. **The dominant compute contract is stable:** storage and API surfaces are `f32`, while reductions that benefit from it use `f64` accumulators (`gemm`, `softmax`, `rms_norm`, `attention`, `conv1d`, `ssm` output accumulation).
3. **INT4 exists as a storage/dequantization building block, not as an active model execution path.** `Int4LinearWeights` can materialize dense weights, but `LinearProjection::project[_into]` currently rejects `Int4Affine` on the host path.
4. **`weights.rs` tracks dtype metadata but does not enforce or decode it.** That is acceptable today because the audited runtime is still synthetic / scaffold-level, but it is a future integration risk once real checkpoint loading starts.
5. **Tolerance policy is already layered, if lightly documented:** kernel unit tests mostly use `1e-6`, some more reduction-heavy paths use `1e-5`, validator default is `1e-4`, and E2E uses `1e-5`.

## Verification

- `cargo test --workspace --quiet` ✅
- `cargo run -p nemotron-validate --quiet` ✅
  - kernel validation: `9/9` pass
  - gpu validation: `5/5` pass
  - observed diffs stayed within the existing tolerance budget

## Conclusion

No code change is required right now for precision correctness in the audited scope. Recommended follow-up work is documentation-oriented:

- document the intentional `f32 storage + f64 accumulation` policy before real cutile kernels land;
- document that current INT4 support is affine packed-weight infrastructure, not full AWQ execution parity yet;
- when real GPU kernels are added, re-audit tolerances because exact host parity will no longer be guaranteed.
