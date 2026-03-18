---
status: implemented-with-open-gaps
goal: Build nemotron-rs (Rust inference framework on cutile-rs) for Nemotron-3-Nano-30B-A3B on RTX 3090, matching vLLM outputs.
prompt: >-
  pick Nemotron-3-Nano-30B-A3B, create environment on remote RTX 3090,
  implement nemotron-rs based on cutile-rs as a deep learning inference
  framework, match outputs of existing framework, add integration tests
created: 2026-03-17T22:17:59Z
finished: 2026-03-18T10:07:20Z
---

# Executed Plan — Build Nemotron Inference Framework

## Summary

Rust workspace with host-fallback kernels, NN layers, model runtime, greedy generation, and validation against vLLM reference fixtures. Kernel parity: **9/9 pass**. Full checkpoint parity: not yet reached.

---

## Phase 0 — Research ✅

Established constraints before writing code.

| Finding | Detail |
|---|---|
| Reference framework | vLLM (`stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ`) |
| Hardware | `ssh tn` — Ubuntu 24.04, RTX 3090 sm_86, 24 GB VRAM |
| Architecture | 52 hybrid layers: 23 Mamba-2, 23 MoE, 6 GQA Attention |
| Quantization | INT4 preferred (17 GB fits 24 GB VRAM) |
| Key insight | MoE uses sigmoid routing with bias correction, not softmax — must validate from reference, not assume from naming |
| Key insight | Attention has no RoPE despite config having `rope_theta` — follow HF reference exactly |

---

## Phase 1 — Remote Environment ✅

| Task | Status | What |
|---|---|---|
| 1.1 CUDA toolkit | ✅ | CUDA 13.2 at `/usr/local/cuda-13.2` |
| 1.2 Rust nightly | ✅ | Via rustup |
| 1.3 LLVM 21 + MLIR | ✅ | `libmlir-21-dev`, `mlir-21-tools` |
| 1.4 Build cutile-rs | ✅ | `~/codes/cutile-rs-upstream` — examples pass on sm_86 |
| 1.5 vLLM reference | ✅ | `~/codes/nemotron-rs/.venv` — AWQ model runs |

---

## Phase 2 — Reference Artifacts ✅

| Task | Status | Output |
|---|---|---|
| 2.1 E2E inference | ✅ | `outputs/reference_inference_vllm.json` |
| 2.2 Kernel fixtures | ✅ | `outputs/reference_kernels/` — 7 kernels with shapes/dtypes |
| 2.3 Layer extraction | 🚫 blocked | vLLM internals incompatible with hook-based extraction |

---

## Phase 3 — Workspace Scaffolding ✅

Five crates, each with a single responsibility:

| Crate | Role |
|---|---|
| `nemotron-kernels` | Host-fallback compute primitives |
| `nemotron-nn` | Layer abstractions composing kernels |
| `nemotron-model` | Config, tokenizer, weights, runtime, generation |
| `nemotron-cli` | CLI binary |
| `nemotron-validate` | Kernel + E2E validation against fixtures |

---

## Phase 4 — Kernels ✅

All host-fallback. GPU backends are future work.

| Kernel | File | Notes |
|---|---|---|
| GEMM | `gemm.rs` | f32 accumulate |
| RMSNorm / gated | `rms_norm.rs` | Row-wise; gated variant for Mamba2 |
| Softmax | `softmax.rs` | Numerically stable |
| SiLU | `activations.rs` | — |
| ReLU² | `activations.rs` | For MoE expert MLPs |
| GQA attention | `attention.rs` | Causal mask, decode offset, grouped heads |
| Causal Conv1D | `conv1d.rs` | **Fix:** reversed tap order for PyTorch parity |
| Selective scan | `ssm.rs` | Optional initial state, D term, softplus dt |
| Embedding lookup | `embedding.rs` | — |
| MoE routing | `moe_routing.rs` | **Fix:** added `moe_route_softmax` for reference parity |
| INT4 quant/dequant | `quantize.rs` | Group-wise with scale + zero-point |

**Design decision — host fallbacks first.** Correct, inspectable CPU paths before GPU claims. Future cutile-rs kernels drop in behind the same API.

---

## Phase 5 — NN Layers ✅

| Layer | File | Composes |
|---|---|---|
| Linear projection | `linear.rs` | GEMM, optional quantized weights |
| Attention | `attention.rs` | Linear × 4 + attention kernel |
| MLP | `mlp.rs` | Linear × 3 + activation |
| Mamba2 mixer | `mamba2.rs` | Conv1D → SiLU → SSM → gated RMSNorm → Linear |
| MoE | `moe.rs` | Router + N expert MLPs + optional shared expert |
| Block | `block.rs` | Pre-norm → mixer → residual |
| Cache | `cache.rs` | KV cache (attention) + conv/SSM state (Mamba) |

**Fix during integration:** RMSNorm in Mamba2 was applied to full sequence buffer instead of row-wise. Corrected.

---

## Phase 6 — Model Runtime ✅

| Component | File | What |
|---|---|---|
| Config | `config.rs` | Parses Nemotron-3-Nano-30B-A3B defaults |
| Tokenizer | `tokenizer.rs` | HuggingFace `tokenizers` crate |
| Weights | `weights.rs` | Direct safetensors header/index parsing — no external API wrapper |
| Runtime | `model.rs` | Embedding → 52 blocks → final RMSNorm → LM head |
| Generation | `generate.rs` | Greedy loop with EOS stopping |
| CLI | `main.rs` | `--model-dir`, `--prompt`, `--max-tokens`, `--preview` |

---

## Phase 7 — Validation ✅ (partial)

| Task | Status | Detail |
|---|---|---|
| Kernel validation | ✅ 7/7 | GEMM, RMSNorm, softmax, SiLU, ReLU², Conv1D, MoE routing |
| Synthetic E2E validation | ✅ 2/2 | Tokenizer → forward → predict → generate → preview on tiny runtime |
| Integration tests | ✅ | `nemotron-model/tests/`, `nemotron-cli/tests/`, `nemotron-validate/tests/` |
| Layer-level validation | 🚫 blocked | Depends on unresolved layer extraction |
| Performance benchmarks | ⏳ pending | — |
| Full checkpoint parity | ⏳ pending | Runtime doesn't yet load the real 52-layer checkpoint end-to-end |

```text
$ cargo test --workspace --quiet    # 156 tests pass
$ cargo run -p nemotron-validate    # summary: 9/9 validations passed
```

---

## Key Decisions

| Decision | Reason |
|---|---|
| Host fallbacks before GPU | Correct numerics first; GPU kernels drop in later |
| Kernel parity before model parity | Isolate mismatches at the smallest testable unit |
| Synthetic E2E fixtures, not fake full-model | Honest about scope; validator labels its own limitations |
| Blocked tasks stay blocked | Per-layer extraction failed repeatedly; hiding it would create false progress |

---

## Deliverables

**Source:** `nemotron-kernels/src/`, `nemotron-nn/src/`, `nemotron-model/src/`, `nemotron-cli/src/`, `nemotron-validate/src/`

**Tests:** `nemotron-model/tests/{model_runtime,weights}_integration.rs`, `nemotron-cli/tests/preview_integration.rs`, `nemotron-validate/tests/validator_integration.rs`

**Data (gitignored):** `data/reference_kernels/{fixtures,manifest}.json`, `data/reference_outputs/{fixtures,manifest,tokenizer}.json`

**Remote:** `~/codes/nemotron-rs/.venv`, `~/codes/nemotron-rs/outputs/`, `~/codes/cutile-rs-upstream`

---

## Next Steps

1. **Performance benchmarking** — add throughput/latency measurement for the current host runtime
2. **Unblock layer extraction** — try HF transformers hooks instead of vLLM internals
3. **Real checkpoint loading** — extend `weights.rs` + `model.rs` to load the full 52-layer AWQ checkpoint
4. **Full output parity** — only claim vLLM-matching once real end-to-end generation comparison passes
