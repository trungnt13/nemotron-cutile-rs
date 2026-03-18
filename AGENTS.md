# Agent Guidelines

## Overview

Rust inference framework for **Nemotron-3-Nano-30B-A3B** — a 52-layer hybrid model with 23 Mamba-2 mixers, 23 Mixture-of-Experts (MoE) layers, and 6 Grouped-Query Attention (GQA) blocks. The goal is output parity with the vLLM reference implementation (`stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ`), targeting INT4 (AWQ) quantization on an RTX 3090.

Key design decisions:
- **Host-fallback first** — all 10 kernels are pure-Rust CPU implementations. GPU kernels via `cutile-rs` are planned but not yet implemented.
- **Kernel parity before model parity** — validate individual kernel outputs against vLLM fixtures before attempting full-model inference.
- **MoE uses sigmoid routing with bias correction**, not softmax — verified from the reference, not assumed from naming.

## Build & Test

```bash
cargo build --workspace          # build all 5 crates
cargo test  --workspace          # run all unit + integration tests (~167 tests)
cargo run -p nemotron-validate   # run kernel + E2E validation against fixtures (9/9 pass)
cargo run -p nemotron-cli -- "Hello, world!"  # generation preview
```

Toolchain requirements:
- Rust edition 2021, MSRV **1.85**
- Workspace resolver `"2"`
- `cutile` is a git dependency (`NVlabs/cutile-rs`) — present in `Cargo.toml` but not yet used in any kernel

Prefer workspace-level dependencies in the root `Cargo.toml` to keep versions consistent across crates. Ask before adding new workspace dependencies.

## Project Structure

```
nemotron-kernels  (10 kernel modules: gemm, softmax, rms_norm, activations,
│                  attention, conv1d, ssm, embedding, quantize, moe_routing)
▼
nemotron-nn       (7 layer modules: linear, attention, mlp, mamba2, moe, block, cache)
▼
nemotron-model    (config, tokenizer, weights, model runtime, generation)
▼
├── nemotron-cli        (binary — thin wrapper over model)
└── nemotron-validate   (validation suite — loads fixtures from data/)
```

Dependencies flow strictly downward. Do not introduce upward or circular dependencies between crates.

| Crate | Role |
|---|---|
| `nemotron-kernels` | Low-level compute primitives (pure math, no I/O) |
| `nemotron-nn` | Neural-network layer abstractions composing kernels |
| `nemotron-model` | Model config, tokenizer, weight loading, generation |
| `nemotron-cli` | Command-line interface binary |
| `nemotron-validate` | Correctness validation against vLLM reference fixtures |

## Code Conventions

- **Error handling:** Crate-specific error enums (e.g., `Mamba2Error`, `ModelForwardError`) for recoverable errors; `anyhow` for top-level binaries only. Do not use `unwrap()` in library code.
- **Kernel API pattern:** Each kernel module exposes a `*Kernel` struct, a `*Shape` config struct, and one or more free functions (e.g., `gemm_into()`, `softmax()`). Follow this pattern for new kernels.
- **Test documentation:** Every `#[test]` function must have a `///` doc comment in the format: `"Verifies that [behavior] when [condition]. This catches [regression]."` This was established in the verification pass and must be maintained.
- **Module layout:** One module per kernel/layer. Public types re-exported from `lib.rs` via a `*Stub` registry.
- **Naming:** Rust standard — `snake_case` for functions/modules, `PascalCase` for types, `SCREAMING_SNAKE` for constants. Kernel names match their mathematical operation (e.g., `gemm`, `rms_norm`, `selective_scan`).
- **Visibility:** Minimize `pub` surface. Internal helpers should be `pub(crate)` or private.

## Testing

Tests live in two places:
1. **Inline unit tests** — `#[cfg(test)] mod tests` inside each source file (~157 tests across all crates)
2. **Integration tests** — `tests/` directories in `nemotron-model`, `nemotron-cli`, and `nemotron-validate`

The **validation suite** (`nemotron-validate`) loads reference fixtures from `data/` and compares kernel/E2E outputs:

```
data/                          (gitignored — not committed)
├── reference_kernels/
│   ├── fixtures.json          kernel test cases (shapes, inputs, expected outputs)
│   └── manifest.json
└── reference_outputs/
    ├── fixtures.json          E2E test cases (prompts, expected tokens/logits)
    ├── manifest.json
    └── tokenizer.json
```

When adding a new kernel or layer:
1. Add unit tests in the source module
2. Add a `///` doc comment to each test function
3. If a vLLM reference fixture exists, add a validation case in `nemotron-validate`
4. Run `cargo test --workspace` to confirm all tests pass

## Boundaries

### Always Do
- Run `cargo test --workspace` before committing
- Follow the layered crate architecture (kernels → nn → model → cli/validate)
- Add tests for new code; document tests with `///` comments
- Update `ai-docs/` for substantial tasks

### Ask First
- Adding or removing workspace dependencies
- Modifying files in `data/` (validation fixtures)
- Architectural changes (new crates, changing crate boundaries)
- Pushing to remote (`git push`) — local commits are fine, but push only with explicit user approval
- Changes to `.gitignore`

### Never Do
- Force-push (`git push --force`)
- Run destructive shell commands (`rm -rf`, `git clean -fdx`) without confirmation
- Use `unwrap()` or `expect()` in library crate code
- Introduce circular dependencies between crates
- Commit secrets, model weights, or data files to the repository

## Current State

| Area | Status | Notes |
|------|--------|-------|
| Host-fallback kernels | ✅ 10/10 implemented | Pure Rust, all pass validation |
| NN layers | ✅ 7/7 implemented | Compose kernels correctly |
| Model runtime | ✅ Working | Config, tokenizer, weights, generation |
| Kernel validation | ✅ 9/9 pass | Against vLLM reference fixtures |
| Unit tests | ✅ ~167 passing | All documented with `///` comments |
| GPU kernels (cutile) | ⏳ Not started | `cutile` dep exists but unused |
| Full checkpoint loading | ⏳ Pending | Needs real AWQ model weights |
| Layer-level validation | 🚫 Blocked | vLLM internals incompatible with fixture extraction |
| CI/CD | ❌ None | No GitHub Actions — all testing is local |
| Doc-tests | ❌ None | Public API examples not yet written |

## Workflow

Every substantial task must follow this sequence unless the user explicitly asks for a narrower spike:

1. **Research** — inspect the current code, references, constraints, and risks before changing implementation.
2. **Plan** — write or update the design/execution plan before coding so scope, assumptions, blockers, and validation strategy are explicit.
3. **Execution / Implementation** — make the code changes against that plan and keep the plan aligned when scope changes.
4. **Alignment** — compare the implementation against the user prompt, reference behavior, and prior decisions; record mismatches or intentional deviations.
5. **Verification** — run the relevant build, test, and validation commands and record the results before handoff.

Do not skip straight to implementation for normal feature work. Start from research, then plan, then execute.

## Commits

Commit after completing each self-contained task — not after every file edit, but once the change compiles and makes logical sense as a unit. This keeps the history bisectable and each commit reviewable on its own.

Write commit messages as a short imperative summary (≤ 72 chars), followed by a blank line and a body that explains *why* the change was made, not just *what* changed. Include a `Co-authored-by` trailer when applicable.

```
Add flash-attention kernel for variable-length sequences

The previous padded-batch approach wasted ~40 % compute on padding tokens.
Flash-attention operates on ragged tensors directly, eliminating that overhead.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

## Parallel Work

Use `git worktree` when working on independent tasks in parallel — never juggle multiple features in a single checkout. Each worktree gets its own build directory, so concurrent `cargo build` invocations won't conflict.

```bash
# create a worktree for a feature branch
git worktree add ../nemotron-cutile-rs--flash-attn -b flash-attn

# when done, clean up
git worktree remove ../nemotron-cutile-rs--flash-attn
```

Name worktree directories `<repo>--<branch>` so they sort next to the main checkout and their purpose is obvious at a glance.

## Documentation

Keep docs **concise**, **intuitive**, and **reasoned**:

- **Concise** — say it once, say it clearly. If a sentence doesn't help the reader act, delete it.
- **Intuitive** — structure docs so the most common question is answered first. Use concrete examples over abstract descriptions.
- **Reasoned** — always explain *why* a convention exists, not just *what* it is. A rule without a rationale will be ignored or cargo-culted.

Update documentation in the same commit as the code it describes — stale docs are worse than no docs because they actively mislead.

## ai-docs Ownership

`ai-docs/` is the shared design-document workspace for research notes, plans, execution summaries, alignment notes, and verification results. **Every substantial task must produce an `ai-docs/` record before it is considered complete.**

### Naming Convention

Use the pattern `executed-plan-<goal-description>.md` where `<goal-description>` is a short kebab-case summary of the task goal. Do not prefix with the project name — the repository already provides that context.

Examples:
- `executed-plan-build-nemotron-inference.md`
- `executed-plan-verify-implementation-tests.md`
- `executed-plan-add-flash-attention-kernel.md`

### Requirements

- Multi-agent work must use `ai-docs/` as the source of truth for handoff state.
- Each agent must leave the docs in a better state than it found them: update status, decisions, blockers, commands run, and next actions.
- Implementation should not begin until the relevant `ai-docs/` document captures research and plan context, unless the user explicitly directs otherwise. If that happens, document the deviation immediately.

### Enforcement Checklist

Before marking any task complete, verify **all** of the following. Do not skip any step — incomplete handoff creates invisible debt.

1. [ ] `cargo test --workspace` passes (or the relevant subset if scoped)
2. [ ] An `ai-docs/executed-plan-*.md` file exists for this task with frontmatter:

```yaml
---
status:        # e.g. complete, implemented-with-open-gaps, blocked
goal:          # one-line goal statement
prompt:        # the user prompt that initiated this task
created:       # ISO 8601 timestamp
finished:      # ISO 8601 timestamp (or blank if not yet finished)
---
```

3. [ ] The body documents: what was done, key decisions, verification results, and next steps (if any)
4. [ ] All changes are committed (`git status` shows clean working tree) with a message following the Commits convention above
5. [ ] If the user has approved, push to remote (`git push`) — do not push without explicit approval

## References

- **vLLM reference model:** `stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ`
- **cutile-rs:** <https://github.com/NVlabs/cutile-rs>
- **Design docs:** `ai-docs/` directory (executed plans with status frontmatter)
- **Validation fixtures:** `data/reference_kernels/` and `data/reference_outputs/` (gitignored)
