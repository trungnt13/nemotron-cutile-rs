# Agent Guidelines

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

## Workflow

Every substantial task must follow this sequence unless the user explicitly asks for a narrower spike:

1. **Research** — inspect the current code, references, constraints, and risks before changing implementation.
2. **Plan** — write or update the design/execution plan before coding so scope, assumptions, blockers, and validation strategy are explicit.
3. **Execution / Implementation** — make the code changes against that plan and keep the plan aligned when scope changes.
4. **Alignment** — compare the implementation against the user prompt, reference behavior, and prior decisions; record mismatches or intentional deviations.
5. **Verification** — run the relevant build, test, and validation commands and record the results before handoff.

Do not skip straight to implementation for normal feature work. Start from research, then plan, then execute.

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
5. [ ] Commits are pushed to the remote (`git push`) — unsynced local commits are invisible to other agents and collaborators

## Project Context

This is a Rust workspace (`resolver = "2"`, edition 2021) with five crates:

| Crate | Role |
|---|---|
| `nemotron-kernels` | GPU compute kernels |
| `nemotron-nn` | Neural-network layer abstractions |
| `nemotron-model` | Model architecture and weights |
| `nemotron-cli` | Command-line interface |
| `nemotron-validate` | Correctness and regression tests |

Run `cargo build` at the workspace root to build everything; `cargo test` to run all tests. Prefer workspace-level dependencies in the root `Cargo.toml` to keep versions consistent across crates.
