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
