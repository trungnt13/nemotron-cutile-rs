---
status: complete
goal: Audit the model-level GPU path for host parity, error propagation, and safe wrapper composition.
prompt: |
  You are auditing the model-level GPU path in /Users/trungnt13/codes/nemotron-cutile-rs.

  Scope:
  - Files: nemotron-model/src/model.rs, nemotron-model/src/generate.rs, and any tightly-related model tests
  - Goal: verify `forward_tokens_gpu`, `predict_next_token_gpu`, and `generate_gpu` correctly preserve host behavior, propagate errors, and compose the lower-level GPU wrappers safely.
  - You may make precise code changes within this scope if you find real correctness issues.
  - Avoid files currently being edited by other agents, especially nemotron-validate/src/main.rs.
  - Run focused tests if you modify code.

  Required workflow:
  1. Investigate thoroughly.
  2. Make necessary fixes in scope if needed.
  3. Run focused validation if changed.
  4. Update SQL on finish:
     - Success: UPDATE todos SET status = 'done' WHERE id = 'audit-model-gpu-path';
     - Blocked: UPDATE todos SET status = 'blocked' WHERE id = 'audit-model-gpu-path';
  5. Return a concise summary with completed work, whether it is fully done, and blockers/follow-ups.
created: 2026-03-19T10:00:17Z
finished: 2026-03-19T10:02:31Z
---

## Research

- Inspected `nemotron-model/src/model.rs` and `nemotron-model/src/generate.rs`.
- Traced `forward_tokens_gpu` through `GpuTensor` and `NemotronBlock::forward_gpu` host-fallback wrappers.
- Reviewed existing GPU model test coverage in `nemotron-model/src/model.rs` and generation coverage in `nemotron-model/src/generate.rs`.
- Verified current repository status before editing because other agents are actively modifying unrelated files.

## Plan

1. Confirm whether the model-level GPU path preserves host behavior for normal and edge cases.
2. If a real correctness gap exists within scope, patch only `nemotron-model` files.
3. Add focused tests that exercise GPU parity and error propagation without touching unrelated crates.
4. Run focused `cargo test -p nemotron-model ...` validation and record results.

## What was done

- Audited `forward_tokens_gpu`, `predict_next_token_gpu`, and `generate_gpu` against their host counterparts.
- Confirmed normal GPU forward/predict composition is structurally safe at the model level: embedding lookup stays on host, `GpuTensor` transfers wrap block-level `forward_gpu`, then final norm and LM head return to the host path.
- Fixed one real host-parity bug: `forward_tokens_gpu` now delegates empty-token inputs to `forward_tokens`, avoiding an early `GpuTensor` zero-dimension error that changed the observable error surface.
- Added focused async tests covering:
  - GPU forward empty-input parity with host errors.
  - GPU generation success parity with host generation.
  - GPU generation error parity for empty prompts without special tokens.

## Key decisions

- Kept the fix in `nemotron-model` only. Lower-level wrappers remain unchanged because the model-level composition already passes consistent row counts and tensor shapes.
- Used host delegation only for the empty-input edge case so the async GPU path preserves host semantics without widening scope into `nemotron-nn` or `nemotron-kernels`.
- Added direct `generate_gpu` tests because the audit found coverage for `generate`, but not for the async GPU generation loop itself.

## Alignment

- The audited GPU path now preserves host behavior for the identified empty-input edge case.
- Error propagation remains aligned with host behavior for normal generation, missing-tokenizer generation, and empty-prompt forward failures within this scope.
- No changes were made outside the requested model-level scope or in files explicitly called out as risky to touch.

## Verification

- Ran: `cargo fmt --all -- nemotron-model/src/model.rs nemotron-model/src/generate.rs`
- Ran: `cargo test -p nemotron-model gpu_`
- Result: 4 focused GPU tests passed:
  - `generate::tests::generate_gpu_preserves_empty_prompt_error`
  - `model::tests::gpu_forward_rejects_empty_input_like_host`
  - `model::tests::gpu_forward_matches_host`
  - `generate::tests::generate_gpu_matches_host_generation`

## Next steps

- If future work broadens beyond the model layer, add direct boundary checks in lower-level `forward_gpu` wrappers so mismatched tensor shapes and row counts fail closer to the API boundary.
- Once concurrent edits settle, consider a clean-tree workspace or worktree for follow-up audit commits.
