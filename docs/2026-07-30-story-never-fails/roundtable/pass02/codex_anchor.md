# Codex Grounded Anchor — Round 2

## Verdict

YES-WITH-FIXES. Revision 1 now has the right boundaries: deterministic
whole-source evidence mechanics, freely replaceable fictional drafts, typed
model-candidate retries, and one final ledger assembly. The coding plan needs
the following exact seams pinned before edits.

## Must fix

1. `invoke_codex_structured` currently owns one ladder and converts
   `StructuredCallFailedError` to `CodexPassError`. Refactor the existing body
   into a one-candidate helper and retain a public wrapper with
   `retry_until_valid=False` by default. The canonical runner opts in. Catch
   only typed ladder exhaustion in the outer loop; all other errors propagate.
2. Installed ComfyUI's `InterruptProcessingException` inherits
   `BaseException`, so the existing `except Exception` blocks already let it
   escape. Preserve that property. Add a lazy interrupt poll before and after
   each model invocation and before each fresh candidate; catch only a missing
   Comfy import, never the interrupt itself.
3. A fresh retry must be visible in the actual prompt. Copy immutable artifact
   inputs and add bounded `candidate_cycle`, unique nonce, and deduplicated last
   findings. Do not recursively include prior prompts or rejected raw drafts.
4. P0 repair closures currently capture the single unwindowed `p0_inputs`,
   allowed-field set, and A0 repair payload. Factor one window invocation so its
   prompt, literal repair, and local validator all close over the same window.
   Rebase only after local acceptance.
5. `p0_source_chunks` needs an explicit overlap argument whose default
   preserves existing behavior. Production uses 239 characters. Shift the next
   start within the same fixed allowance; never make a prompt larger than the
   measured budget, and reject an overlap that cannot make forward progress.
6. RSS projection must make `full_text` available to the chunker even when a
   derived alias contains it. Do this only for RSS; preserve pinned-premise
   projection behavior.
7. Define deterministic merge helpers as pure functions with exact selection
   order, dedupe identity, contiguous IDs, number-parent remapping, and a final
   immutable-A0 validation. Never call `source.find` during global rebasing;
   duplicate quotes must retain their accepted window occurrence.
8. P5 can be marked accepted in the call journal before post-P5 safety cleanup.
   If safety rejects it, mark that exact candidate as rejected by safety and
   continue. Only the last safe P5 candidate may be labeled final acceptance.
9. Source provenance must distinguish the fetch-selected body hash from the
   later normalized seven-field A0 digest. Do not present them as the same
   coordinate receipt.
10. Keep the existing 48,000-byte custom-premise limit. Expand only
    registry-owned RSS admission to a precisely tested envelope compatible with
    the 2 MiB fetch owner.

## Keep narrow

- No model-driven global fact selector.
- No subjective craft reviewer.
- No LLM ledger patcher.
- No retry around `_assemble_ledger`, cleanup, freeze, save, reopen, voice
  configuration, or final hash proof.
- No Pro-runner claim in the canonical chunk.
- No workflow JSON change.

## Required firing tests

- Candidate exhaustion twice then acceptance at P0 and P5, with distinct prompt
  nonces and unchanged upstream artifacts.
- Comfy interrupt propagates rather than becoming `CodexPassError`.
- Provider/config errors are invoked once and remain loud.
- P5 safety rejection changes journal disposition and causes a new full spoken
  candidate before ledger assembly.
- A 240-character quote across a nominal P0 cut is fully visible, rebases to the
  later duplicate occurrence, and survives final A0 validation.
- More than six nonempty windows select deterministic beginning/middle/tail
  facts; number references follow canonical fact renumbering.
- RSS above 48,000 bytes succeeds, custom premise above 48,000 bytes still
  fails, and the security-owned maximum is never silently sliced.
- Multiple RSS content alternatives, URL/RSS tie policy, failed scrape
  preservation, 12,000+ tail retention, and distinct body/A0 receipts.
