# A2 finished-code review judgment

Driver/judge Codex. This is one scoped finished-diff review following the
converged cross-machine R1-R4 design. CLI reviewer: Gemini 3.8 Flash (High),
through Antigravity. The driver anchor preceded dispatch. Reviewer source:
antigravity.md. No new architecture campaign or model/GPU qualification.

## Grounded decisions

1. Accept the constrained route's EOS collection defect. Comparing the final
   token to a list could falsely report output_limit for a completed reply at
   exact capacity. Normalize scalar/list/tuple/set EOS values and test all four
   with one remaining token and provider-capacity messages. Fixed.
2. Accept the need for a clear invalid-budget diagnostic; reject the proposed
   inference that unmarked max_new_tokens=None means full capacity. The actual
   writer/base/polish paths require the provider-capacity message contract.
   Inspection must describe that same request, not authorize a larger budget.
   inspect_native_prompt_fit in _otr_model_loader now raises TypeError before
   preparation for unmarked None. Tests prove no tokenization, device move or
   generation. Marked None remains covered across all four native routes.
3. Keep inspect_structured_fit.max_new_tokens required. An inspector must be
   told the actual planned output budget; a default could inspect a different
   request from the later call. Explicit marked None is supported.
4. Duplicate flag reads are optional cleanup, not a correctness defect. No
   extra refactor in this chunk. Accept the HF reuse-key tuple annotation fix.
5. Keep check_context_window as a documented compatibility no-op. It no longer
   rejects a model for being smaller than a project estimate. Unsupported GPTQ
   keeps its actual NotImplementedError.
6. Reject deletion of HARD_VRAM_CONTEXT_LIMIT as dead code: Sci-Fi's existing
   repair-turn heuristic still reads it. That char-estimate heuristic remains
   outside A2 and is explicitly disclosed in GO_FORWARD.

## Independent follow-up

Einstein's load/config audit found a real sparse-AutoConfig downgrade: later
missing metadata replaced known snapshot capacity with an estimate. The final
loader adopts AutoConfig/model config only when native capacity is present;
loader-boundary regressions cover sparse config and first-download resolution.

Euler independently checked both final CLI dispositions and the resulting EOS
and invalid-budget code/tests; no remaining finding. Dewey checked the optional
loader signature, direct-loader root setup and stale minimum-window fixtures;
those test contracts now match actual capacity and preserved old call shapes.
The full-suite zero-room fixture now uses zero remaining tokens; 42 remaining
tokens are usable under the operator's one-token default.

Validation and final file hashes are in
docs/2026-09-11-my-story-cross-machine/a2_receipt.md and a2_*.json. Baseline
failures are compared by identity and assertion evidence, not count alone.
The CLI's system-Python observations do not replace the real Windows venv runs.
No dollar-cost receipt was returned by the local CLI; no spend is invented.
