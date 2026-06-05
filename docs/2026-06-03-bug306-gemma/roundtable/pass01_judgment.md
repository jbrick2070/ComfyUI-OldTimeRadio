# BUG-306 roundtable -- pass01 judgment (Claude is the judge)

Panel: 6 requested, **3 usable** (openai/gpt-5.5, google/gemini-3.1-pro, x-ai/grok-4.3).
3 FAILED empty (Opus, Sonnet, DeepSeek-v4-pro -- reasoning models that spent the
2000-token cap on hidden reasoning; `finish_reason=length`. Same class as
BUG-294/301: a reasoning model needs a higher output-token floor. A re-run at
`--max-tokens 8000 --models <the 3>` would recover them; not needed -- the 3
usable reviews converged.) Actual spend: ~$0.11.

## Grounded claims

CONFIRMED (verified in `nodes/_otr_model_catalog.py`):
- **Removing the curated row is NOT sufficient to make gemma-4-12b safe.**
  `validate_model_id` (L790-841) admits on 3 paths: Path 1 curated (L830),
  **Path 2 locally-scanned + on_disk (L834-837)**, **Path 3 arbitrary HF id when
  auto-download/allow-remote (L839-841)**. gemma-4-12b-it is on disk (22 GB
  cached) so Path 2 admits it; `OTR_MODEL_CATALOG_AUTO_DOWNLOAD` **defaults "1"**
  (L815-817) so Path 3 admits it too. A stale/manual workflow pinning the id, or
  a cache pick, still reaches the loader -> crash -> episode abort. [GPT nailed
  this; Gemini started the same trace but was truncated; Grok focused on the
  workflow-pin sub-case.]
- **The row's `vram_fit_tier="PASS"` misreports an unloadable model** ->
  `check_vram_fit` would say PASS. [Grok MUST-FIX #5, GPT MUST-FIX #2.]
- **The doc was a question-list, not a buildable plan.** [Grok VERDICT "no" --
  fair; the synthesized pass01_plan.md now states a single chosen plan + PD3/PD6
  checks.]

ACCEPTED:
- Smallest correct change for the catalog side = **delete the row** (a one-line
  literal), NOT add an `available`/`unavailable_reason` field (Grok MUST-FIX #3,
  CUT #4 -- adding availability semantics is an overhaul for one row). BUT,
  grounded caveat above: deletion alone is insufficient, so the plan ADDS a
  fail-closed block in validation.

REJECTED (panel agreed, consistent with the stated constraints):
- Option C upgrade/patch transformers -- disallowed by constraint 1 (protect the
  cu130 Blackwell stack; IndexTTS2/Chatterbox venv-brick precedent). [Grok CUT #1]
- Option D wait/pin -- non-actionable, no code change now. [Grok CUT #2]
- Option E sidecar IPC -- over-engineering for one writer row. [Grok CUT #3]

UNVERIFIABLE FROM THIS GROUNDING (verify-at-build):
- Option B (writer load-failure fallback) -- needs `_otr_model_loader` (the
  Selector/`load_llm` caller) + the StylePicker `StyleGenerationFailedError`
  raise path; only `_otr_model_catalog.py` was grounded this pass. [Grok MUST-FIX
  #4 -- correct to flag.] Treated as a separate, larger follow-up.
- CURATED_CONTEXT_OVERRIDES omits gemma-4-12b (Grok SHOULD-FIX #2) -- moot once
  the row is removed.

## Convergence
One pass produced a clear, grounded, actionable plan. No new material item is
likely from a second paid pass on a decision this bounded -> CONVERGED. Stop.
