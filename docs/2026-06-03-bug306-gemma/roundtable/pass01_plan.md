# BUG-306 -- hardened handling plan (post-roundtable, grounded)

## The decision that drives everything
What guarantee do we want?
- "gemma-4-12b can't be **picked from the UI**" -> removing the catalog row is enough.
- "gemma-4-12b can't **crash a run**" (stale/manual workflow pin, or cache pick) ->
  removal is NOT enough, because `validate_model_id` Path 2 (on-disk cache) and
  Path 3 (auto-download default ON) still admit it. We want this stronger one.

## Chosen fix (surgical; closes both the UI and the crash path)

### Step 1 -- remove the curated row
Delete the `CuratedModel(repo_id="google/gemma-4-12b-it", ...)` literal from
`CURATED_LLM_MODELS` in `nodes/_otr_model_catalog.py` (lines ~149-166). Leave a
one-line comment in its place recording the reason + date so a future re-add is
self-documenting:
```python
# google/gemma-4-12b-it removed 2026-06-03 (BUG-306): model_type `gemma4_unified`
# is not registered by transformers 5.5 -> unloadable. Re-add when transformers
# supports it. See docs/2026-06-03-bug306-gemma/.
```
PD3 check (REQUIRED before calling it done): grep every workflow JSON under
`workflows/` for `gemma-4-12b` -> must be zero (the canonical workflow pins
Mistral-Nemo). If any pins it, repoint to DEFAULT_LLM.

### Step 2 -- fail closed so it can't crash a run via Path 2/3
Add a known-unsupported guard so the id is rejected with a clear, actionable
error BEFORE the loader ever tries it (instead of 5 Selector retries ->
StyleGenerationFailedError -> episode abort). Smallest place: a deny check at the
top of `validate_model_id` (right after `_structural_reject`, before Path 1), or
inside `_structural_reject`:
```python
_UNSUPPORTED_ARCH_IDS = {"google/gemma-4-12b-it"}  # gemma4_unified vs transformers 5.5 (BUG-306)
if normalized in _UNSUPPORTED_ARCH_IDS:
    raise UnknownModelError(
        f"{normalized!r} declares model_type 'gemma4_unified', which the "
        f"installed transformers does not register -> unloadable. Use a "
        f"supported writer model (default {DEFAULT_LLM}). Re-enable when "
        f"transformers supports gemma4_unified. (BUG-306)"
    )
```
Optional stronger variant (more general, more code): read the model's
`config.json` `model_type` at validation and deny any `gemma4_unified` repo, not
just this one id. Recommend the explicit-id set for now (smallest correct change);
revisit if more gemma4_unified models appear.

### Step 3 -- tests + housekeeping
- Test: `validate_model_id("google/gemma-4-12b-it")` raises `UnknownModelError`
  EVEN with the model on disk and `OTR_MODEL_CATALOG_AUTO_DOWNLOAD=1` (proves
  Path 2 + Path 3 are closed).
- Test: `build_dropdown_choices()` no longer lists gemma-4-12b.
- Update/remove the session's gemma-4-12b catalog tests + `docs/model-license-
  google--gemma-4-12b-it.md` / `docs/model-license-audit-targets.txt` entry
  (keep the license doc as documentation if desired -- harmless, but drop it from
  any "active catalog" assertion).
- Run full `tests/` + Bug Bible. PD6: the guard lives in the catalog validator,
  NOT a node widget -> no new `model_id` widget, no workflow re-wire.

## Optional follow-up (separate, larger -- NOT in this fix)
**Writer load-failure fallback (Option B).** If ANY chosen writer model fails to
load, fall back to `DEFAULT_LLM` and continue instead of aborting. This fixes the
general "one bad model kills the episode" gap, not just gemma. It needs grounding
against `nodes/_otr_model_loader.py` (the Selector / `load_llm` caller) and the
StylePicker `StyleGenerationFailedError` path -- neither was in this roundtable's
grounding -- plus care to (a) surface the substitution in the ledger/log, (b) not
silently mask a real misconfig, (c) stay idempotent + VRAM-safe. Recommend as its
own small sprint after the surgical fix lands.

## Invariants guarded
PD3 (workflow-JSON grep before done) - PD6 (guard is in the catalog validator, no
node widget) - no-overhaul (delete one row + a deny set; no `available` field, no
loader refactor) - offline-first / Blackwell stack untouched (no transformers
change) - audio-king N/A (no audio path).

## Rejected (panel-agreed): upgrade transformers (stack risk), wait/pin
(non-actionable), sidecar (over-engineering for one row).
