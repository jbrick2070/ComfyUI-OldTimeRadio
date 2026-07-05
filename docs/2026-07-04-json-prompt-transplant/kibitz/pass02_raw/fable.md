# r2 Review -- Fable panelist (coding plan / implementability)

## VERDICT
**NOT code-ready -- GO-WITH-FIXES.** The r1 synthesis is sound, but per-chunk diffs don't exist yet, and two MF resolutions collapse into each other in a way the plan hasn't noticed: under MF-C6, the two MF-C1 identity sites need **zero Phase A production edits** -- defer is free, not a compromise.

## MUST-FIX

**MF-R2-1 (resolves MF-C1): pick DEFER -- it falls out of MF-C6.** If the science pack empty-overrides `outline_system` and `line_composer_system`, production keeps its literals and no loader touches `_otr_outline.py:532` or `_otr_line_composer.py:1174` in Phase A. Non-science banks that would consume these seams only go live in Phase B. Extracting them now buys nothing and risks the `:1847` identity check plus the router's import-time bindings (`_otr_creative_prompt_router.py:43-47,61-64`) and line_composer's direct fallback binding (`:2060-2061`). If extraction proceeds anyway, module-level single-assignment rebind is identity-safe (router imports at runtime, so chunk order is soft), but ship the identity pytest: `resolve_creative_system_prompt(default, phase) is module._SYSTEM_PROMPT`.

**MF-R2-2: `line_grounding` breaks the empty-override pattern twice.** (a) `profiles.py:60-65` hard-errors on empty `line_grounding` -- an all-empty science pack cannot pass `resolve_profile()`, contradicting MF-C6 as stated. (b) The production rider is **conditional two-variant Python with an f-string** (`_otr_line_composer.py:1621-1636`, `{req.conflict_object}`) -- neither empty-override nor literal-move works. Fix: per-bank required-seam declaration in `banks.json` (science declares none required) + defer `line_grounding` extraction to Phase B. Note `science_news_default.json:11` currently carries the KILL-1-superseded paraphrase -- must be dropped per MF-C6.

**MF-R2-3: MF-C4 extractor must not bypass `resolve_profile()`.** `get_pack_prompt_or_none()` reading pack JSON directly recreates the parallel-helper anti-pattern MF-C4 forbids; going through `resolve_profile()` trips MF-R2-2(a). The chunk plan must sequence the profiles.py required-seam relaxation **before** the extractor chunk.

**MF-R2-4: `production_mirror/` does not exist at `7df7c80`** (glob empty; `compat.py:20,66,72,108` cites it). MF-C5's "accept drift" option is unavailable -- there is nothing to drift from. Mirror creation at `a7bdc42d` must be chunk 1.

## SHOULD-FIX

- **SF-R2-1:** The 4 new seams map cleanly: `_MACRO/_PHASE/_BEAT_SYSTEM_PROMPT` at `_otr_outline.py:1102/1115/1130`, consumed at `:1868/:1996/:2101` with **no identity checks** -- extraction-safe, unlike the legacy pair. State this asymmetry in the seam table.
- **SF-R2-2:** Document that seam `outline_system` maps to the **legacy** `:532` prompt (per `:1045` comment, superseded by the staged prompts) -- the table should mark it legacy/identity-bearing.
- **SF-R2-3:** Appending 4 seams makes `TEMPLATE_SEAMS` 18 entries, not "14" -- fix the count language (MF-C8 hygiene).

## Grounding table

| Claim | Source | Status |
|---|---|---|
| Identity check `resolved is _SYSTEM_PROMPT` | `_otr_outline.py:1847` | CONFIRMED |
| Router import-time bindings | `_otr_creative_prompt_router.py:43-47,61-64` | CONFIRMED |
| line_composer fallback `system = _SYSTEM_PROMPT` | `_otr_line_composer.py:2060-2061` | CONFIRMED |
| Macro/phase/beat prompts, no identity check | `_otr_outline.py:1102,1115,1130,1868,1996,2101` | CONFIRMED |
| `line_grounding` required non-empty | `profiles.py:60-65` | CONFIRMED |
| Rider is conditional f-string | `_otr_line_composer.py:1621-1636` | CONFIRMED |
| Science pack paraphrase present | `science_news_default.json:8,11` | CONFIRMED |
| `TEMPLATE_SEAMS` = 14 entries | `contracts.py:25-42` | CONFIRMED |
| `production_mirror/` absent at 7df7c80 | glob `*mirror*/**` empty; `compat.py:108` | CONFIRMED |
| Empty-override test | `test_transplant_modules.py:70-77` | CONFIRMED |
