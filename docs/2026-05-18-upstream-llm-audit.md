# Upstream LLM audit — Path C step 1

**Scope:** every LLM call site in the writer pipeline (story brief
→ style → cast → outline → per-line → reviewer/doctor).

**Verdict rule:** OVER-LOADED if any of:
- max_new_tokens > 300
- output schema has > 5 fields (counting list-of-objects as
  fields × list length)
- single call is solving more than one independent
  sub-problem

**Decision:** read-only. No code change in this commit. Flagged
sites become candidates for break-down after the outline refactor
proves the pattern.

---

## Summary table

| Site                       | File                          | max_new_tokens | max_attempts | Schema shape                              | Verdict          | Status from logs |
|----------------------------|-------------------------------|----------------|--------------|-------------------------------------------|------------------|------------------|
| story_brief reflection     | _otr_story_brief.py:55,508,618 | 160            | 3            | StoryBriefModel (~5 fields)               | FINE             | not exercised this sprint |
| style_picker inventor      | _otr_style_picker.py:95,470   | 80             | 3            | 5 snake_case slug strings                 | FINE (2-pass)    | GREEN retests #8-#11 |
| style_picker chooser       | _otr_style_picker.py:99,521   | 16             | 1            | 1 slug pick                               | FINE             | GREEN |
| news_interpreter           | news_interpreter.py:549,640   | 400            | 3            | key_terms list (~5-7 strings)             | FINE             | GREEN retests #8-#11 |
| casting                    | _otr_casting.py:298,386       | 250            | 3            | single cast row (4-5 fields)              | FINE             | GREEN retests #8-#11 |
| **outline**                | **_otr_outline.py:885,982**   | **1500**       | 3            | **Outline w/ N beats, each 6 fields**     | **OVER-LOADED**  | **RED retests #8-#11** |
| per_line composer          | OTR_LedgerScriptWriter.py:1012,1162 | 200 cap   | 3            | single line (3 fields)                    | FINE             | not reached |
| ledger reviewer audit      | _otr_ledger_reviewer.py:104,428 | **2000**     | (TBD)        | ReviewerEdit list                         | **OVER-LOADED**  | not reached |
| ledger reviewer doctor     | _otr_ledger_reviewer.py:106,823 | **3500**     | (TBD)        | repair patch                              | **OVER-LOADED**  | not reached |
| writer fanout slot probe   | OTR_LedgerScriptWriter.py:771 | 24             | 1            | warmup                                    | FINE             | (not generation) |

(per_line composer's cap is the scaling ceiling; attempt-1 scales
with target_words, capped at 200. Already broken down per beat
per Pattern 4b; structurally identical to the outline refactor
this commit will land.)

## Detail per site

### story_brief reflection -- FINE
Single small reflection step. 160-token output, 3 attempts.
Schema is the StoryBriefModel (~5 fields). Not exercised in
§3.7 retests because the pipeline failed earlier. Memory entry:
"Story brief is GREEN per Sprint complete 2026-05-10."

### style_picker -- FINE (already 2-pass)
- Inventor: 80 tokens, 3 attempts at temperatures (0.6, 0.7, 0.7).
  Output is 5 snake_case slug strings.
- Chooser: 16 tokens, 1 attempt at temp 0.10. Picks one.
- Combined: ~96 tokens of LLM output, lowest of any site.
- 100% pass rate in retests #8-#11.

### news_interpreter -- FINE
- 400 tokens, 3 attempts.
- Output: small `key_terms` list (~5-7 strings).
- Memory entry: "news_interpreter sprint COMPLETE 2026-05-10
  (6f3218d/70d25eb/f518fb3/9f82685/4f45c7c)".
- Pass rate retests #8-#11: 100% (occasionally needs 2 attempts;
  always succeeds within 3).

### casting -- FINE
- 250 tokens, 3 attempts.
- Output: single cast row per call (4-5 fields:
  name + voice + gender + maybe namesake).
- Called once per character (so num_characters total invocations),
  each independent.
- Memory entry: "Cast Contract Phase 0+ ... 106/106 cast suite
  passing".
- Pass rate retests #8-#11: 100% (typically 1 attempt).

### outline -- OVER-LOADED ★ THIS COMMIT'S TARGET
- **1500 tokens max_new_tokens**, 3 attempts (2 fresh + 1
  repair at temp 0.30).
- Output schema: full `Outline` pydantic model with `beats[]`
  list (each beat has `beat_id`, `arc_phase`, `intent`, `mood`,
  `target_words`, `speaker`, plus pattern constraints on
  beat_id and >= 3 on target_words).
- For act_count=3 + include_act_breaks=False: 14 voiced beats +
  2 announcer = 16 beats. Each beat has 6 fields. Total schema
  surface ≈ 96 leaf fields.
- Plus per-phase aggregate constraints:
  - per_phase_words sums must land in `[allowed_lo, allowed_hi]`
  - per_phase_beats counts must land in `[allowed_lo, allowed_hi]`
  - arc_phase tags must match phase order
- **Pass rate retests #8-#11: 0%.** Both Mistral-Nemo and
  Gemma-4-E4B-it fail this call after 3 attempts on every iter.
  Failure flavors:
  - Mistral over-produces (4 beats vs 5-8; m-prefix IDs)
  - Gemma under-produces (60-65 words vs 67-101) then
    over-corrects on repair (290-320 words vs 67-101)
- **Diagnosis:** single mega-call asking the LLM to solve
  N independent sub-problems (logline + theme + per-phase
  allocation + N beats + N moods + N target_words + N intents)
  in one structured pass. The LLM can satisfy any individual
  constraint but cannot satisfy all simultaneously under
  pydantic strict validation.
- **Refactor:** see step 2 commit. Break into
  `1 + act_count + num_beats` smaller calls (~16 at smoke).
  Each call < 200 tokens output, < 5 schema fields. Independent
  retry per call. Python combines the results into the same
  `Outline` shape the canon extractor consumes.

### per_line composer -- FINE
- 200 tokens cap, 3 attempts.
- Output: single line of dialogue (3 fields: speaker + line + intent).
- Called per beat (so num_beats total invocations), each
  independent.
- Already broken down per Pattern 4b (the outline refactor mirrors
  this structure).
- Not exercised in §3.7 retests because outline blocks first.

### ledger reviewer audit -- OVER-LOADED (flag, not addressed)
- **2000 tokens max_new_tokens** at _otr_ledger_reviewer.py:104.
- Output: ReviewerEdit list (variable count, structured patches).
- Runs AFTER per-line composition. Not reached in §3.7 retests.
- **Flag:** when outline + per-line pipeline is unblocked, this
  becomes the next over-loaded site. Same break-down pattern
  likely applies (one reviewer pass per beat instead of one
  pass over all beats).
- **Recommendation:** defer until outline refactor proves the
  pattern and exercise reaches this stage. Then audit and
  break down with the same recipe.

### ledger reviewer doctor -- OVER-LOADED (flag, not addressed)
- **3500 tokens max_new_tokens** at _otr_ledger_reviewer.py:106.
- Output: repair patch (variable count, structured patches).
- Highest token cap in the pipeline. Runs AFTER audit produces
  a list of issues; doctor applies repairs.
- **Flag:** same recommendation as audit. Likely needs per-edit
  break-down. The 3500-token ceiling alone is a strong signal
  this call has the same kind of compliance pressure outline
  has.
- **Recommendation:** defer until pipeline reaches this stage
  in §3.7 retest #12+.

## What this audit recommends now

1. **THIS COMMIT (Path C step 1):** ship this audit doc. No
   code change.
2. **NEXT COMMIT (Path C step 2):** outline refactor. Break the
   single 1500-token call into `1 + act_count + num_beats`
   smaller calls. Each under 200 tokens output, under 5 schema
   fields. Combined deterministically in Python.
3. **FUTURE COMMITS (post-retest #12 GREEN):** ledger reviewer
   audit + doctor break-down. Same recipe. Defer until pipeline
   exercises these stages so the failure flavors are observable
   first.

## Update 2026-05-18 (Path F)

Retest #12 surfaced a separate defect class: MusicGenTheme
crashed on LLM-invented style slug `'station_supply_arrival_protocol'`
because its hardcoded `_STYLE_PALETTE` only carried the 10
canonical preset slugs. Style picker's "let the story decide"
invent design and the palette lookup were structurally
incompatible.

Path F refactor (commit landed in this audit's next commit):
**MusicGenTheme reads the meta brief, not the style slug.** The
new `_compose_music_prompt(meta, cue_id)` pulls
`story_brief_terms.atmosphere` (mood), `story_brief_terms.setting`
(scene), and `gen_params_initial.period_voice` (if present) and
combines them with cue-specific musical-character templates
(opening / closing / interstitial). The style slug is logged
diagnostically but no longer drives the prompt.

Retired surfaces (per no-legacy-back-compat):
- `musicgen_theme._STYLE_PALETTE` import (still exists at
  `_otr_style_palette.STYLE_PALETTE` for the freeze cascade's
  writer-slug drift validation)
- `musicgen_theme._resolve_cue_from_style`
- `musicgen_theme._apply_story_brief_mood_prefix`
- `tests/test_musicgen_news_brief_used.py` (deleted)
- `tests/test_musicgen_style_palette.py` (deleted)
- `tests/test_story_brief_musicgen_c5g.py` (deleted)

MusicGenTheme is NOT an LLM call (audio model, not text model)
so it doesn't appear in the upstream LLM audit table above. The
refactor is included here because it falls in the same Path
C/F family: each LLM (or LLM-like) consumer should pull
narrative signal from the meta brief, not from thin abstraction
layers that drift.

The audit table above for the writer pipeline is unchanged --
this update is downstream of the writer.

## What this audit does NOT recommend

- Do NOT break down `news_interpreter`, `casting`, `style_picker`,
  `story_brief`, or `per_line composer`. They are FINE.
- Do NOT touch the writer's slot probe / warmup calls
  (24 tokens). They are scaffolding, not generation.
- Do NOT change the writer's two-model selector. Both slots
  receive the same Gemma-4-E4B-it model id from the workflow's
  current widget values; the routing is correct.

---

## Constants reference

| Constant                       | File                     | Line | Value |
|--------------------------------|--------------------------|------|-------|
| _REFLECTION_MAX_NEW_TOKENS     | _otr_story_brief.py      | 55   | 160   |
| _INVENTOR_TEMPERATURES         | _otr_style_picker.py     | 92   | (0.6,0.7,0.7) |
| _INVENTOR_MAX_TOKENS           | _otr_style_picker.py     | 95   | 80    |
| _CHOOSER_MAX_TOKENS            | _otr_style_picker.py     | 99   | 16    |
| news_interpreter max_new_tokens| news_interpreter.py      | 549  | 400   |
| casting max_new_tokens         | _otr_casting.py          | 298  | 250   |
| **outline max_new_tokens**     | **_otr_outline.py**      | 885  | **1500**  |
| per_line max_new_tokens_cap    | OTR_LedgerScriptWriter.py| 1012 | 200   |
| _AUDIT_MAX_NEW_TOKENS          | _otr_ledger_reviewer.py  | 104  | 2000  |
| _DOCTOR_MAX_NEW_TOKENS         | _otr_ledger_reviewer.py  | 106  | 3500  |
