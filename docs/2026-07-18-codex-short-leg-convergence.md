# codex short-leg convergence -- root fix (2026-07-18, kibitz r2)

Follow-on to `docs/2026-07-18-codex-p2p5-short-leg-fix.md`. After the P2
cast-name + P5 self-vocative fixes landed, LIVE 30w/120w codex_v4 Mistral-Nemo
legs still failed -- but on OTHER gates, a different one each draw. Root-caused
and hardened via a `kibitz` r2 pass (Codex gpt-5.5 + Antigravity "Gemini 3.5
Flash (High)"); both agents converged INDEPENDENTLY on the design below and every
claim was grounded against the files. (Full run + judgment:
`kibitz-runs/2026-07-18-codex-short-leg-convergence/r2/` -- gitignored scratch.)

## Axes (LIVE evidence)
- **30w** -> P5 `invalid accepted-order boundary` + `every locked cast row must
  own a voiced line`.
- **120w** -> `P3_rewrite` `scenes[].description` `string_too_long`.

None were the P2/P5 fixes; local Mistral-Nemo trips the codex structural/schema
gates stochastically and the 2-swing bounded repair does not converge.

## Fix (this chunk: Axis 1 + 2)
- **Axis 1 -- Python owns the script graph coordinates.** `beat_id`/`shot_id`/
  `boundary` are 100% derived from the accepted score
  (`_accepted_script_line_metadata`), so `_normalize_script_line_coordinates`
  rewrites them from the score in the `capture` wrapper BEFORE pydantic
  validation (the fields are non-nullable, so a post-validation fix is too
  late). The model can no longer fail P5 on a mechanical coordinate it should
  never author. `beat_id` moved out of `_SCRIPT_LINE_AUTHORED_FIELDS`.
- **Axis 2 -- P5 coverage reconciled to the score.** `compile_radio_score_draft`
  makes cast coverage ADVISORY (a reconciled short draft may leave a planned
  cast member beat-less; it carries no lines) but `validate_spoken_text_and_roster`
  FATALLY required every locked row voiced -- a systematic contradiction that
  failed every short leg whose draft left the announcer beat-less. Coverage is
  now measured against the score's scheduled speakers
  (`{beat.char_id for scene in score.scenes for beat in scene.beats}`), names the
  missing id, and a coverage rejection routes to a targeted LLM repair
  (`_SCRIPT_ARTIFACT_COVERAGE_REPAIR_RULES`). Fail-closed on exhaustion.

## OPERATOR FLAG
Axis 2 softens the P5 coverage gate from "every locked cast row" to "every
score-scheduled speaker" -- a gate the task said to keep fatal. It is the root
cause of the systematic short-leg failure and is consistent with the shipped
advisory-coverage policy (`ed7b37de`) and Gate 3 ("no count field gates
production"). The FORWARD dangling-reference gate (every line `char_id` is a
locked cast id) and graph closure stay fatal. Called out for the eyeball.

## Invariants kept FATAL
Line-ID set closure, forward speaker/`char_id` locking, `shot_index`/`cast_id`/
`fact_id`/`cue_id`/`unused_shot`/graph closure/G9 SFW, the self-vocative repair.
Axis 1 touches only score-derived coordinates -- never `text` or `char_id`.

## Proof
Full suite 8087 passed / 32 skipped / 1 xfailed; Bug Bible 17; AST + no-BOM.
Focused: coverage-vs-roster, coordinate normalization, coverage repair-rule
branch. LIVE 30w/120w legs pending.

## Still open (next chunk)
**Axis 3 -- `P3_rewrite` over-cap `description`.** The existing `_p3_text_patch`
(author-bounded shortening) only fires when the CURRENT repair-factory error is a
`ValidationError`; a `string_too_long` that emerges on the typed-repair RESPONSE
never gets a patch swing. Add a late-recovery patch on draft-pass exhaustion
(reuse `_derive_p3_text_patch_targets` + `_p3_text_patch_preflight` +
`_run_p3_text_patch`). No new clamp; prose-protection policy untouched.
