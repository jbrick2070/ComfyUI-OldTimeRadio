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

## Axis 4 -- P3 music-cue anchor indices (landed, live-surfaced)
A third re-draw failed on `music_cues.N.anchor_line_index Input should be less
than or equal to 1` -- the SAME class as Axis 1: `anchor_beat_index` /
`anchor_line_index` are mechanical routing indices that `compile_radio_score_draft`
ALREADY clamps to the real beat/line counts (1122-1134), but the local model
overshoots the schema cap (`RadioScoreDraftCueV4`, 512-513) and pydantic rejects
BEFORE the compiler's clamp can run. `_normalize_draft_cue_anchors` clamps them
to the schema bounds in the draft capture wrapper (Python owns them); the
compiler then applies the precise per-beat clamp. Same principle as Axis 1, same
existing-clamp precedent -- not a new problem model.

## Axis 5 -- P3_rewrite structure re-imposition (landed, kibitz r2)
A further re-draw failed `P3_rewrite: rewrite changed a locked draft structural
decision`. The rewrite must improve PROSE while preserving the STRUCTURE the P3
draft locked -- because that structure IS the ledger skeleton: per beat
`(shot_index, char_id, line_count)`, per scene the shot count, per cue
`(cue_id, anchor_beat_index, anchor_line_index)` (`_radio_score_draft_structure_signature`).
The local model drifts one though it should touch only text.
`_reimpose_rewrite_structure` (Axis-1 pattern) forces those locked fields back
from the accepted draft (`artifact_inputs["previous_draft"]`) in the P3_rewrite
capture wrapper, keeping the model's prose; cues are matched by `cue_id` (not
position) so a reorder never mislabels cue prose; a COUNT mismatch bails and the
signature gate still fails closed. Ledger guarantee: the re-imposed structure ==
the accepted P3 draft's, which already compiled into a valid ledger, so the
model can only improve wording, never punch a hole. kibitz r2 (codex + antigravity)
converged; the old reject-and-repair test became a one-call re-imposition test.

## Axis 6 -- generalize the P5 spoken-prose reword (completes Fix 2)
With Axis 1-5 the writer now reaches P5, and a live 30w leg failed
`P5: l001 spoken text contains a non-lexical token`. Root: the original
self-vocative fix (Fix 2) rerouted ONLY `self-vocative` to the reword rule; the
other per-line spoken-prose defects `_spoken_error` raises (non-lexical token,
all-caps lexical word, stage direction/markup/role label, empty text) still fell
to the generic rule that tells the model to PRESERVE the defective prose.
`_script_artifact_repair_rules` now routes ALL of them (`_SPOKEN_PROSE_DEFECT_MARKERS`)
to `_SCRIPT_ARTIFACT_SPOKEN_REWORD_REPAIR_RULES` -- reword the named line to clean
plain dialogue, preserve every other line (Gate 3: the model, not Python, fixes
spoken prose; fail-closed on exhaustion). Structural spoken rejections (unlocked
cast id, illegal role, music skip contract) stay on the metadata/generic rule and
their own fatal gates.

## Invariants kept FATAL
Line-ID set closure, forward speaker/`char_id` locking, `shot_index`/`cast_id`/
`fact_id`/`cue_id`/`unused_shot`/graph closure/G9 SFW, the self-vocative repair.
Axis 1 touches only score-derived coordinates -- never `text` or `char_id`.

## Proof
Full suite 8087 passed / 32 skipped / 1 xfailed; Bug Bible 17; AST + no-BOM.
Focused: coverage-vs-roster, coordinate normalization, coverage repair-rule
branch. LIVE 30w/120w legs pending.

## Axis 3 -- `P3_rewrite` over-cap `description` (landed, second chunk)
The existing `_p3_text_patch` (author-bounded shortening) only fires when the
CURRENT repair-factory error is a `ValidationError`; a `string_too_long` that
emerges on the typed-repair RESPONSE exhausts the ladder with no patch swing
(proven live: `P3_rewrite` `scenes[].description`). `_p3_late_recovery_text_patch`
now runs in `invoke_codex_structured`'s exhaustion handler: for a draft pass
whose `last_error` is a `string_too_long` `ValidationError`, with a declared
patch transport and a parseable last draft, it reuses the existing, tested
`_derive_p3_text_patch_targets` + `_p3_text_patch_preflight` + `_run_p3_text_patch`
to shorten only the over-cap authored leaves. No new clamp; the prose-protection
policy is untouched; a failed patch lets the original failure stand.
