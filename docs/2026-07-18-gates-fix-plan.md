# Short-episode structural-gate fix -- change under review

**Date:** 2026-07-18. **Branch:** v2.0-alpha. **Baseline:** HEAD `c507acff`.
**Reviewers:** kibitz local panel (Codex gpt-5.6-sol + Antigravity Gemini 3.5 Flash High).
Claude is anchor + judge; ground every claim against the real Windows files.

## Problem (live-proven)

Short canonical smokes fail in the WRITER on deterministic STRUCTURAL gates, on the
free local Mistral-Nemo default:

- `scifi_codex_v4` @120w: `RadioScoreDraftCompileError code=beat_count` -- "flattened
  draft beat count 6 must equal accepted advisory count 12" (`_otr_scifi_codex.py`
  compile). @30w a P5 validation retry-storm also OOMs.
- `scifi_fable2` @120w: `Fable2ScriptError` -- `WORD_BUDGET 87 outside 95-145`,
  `SCENE_COUNT 2 expected 1` (`_otr_scifi_fable2.py` `_run_markup_ladder`).

Root (codex): `_otr_scifi_codex.py` line ~3297 derived the beat count purely from cast
size -- `beat_ids = range(max(3, min(12, len(p2.cast) * 3)))` -- IGNORING the word
budget. 4 cast -> 12 beats demanded whether the episode is 30 or 720 words; P3 then
requires the draft to match that count EXACTLY. Root (fable2): `_script_budget_defects`
word/scene COUNT bands raise `Fable2ScriptError` on reroll exhaustion.

## Governing contract (this is enforcement, not a new policy)

`docs/SOURCE_BANK_PREFLIGHT.md` Gate 3 (all **Hard**):
- L159-162: "Models are never asked to calculate, report, or enforce exact word, line,
  item, or coverage counts... **No model-produced or unused count field can gate
  production.**"
- L166-167: "`target_words` is advisory and recorded. It does not trigger deterministic
  trim, padding, culling, rewriting, or **a fatal quota gate**."
- L157-158: output reservations "scale from the artifact's real size driver such as
  **line count**... not only `target_words`."
Gate 6 L275-282: a live **30-word** run and 30-word canonical smokes are a HARD gate.

Git confirms these gates are recent (regression): codex lane + cast*3 count = `c22eef0a`
(v4 bake-off, ~1wk old); exact beat gate = `c942b2ae`; fable2 budget gate = `95582643`
("Enable Fable2 S2 full loop for 120-900 words" -- short never supported). The old
science_news lane ran any length because it had none of these.

## THE LAW interplay

GO_FORWARD THE LAW: "an audit may improve a story, it may never fail one -- only
DETERMINISTIC validators may end an episode." These ARE deterministic, so THE LAW as
written permits them. Gate 3 is the stricter, controlling rule for COUNT/quota gates:
counts must never gate production. SFW/safety deterministic gates (G9) remain fatal.

## Change implemented (needs review)

1. **codex beat count scales to size driver** (`_otr_scifi_codex.py` ~3297):
   `_cast_n=max(1,len(p2.cast)); _words_cap=max(3, requested_words//15);
   _beat_n=max(_cast_n,3,min(12,_cast_n*3,_words_cap))`. Keeps >= cast (so
   `cast_coverage` stays satisfiable), floors at 3, caps at 12, scales down by words.
2. **codex compile reconciles a beat-count mismatch instead of raising**
   (`compile_radio_score_draft` ~866): if `flat_draft_beat_count != len(advisory_rows)`
   (and >=1), REBUILD `advisory = make_advisory_word_blueprint(advisory_total_center,
   [b000..b{N-1}])` to the draft's ACTUAL count, then compile positionally as before.
   Zero downstream hole claimed: the compiler indexes `advisory_rows[global_beat_number]`
   positionally and emits `beat_id=advisory_row.beat_id`; a rebuilt N-row advisory is
   valid for any N. A `<1` beat draft still raises (ledger needs >=1 beat).
3. **fable2 budget defects become advisory** (`_run_markup_ladder`): reroll only while
   `budget_rerolls < _MAX_BUDGET_REROLLS`; once spent, fall through to the normal accept
   path and record `advisory_budget_defects`, instead of `raise Fable2ScriptError`.
   PARSE defects (`parsed is None`, MISSING_END, BAD_LINE_SHAPE) STILL fail closed.

## Test impact (4 focused failures -- plan to update, not silence)

All 4 encode the OLD count-gate; rewriting to the new contract:
- `test_fable2_runner_ladders::test_budget_exhaustion_fails_loud` -> assert ACCEPT +
  `advisory_budget_defects` contains WORD_BUDGET (not raise).
- `test_scifi_codex_lane::test_p3_beat_count_error_reports_expected_count` -> assert a
  popped-beat draft now COMPILES and the score's beat count == the draft's actual count.
- `test_scifi_codex_lane::test_draft_compiler_rejects_unowned_or_invalid_runtime_decisions`
  -> drop the `beat_count`/`missing_beat` case; KEEP shot_index/unused_shot/cast_id/
  cast_coverage/fact_id/cue_id/cue_anchor cases (untouched integrity gates).
- `test_scifi_codex_lane::test_p3_base_and_repair_bind_locked_total_to_per_scene_cap` ->
  make the "wrong" first draft a PER-SCENE-CAP violation (6 beats in 1 scene, total still
  6) so the untouched per-scene-cap repair path still fires (2 calls) + keep prompt asserts.
Full suite (~8000) + Bug Bible not yet run. A KNOWN-FAIL-GUARD flags new failures.

## Questions for the panel (break the framing)

1. Is compiler-level reconcile the right SEAM, or should reconcile-on-exhaustion live in
   the P3 ladder (give the model one repair toward the intended count, then reconcile)?
2. Any REAL downstream consumer of the codex beat count that a silent reconcile breaks
   (receipts, shot/audio mapping, captions, credits, obs_publish) -- a ledger hole?
3. Should `unused_shot` + `cast_coverage` ALSO be relaxed for thin episodes (same Gate-3
   class), or are they legitimate ledger-integrity gates to keep fatal? They were not the
   proven blocker but will now be reached (beat_count no longer short-circuits).
4. Is `//15` words/beat a sound floor, or is there a principled size driver (line count)?
5. Does accepting a truncated/off-band draft harm TTS/video/captions, or is it safe +
   recorded per Gate 3? Any 30-word residual failure mode not covered here (fable2 PARSE)?
