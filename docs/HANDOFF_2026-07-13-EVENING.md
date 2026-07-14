# Handoff -- 2026-07-13 evening

**Branch:** `v2.0-alpha`. **HEAD == origin == `9468028c`.** Tree clean.
**Suite:** 7885 passed / 31 skipped / 1 xfailed. **Bug Bible:** 17/16/3.
12 commits today, 26 files, +2171 / -2552.

## THE LAW (yours, and it now holds everywhere)

> AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE.

Every LLM veto in the codebase is gone. What each one claimed to protect is now a
deterministic contract with a repair route at a rung that can act on it.

## GREEN LIVE LEGS (RESULT SUCCESS + obs_publish OK + asset on disk)

| Bank | Words | Prompt | Asset |
|---|---:|---|---|
| `original_codex56sol` | 30 | `fb34bf4f` | 54.5 MB -- *The Echoing Library* |
| `original_codex56sol` | 120 | `9874b749` | 84.3 MB -- *The Rhythm's Recollection* |
| `original_codex56sol` | 420 | `b9c49e0d` | 58.6 MB -- *The Timetable Trail* |
| `scifi_gemini` | 30 | `12f7ecde` | 46.0 MB -- *Temporal Tapestry* |

The 420-word leg cleared **every pass on its first attempt**.

## WHAT WAS RIPPED

* `original_codex56sol`: P4 (fair play), P7/P8 (blind listener + retake), P9 (final
  contract audit), and the P5 score-intent anchor patch. **9 LLM passes -> 4** plus
  one per-line script patch. Fair play is now a DETERMINISTIC CONTRACT -- the device
  anchor is SPOKEN on a clue line before the reveal line -- repaired by the bounded
  line patch that already existed.
* `scifi_gemini`: `sfw_pass`, the P5-recheck, `SciFiGeminiRewriteExhaustedError`. The
  spoken-text check moved INTO the P4/P6 ladder with a cast-name/source-acronym
  exemption. (The outline seam ORDERS all-caps cast names, and the validator killed
  any all-caps token -- outside every ladder.)
* `scifi_sonnet`: the `severity`/`invented_fact_flags`/`sfw_pass` veto and
  `SonnetAuditExhaustedError`. Replaced by `ungrounded_lines`: a factual line must
  cite a real dossier fact and may speak only numbers the source states -- a PROOF,
  which now feeds the existing script doctor instead of raising at the end.
* `scifi_fable2`: the P8 LLM ledger audit, which raised AFTER `_assemble` and two
  `led.save()`s -- a complete, persisted ledger thrown away on a model opinion. Its
  lexicons were the PERIOD lane's, so ordinary sci-fi words ("machine", "computer")
  corroborated hallucinated flags. Now `_assert_no_weapons_or_smoking`.
* `original_radio`: `corroborate_hard_finding` was a RAW SUBSTRING scan over
  `finding.detail + script`, so "gun" fired on "begun" AND a judge that wrote "the
  scene mentions a gun" corroborated ITSELF against a clean script. Now word-boundary,
  script-only.

## WHAT WAS ADDED

**G9** in `_otr_ledger_freeze.run_gap_audit` -- the first working SFW ship-stop. A
word-boundary `DEFAULT_PROFANITY_TERMS` scan over spoken ledger text, on the one path
every lane crosses, raising at Phase 10. Before this, `_otr_ledger_scrub` ran only on
`run_story_spine=True` lanes and NOTHING read its verdict; codex and fable2 had no
profanity check at all. Ripping the LLM `sfw_pass` gates without this would have
widened a hole that was already open.

## THE BUG CLASS THAT COST TWELVE LIVE ROLLS -- "the lost anchor"

A pass hands an LLM an IMMUTABLE string Python already owns -- a constraint-draw
field, a dealt card, a locked speaker, a coordinate from an accepted artifact -- and
asks for it back verbatim. The model paraphrases. Python compares exactly and kills
the episode over a copy of its own input.

**RESTORING AN INPUT IS NOT AUTHORING.** Restore it when the correction is FORCED
(exactly one value possible); return it to the model when it is not.
`_restore_slate_immutables`, `_restore_thread_lost_objects`,
`_project_announcer_char_id`, `_project_arc_phases`, `_drop_unknown_clue_ids`,
`_restore_anchor_in_text`. A Sonnet fan-out found the same class in fable2 (the dealt
card, the locked cast names) and scifi_codex (`beat_id` refused while its two tuple
siblings were restored from the same accepted-score tuple).

Three more laws, each paid for with a live roll:

1. **A repair prompt that does not fit is worse than no repair.** P5's repair was
   5,772 tokens into a 4,592-token window; PROMPT_GUARD silently truncated the
   contract and the model answered from the fragment it could still see. Bound the
   repair context (`_repair_inputs`).
2. **A bounded repair must ask for the unit the model can deliver.** The script patch
   asked for every planned line in one call, so a model that fixed two of three
   failed the whole batch -- and the two good lines went in the bin with the missing
   one. One line, one call.
3. **"It is broken" is not a repair prompt.** Name the missing object, the unassigned
   clue, the exact string, the beats that can carry it. Three validators returned
   nothing but "wrong" and the ladder burned its rungs guessing.

PBUG-20260713-15..18 logged and closed by `fb34bf4f`. -14 marked SUPERSEDED.

## THE ONE OPEN BLOCKER

`scifi_sonnet` now dies at **P0**, before any of today's code runs:

```
P0 failed: prompt requires 5424 input tokens but only 5192 fit
(context_cap=8192, max_new_tokens=3000)
```

This is `prompt_must_fit=True` doing its job -- the lane refuses to silently truncate
-- and it is the **pre-existing context/cap item** (GO_FORWARD item 7), not a
regression. The RSS article that day was simply long. All of today's sonnet fixes are
in and unit-proven; the lane cannot be live-proven until P0 fits.

**This is the next thing to fix**, and it is the same lesson as PBUG-15 seen from the
other side: the base prompt, not just the repair prompt, must be measured against
`context_cap - max_new_tokens`.

## STILL TO LIVE-PROVE

`scifi_fable2`, `scifi_codex`, `original_radio`, and the four legacy banks
(`science_news`, `media_archive`, `public_domain_story`, `shakespeare`) -- 30w, then
120w. Everything is committed, tested, and pushed; these are GPU-time, not coder-time.

## HOW TO RUN A LEG

```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 `
  -Profile none -Words 30 -Set OTR_LedgerScriptWriter.source_bank=<bank>
```

It resets selectively (positive OTR-server ownership -- it will not touch your GUI),
picks a free port, loads the real `workflows/otr_canonical.json`, and prints
`RESULT SUCCESS` plus the `obs_publish OK` path. Always `Test-Path` the asset.
