# Regression review target -- the 2026-08-14 story-cleanup tranche

**This is a BUG HUNT on shipped code, not a plan review.** Everything below is
already committed and pushed on `v2.0-alpha`. The full suite is green
(10422 passed / 110 skipped / 1 xfailed) and one live episode was generated and
read. Green tests and one good episode are exactly what this review must not
take as proof.

**What to hunt:** real defects, unowned ledger fields, invariants broken in
passing, and anything a green suite would not catch. Read the actual files.

## Commits under review

| commit | what it changed |
|---|---|
| `8da8f457` | `speaker` on ledger line rows (PBUG-20260814-01) |
| `fa001688` | P6 news coda as its own pass (PBUG-20260814-02) |
| `4d905037` | per-beat dialogue + scene review (PBUG-20260814-03) |
| `9ef9fead`, `665ef39a` | `scripts/otr_ledger_view.py` inspector + live window |

## 1. The speaker field -- `nodes/production_ledger.py`

`Ledger.set_lines()` rebuilds every line row as a literal dict that ENUMERATES
the keys it keeps, and `speaker` was absent while the sibling `set_beats()`
carried it. Every published ledger shipped `speaker: null` on every spoken row.

Changed:
- `set_lines()` now emits `"speaker": _safe_str(raw.get("speaker")) or None`.
- `init_lines_from_outline()` stamps the outline beat's speaker on the line row.
- `_otr_scifi_codex._assemble_ledger` takes `b.speaker` off the OWNING BEAT.
- `_otr_scifi_fable2._spoken_row` gained a required `speaker`; `_beat()` now
  READS it back off the line row instead of being handed it separately.
- `story_orchestrator._emit_partial_ledger` carries the streamed name.
- `tests/fixtures/fable2/golden_s1b_assembly.json` regenerated.

**Questions for the reviewer.** Is there any OTHER producer of ledger line rows
that still does not supply `speaker`? Does any consumer key on the field's
ABSENCE rather than its value? Does the fable2 change alter row ordering or the
proof map? Is `speaker` now carried through the freeze cascade, the scrubber,
and the reviewer without being dropped again downstream of `set_lines()`?

## 2. The news coda -- `nodes/_otr_scifi_codex.py`

The coda was prompt text concatenated onto the P3 and P5 system messages. No
output type, nothing reserving a final row, nothing verifying afterwards. The
published episode had one announcer row, in the middle.

Now `P6` is its own pass, run AFTER P5, reusing the existing
`codex_coda_contract_system` seam. New: `NewsCodaV4`,
`_news_coda_source_anchors`, `_names_a_source_anchor`, `_news_coda_findings`,
`_call_news_coda`, `_assert_news_coda_is_last`, and reserved ids
`_NEWS_CODA_LINE_ID = "l999"` / `_NEWS_CODA_BEAT_ID = "b999"`.

Behaviour: the ladder's `post_validator` is deliberately empty and
`retry_until_valid=False`; the detector runs OUTSIDE the ladder so a firing
verifier triggers ONE bounded CLEAN (the draft returns with `previous_attempt`
and `unmet_requirements`), never a refusal and never a reroll. Three outcomes,
all of which continue the render: `clean`; `unclean` (ships, flagged in
`meta.scifi_codex.news_coda.status`); `absent` (nothing appended, nothing
invented). Budget `_NEWS_CODA_MAX_OUTPUT_TOKENS = 384`.

`_assemble_ledger` gained `news_coda: str | None` and Python-appends the row
last, with its own beat on the LAST scene's LAST shot.

**Questions for the reviewer.** Can `l999`/`b999` collide with a compiled id
under any act count? The coda row is NOT in the P5 script artifact but IS a
voiced ledger row -- does that break `_otr_content_authorship.build_receipt`,
`require_voice_coverage`, `_CodexTailFinalizer._proof`, the freeze gap audit,
per-beat audio slicing in `scene_sequencer`, still-image keying, or the video
tail window? Music cues carry `anchor_line_id`; the coda is appended after the
score's lines -- can the closing cue now anchor BEFORE the last spoken row, and
does anything downstream care? Does `stamp_word_counts` / `_OTRWD.stamp_actual`
count the coda correctly? What happens on a two-cast episode where the announcer
owns no score beat?

## 3. Per-beat dialogue -- `nodes/_otr_scifi_codex.py`

`_call_script_text_draft` kept its name and place but stopped being a single
call. Per scene: one `P5B` dialogue job per accepted beat, then one `P5R` scene
review. New `BeatTextDraftV4` (array ceiling = the BEAT's, not the script's),
`SceneReviewDraftV4`, `_closed_rows_findings`, `_beat_dialogue_inputs`,
`_call_beat_dialogue`, `_call_scene_review`. Budgets
`_BEAT_TEXT_MAX_OUTPUT_TOKENS` and `_SCENE_REVIEW_MAX_OUTPUT_TOKENS` replace the
whole-provider-window reservation; `max_new_tokens` was REMOVED from
`_call_script_text_draft`'s signature.

The accepted rows are assembled into the same `ScriptTextDraftV4` the whole-play
pass used to return, then compiled/canonicalized/validated by unchanged code.
The canonical-surface hygiene check MOVED into the per-beat validator, because
per beat the canonicalization happens too late to reroll.

`_script_artifact_inputs` was trimmed: it no longer builds `accepted_line_graph`,
`accepted_line_ids`, `accepted_line_count`, `music_cues` or the word steer, and
`_script_artifact_context` was deleted. `ActSteerV4` is still used elsewhere.

**Questions for the reviewer.** Is `rows_so_far` unbounded -- can a 12-beat
episode's prompt outgrow the context window on the last beat or the last scene
review, and what happens when `prompt_must_fit` cannot reserve the budget? Are
the per-job budgets big enough for a legitimately long beat, given a schema
ceiling of 12000 chars per line? Does removing the whole-play view cost the
writer anything the per-beat window does not restore? If a scene review returns
a row UNCHANGED but the beat writer had already been rerolled, is the accepted
text still the one that was validated? Does the `P5B`/`P5R` id show up anywhere
that pattern-matches pass ids (journals, receipts, soak parsers, dashboards)?
Is anything still reading `journal["script_transport"]["output_budget_mode"]`
and expecting `provider_capacity`?

## 4. The inspector -- `scripts/otr_ledger_view.py`

New, ~700 lines. Reads a ledger and grades F1 (action in a spoken row) and F2
(speaker mismatch) and nothing else. `--watch` is a live window; `--ladder`
censuses retry ladders from the call journal; `--html` writes a page.

It is READ-ONLY by contract and there is a test asserting it never writes to a
ledger.

**Questions for the reviewer.** Does it write anything anywhere it should not?
Are the F1 regexes still over-firing on ordinary dialogue (six false positives
out of six findings were already fixed once)? Does `--watch` handle a ledger
that is being rewritten underneath it, a truncated JSON mid-save, or a missing
episode tree? Does `ladder_census` mis-read any journal shape -- deterministic
repair, repair-owner-exhausted, multi-cycle?

## Standing invariants any fix must not break

- A ripped or changed pass may not leave a ledger field unowned. Enumerate the
  fields, give each exactly one owner.
- Code may DETECT and explain. Only a MODEL pass may rewrite prose.
- No word-count authority anywhere, in any form. `act_count` 1..8 is the only
  knob that shapes an episode; length is an observation.
- Runaway guards are code-side and stay. Right-size the job; never raise the
  guard.
- One prompt per job for every model tier. Vary the JOB SIZE, not the text.
- The sealed ledger holds announcer speech, character dialogue and music cues.
  Music rows are load-bearing for audio slicing, still keys and the video tail.
- `workflows/otr_canonical.json` is the source of truth; code not wired into it
  is dead.
