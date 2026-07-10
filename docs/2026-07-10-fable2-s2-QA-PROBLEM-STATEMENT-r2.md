# QA PROBLEM STATEMENT ROUND 2 -- audit the fixes, pre-audit S2

**Date:** 2026-07-10 (afternoon). **Repo:** ComfyUI-OldTimeRadio, branch `v2.0-alpha`,
HEAD `d67f0b13`. **Audience:** the same external analyst as round 1. Read the REAL
Windows files under `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`.

**FROM THE OPERATOR: your round-1 analysis
(`docs/2026-07-10-fable2-s1b-QA-ANALYSIS.md`) was folded the same day and it worked --
both fable2 lanes are now GREEN ("Einstein's Echo" default lane 570s; "The Butterfly's
Gambit" LTX lane ltx_audio_in + stills, 1787s, both in otr\obs). Round 2: (A) AUDIT THE
FIXES WE MADE from your findings -- hold them to the same standard you held our code;
(B) DESIGN-REVIEW the S2 work BEFORE we write it; (C) define what "production ready"
means measurably. Write your findings to a file. Cite verified file:line; never present
an unverified claim as fact.**

## A. Audit the round-1 fix implementations (commit `af378aad`)

1. **Ownership-aware disk merge** -- `nodes/production_ledger.py`,
   `_MERGE_OWNED_ROW_FIELDS` inside `_merge_with_disk` (10 fields: text, char_count,
   word_count, skip, tts_skip_reason, reviewer_skip_reason, speaker_role, char_id,
   boundary, compose_flags).
   - Is the denylist COMPLETE? Sweep every field the Ledger class writes/manages on
     lines/clips/music rows (traits, beat_intent, arc_phase, target_words,
     dialogue_slot_id, reviewer_note, needs_render_realign, line_id/beat_id/shot_id,
     start_s/dur_s, bark_wav_path...) and classify each as OWNED (never restore) vs
     DURABLE out-of-band (must keep restoring -- BUG-LOCAL-108 family). Flag any
     misclassification with the failure it would cause.
   - Did ANY existing code path RELY on the old falsy-restore behavior (intentionally
     clearing a field expecting disk to resurrect it)? The change touches every save on
     every lane; clips[] and music[] ride the same loop.
   - Concurrency/rebind: save() rebinds self.data to the merged dict; confirm no caller
     between the reviewer's save and Phase 10 holds a detached reference the fix's
     semantics change (the BUG-LOCAL-273 refresh pattern).
2. **Doctor skip contract** -- `nodes/_otr_ledger_reviewer.py` skip branch now stamps
   `tts_skip_reason = "reviewer_skip: <payload>"`. Verify every consumer of
   tts_skip_reason accepts that vocabulary (freeze gap audit, TTS skip paths, soak
   telemetry), and whether the "retire doctor skip entirely" recommendation should
   still happen (the reviewer's own guidance text discourages mutes).
3. **5B/5C lane capability gate** -- `nodes/_otr_freeze_cascade.py`
   `_legacy_line_compose_applicable` (pack-seam check, fail-OPEN on resolution error)
   + `StoryCriticReport.clean()` substitution at the 5B entry.
   - You recommended gating ALL legacy text-mutating passes. We gated ONLY 5B/5C: the
     SCRIPT DOCTOR still ran on the green LTX roll (verdict frozen_with_doctor_edits) --
     so doctor rewrites still mutate fable2 text after its proof map is sealed TODAY.
     Give the concrete gate design for the doctor (call site
     `nodes/_otr_freeze_cascade.py` phase 1+2 -> `_OTRLR.review_ledger(generate_fn,
     led)`): parameter vs cascade-level skip, what deterministic structural validation
     must SURVIVE for fable2 (cast repairs? which?), and what verdict/receipt plumbing
     expects reviewer output.
   - Is fail-OPEN right for the gate's error path, or does it reopen the corrupt-row
     class for a lane whose pack momentarily fails to resolve?
   - Stage-7/A2: with clean() critic reports, verify NO path still enters the Stage-7
     escalation or A2 for an inapplicable lane under BOTH default and enabled
     escalation env settings (`_otr_story_select.py`, `_otr_reroll_escalation.py`).

## B. Design-review the S2 work BEFORE code

4. **text_for_tts route** (your round-1 recommendation for proof-preserving TTS
   pronunciation): nothing consumes that schema field today. Specify: which adapters
   must read it (bark batch, kokoro announcer, indextts2/chatterbox/dia sidecars --
   `nodes/_otr_tts_engine_sidecars*`, `nodes/scene_sequencer.py` announcer bus), where
   Phase 7's current normalizations live (`nodes/_otr_readiness.py:264-268`) and which
   of them (numbers->words, "Dr."->"Doctor") must move INTO text_for_tts generation,
   the migration posture for legacy lanes (byte-identical requirement for
   science_news), and the proof-map invariant test that locks canonical text.
5. **Inter-scene music wiring** (your round-1 P1): give the exact node/link/widget
   deltas for `workflows/otr_canonical.json` + `nodes/stable_audio_theme.py` +
   `nodes/scene_sequencer.py` + EpisodeAssembler to render/place every
   `ledger.music[]` cue at its authored boundary (cue-list manifest replacing the
   fixed 3-slot contract). Positional widgets law: `widgets_values` is POSITIONAL --
   only APPEND new optional widgets at the END (BUG-LOCAL-097). Flag every consumer of
   the current 3-output theme contract.
6. **S2 full-loop contracts** (P2a select, P4 critic, P5 revision, keep-better-draft
   judge) per `docs/2026-07-10-scifi-fable2-architecture.md` s13 S2 + your round-1
   section 4: turn your P5 contract sketch into acceptance-test-shaped requirements we
   can implement against (validate_revision_contract inputs/outputs; the deterministic
   whole-play _defect_score axes; the atomic artifact-update set on a winning
   revision: parsed/proof_map/draft hashes/P6 input/P8 input). Also: per-scene word
   allocation (integer allocation summing to target; band per scene), the P3 receipt
   stamping the ACTUAL max token budget used, and pitch-id/{1,2,3} + select-member
   validation in full mode.
7. **Caption/credits sentinel alias + HuMo stale guard** (your round-1 3.2/3.3): we
   have not fixed these yet. Confirm your file:line pins still hold at HEAD and give
   minimal patches + regression sketches so S2 can land them as a small chunk.

## C. Define "production ready" measurably

8. **Soak acceptance metrics for S3**: propose concrete thresholds from the code's
   retry budgets (e.g. green-roll rate over N random-RSS rolls at 30w and 350w;
   per-gate reroll/discard rates: dossier drops, read outer-retries, P3 ladder depth
   used, casting repair rate; freeze verdict distribution; wall-clock envelopes per
   lane). State what telemetry already exists in meta (pass_receipts, parse,
   audit.discarded, normalizations) vs what must be added to measure these.
9. **Content pre-screen of the two published episodes** (transcripts are in each
   episode's ledger under `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\`):
   flag with quotes -- example-name leakage recurrence, register distinctness between
   the two cast voices, coda/news-read seam quality, any SFW/era slips. Advisory for
   the operator eyeball, not gates.

## Constraints (unchanged from round 1)

No fallbacks; fail loud; science_news byte-identical; canonical workflow changes ship
WITH their code in the same change; UTF-8 no BOM; SFW; the ratified architecture doc
governs unless live evidence justifies a documented deviation. Deliverable: one
analysis file, findings ordered P0/P1/P2, each with file:line, the concrete fix, and a
regression-test sketch.
