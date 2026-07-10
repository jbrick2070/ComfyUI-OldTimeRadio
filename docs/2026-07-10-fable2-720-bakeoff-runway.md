# 720-word Story-Engine Bake-off -- the runway (2026-07-10 evening)

**Goal (operator directive):** a REAL bake-off of the story engines at 720 words --
who writes the best episode. Contenders: (A) `scifi_fable2`, (B) `original_radio`,
(C) `science_news`, (D) a **GPT-authored from-scratch sci-fi pack** running through
the fable2 pipeline (same code, different creative voice -- measures pack
authorship, not plumbing). Blind operator judging.

**Baseline:** HEAD `47bf50f2` (r2-QA P0 fold shipped: FreezePolicy readonly freeze
boundary, live-root veto, row-level merge ownership; suite 7463/31/1 + Bug Bible
green). Governing spec for every precursor = `docs/2026-07-10-fable2-s2-QA-ANALYSIS-r2.md`
(acceptance-shaped contracts + exact workflow deltas; claims verified at HEAD).

**Why no multi-act scoping is needed:** 720 words sits inside S2 full-loop mode
(120-900). Act-chunking is a >900w feature, deferred post-S3 by the ratified
architecture. Nothing here waits on it.

## The chunks (each: code + tests -> suite + Bug Bible -> commit AND push)

**C1 -- P1.1 ownership/revision merge contract** (small, foundation for C2/C3).
Adopt the r2 table: lines/music/clips owned-vs-durable classification; durable
renderer fields survive only with matching source identity (prompt/spec hash for
music cues, canonical-text hash for `text_for_tts`). Table-driven regression per
field: replacement, deliberate clearing, same-spec persistence, changed-spec
invalidation. Retire only explicitly superseded legacy merge tests; keep a legacy
read-compat migration test.

**C2 -- P1.3 text_for_tts delivery routing** (proof-preserving pronunciation).
Order inside the chunk: (1) capture the science_news byte-parity fixture FIRST
(serialize synthesis request + ordering + canonical text pre-change); (2) Phase 7
for fable2 writes ONLY `text_for_tts` + `text_for_tts_source_sha256` + a
normalization receipt -- never canonical text/counts/proof (numbers->words and
Dr.->Doctor move INTO delivery-field generation; re-enables the pronunciation we
deliberately switched off in the P0 fold); (3) voice common resolves
`delivery = line.get("text_for_tts") or canonical` -- for fable2, absent/empty/
hash-mismatch is TERMINAL, no fallback; (4) assert science_news exact parity vs
the fixture; (5) full-cascade spies: every synthesis input == text_for_tts,
canonical text/proof unchanged.

**C3 -- P1.4 authoritative cue manifest + canonical workflow wiring** (the only
chunk that touches `workflows/otr_canonical.json`; code + JSON in the SAME
commit). `ledger.music[]` becomes the authored queue (cue_id, anchor_line_id,
placement, generation_prompt, target_duration_s; `set_music()` must stop
discarding anchors). StableAudioTheme: fixed 3-slot outputs -> `cue_audio_clips`
+ `cue_manifest_json` + render_log + done; renders EVERY ledger prompt straight
to the episode audio dir. SceneSequencer inserts inter-scene cues at authored
boundaries; EpisodeAssembler shifts to master time. Exact graph delta (r2-spec'd):
delete links 241/242/243; APPEND inputs 7 (`music_cue_audio`) + 8
(`music_cue_manifest_json`) on node 3 SceneSequencer and node 7 EpisodeAssembler;
add links 280 `[83,0,3,7,AUDIO]`, 281 `[83,1,3,8,STRING]`, 282 `[83,0,7,7,AUDIO]`,
283 `[83,1,7,8,STRING]`; `last_link_id` 283. No widget change (crossfade, if ever,
appends at END -- BUG-LOCAL-097). science_news gets a legacy manifest reproducing
its present 3 slots exactly. Re-validate: OTR_WorkflowValidator + JSON round-trip
+ link/widget audit in the same commit.

**C4 -- P1.5 S2 full loop** (the big one; /kibitz the coding plan BEFORE writing
it -- this doc is that kibitz input). Contracts per r2: mode bounds (30-119
compact + P2a/P4/P5 stamped skipped; 120-900 full; 901+ rejected pre-creative);
exactly 3 pitch cards ids {1,2,3} + validated select; exact quotient/remainder
scene allocation (350 -> 117/117/116; 720 -> per-skeleton, sums exact, per-scene
band validated in P3 AND P5); P3 receipts record ACTUAL max token budget +
attempts; P4 critic + P5 revision consume normalized markup + all protected
artifacts; pure `validate_revision_contract(...)` (rejects title/cast/speaker-set/
scene-order/skeleton/coda-boundary/music changes, unnoted edits, budget
violations; malformed rules fail loud); deterministic lexicographic
keep-better judge (draft 2 wins only if eligible AND strictly lower); atomic
immutable FinalDraft propagated to P6/P7/P8/lines/proof -- never a mixed-draft
artifact. Acceptance matrix at 30/119/120/350/900/901 + fully mocked 350w loop
+ negative revision cases per protected field.

**C5 -- live proof rolls.** 350w fable2 smoke (default lane) -> fold failures
root-cause (kibitz every failure, standing rule) -> 720w verification roll,
default lane + one LTX lane roll. Freeze must be `frozen_clean` with
`content_mutations: 0` receipts; assets Test-Path'd in `otr\episodes\<ep>\` /
`otr\obs\`.

**C6 -- P2.2 caption/credits sentinel alias + HuMo stale guard** (small; makes
the rendered artifacts judgeable -- ANNOUNCER labels currently drop off sentinel
rows). Route captions/credits through the alias-aware cast resolver; HuMo stale
guard matches role/source-family/shot-lock predicate instead of a literal
char_id.

**C7 -- the bake-off itself.**
- One pinned news story for all four contenders; seeds pinned via
  OTR_CAST_SEED/OTR_STYLE_SEED for reproducibility (creative RNG is OS-entropy
  otherwise).
- Contender D: GPT (newest available via OpenRouter) authors a complete
  fable2-pipeline pack from scratch -- all 10 seams + bank row (new bank id,
  e.g. `scifi_gpt_pack`), subject to the same lint/registry gates; NO pipeline
  code changes allowed. Claude may fix JSON-shape errors only, never wording.
- 4 episodes x 720 words, rendered end to end; labels blinded (A/B/C/D shuffled)
  for the operator listen.
- Scorecard: story coherence, character distinctness, dialogue quality,
  ending earned, news-seam integration, would-listen-again. Operator judgment
  is final; no automated content gate (r2 P2.4).
- Known honesty flags going in: original_radio 420w run undershot (239/420);
  science_news is untuned at 720w. Report actuals, don't excuse them.

## Deliberately deferred (not on this runway)
P2.1 retire-doctor-skip (global reviewer change; post-bake-off), P2.3 full S3
soak cohorts (4x30 rolls -- after the bake-off proves the loop), >900w
act-chunking, cloud OpenRouter slot pins (separate ratify gate).

## Open questions for the kibitz panel
1. Chunk order: any hidden dependency that breaks C1->C4 sequencing (e.g. does
   C3's set_music anchor change need C1's music identity hash first)?
2. C4 blast radius: does the atomic-FinalDraft propagation require touching the
   P6/P7/P8 call signatures in `_otr_scifi_fable2.py`, and is a staged
   two-commit split safer?
3. 720w single verification roll vs 3 rolls before calling C5 green?
4. Contender D ground rules: is seam-count parity (10 seams) the right fairness
   bar, or should D also author its own banks.json fetcher config?
5. Anything in the r2 spec that contradicts the shipped P0 fold at `47bf50f2`
   (receipt shape, policy names) that the C2-C4 specs should be re-based on?
