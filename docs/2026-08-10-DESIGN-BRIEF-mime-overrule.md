# DESIGN BRIEF - mime overrules the audio dropdowns

Date: 2026-08-10. Status: **R1 COMPLETE - architecture decided.** See
`kibitz-runs/2026-08-10-mime-overrule/r1/` (codex.md + judgment.md). Headline
outcomes: a registered PRE-AUDIO owner node renders mime assets before any
TTS/music runs (ShotLock unlocks on audio_done and cannot be the planner);
stems substitute at line_id BEFORE SceneSequencer's timeline mutation (the
:1296-1320 list is themes-only - my error, corrected); rows stamp
`audio_owner="self_scored_video"` (never `skip`); the overrule binds at the
ROLE SLOT; new pre-audio `mime_target_s` duration authority; episode mime
beats HARD-CAPPED at 200 canvas frames / 8.000 s (f277 measured over the
VRAM gate); orthogonal `audio_authority="self_scored"` capability (no new
family, no new role); episode-wide boot-intersection check; render-all-mime-
before-audio failure protocol with atomic clip/stem receipt pairs; candidate
B cut from production (runner core survives as audition/replay). ONE open
operator question: what happens to a mime-cast row's authored dialogue text
- no caption, intertitle, or accessibility-only? r2-r4 run near this spec's
build, after the main transplant.
Follow-up spec to `2026-08-09-SPEC-lab-findings-into-otr.md` (which ships the
H3 adapters + the standalone mime runner). This brief is the kibitz r1 input
for the NEXT spec. The operator is explicitly OPEN TO IDEAS on mechanism.

## Fixed deliverable (operator-approved 2026-08-10): the cheap dropdown

Ships WITH this spec, no frontend JS ever:
- A static option added to the TTS and music model dropdowns - plain menu
  entries, positional-widgets_values-safe, self-documenting in any saved
  graph. LABEL COPY IS USER-GUIDANCE, not status (operator 2026-08-10:
  "make it user friendly"): the entry should tell the user what to do,
  e.g. `n/a - mime video provides this beat's audio`. The LABEL alone must
  carry the full meaning: tooltips are secondary-only (operator: ComfyUI
  tooltips are "a bit buggy") - nice when they render, never load-bearing.
  Exact strings get operator approval at build, same as public ids.
- Plan-time enforcement both directions: mime cast + audio dropdowns on
  anything -> overrule proceeds, stamped in the ledger; `n/a` picked with NO
  mime cast -> NAMED refusal before any render spend.
- Tooltips carry the explanation ("ignored when a mime beat is cast").
- Frontend reactivity (auto-greying) is explicitly OUT until ComfyUI exposes
  a stable dependent-widget API; the semantic lives in policy JSON so the
  future UI hop is cosmetic.

## The fixed part (operator rulings, not up for debate)

1. **UX:** the mime video engine is a first-class DROPDOWN choice. Selecting
   it for a beat means that beat renders NO TTS and NO music - the video's
   invented score IS the beat's audio. "The mime video dropdown overrules any
   TTS or music dropdowns. That's the easiest architecture."
2. Mime never lip-syncs; the body carries the sequence. Prompts describe the
   scene (score request derived from beat intent allowed); never constrain
   audio negatively.
3. Non-mime beats keep V-1 exactly as-is: real TTS/music, frozen master,
   byte-identity mux. The episode's delivered audio must remain THE frozen
   master end to end.
4. Quality gates stay human: any mime-in-episode output passes Jeffrey's ear
   before promotion.

## The hard problem (why this is not a line item)

OTR is audio-first: SceneSequencer/EpisodeAssembler build and SHA-freeze the
master WAV (scene_sequencer.py:1401-1422, `_stamp_master_audio_identity`)
BEFORE any video renders; every video engine is silent by contract and the
mux asserts decoded-PCM identity against the master
(otr_master_audio_mux.py:264-294). Beat durations derive from TTS sample
counts (`compute_clip_budget` -> `coverage_plan.partition_beat`, exact-sum).

A mime beat inverts this: its audio does not exist until its video renders.
Known legal insertion point for model audio: as a segment in the assembly
list BEFORE the freeze (scene_sequencer.py:1296-1320) - so the master simply
CONTAINS the mime window and V-1's identity assert still holds.

## Candidate architectures (r1 should rank these and may propose others)

A. **Phase inversion for mime beats:** planner detects mime-cast beats;
   their duration comes from script target seconds snapped to the H3 grid
   (129..377 canvas frames); their video renders in a NEW early phase; the
   decoded stem enters assembly pre-freeze as that beat's audio segment;
   the silent video clip then flows through the normal video phase (its own
   encode stays -an; the master carries its sound).
B. **Stem-first two-step:** the standalone runner (shipping this build)
   pre-renders mime clips as assets; the episode script REFERENCES a
   pre-rendered mime asset (clip + stem pair); assembly treats the stem as
   ordinary source audio (like music beds today) and the video side treats
   the clip as a pre-rendered segment. No new render phase inside the
   episode pipeline; determinism and receipts come free; cost = a manual or
   scripted pre-pass.
C. **Post-freeze substitution** - REJECTED up front: breaks the byte-identity
   assert; do not resurrect.

## Constraints the winner must satisfy

- NO FALLBACKS law: a failed mime render refuses loud at plan or render
  time; it never silently degrades to music/TTS.
- One-face policy, mouth policy, content-oracle motion checks: mime beats
  must not weaken any (family stays a shipped value; motion checks apply).
- The registry boundary from kibitz r3 MUST-FIX 1: whatever registers into
  the dropdown needs an enforced standalone/overrule semantic - the dropdown
  is the unfiltered registry and role_compat cannot exclude; name the exact
  enforcement point (ShotLock plan time + director validate).
- Ledger/receipts: the beat's audio provenance must be stamped (engine stem,
  duration, SHA) with the same rigor as TTS rows.
- Duration authority: mime beats use script-target seconds on the H3 grid;
  the composite's audio-wins cross-check must still hold for the whole
  episode.
- Boot: mime renders need the sage-free H3 lane; per lane-independence
  doctrine an episode containing mime beats runs on the H3-compatible boot.

## R1 questions

1. A vs B (or better): which is the smallest architecture that honors the
   fixed UX? Note B delivers the UX ("pick mime in the dropdown") only if
   the dropdown pick can trigger/consume the pre-pass - is that actually
   simpler than A once wired?
2. Where exactly does "overrule" bind - script parse, ShotLock, or
   SceneSequencer - so TTS/music generation is SKIPPED (not rendered and
   discarded) for mime beats?
3. What is the failure story (mime render fails at hour 2 of an episode)?
4. What does the mime ROLE look like - does it ride music_visual, or is
   this finally the expensive new-role path, and if so what is the minimum?
5. What of the runner survives as the episode implementation's core?
