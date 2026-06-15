# OTR opener-still + image-model fix plan (roundtable pass00) — GROUNDED

> **Read this first, panel.** The PREVIOUS roundtable on this bug converged on a
> plan that was WRONG because it misread the code (it claimed "no music-mirror
> exists" and "synthesize a music_open line in EpisodeAssembler" — both refuted
> by live instrumentation, see below). This plan leads with VERBATIM runtime
> instrumentation + exact code locations. Critique THESE; if you propose a change,
> name the exact file/function and say what real input proves it.

## Context — grounded via live instrumentation (HEAD `8a517b1`, render `…171037`)

Three remaining defects in the video/image pipeline after BUG-400/401/402 shipped.
A live 30-word smoke with `[BUG-403/404 instr]` logging produced (verbatim):

```
build_clip_manifest opener: bid=b000_music_open line_hit=False shot.start_s=0.0 -> row.start_s=0.0 tfc=238 exists=True
plan_timeline_segments: positioned=True ... seg order=0 source=clip shot=shot_b000_music_open n=238
_resolve_title_timing: music_open_line=None first_dialogue_f=None resolved_open=False -> {music_open_start_f:0, music_open_end_f:25}
```

Operator look-QA of the published 17:10 obs final, opener frame (t≈0:08, the
`[0,9.5s)` head-gap): **title header + green CRT scopes PRESENT, but the centre
image (the opener still) is BLACK.** Dialogue beats DO show their centre stills.

**What is already CORRECT (do not "fix" these):**
- `otr_shot_lock.derive_opening_music_beat` synthesizes the `b000_music_open` beat
  (`_start_s=0.0`, `dur_s=first_line.start_s`). `build_execution_plan` (L721-728)
  stamps the shot `start_s=0.0`.
- `render_driver.build_clip_manifest` resolves the b000 row `start_s=0.0` (line
  lookup misses → falls back to `shot.start_s`). `plan_timeline_segments` runs
  **positioned** and places b000 at `[0,238)`. Placement is NOT the bug.
- The all-black opener seen earlier was a BUG-402 casualty (blend → source-copy);
  402 is fixed and the blend now succeeds (scopes + captions return).
- A music-mirror EXISTS (`scene_sequencer.py` BUG-130, ~L1347) and ShotLock makes
  the beat; do NOT add a second music_open line (it would also break
  `derive_opening_music_beat`, which reads `lines[0].start_s >= 2.0`).

## Invariants to guard (reject any fix that breaks one)
- Audio spine FROZEN: `test_audio_byte_identical` stays GREEN (visual/metadata only).
- Workflow source of truth = `workflows/otr_scifi_16gb_full.json`; wire any node/
  widget change in the SAME change (these fixes look code-only — confirm).
- Single resident heavy ≤14.5 GB; 100% local; UTF-8 no BOM; SFW.
- LOUD fallbacks: a missing model/asset must log+restamp, never silently degrade.
- `plan_timeline_segments` positioned/sequential contract: one untimed row must
  not silently flip the whole composite (latent `all()` trap — defensive only).

## FIX 1 — BUG-403-remainder: opener centre still is BLACK (real root)
**Grounded root:** `render_driver.build_clip_manifest` (L1400) sets each row's
still via `"init_image": _portrait_index(led).get(row_char, "")`. The opener beat
has `char_id=""`, and the minted opener still (`still_b000_music_open`, a NON-cast
image recorded under `ledger['images']`, NOT in the char-id-keyed
`_portrait_index`) → `init_image=""` → flux_still renders an empty centre. Dialogue
beats carry real char_ids (c01/c02/c03) that `_portrait_index` has, so they show.
**Proposed fix:** resolve the opener beat's `init_image` from the opener still
(`still_b000_music_open` in `ledger['images']`) when `char_id` is empty / the beat
is the synthetic opener — instead of the empty char-id key. Keep it LOUD if the
opener still is absent. (Verify how `build_request_from_shot` L660-691 consumes
`init_image` so the fix actually reaches flux_still's `asset_refs.init_image`.)

## FIX 2 — BUG-405: per-role IMAGE-model selection ignored
**Grounded symptom:** operator set (BEFORE the render) `announcer_image_model=
lumina_image`, `music_image_model=qwen_image`, `other_beats_image_model=hidream_i1`;
the render minted EVERY still with `flux_gen1` (`[OTR.image.flux_gen1] minted
portrait` ×6).
**Grounded (partial):** the dispatcher DOES read the per-role model
(`resolve_engine_for_role` L126, reading `image_policy["image_models"][slot]`) and
`assert_usable`s it (L406). On failure it appends a warning + `continue` — it
**SKIPS the object** (L407-409), it does NOT silently fall back to flux_gen1. So
the operator's "weights not downloaded → fallback" would instead produce a SKIP
(no still), which we did NOT see — 6 stills minted. => EITHER (a)
`image_policy["image_models"]` carried `flux_gen1` (the GUI pick never reached the
policy; the SAVED `otr_scifi_16gb_full.json` ImageDirector widgets are all
flux_gen1), OR (b) the per-object `gen_fn` handed to the dispatcher is a
flux_gen1-only mint (engine_id validated but generation hardcoded to one engine,
CW-era). **NEXT (verify-at-build):** capture the run's actual `image_policy_json`
(lumina/qwen/hidream vs flux_gen1?) and identify what `gen_fn` the dispatcher
receives + whether it is engine-specific. **FIX:** route generation to the
SELECTED engine's mint; if a weight is genuinely missing, fail LOUD naming it
(no silent flux_gen1). Panel: which fork is more likely, and where exactly does
the per-role selection get dropped?

## FIX 3 — BUG-404: title-card window collapses to ~1s
**Grounded root:** `video_engine._resolve_title_timing` gets `first_dialogue_f=
None` because the legacy `[Video]` render calls it on PRE-audio-timing lines
(no `start_s`); the role set `("announcer","character")` matches, so it's a
missing-`start_s` input, not a role bug. With no first-dialogue and no music_open
line it falls to the envelope fallback → `[0,25)` (~1s). The static header shows;
the proper `[0, first_dialogue)` window does not.
**Proposed fix:** feed `_resolve_title_timing` the audio-timed lines (post-
EpisodeAssembler shift) so the window spans the real head-gap `[0, first_dialogue)`.
Confirm WHERE the legacy `[Video]` render gets its `led` and whether it can read
the shifted ledger.

## Open questions for the panel
1. FIX 1: is resolving the opener `init_image` from `ledger['images']` the right
   seam, or should the opener still be added to `_portrait_index` under a stable
   key? Which avoids regressing dialogue-beat portraits?
2. FIX 2: most likely place the per-role image model is dropped (director vs
   policy-json vs dispatcher)? Any silent-fallback that should be LOUD?
3. FIX 3: is fixing the legacy `[Video]` title timing worth it given the new
   video platform, or should the card be drawn on the positioned composite floor?
4. Are these truly independent, or does one fix subsume another?
