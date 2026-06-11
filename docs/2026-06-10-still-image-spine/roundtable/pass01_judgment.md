# Pass 01 judgment -- still-image spine (3 API panelists + Fable as independent panelist, ~$0.03)

Verdicts: gpt-5.5 no, gemini-3.1-pro no, deepseek-v4-pro no, fable
yes-with-fixes. All four found the SAME structural gaps (the statement was a
problem statement, not a plan -- "no" expected). OPERATOR NORTH STAR applied to
every judgment: the 6/5 pipeline (otr_video_plan.py, deleted at e74a3ce,
preserved at docs/2026-06-10-brief-downstream-gaps/legacy_otr_video_plan_
e74a3ce.py.txt) PROVABLY produced top-notch FLUX radio stills + awesome motion;
panel advice is taken with a grain of salt wherever it invents instead of
restores.

## ACCEPTED (consensus, grounded)

1. **Still-object schema + scene-still prompt source** (GPT-1, Gem-3, DS-1,
   Fable-1/4): extend the image platform with scene-still objects
   `{object_id: still_<beat_id>, kind: portrait|scene_open|scene_beat, role,
   beat_id, w, h, prompt, prompt_hash, source}`. RESOLUTION of the ordering
   trap (Fable-1: b000 is ShotLock-synthetic but image gen runs BEFORE
   ShotLock in the graph): the image node derives scene targets from the SAME
   pure helpers ShotLock uses (`derive_opening_music_beat(ledger, fps)` + the
   role mapping over lines) -- no graph reorder, b000's still exists before
   ShotLock plans the shot.
2. **6/5 prompt composition for stills** (operator north star + DS cut-1 +
   GPT S-7): the legacy `compose_shot_prompt` LAYER ORDER restored with the
   shared subject replacing the dead scene_visual layer:
   `[macro-radio subject (opens) | portrait_prompt (chars)] + [setting terms
   top-2] + [shot/framing hint] + [TRIMMED era tail] + [style tail]`.
   Subject table lives in `_otr_story_brief_helpers` (ONE source of truth;
   the round-5 driver branch refactors to call it -- Fable-4, GPT S-2, Gem-3;
   parity test: driver text prompt and still prompt share the leading
   subject).
3. **Dispatcher fixes** (Gem-1/2, GPT-2/3/8, DS-2): role from the payload
   (not hardcoded character_video); per-kind engine slot resolution
   (announcer/music/other image-model slots actually honored); `episode_id`
   input -> save EVERY still to `episodes/<ep>/stills/`; ledger images[]
   carries the episode-local path; the global pool retires only after a
   tracked-reader sweep (Fable-3; staging/consumers verified).
4. **Driver init-selection by engine family** (GPT-2, DS-3, Fable):
   `audio_driven_face -> portrait(char_id)`; `image_to_video/static_motion ->
   scene still(beat_id)`; missing still -> LOUD + today's text/floor path
   (never a silent empty init into a fail-closed engine).
5. **Conditioning build order** (GPT cut-1, DS-6, Fable-2): v1 =
   still_kenburns drifts the scene still (verify it accepts external init --
   DS-4) + wan_i2v(init=scene still) where enabled; **LTX img2vid CUT from
   v1** (wrapper-band risk proven tonight); LTX keeps its round-5 text path.
   Acceptance must be passable with kenburns alone.
6. **Dimensions per kind** (GPT-5): scene stills render LANDSCAPE (canvas-
   matched /32), portraits keep 832x1216.
7. **Sequencing + VRAM** (DS-5, Fable-7): stills mint in the image phase
   (image_done gate wired to the render node -- verify the gate input exists,
   else add); never lazily mid-episode under a resident heavy engine.
8. **Determinism + cache** (Fable-8, GPT-4): still seeds from the V-7
   request-hash scheme; cache key gains kind/w/h; cached content always
   materializes into the CURRENT episode's stills/ + a fresh ledger row.
9. **Observability** (Fable-5, GPT S-3, GPT-10): trace + manifest rows gain
   `init_image`, `init_source` (portrait|scene_still|none) for every beat;
   acceptance asserts per-family conditioning, not "at least one".
10. **Era-tail diet with knobs** (all + GPT-9): explicit helper profiles --
    stills = atmosphere line + palette top-2 + lighting top-2 (~120 chars);
    video call sites unchanged (240-cap already trims). Portrait-hash churn
    accepted once, in this slice, deliberately.
11. **S4 (M4->HuMo) OUT of scope** (GPT S-1, DS S-2, Fable-5) -- separate
    ticket; only the trace stamps land here.
12. **stills_manifest.json + contact sheet** (GPT optional): the manifest
    yes (cheap, operator-inspectable); contact sheet optional later.

## REJECTED (grain of salt, with reasons)

- **Gem cut-1 (drop structure-based synthetic-open detection, use role
  alone)**: contradicts the r5 pass03 convergence AND tonight's scene_broll
  fixture proof; the refined detector (suffix definitive, empty-sids only for
  open roles) already covers both failure modes. No change.
- **GPT-6's "either/or" framing on LTX img2vid**: resolved harder -- CUT from
  v1 entirely (not "optional branch"); revisit as a probe after the spine
  ships and the operator eyeballs kenburns/wan conditioning.
- **DS cut (drop per-beat brief integration for stills) taken PARTIALLY**:
  the fixed macro subject leads (restoration), but palette/setting terms stay
  (the 6/5 stills were per-episode-palette tinted -- that IS the legacy look;
  dropping them would invent a new look, violating the north star).
- **Any new "primary model" abstraction** (GPT cut-4): agreed, rejected --
  registry slots already exist.

## VERIFY-AT-BUILD (carried)

still_kenburns external-init support; wan_i2v dimension snapping on landscape
stills (one probe clip); the render node's gate input for image_done; the
global-stills reader sweep before retiring the pool; LTXVImgToVideo node
presence (future probe only).

## Convergence

All four panelists found the same gaps; the synthesized plan (pass01_plan.md)
resolves each with grounded mechanics. Pass02 = convergence check on the PLAN.
