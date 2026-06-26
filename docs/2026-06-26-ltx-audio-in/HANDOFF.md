# LTX audio-in -- handoff (2026-06-26)

## FINAL SCOPE (operator 2026-06-26): collapse to ONE engine + robust still logic
Operator directive: REMOVE the two legacy LTX-AV engines and keep ONE audio-in
engine; make the still logic the MOST ROBUST (works regardless of which video /
still model is chosen). Roundtable authorized if the builder is genuinely torn --
but the approach below is converged, so build it; only roundtable a real fork.

### A. Collapse to one engine
- DELETE `LtxAvTalkEngine` + `LtxAvMusicEngine` from `eng_ltx_av.py`; keep the
  single unified engine (`ltx_audio_in`, or rename to `ltx_av` -- operator's pick;
  ltx_av reads cleaner as the twin of `ltx_video`). It is `_is_talk=True` (I2V on
  any still), `accepts_still=True`, roles = announcer_visual + music_visual +
  character_video, `default_roles` = the bookend roles (so it is the DEFAULT the
  two legacy ones used to be), required text_prompt + audio_ref + init_image, NO
  fallback.
- REMOVE the `ltx_av_talk` + `ltx_av_music` rows from `registry.py` CAPABILITIES
  and from `scripts/otr_video_dep_pilot.py` OPT_IN_ENGINES (keep the one).
- REPOINT the canonical `workflows/otr_scifi_16gb_full.json`: any bookend role
  widget set to `ltx_av_talk` / `ltx_av_music` -> the unified engine (CLAUDE.md
  Section 0: JSON change in the SAME commit; re-run OTR_WorkflowValidator + the
  widget/link audit). Grep the repo for the two old names -> ZERO live references.
- Update the tests (test_ltx_audio_in_engine + any that name the two variants).

### B. Robust still logic (the durable rule -- model-agnostic)
The still PRESENCE + feeding is driven by ONE capability, per beat, so ANY
video/still model combo gets the right still or correctly skips:
- For EVERY beat, the still dispatcher already keys on
  `engine_consumes_still(eng)` (accepts_still / required_inputs init_image). KEEP
  that as the single gate.
- CHARACTER beats -> the character PORTRAIT still (already works for accepts_still
  engines; a procedural engine like still_kenburns correctly skips). For
  "z-image-turbo intro stills" the char-beat engine MUST be an accepts_still still
  lane (still_parallax), NOT the procedural still_kenburns.
- BOOKEND beats (b000 music-open, announcer open, announcer close, closing music)
  -> the SAME shared `scene_open` radio bookend still (the control-room still that
  ALREADY renders today -- see signal_lost_bells_beneath_sardis). The still-spine
  must EMIT that one scene_open still whenever ANY bookend role's engine
  `engine_consumes_still`, and `render_driver` must feed it as init_image to EVERY
  bookend beat whose engine consumes a still (today the music-open beat lands
  init_image='' at ~render_driver.py:892 instead of the scene_still). One still,
  start + end, fed to whatever engine the role has -> model-agnostic by
  construction; an engine that ignores stills (T2V / procedural) simply does not
  read it. This is the operator's "use the same bookend still as in this".
- FALLBACK posture (NO silent fallback, per the no-fallbacks contract): if a
  bookend engine consumes a still but the scene_open still could not be generated
  (e.g. no image model), the still-spine must GUARANTEE it earlier (emission is
  gated on the engine's need) rather than fail at render. If genuinely torn on the
  guarantee-vs-fail-loud edge, THAT is the spot to /roundtable.

### Validate (one smoke), then relaunch the soak
Boot `_otr_overnight_420_boot.cmd`; smoke `_otr_combo_soak.py` with the unified
engine on bookends + still_parallax char beats + indextts2, 80w/act 1, until an OBS
final lands in output/otr/obs. Then relaunch `_otr_overnight_420_soak.py`.

---


## Status
- **`ltx_audio_in` engine SHIPPED** (`8dfd56b8`): the unified, agnostic LTX audio-in
  lane (I2V on any still + audio, music OR voice). Correct + suite-green + declared
  (registry CAPABILITIES) + dep-pilot entry + `tests/test_ltx_audio_in_engine.py`.
- **NOT yet rendering end-to-end.** One blocker remains, diagnosed to the function.

## The blocker (precise)
A live smoke (`ltx_audio_in` bookends, `still_kenburns` beats, indextts2, 80w) FAILED:
```
shot shot_b000_music_open engine 'ltx_audio_in': GraphExecutionError:
  ltx_audio_in (talk) requires init_image (got '')
```
Root cause chain:
1. `ltx_audio_in` is `_is_talk=True` (I2V) -> needs `init_image`.
2. `accepts_still=True` is correct, and `OTR_ImageGenDispatcher.engine_consumes_still`
   reads it fine -- BUT the dispatcher only MINTS what `derive_image_prompts` hands it
   as `objects`.
3. `derive_image_prompts` -> `derive_scene_still_targets`
   (`nodes/otr_meta_brief_image_prompt.py:249`) emits scene stills for the
   ANNOUNCER bookend (it needs one for HuMo, see the comment ~line 115) and
   open/outro, but **does NOT emit a scene-still target for the MUSIC bookend**
   (`b000_music_open`). So the dispatcher has no object to mint for b000 ->
   `init_image=''` -> the I2V engine fails LOUD (no fallbacks).
4. The infra to CARRY it already exists end-to-end: the dispatcher's
   `kind == "scene_open"` radio-bookend still + `radio_bookend_seed=4242`
   (`otr_image_gen_dispatcher.py:121-128`), the role->image-slot map
   (`announcer_visual->announcer_image_model`, `music_visual->music_image_model`,
   :144), and the render_driver `init_source=scene_still` path
   (`render_driver.py:~901`). Only the EMISSION is missing.

## OPERATOR REFRAME (2026-06-26, the correct fix direction)
The radio bookend still ALREADY exists and renders fine -- a prior episode
(`signal_lost_bells_beneath_sardis_..._final.mp4`) opens on a perfect control-room
`scene_open` still under the announcer. So do NOT "emit a new" music still. Instead
REUSE THE SAME `scene_open` radio bookend still for the MUSIC bookend beat (and the
closing bookend), so the bookend ALWAYS has its still at start + end regardless of
which video/still model is selected -- "use the same bookend still as in this".

Pointer: `render_driver` sources the still per beat at ~`render_driver.py:790-911`.
The `b000_music_open` beat (char_id="") currently lands `init_image=""` (~:892)
instead of the existing `scene_open` ST-3 radio-bookend still (the `init_source=
scene_still` path, ~:901/:911). The fix: make the music-open (and closing-music)
bookend beat resolve to the SAME `scene_open` radio bookend still the announcer open
uses -- one shared bookend still, fed to whatever engine the bookend role has
(ltx_audio_in I2V, humo, etc.). It is model-agnostic by construction then.
Confirm the `scene_open` still object is generated for the episode (it is when the
announcer open is present) and just route it to the music bookend beat's init_image.

## The fix (earlier framing -- SUPERSEDED by the reframe above; kept for context)
In `derive_scene_still_targets` (otr_meta_brief_image_prompt.py:249): when the
**music bookend** beat's video engine consumes a still (gate on
`engine_consumes_still` / `accepts_still` via the image policy, like the
announcer/open targets already do), EMIT a `scene_open`/scene-still target for it
(role `music_visual`, a music-scene prompt, the `radio_bookend_seed`). Then the
dispatcher mints it on the `music_image_model` slot and render_driver feeds it as
`init_image`. Keep it GATED on the engine consuming a still so the existing
no-still bookend engines (`ltx_av_music` T2V, visualizer) are unchanged.

ALSO note the char-beat half of the recipe: `still_kenburns` is a *procedural floor*
(`accepts_still=False` -> "ignores init_image"), so it does NOT animate a
z-image-turbo still. For "z-image-turbo intro stills", the char-beat engine must be
an `accepts_still=True` still lane (e.g. `still_parallax`), not `still_kenburns`.
Re-confirm the operator's intended char-beat engine before wiring the soak recipe.

## Validate
Re-run the smoke (`scripts/_otr_combo_soak.py` with
`OTR_COMBO_ANNOUNCER/MUSIC=ltx_audio_in`, a real `accepts_still` char-beat engine,
`OTR_SOAK_TARGET_WORDS=80 OTR_SOAK_ACT_COUNT=1`) on the live :8000 server
(`scripts/_otr_overnight_420_boot.cmd`, which sets `OTR_ENABLE_LTX_AV=1` + indextts2)
until an OBS final lands in `output/otr/obs`. THEN relaunch the overnight soak
(`scripts/_otr_overnight_420_soak.py`, bookends already = ltx_audio_in).

## Operator decision pending
Char-beat engine for "z-image-turbo intro still" (still_parallax vs other) -- confirm
before the soak relaunch.
