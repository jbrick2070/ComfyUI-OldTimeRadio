# LTX audio-in -- handoff (2026-06-26)

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

## The fix (scoped)
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
