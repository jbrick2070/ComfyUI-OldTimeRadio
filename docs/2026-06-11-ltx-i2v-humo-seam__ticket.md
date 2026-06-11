# LTX-I2V + HuMo-seam ticket -- record (2026-06-11, overnight session)

Operator ticket: feed the episode's FLUX scene stills into LTX as init images,
fix the reddish prompt drift, kill the microphones on HuMo character beats.

## Part A -- reddish drift fix (SHIPPED)
`render_driver` LTX scene composer now finishes with the TRIMMED still-profile
era tail: `finish_visual_prompt(..., era_profile="still")` (new pass-through
parameter; default `"full"` keeps every other call site byte-identical). Video
prompts now share the stills' palette diet (atmosphere + palette top-2 +
lighting top-2, ~120 chars) instead of the full top-3 palette tail that
dragged LTX scene clips reddish. The `get_open_subject` lead is unchanged
(parity tests green). Tests: `test_brief_prompt_finishing.py` (era_profile
default byte-identity, still-tail composition, LTX-site still-tail assert).

## Part B -- OTR_ENABLE_LTX_I2V img2vid branch (SHIPPED dark; PROBE PASSED)
- Adapter branch INSIDE `eng_ltx_video` only: `OTR_ENABLE_LTX_I2V=1` + an
  on-disk request init image -> the image-conditioned wrapper graph
  (`LoadImage -> LTXVImgToVideo -> LTXVConditioning -> KSampler -> VAEDecode`);
  flag unset (DEFAULT) or init missing -> the round-5 text path, with the
  missing-init case logged LOUD. ENGINE_FAMILY stays `text_to_video`; no
  registry/family change; no new widgets.
- Driver: with the flag on, an `ltx_video` shot whose beat has a ledger scene
  still gets `init_source=scene_still` stamped on the request/trace; no still
  = LOUD text-path fallback (never silent).
- Decode band UNTOUCHED: `_ltx_frame_length` floor/cap (169f default) governs
  BOTH paths.

### PROBE RESULTS (live, RTX 5080, ComfyUI server :8000, 2026-06-11 ~03:30)
- `/object_info`: `LTXVImgToVideo` PRESENT; required inputs = `positive,
  negative, vae, image, width, height, length, batch_size, strength` --
  **`strength` is REQUIRED on this install** (the ticket graph initially
  omitted it; adapter now sends `OTR_LTX_I2V_STRENGTH`, default 1.0).
- One-clip probe (`scripts/_otr_ltx_i2v_probe.py`, init = episode still
  `still_b005_dc2731c9b814.png`, 1472x832, length 169, 12 steps, vp9 webm):
  **COMPLETED, no decode error.** ffprobe on the output:
  `width=1472 height=832 r_frame_rate=25/1 nb_read_frames=169`.
- Dimension/crop behavior: output EXACTLY the asked 1472x832x169 -- no crop,
  no resize drift with a same-canvas init still; the 169f decode floor holds
  on the i2v path (the tensor 256-vs-128 band behaves as on txt2vid).
- NOT yet verified (next GPU sitting, before enabling more beats): `strength`
  semantics sweep (1.0 vs lower -- how hard the init pins composition), VRAM
  peak under the i2v graph vs the 14.5 GB ceiling on a FULL-length episode
  run, and the look-QA of init-conditioned motion vs the text-only opens.

## Part C -- M4 -> HuMo creative seam + gear scrub (SHIPPED)
- `build_request_from_shot`: CHARACTER face beats (audio_driven_face, role
  not announcer/music) now run the broadcast-gear scrub over the M4 creative
  prompt (`_GEAR_WORDS_RD` local mirror of
  `otr_meta_brief_image_prompt._GEAR_WORDS`, LOCKSTEP pinned by test); a face
  beat with NO creative prompt -- the proven microphone re-introduction path
  (the studio default re-dressed the gear) -- logs LOUD and renders on the
  gear-free `_CHAR_FACE_FALLBACK_PROMPT`, stamped
  `_prompt_source=default_scrubbed`. ANNOUNCER beats exempt (radio-styled BY
  DESIGN; keep the studio default, stamped `default`). NO negations anywhere
  (the c01 giant-mic lesson: scrub the output, never add "no microphone").
- eng_humo consumes the request prompt already (`plan.text_prompt` ->
  `positive`); with the stamps, every HuMo trace row now proves which prompt
  source reached the engine.
- Tests: `test_brief_prompt_finishing.py` (scrub on character beats,
  announcer exemption, LOUD scrubbed fallback, default stamp, regex lockstep).

## Invariants held
Audio ledger frozen (`test_audio_byte_identical` green); saved json untouched
(no wire was needed); suite + Bug Bible green at every commit; all new
behavior env-gated or LOUD.
