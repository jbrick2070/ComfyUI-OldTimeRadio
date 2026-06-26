# LTX audio-in consolidation + robust capability-driven still routing -- BUILD PLAN (pass00)

Operator directive (2026-06-26): "remove the two legacy and ensure the still
logic is most robust." Collapse the LTX audio-in lane to ONE engine and make the
scene-still routing model-agnostic (works regardless of which video / still model
a role is wired to). This plan is the thing the roundtable hardens.

## Goal / end state

- ONE LTX audio-in engine. `ltx_video` = the regular (no-audio) LTX lane (FROZEN);
  `ltx_audio_in` = the audio-in lane (I2V on WHATEVER still + the shot audio, music
  OR voice; there is no separate LTX "lip-sync" parameter -- talk is just I2V on a
  face still). DELETE `ltx_av_talk` (`audio_driven_face`) and `ltx_av_music`
  (`audio_conditioned_video`, T2V, accepts_still=False).
- The scene-still routing in render_driver is driven by the engine's DECLARED
  capability (`accepts_still`), not by a patchwork of family-sets + hardcoded
  engine-name branches. Any current or future video/still engine that consumes a
  still gets the right still (wide scene plate for scene engines; the character
  PORTRAIT for audio_driven_face; mesh FODDER for mesh engines), or correctly
  skips -- with a LOUD degrade when a needed still is missing. NO silent fallback.

## Invariants (a "fix" that breaks one is rejected)

1. `test_audio_byte_identical` stays green -- the frozen master-audio spine is
   untouched; the video lane is always silent (only OTR_MasterAudioMux adds audio).
2. CLAUDE.md Section 0: the canonical workflow JSON change ships in the SAME commit
   as the code; re-validate (OTR_WorkflowValidator + JSON round-trip + widget/link
   audit). `widgets_values` is positional -- never insert mid-list.
3. NO FALLBACKS (547671d): a failed render fails LOUD. The legacy `SYNTH_FALLBACKS`
   `ltx_av_talk->humo` / `ltx_av_music->ltx_video` entries are REMOVED, not
   re-pointed (they contradict the engines' declared `fallback_engine=None`).
4. The 14.5 GB ceiling guard (NVML) is preserved for the heavy LTX-AV lane.
5. The portrait-vs-wide still distinction is preserved: audio_driven_face (HuMo)
   conditions on the 832x1216 vertical PORTRAIT; wide engines (render_aspect=wide:
   flux_still/flat_still/ltx_video/ltx_audio_in) condition on the 16:9 scene still
   and NEVER let the portrait leak into a wide frame (BUG-LOCAL-403 / the 2026-06-20
   "radio booth images" pillarbox fix).
6. The mesh-fodder path is preserved: `requires_mesh_fodder` engines (mesh_stage)
   get clean fodder, NOT the scene still (the clay-blob guard at render_driver:842).
7. UTF-8, no BOM, ASCII source. SFW. Run the regression suite + Bug Bible after the
   change; commit AND push to v2.0-alpha same session.

## Grounded current state (verified against the files)

`nodes/_otr_video_engines/eng_ltx_av.py` holds THREE engines over one
`_LtxAvBase`: `ltx_av_talk` (audio_driven_face, _is_talk, fallback None),
`ltx_av_music` (audio_conditioned_video, accepts_still=False, T2V,
default_roles=(music_visual, announcer_visual)), and the new `ltx_audio_in`
(audio_conditioned_video, accepts_still=True, _is_talk, NO default_roles).
`_build_graph` already does "i2v on ANY init image, else empty-latent t2v", so the
render core needs NO change for the unified engine -- only routing + wiring.

The scene-still routing in `render_driver.py` is a PATCHWORK, each piece tied to a
shipped bug:
- `_SCENE_INIT_FAMILIES = {image_to_video, static_motion}` (line 479): the
  family-keyed branch (842) routes the beat scene still, guarded by
  `not _requires_fodder` (the mesh clay-blob guard).
- name-branch `flux_still`/`flat_still` (869): scene still, wide-only, clears the
  portrait so it can't leak into a wide frame; LOUD if missing.
- name-branch `ltx_video` + OTR_ENABLE_LTX_I2V (906): scene still; stamps
  `_i2v_still_missing` LOUD if absent.
- `audio_driven_face` keeps the character portrait (asset_refs.init_image, set
  upstream); it is NOT in the scene-still branches.
- `ltx_audio_in` (family audio_conditioned_video, accepts_still=True) matches NONE
  of these -> `init_image=""` -> render_clip raises "ltx_audio_in (talk) requires
  init_image" on b000_music_open. THIS is the bug.

Other name-keyed couplings to the two legacy names:
- `SYNTH_FALLBACKS` (63), `ENGINE_FAMILY` (67-81), the canvas clamp
  `if engine_id in ("ltx_av_talk","ltx_av_music")` (1082), the scene-prompt branch
  `("ltx_video","wan_i2v","ltx_av_music")` (1163), `_LTX_OPEN_ENGINES` (1731),
  `_uses_ambient_master_audio` (the audio_conditioned_video ambient-slice path,
  ~965).
- `config/profiles/16gb_full.json` role_overrides: announcer_visual + music_visual
  = `ltx_av_music` (the live DEFAULT wiring).
- `workflows/otr_scifi_16gb_full.json`: node-87 announcer/music engine widgets =
  `ltx_av_music` (asserted by test_workflow_live_passes_validator wv87).
- registry `CAPABILITIES` rows + the dep-pilot OPT_IN_ENGINES list.
- ~12 test files name the two engines (see Part D).

## Part A -- collapse to one engine (`eng_ltx_av.py`)

DELETE `LtxAvTalkEngine` and `LtxAvMusicEngine`. Keep `LtxAudioInEngine`
(name=`ltx_audio_in`). Give it `default_roles = ("music_visual",
"announcer_visual")` so it inherits the per-role DEFAULT the music engine held
(otherwise those roles lose their audio-in default). Keep `accepts_still=True`,
`_is_talk=True`, roles = (announcer_visual, music_visual, character_video),
required = (text_prompt, audio_ref, init_image), fallback None. Update `__all__`.

OPEN Q (panel): keep the public id `ltx_audio_in`, or rename to `ltx_av`? Rename
is cosmetic but multiplies the JSON/profile/test churn. Recommend KEEP
`ltx_audio_in` (already wired in tests/dep-pilot/soak); note it cleanly as
"ltx_video = no-audio, ltx_audio_in = audio-in".

## Part B -- robust capability-driven still routing (`render_driver.py`)

Replace the three scene-still branches (the `_SCENE_INIT_FAMILIES` family branch +
the `flux_still`/`flat_still` name branch + the `ltx_video` name branch) with ONE
capability-driven rule, evaluated per beat AFTER fodder/portrait are resolved:

    eng = shot.engine_id
    if engine_consumes_still(eng) and not _requires_fodder
            and engine_family(eng) != "audio_driven_face":
        # wide scene engines: condition on the beat scene still, wide-only,
        # clear any portrait so it can't leak into a wide frame.
        still = _still_index(ledger).get(still_pool_key or beat_id, "")
        if still:
            init_image, init_source = still, "scene_still"
        else:
            init_image, init_source = "", "missing_scene_still"   # LOUD; stamp
            _LOG.warning(... MISSING-STILL (LOUD) ...)

where `engine_consumes_still(eng)` reads the engine's declared `accepts_still`
(via the video registry), with a safe default for unregistered/floor engines.
audio_driven_face is explicitly excluded (keeps the portrait, unchanged). mesh via
`_requires_fodder` (unchanged). This SUBSUMES flux_still/flat_still/ltx_video/
ltx_audio_in and any future accepts_still engine. Preserve the existing LOUD
`missing_scene_still` stamp + the `_i2v_still_missing` trace semantics so the
degrade stays visible. Keep OTR_ENABLE_LTX_I2V as a kill-switch for the LTX rows
(or retire it -- panel to weigh; default behavior must not change).

OPEN Q (panel): is gating on `accepts_still` exactly right, or should it be a
distinct `engine_consumes_still` that also checks `required_inputs` contains
init_image? `ltx_av_music` was accepts_still=False ON PURPOSE (T2V reacts to the
track) -- after deletion the only audio_conditioned_video accepts_still engine is
ltx_audio_in, but the VISUALIZER is also audio_conditioned_video and must NOT get
a still. Confirm the visualizer's accepts_still=False so the capability gate
excludes it. (Verify-at-build: enumerate every registered engine's accepts_still.)

## Part C -- wiring (same commit as the code; JSON re-validated)

1. registry.py: delete the `ltx_av_talk` + `ltx_av_music` CAPABILITIES rows + the
   two entries in the engine list (291-292). Keep `ltx_audio_in`.
2. scripts/otr_video_dep_pilot.py: delete the two OPT_IN_ENGINES entries; keep
   ltx_audio_in. (Guard tests: every-registered-engine-declared +
   pilot-covers-flag-gated must still pass.)
3. render_driver.py name-maps: drop the two from SYNTH_FALLBACKS, ENGINE_FAMILY,
   _LTX_OPEN_ENGINES; add `ltx_audio_in` where the audio-in lane belongs
   (ENGINE_FAMILY audio_conditioned_video; _LTX_OPEN_ENGINES yes). Canvas clamp
   (1082): replace the name-set with `requires_flag == OTR_ENABLE_LTX_AV` /
   family-or-capability so the AV canvas clamp follows the engine, not the name.
   Scene-prompt branch (1163): `ltx_av_music` -> `ltx_audio_in`.
   `_uses_ambient_master_audio`: ensure ltx_audio_in's audio_conditioned_video
   family still gets the bounded ambient slice for no-timing music beats.
4. config/profiles/16gb_full.json: role_overrides announcer_visual + music_visual
   -> `ltx_audio_in`.
5. workflows/otr_scifi_16gb_full.json: node-87 announcer/music engine widgets
   `ltx_av_music` -> `ltx_audio_in` (positional widgets -- replace value in place).
   Re-run OTR_WorkflowValidator + round-trip + widget/link audit.
6. otr_scifi_16gb_full_api.json is a stale generated copy (NOT canonical) -- leave
   or regenerate; do not hand-edit as if canonical.

## Part D -- tests (update in the same commit)

- test_ltx_audio_in_engine.py: drop `test_two_legacy_variants_unchanged`; add a
  test asserting `ltx_av_talk`/`ltx_av_music` are GONE from the registry and the
  capability map, and that `ltx_audio_in` is the default for music_visual +
  announcer_visual.
- test_video_ltx_av.py: rewrite the two-engine assertions to the one engine.
- test_ltx_av_driver_wiring.py: ENGINE_FAMILY/SYNTH_FALLBACKS now key ltx_audio_in;
  the music-open scene-prompt test targets ltx_audio_in.
- test_capability_profiles.py: role_overrides now ltx_audio_in.
- test_workflow_live_passes_validator.py: wv87 now ltx_audio_in.
- test_image_platform_c1.py: the accepts_still opt-OUT case used ltx_av_music
  (False) -- repoint to a still-opt-out engine that survives (visualizer), and add
  ltx_audio_in as the accepts_still=True case.
- test_video_ledger.py, test_video_motion.py,
  test_video_render_driver_perbeat_audio.py, test_still_aspect_and_labels.py,
  test_tested_only_dropdown_gate.py, test_ltx_open_health.py: replace the two names
  with ltx_audio_in.
- tests/debug_prompt.json: ltx_av_music -> ltx_audio_in.

## Part E -- validate + ship

Windows venv `pytest -q -p no:cacheprovider` (OTR) + the Bug Bible regression
(survival-guide repo, relative path). AST-parse + no-BOM on every touched .py.
Commit AND push to v2.0-alpha; verify HEAD==origin. THEN reset the box (selective
CIM kill), boot the UTF-8 launcher, smoke ONE short episode (ltx_audio_in bookends
+ still_parallax char beats + indextts2) until an OBS final lands in
output/otr/obs, then relaunch the overnight 420 soak.

## Forks for the panel (the genuine uncertainty)

F1. Still routing: the capability-driven UNIFICATION in Part B (subsume all three
    branches) vs a NARROW safe-mirror (just add an `ltx_audio_in` branch cloning
    the proven `ltx_video` I2V branch, leave the patchwork). Robust vs low-risk.
F2. The exact capability to gate on (`accepts_still` alone vs accepts_still AND
    init_image in required_inputs). Which excludes the visualizer + any T2V
    audio-reactive engine cleanly?
F3. Engine id: keep `ltx_audio_in` vs rename `ltx_av`.
F4. OTR_ENABLE_LTX_I2V kill-switch: keep (now governing the unified capability
    branch for LTX rows) vs retire. Must not change default behavior.
F5. Any guard in the three deleted branches that the unified rule would silently
    drop (the `_requires_fodder` mesh guard; the portrait-clear; the
    station_card ex\-exclusion; the `_i2v_still_missing` stamp)?
