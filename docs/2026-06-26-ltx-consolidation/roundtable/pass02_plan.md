# FINAL BUILD SPEC (pass02) -- LTX audio-in consolidation + role-driven routing

Synthesis of R1+R2 (GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro, all grounded).
CONVERGED. This is the build spec. Sequenced value-first in committable chunks so
each ships green independently.

## The corrected design (what R2 changed from pass01)

`ltx_audio_in` is render_aspect="wide" (eng_ltx_av:153). So a CHARACTER beat on it
must take the beat's WIDE scene still (the image phase mints kind=scene_character
for char beats), NOT the vertical portrait -- the portrait is ONLY for
portrait-aspect audio_driven_face (HuMo). This means ltx_audio_in routes the still
EXACTLY like the existing wide still-consumers flux_still/flat_still (cheap_families,
render_driver:869) -- which already handle the char-beat wide-still + portrait-clear
correctly. So the STILL fix is to add ltx_audio_in to that wide-still branch, NOT a
risky full-capability rewrite. The role-driven AUDIO + PROMPT fixes still apply (a
character beat must get clean own-voice audio + a character prompt, not the ambient
master slice + scene prompt). Missing-required-still is ALREADY terminal+LOUD:
render_clip raises GraphExecutionError on empty init_image and there are no
fallbacks -- so the real job is to ENSURE the still is minted + routed.

## CHUNK 1 (the unblocker -- ship first): still generation + routing + audio/prompt

C1a. STILL GENERATION (the original blocker): the MUSIC bookend beat
(b000_music_open) gets NO scene-still target today (derive_scene_still_targets,
otr_meta_brief_image_prompt.py emits for announcer/open/outro only). Make the music
bookend reuse the SAME scene_open radio-bookend still as the announcer open (operator:
"use the same bookend still"). Emit a scene-still target for the music bookend
(role music_visual) keyed so the dispatcher mints it on the music_image_model slot,
seeded radio_bookend_seed. GATE on the music engine consuming a still (so a no-still
music engine -- visualizer -- is unaffected).

C1b. STILL ROUTING (render_driver:869): add "ltx_audio_in" to the wide-still
branch `if engine_id in ("flux_still","flat_still")` -> add ltx_audio_in. It then
conditions on the beat's wide scene still (scene_character for char beats, the
radio-bookend still for announcer/music), portrait cleared -- no portrait clobber.

C1c. AUDIO (render_driver:723 _uses_ambient_master_audio): exclude character-face
beats. New signature accepts a precomputed is_char_face flag:
`_uses_ambient_master_audio(engine_id, family, is_char_face)` -> returns False when
is_char_face. Update the single call site in build_request_from_shot (it already
computes the classifier -- see C1e). Bookend (announcer/music) beats are NOT
char-face, so they keep the ambient slice (correct).

C1d. PROMPT (render_driver:1163 scene-prompt branch + the ~1133 audio_driven_face
prompt-fallback): (i) add ltx_audio_in to the scene-prompt engine set; (ii) gate
the WHOLE scene-prompt branch on `not is_char_face`; (iii) broaden the
`elif _fam == "audio_driven_face"` char-fallback gate to `is_char_face` so an
ltx_audio_in CHARACTER beat with no M4 prompt gets the gear-free char fallback, not
the generic radio-studio default. Also gate the "stable centered subject" clause
(~1124) on is_char_face, not engine_id.startswith("ltx").

C1e. CLASSIFIER (one shared helper, render_driver): define
`_is_character_face_beat(shot, line)` ONCE -- role=="character_video" PRIMARY;
char_id (from shot OR line) only when role is missing/legacy; OR
family=="audio_driven_face" and role not in (announcer_visual,music_visual). Replace
the existing inline `_is_char_face_beat` (~1100) AND wire it into C1c/C1d so the
three axes never drift. Scene roles (scene_broll/background_abstract) with a stray
char_id do NOT route to char-face (role is primary).

C1 tests: a table-driven build_request_from_shot routing-matrix test asserting real
request fields (asset_refs.init_image, audio_ref.path, text_prompt source,
observability.init_source) for: ltx_audio_in announcer/music -> scene_still +
ambient slice + scene prompt; ltx_audio_in character -> wide scene_character still +
clean per-line audio + char prompt; flux_still/flat_still unchanged; HuMo -> portrait;
ltx_video missing still -> text-only LOUD; visualizer/station_card -> no still.
Plus the existing suites. COMMIT+PUSH.

## CHUNK 2 (the consolidation -- ship second): delete the two legacy engines

C2a. eng_ltx_av.py: DELETE LtxAvTalkEngine + LtxAvMusicEngine; set
LtxAudioInEngine.default_roles=("music_visual","announcer_visual"); rewrite the
module/class docstrings (drop "two adapters/back-compat/dark"); __all__ =
["LtxAudioInEngine"].
C2b. registry.py: delete the two CAPABILITIES rows + the two list entries;
VALIDATED_ENGINES remove the two, ADD ltx_audio_in (provenance: same GPU-proven
_LtxAvBase+weights, I2V branch ltx_av_talk exercised; Chunk-3 smoke confirms live).
C2c. render_driver name-maps: SYNTH_FALLBACKS drop both (add NONE for ltx_audio_in);
ENGINE_FAMILY["ltx_audio_in"]="audio_conditioned_video", drop both;
_LTX_OPEN_ENGINES = frozenset({"ltx_video","ltx_audio_in"}); canvas clamp (1082)
name-set -> ("ltx_audio_in",) + fix its log message.
C2d. scripts/otr_video_dep_pilot.py: drop the two OPT_IN_ENGINES entries.
C2e. config/profiles/16gb_full.json: announcer_visual + music_visual -> ltx_audio_in.
C2f. workflows/otr_scifi_16gb_full.json: node-87 announcer/music widgets
ltx_av_music -> ltx_audio_in (positional value in place); re-validate
(OTR_WorkflowValidator + round-trip + widget/link audit).
C2g. tests: drop test_two_legacy_variants_unchanged; assert the two GONE from
registry+CAPABILITIES+VALIDATED_ENGINES + ltx_audio_in default for music/announcer;
rewrite test_video_ltx_av, test_ltx_av_driver_wiring, test_capability_profiles,
test_workflow_live_passes_validator (wv87), test_image_platform_c1 (opt-out ->
visualizer; add ltx_audio_in required-still case), test_video_ledger,
test_video_motion, test_video_render_driver_perbeat_audio, test_still_aspect_and_labels,
test_tested_only_dropdown_gate, test_ltx_open_health, tests/debug_prompt.json.
COMMIT+PUSH.

## CHUNK 3: smoke + soak
Reset box (selective CIM kill), boot UTF-8 launcher, smoke ONE short episode
(ltx_audio_in bookends + still_parallax char beats + indextts2) -> OBS final in
output/otr/obs; if green, add ltx_audio_in to VALIDATED_ENGINES is already done in
C2b (confirmed live). Relaunch the 420 soak.

## Verify-at-build (grep/enumerate before deleting)
1. grep ALL refs to LtxAvTalkEngine/LtxAvMusicEngine/ltx_av_talk/ltx_av_music across
   code+tests+JSON+fixtures+launchers (GPT #11) -> zero live refs after.
2. enumerate each engine (name, family, render_aspect, accepts_still,
   required_inputs); confirm station_card route none (no init_image required + not
   in the wide-still branch), visualizer none, wan_i2v via _SCENE_INIT_FAMILIES
   unchanged.
3. render_single(): a smoke through render_single("ltx_audio_in") does NOT apply the
   512x288 AV clamp (GPT #12) -- the soak smoke goes through the full pipeline
   (build_request_from_shot), which DOES clamp; if a render_single path is used,
   add the clamp. Verify the smoke path.
4. default-role resolution consults default_roles for ltx_audio_in.
5. no runtime/validator consumer of otr_scifi_16gb_full_api.json (else regenerate).

## Resolved forks (final)
F1 robust-vs-narrow: role-driven AUDIO/PROMPT (robust) + join the existing wide-still
   branch for the STILL (low-risk, already char-aware). Best of both.
F2 capability: the wide-still branch is by-name (flux_still/flat_still/ltx_audio_in);
   a full accepts_still/required_inputs capability gate is a FUTURE cleanup (the
   enumeration test pins behavior now). Not blocking.
F3 keep ltx_audio_in name. F4 OTR_ENABLE_LTX_I2V scoped to ltx_video only.
