# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> **LATEST SESSION -- 2026-06-22 (CODER+ROUNDTABLE, very long; PER-SEGMENT RMS LOUDNESS LEVELING SHIPPED +
> PUSHED + EAR-VALIDATED; CAPABILITY-ROUTING roundtable R1 CONVERGED, not built).**
> **(1) AUDIO LEVELING (baked-in default):** even out shot-to-shot dialogue levels. 4-round live roundtable
> (gpt-5.5+gemini-3.1-pro+deepseek-v4-pro+grok-4.3 + Claude judge, ~$0.52) -> implemented in
> `nodes/scene_sequencer.py`: per-dialogue-clip RMS leveling (`_level_dialogue_clip` -> `_loudness_normalize_clip`,
> target -16 dBFS, peak-safe cap + noise gate + float64 RMS; the 3 dialogue call sites :747/:753/:775; SFX
> :726 stays peak). **RMS is the BAKED-IN DEFAULT** (`OTR_SEGMENT_LOUDNORM` defaults `rms`; `=peak` escape
> hatch) per operator "don't hide behind a switch." Episode master UNCHANGED (makeup +4 dB) + target ~= what
> peak-norm produced, so the per-line BALANCE evens WITHOUT shifting overall loudness. **Voice-MODEL-AGNOSTIC**
> (downstream of TTS; leveled bark chars + Kokoro announcer in one render). Commits `82fbd4e` (CLAUDE.md S8
> roundtable defaults) + `8ea03e9` (leveling) PUSHED to v2.0-alpha; full suite **4947 pass/34 skip** + Bug Bible
> 16/7/3 green; proven LIVE (`[loudnorm]` preflight, target -16) + operator EAR-VALIDATED ("swift sounds ok").
> Docs `docs/2026-06-22-loudness-normalization/` (FINAL_PLAN + pass00-04 + OPERATOR_NOTES +
> `tools/measure_dialogue_rms.py`; old-path dialogue measured -13..-24 dBFS, mean -17.3). **UNCOMMITTED:**
> `tests/_run_baseline.py` workflow-path bug fix (two-`..`->one; had silently broken the gated
> `OTR_REGRESSION_RUNTIME` audio capture). **Byte-identical GOLDEN re-baseline DEFERRED** (default-peak keeps the
> suite green; the rms golden is a future headless capture -- must force bark, indextts2 isn't headless-installed).
> **(2) CAPABILITY-ROUTING (NEW; roundtable R1 CONVERGED; NOT built):** live wall during a 100%-Wan eyeball --
> `wan_i2v` rejected from the announcer slot by the `roles` whitelist DESPITE being input-compatible (the
> announcer supplies `init_image`). Operator directive: declare capabilities ONCE per model, model-agnostic
> downstream (HuMo/LTX-AV = audio-in specials, the rest = still+prompt -> all slots); **HARD: strict SUPERSET,
> do NOT break working models.** R1 CONVERGED (`docs/2026-06-22-capability-routing/roundtable/pass01_plan.md`):
> make `roles` an OPTIONAL override (empty -> pure capability so wan unblocks; set -> enforced, no over-match) +
> decouple `default_roles` from eligibility + confirm the descriptor `roles` source in `otr_video_director.py` +
> handle ASPECT (wide vs portrait) + unify the render-gate source (FAMILY_REQUIRED_INPUTS <- required_inputs) + a
> GENERATED before/after eligibility test + eligibility != auto-selection + VIDEO-only v1 (defer image).
> **NEXT = capability-routing R2 (coding) -> R3 (wiring) -> R4 + build; the audio leveling is DONE.** (Wan
> roadmap item 3: wan_i2v fits b-roll music/beats, not the audio announcer -- the no-silent-swap safety fired
> correctly; the capability fix makes wan a b-roll announcer.)

> **LATEST SESSION -- 2026-06-22 (CODER, marathon; HEAD `08010ec` == origin/v2.0-alpha; full suite 4934
> pass/34 skip, Bug Bible 16/7/3, audio byte-identical 9/0). THREE sprints SHIPPED + PUSHED end to end, each
> chunk suite+Bug-Bible green -> commit+push to v2.0-alpha:**
> 1. **BARK ARTIFACT (upstream-TTS only, spine frozen):** B0 fixture+spine-verify `ffe5e82` -> B1 speech-only
>    dialogue mode + first-line [clears throat] gate `467fb05` -> B2 thread the dropped per-line seed
>    (determinism) `9e68ad5` -> B3 chunk-split hardening `e1b8196` -> QA >4kHz high-band edge metric + scan
>    `e93b54e`. Cleaned-bark re-soak (mesh 3D bookends + z_image_turbo char stills) rendered "Spinning
>    Contamination" to obs; metric before/after = pre-B1 1/14 flagged vs cleaned 0/6. min_eos_p kept 0.1;
>    panel-cut the per-chunk trim + FFT reroll loop.
> 2. **STAGE-DIRECTION LEAK (3-pass roundtable, ~$0.51):** bare undelimited stage directions ("twirls his pen
>    nervously Look...") were leaking into the frozen ledger text -> bark spoke them + captions showed them.
>    Chunk 1 pure scrub+detector+62-case corpus `8c40182` -> Chunk 2 scan + PRECISION GATE (489 ledgers, 20
>    would-mutate, 0 false positives) `e2dd95a` -> Chunk 4 freeze floor in `_strip_stage_directions` `2278bd2`
>    -> Chunk 5 prompt + the S1 music-3681 patch `9142b2f` -> Chunk 3 reroll in compose_line `6ce724d`. Fresh
>    render shipped a CLEAN ledger; the fixed freeze strips the exact screenshot lines.
> 3. **STORY-QUALITY R2 -- ALL 8 LEVERS + Final-QA scan (the operator-requested craft lift):** S1 `5396c05`
>    (music-text suppression) -> S2 `118088f` (announcer close = concrete image, not thesis) -> S3 `1887803`
>    (cliche + flat stage-business reroll) -> C0 `687f766` (action-verb beat intents + wants_are_default) ->
>    C3 `981db60` (contrasting speech_signatures) -> C4+C5 `3e19906` (escalation prompt + on-the-nose reroll)
>    -> C1+C2 `38c2c10` (specificity anchors + central object, 3-pass roundtable-converged ~$0.25) -> Final QA
>    `08010ec` (story_quality_scan extended with the lever metrics). ALL content-only, ledger schema FIXED
>    (free-form meta), NO workflow-JSON change, model-agnostic, deterministic.
> **OPERATOR VERIFY-AT-BUILD (carries forward):** the live `test_audio_byte_identical` baseline (indextts2)
> may need RECAPTURE -- the R2 prompt levers intentionally change generated dialogue (the clean fixture is a
> no-op, but a fixed-seed baseline close that newly trips a gate would shift); run the live gate + a re-soak
> read at leisure. **NEXT = OPERATOR DECISION (no sprint started this hand-off): pick the next forward-order
> item** (the S3 forward order -- 3D item 5, distribution item 6 -- + the carried per-segment LUFS/RMS plan
> are all still PARKED + untouched).

> **PRIOR SESSION -- 2026-06-22 (CODER, very long session; HEAD `90ddfca` == origin/v2.0-alpha; full suite
> 4740 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green). HANDOFF FOR A FRESH CODER WINDOW: the
> NEW CURRENT STEP is the STORY-QUALITY R2 coding sprint -- build-ready in `docs/2026-06-22-story-quality-r2/
> SPRINT_PLAN.md` (see CURRENT STEP, section 1).** SHIPPED + PUSHED this session, each suite+Bug-Bible green:
> - **3D image streams (all 7 chunks)** done + GPU-verified end-to-end earlier in the session (clay-blob fixed;
>   the block below). Then **mesh-quality v1.1+v1.2**: tighter fodder (`ea03203`), gradient sculpt surface
>   (`f8a48b9`), and the lighting fix `e485258` -- the meshes were WHITE MARBLE because WORKBENCH MATCAP
>   ignores the vertex albedo; switched to STUDIO lighting so the gradient + 3D form render (CPU-Blender
>   verified). Both 30w all-3D GPU smokes published to obs.
> - **OpenRouter writer fix** (`d249743`): the nightly frontier writers were 0-lining via `finish_reason=length`
>   (reasoning `~latest` models burning the output budget). Fix = `OPENROUTER_REASONING_EFFORT=none` (set in the
>   server boot + the User env for manual runs) + a one-line "reasoning_effort ACTIVE" log so it's EVIDENT on any
>   run. Roundtable-converged. Confirmed live (remote preflight wrote a full story; near-every leg now writes).
> - **ltx_av_music render crash** (`fadbe60`) + **beat-accurate audio** (`90ddfca`): a music beat with no
>   per-line timing FamilyInputGap-crashed ltx_av_music (ShotRow has no start_s/dur_s). Fix = synthesize a
>   BOUNDED master slice (target_frame_count/fps at the beat's cumulative position, anchored on preceding beats'
>   real line timing -> respects the opening-music offset); audio_driven_face excluded. Roundtable-converged,
>   +4 tests. NOTE: both render_driver fixes are IN-PROCESS -> live on the NEXT server boot, not the still-
>   running soak.
> - **Nightly 10h anthology soak** (local tooling `scripts/_otr_nightly_anthology_soak.py` +
>   `_otr_nightly_soak_boot_launch.ps1`, gitignored): 384w max-creativity, rotating frontier OpenRouter writers,
>   mixed video/3D bookends, rotating approved image models on character beats, rotating voices, dynamic news,
>   720w visualizer story-tests every 4th leg; smoke-first + remote preflight + soak-only local fallback. RAN
>   ~13h, feeding `otr/obs` for the operator's OBS->YouTube broadcast loop.
> - **STORY-QUALITY R2 CAMPAIGN -- CONVERGED, BUILD-READY (the NEXT step).** Grounded in the 13h soak's REAL
>   stories (opus = genuinely good; weak/local = generic + cliche + stage-business + music/non-dialogue text
>   bleeding into the caption track). 3 roundtable passes (pass01 structural + pass02 coding) + Claude's own
>   creative pass (pass01b) + seam-location -> `SPRINT_PLAN.md`: 8 ordered green chunks (S1 music/non-dialogue
>   caption suppression; S2 announcer-close final-image; S3 cliche/stage-business reject gate; C0 non-default
>   wants + action-verb in outline Stage 3; C1 specificity anchors; C2 central story-object; C3 contrasting
>   voice signatures; C4+C5 escalation+subtext). HARD: ledger schema FIXED (new fields ride free-form `meta`,
>   NO Pydantic fields), NO workflow-JSON change, reuse the EXISTING Sprint-5C `reroll_hint` loop +
>   `speech_signature` (already wired in `_otr_line_composer.compose_line_draft`), model-agnostic (every gate is
>   one opus passes -> lifts the weak end, never rewrites opus), craft-ONLY (no word/beat count). Located seams +
>   judgments in `docs/2026-06-22-story-quality-r2/`. >>> THE CODER WINDOW BUILDS SPRINT_PLAN.md, S1 FIRST.
> - OPEN (operator call): restart the soak so the ltx_av + reasoning fixes go live now, vs let it ride to the
>   10h cap and they apply next boot.

> **LATEST SESSION -- 2026-06-21 (CODER; 3D IMAGE STREAMS -- ALL 7 CHUNKS SHIPPED + PUSHED + GPU-VERIFIED
> END-TO-END; HEAD `555788e` == origin/v2.0-alpha; full suite 4733 pass/33 skip, Bug Bible 16/7/3, audio
> byte-identical green per chunk).** The clay-blob root cause is FIXED: `mesh_stage` no longer meshes the
> per-beat cinematic scene still (the whole environment) -- it now meshes a CLEAN isolated subject over a
> generated background plate. Seven ordered green chunks, each suite+Bug-Bible -> commit AND push ->
> HEAD==origin verified: **(1)** `b11759f` `requires_mesh_fodder=True` capability flag on MeshStageEngine
> (routing gate; capability-not-engine-name). **(2)** `a5be9b3` `build_request_from_shot` routes fodder
> engines to a clean `mesh_fodder` still BEFORE the `_SCENE_INIT_FAMILIES` override (skips it for fodder
> engines; LOUD `missing_mesh_fodder`, never meshes the environment); engine map read post-`OTR_FORCE_ENGINE_MAP`.
> **(3)** `af779bd` the prompt FORK -- `OTR_ImageDirector` forwards `mesh_fodder_roles` (per-role engine
> capability), `OTR_MetaBriefImagePromptGen` mints `mesh_fodder` (isolated subject) + `scene_background_plate`
> (subject-free world) with checked-in pos/neg scaffolds instead of one cinematic scene still. **(4)** `0ccb6ff`
> kind-specific indices -- `_portrait_index` kind-filtered (fodder can't leak to HuMo), `_still_index`
> prioritizes `scene_background_plate`, dispatcher carries `mesh_subject_id` onto the ledger row. **(5)** `84962bd`
> mesh cache keys on `mesh_subject_id` (char_id|object_id), not the per-beat still hash; render_driver stamps it.
> **(6)** `a2df9bf` announcer/music subject policy (announcer meshes the announcer via char_id; no-char beats
> mesh a stable `obj_<beat>` object; LOUD on bare `uncast`). **(7)** `555788e` opaque straight-alpha source-over
> composite (mesh over plate, `format=rgb`) -- kills the double-exposure ghost; blend kept as
> `OTR_MESH_COMPOSITE_STYLE=blend` opt-in; real-ffmpeg opacity regression. All content-only / capability-gated;
> ZERO workflow-JSON change; ledger schema `l3-2026-05-14` untouched. **GPU SMOKE (all-slots mesh_stage, 30w,
> real `otr_scifi_16gb_full.json`, FLOOR lane, bark voice):** SUCCESS in 20:17. The dispatcher minted 5
> `mesh_fodder` + 5 `scene_background_plate` objects (every beat -- music/announcer/character), NO cinematic
> scene stills; render_driver logged `meshing CLEAN fodder ... scene still is background plate only` for all 5
> beats; cache keys were the real subjects (`obj_b000_music_open`, `c01`, `c02`, `c03`) -- NEVER `uncast`; the
> VRAM barrier fired before each Blender spawn; `obs_publish OK`. Final = `otr/obs/
> signal_lost_colony_found_20260621_002009_silent_procgen_blended_final.mp4` (20.8 MB, 64.5s). Logs/proof in
> `docs/2026-06-21-3d-image-streams/` (smoke_server.log + the obs mp4 copy). KNOWN COSMETIC NIT (non-blocking):
> the render_driver "meshing CLEAN fodder" log prints `subject=?` for a no-char beat (the cache id is correct,
> `obj_<beat>`); a one-line log polish if it bugs you. **NEXT = 3D v1.5 (DEFERRED, panel-agreed): Cycles +
> 3-point lighting + multi-view texture bake** (the lit/textured tier, on top of clean fodder).

> **LATEST SESSION -- 2026-06-20 (CODER; BUG 1 SHIPPED + GPU-VERIFIED end-to-end; HEAD `9f03abd` ==
> origin/v2.0-alpha; suite 4633 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green).** TOP-PRIORITY
> VISUAL FIX **BUG 1 DONE** -- landscape character beats now show the CHARACTER full-frame 16:9, not the
> radio booth. Two commits: **`7e765b7`** (a) render_driver: flat_still/flux_still are render_aspect=wide ->
> REVERTED the `8bc5381` portrait-conditioning branch; they condition on the beat's scene still, never the
> 832x1216 vertical portrait (missing-still clears init_image so it can't leak); (b) image phase mints a
> per-beat `scene_character` still leading with the character, radio-booth tail dropped. **`9f03abd`** the
> ROOT CAUSE the live render exposed: `"character"` (the CANONICAL writer speaker_role) was MISSING from
> `SPEAKER_TO_VIDEO_ROLE` -> character beats fell through to `background_abstract` and got POOLED as
> other-beats (the deleted 8bc5381 branch had masked it) -> added `"character"->CHARACTER_VIDEO`; PLUS
> BEAT-AWARE stills (operator: per-shot/beat, regardless of image model, video lane too) -- each character
> beat's still is LLM-composed from that beat's `beat_intent`/`traits`/`text` (temp=0, mirrors the portrait
> path; person-guard + gear-scrub + era/grade tail + no-text clause; deterministic per-character fallback).
> Content-only, ZERO workflow-JSON change. **GPU VERIFY (z_image_turbo, FLOOR lane, all-roles flat_still,
> 60w/act1/2chars):** the 3 character beats b002/b003/b004 resolved `kind=scene_character role=character_video`
> (NO other_pool), 3 DISTINCT prompt hashes `source=char_scene_llm`, render_driver conditioned each on its
> scene_character still ("portrait never used"), `audio_byte_identical OK`, `obs_publish OK`. Frames pulled
> from the FINAL obs mp4 at each beat timestamp show the CHARACTER full-frame 16:9 with the matching SDH
> caption -> beat-synced end-to-end. Before/after + final-frame proofs in `docs/2026-06-20-visual-fixes/`.
> GOTCHAS: (1) z_image_turbo needs `OTR_ENABLE_ZIMAGE=1` + `OTR_ZIMAGE_UNET=z_image_turbo_nvfp4.safetensors`
> at server boot (only the nvfp4 quant is on disk, not the bf16 the engine defaults to). (2) the repeated
> "GULLIVER REEVES" cast is the `OTR_C7=1` byte-identity mode pinning OTR_CAST_SEED/STYLE_SEED=42 (my verify
> runs) -- production (no C7) rolls a fresh cast per episode. **ALSO THIS SESSION: BUG 2 DONE (verify-first,
> no change -- the animated title card + rolling credits already work) + BUG 3 DONE (commit `9f76937`: the
> forensic model/engine/SYSTEM dossier now scrolls on the credits via `_build_hud_dossier`; content-only, no
> JSON; +7 tests). ALSO BUG 4 DONE (commit `13017ec`: the ALWAYS-ON audio bars are a SEPARATE overlay layer,
> default ON, decoupled from the scene-aware floor -- new `audio_bars` widget wired IN the workflow JSON
> same-commit [node 93], a separate bars pass lighten-blends the green strip at 0.60 with captions ON TOP;
> +6 tests). HEAD `13017ec` == origin, suite 4646 pass, Bug Bible 16/7/3, JSON intact+extended. The 2026-06-20
> ALL FOUR visual fixes (character aspect / title card / credits dossier / audio bars) are SHIPPED. **NEXT =
> the 3D PoC build (`docs/2026-06-20-mesh-stage-texturing/` pass03+pass04 + the 4-item PIN list; chunks
> 6+7+3 + GPU smoke).**

> **LATEST SESSION -- 2026-06-20 NIGHT (CODER, autonomous "go all night"; HEAD `8de1057` == origin/v2.0-alpha;
> suite 4625 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green).** BUILT + PUSHED the CORE of the
> 3D textured-hero PoC (chunks 2+4+5 paired so producer+consumer land together -- no dead wiring), commit
> `8de1057`: **(2) single-view VERTEX-COLOR PROJECTION** in `scripts/otr_mesh_stage_blender.py` -- `--portrait`
> via `bpy.data.images.load`, a per-vertex point-domain `otr_proj` color attribute painted from the front Y/Z
> projection, set active+render so WORKBENCH `color_type='VERTEX'` draws it; GLB stays geometry-only
> (MESHER_VERSION NOT bumped); bounded hero arc (`_build_turntable` start_angle+arc clamped to MAX_ARC=45deg,
> `frames==1`->one keyframe); selftest projects a deterministic gradient + fails nonzero unless a NON-UNIFORM
> attr exists; pure helpers (`arc_keyframes`/`project_uv`/`sample_image`/`clamp_arc_degrees`) CPU-tested.
> `eng_mesh_stage.build_blender_cmd` plumbs `--portrait`/`--start-angle`/`--arc-degrees` (appended only when set
> -> legacy invocation byte-identical); `render_clip` passes the resolved still. **(4) C1 STAMP**
> (`render_driver.build_clip_manifest`): mesh_stage directory rows get `bg_still_path` from the per-beat scene
> still (the existing per-beat coverage already mints one via the per-role image engine -> image-model AGNOSTIC),
> fail-closed `os.path.isfile` + LOUD warn on absence. **(5) C1 COMPOSITE** (`otr_silent_composite.py`):
> `bg_still_path` carried through `plan_timeline_segments` + a still-aware `-loop 1` bg branch in
> `_encode_segment_from_dir`; ZERO new graph links/widgets (rides the existing 92->84 manifest channel); every
> non-mesh beat omits the field -> floor/black bg byte-identical. +13 unit tests. **REAL-BLENDER VALIDATION
> (CPU-only, did NOT touch the running soak's GPU):** the stage selftest ran through Blender 4.5.10
> (`C:\ComfyUI-Models\tools\blender-4.5.10\blender.exe`) WORKBENCH -> exit 0, 3 RGBA frames of DIFFERING sizes
> (47k/49k/50k = the arc moves + the projection is non-uniform); proof frame
> `docs/2026-06-20-mesh-stage-texturing/selftest_proof_frame_0001.png` shows the cube TEXTURED with the gradient
> (green/teal/blue/magenta), NOT flat gray -> projection works visually in real Blender. **DELIBERATELY NOT
> LANDED BLIND tonight (need the GPU render loop the soak occupies):** Chunk 6 (costly_choice_beat trigger ->
> route that beat to mesh_stage; the no-portrait->still_parallax fallback is ALREADY in place), Chunk 7 (JSON
> wiring on node 87/88), Chunk 3 (2D-ellipse contact shadow, wants GPU-render tuning), and the GPU smoke
> acceptance. Full detail: `docs/2026-06-20-mesh-stage-texturing/BUILD_RESULTS.md`. **SOAK (watched):** the
> 864-word frontier all-night soak on :8011 is HEALTHY -- smoke_0001 PASSED (status=success, 480 words; indextts2
> did NOT crash the server), now grinding the 48-leg matrix (leg_0002+). NEXT = (a) keep watching/triaging the
> soak, (b) when it frees the GPU, run the mesh_stage GPU smoke then land chunks 6+7 (+3) VALIDATED.

> **LATEST SESSION -- 2026-06-20 (CODER + 3D ROUNDTABLE CAMPAIGN; HEAD `f99af26` == origin/v2.0-alpha;
> suite 4616 pass/33 skip, Bug Bible 16/7/3, audio byte-identical green).** SHIPPED + PUSHED (each
> suite+BugBible green): **forensic episode treatment** -- `_write_story_treatment` now emits a full
> spec sheet (LLM config: both slot models + creativity/temp/words; STORY SPINE: news premise + opposed
> wants; per-role RENDER ENGINES video+image; SYSTEM: CPU/RAM/GPU/CUDA/torch; new `nodes/_otr_sys_specs.py`)
> (`e8f3094`); **resolved OpenRouter concrete-model + cost capture** per run for historical accuracy
> (`f03db0a`, `_record_resolved`/`resolved_models_snapshot`, cleared per episode in reset_run_budget);
> **Kokoro char-voice DEEP pool** -- the bank had ONE kokoro entry (announcer-only) so chars collapsed to
> one voice; registered the full English Kokoro-82M set (DOWNLOADED the 15 missing -> 28 on disk, 15F/13M)
> as char_voice entries so CastLock assigns DISTINCT gender-matched voices like Bark (`dee7d5a`+`f99af26`;
> also fixes the missing `bf_emma` announcer voice). **3D ROUNDTABLE CAMPAIGN CONVERGED (docs only, NO 3D
> code yet)** -- `docs/2026-06-20-mesh-stage-texturing/roundtable/`: the "plaster of paris" blob is NOT
> WorldMirror (not wired into OTR at all) -- it's **`mesh_stage`** rendering a geometry-only Hunyuan3D-2mv
> GLB through Blender WORKBENCH matcap (flat gray `(0.78,0.78,0.78)` when the mesh has no vertex colors).
> R1 architecture -> R2 coding-plan (image-AGNOSTIC stills) -> R3 hardened+build-pin-list -> R4 wiring all
> converged via the live panel (GPT-5.5+Gemini-3.1-pro+Grok-4.3, ~$0.41). **PoC = a single-view-TEXTURED
> mesh_stage hero beat (vertex-color projection in Blender) over a GENERATED background, bounded camera
> arc, ledger-driven (costly_choice_beat), no-portrait->still_parallax; CAMERA-ONLY motion (lip-sync/rig =
> the deferred character_3d lane, OUT).** Wiring = C1 (per-clip `bg_still_path` in the manifest; ZERO new
> links/widgets); the exact 3-function composite patch + a 4-item build-time PIN list are in
> `pass03_plan.md` (build) + `pass04_plan.md` (wiring). Build is sprint-ready + OPERATOR-GATED.
> **NEW OPEN TICKET -- audio-reactive bottom BARS don't show:** painted by OTR_SceneAwareScopes as the
> FLOOR (under the clip) + suppressed for portrait/un-probeable/credits. Fix = a dedicated ALWAYS-ON bars
> overlay at the POST stage, decoupled from scene-aware suppression, captions stay above. Plan:
> `docs/2026-06-20-audio-bars-fix/BARS_FIX_PLAN.md`. **RUNNING NOW (next window WATCHES it):** an
> ALL-NIGHT 864-word FRONTIER story soak on :8011 -- 4 frontier creative (Gemini/Grok/GPT/Opus) + Opus
> tech, indextts2 voice, z_image character stills (flat_still), DIVERSE seeds (a different news story per
> leg via the new `OTR_SOAK_DIVERSE_SEED=1` driver knob), 48 legs/8h cap; output
> `docs/2026-06-21-allnight-864-frontier/`. **RISK: indextts2 is the headless-fragile sidecar -- a prior
> leg crashed the server; the smoke-leg-first guard aborts cleanly if it fails -> FALLBACK = kokoro (now
> a deep 28-voice pool).** Opus-tech `~latest` throws transient OpenRouterModelGoneError(404) -> the
 writer gracefully SKIPS those slots (transport fail-soft), stories still write. **SOAK DONE -- 35 legs/8h
> (results docs/2026-06-21-allnight-864-frontier/story_soak_results.csv); GPU now FREE.**
> **>>> NEW OPERATOR DIRECTIVE + TOP-PRIORITY VISUAL FIXES (2026-06-20, docs/2026-06-20-visual-fixes/
> VISUAL_FIXES_PLAN.md):** DIRECTIVE -- ONLY HuMo(portrait)+maybe 3D use the VERTICAL portrait; EVERY other
> path uses LANDSCAPE 16:9. **BUG 1 (do FIRST):** character-beat stills look like bookends/scene shots, not
> characters -- commit `8bc5381` wrongly conditions flat_still/landscape engines on the 832x1216 portrait
> (pillarboxed -> radio-booth floor sides = the "radio booth images"). FIX = (a) render_driver: landscape
> engines NEVER use the vertical portrait (gate on `render_aspect`, portrait ONLY for HuMo/3D); (b) image
> phase MINTS a 16:9 CHARACTER shot for character beats (image-model agnostic) so the character shows
> full-frame. **BUG 2 (verify-first):** the soak log shows rolling credits + procgen DID render
> (`BUG-410 credits scroll restored 6441 frames`, PostUpscaleProcgenBlend ran) -- so verify the animated
> START/END title card on a real obs render BEFORE changing; if missing/covered in still-only, make
> titles+credits an ALWAYS-ON procgen overlay (engine-independent). **BUG 3:** burn the image/audio/LLM
> model detail ONTO the rolling credits (the forensic detail I added is only in the _treatment.txt
> sidecar). **BUG 4:** audio bars always-on overlay (docs/2026-06-20-audio-bars-fix/). BUGs 2/3/4 share the
> always-on procgen-overlay layer. **NEXT = (a) the VISUAL FIXES above (BUG 1 first, GPU-verify each), (b)
> BUILD the 3D PoC (pass03+pass04 + PIN list), (c) watch nothing -- soak is done.** S3 forward order (3D =
> item 5) UNCHANGED.

> **LATEST SESSION -- 2026-06-19 (CODER; STORY-QUALITY PHASE 1 BUILT + PUSHED + RE-SOAK PASS;
> HEAD `ee83a88` == origin/v2.0-alpha; full suite 4600 passed/33 skipped, Bug Bible 16/7/3,
> test_audio_byte_identical green).** Built the roundtable-converged Phase-1 plan
> (`docs/2026-06-19-story-quality-analysis/roundtable/R2/pass02_plan.md`) -- the news->story fix
> -- in 6 dependency-ordered chunks, each suite+BugBible+no-BOM+AST green, committed AND pushed
> per chunk: **Chunk 1 A1** (`a1b05bb`, neutralize the dead `editor_constraints` diagnostic;
> run_story_qa already gated-off); **Chunk 3 A4** (`96db9aa`, new pure `nodes/_otr_anti_loop.py`
> -- deterministic near-dup + 'What if...?' loop detection over voiced lines, announcer taglines
> exempt, wired into the spine as Stage 3.6 recomposing CHARACTER targets); **Chunk 2 A3/A2**
> (`9a792b6`, run the mechanical floor UNCONDITIONALLY after the critic + UNION into
> reroll_targets since run_story_critic returns clean() identically on failure; replace the
> reroll-exhaustion needs_full_rerun terminal-skip with REPAIR-THEN-SHIP -- log residuals to
> `meta.a2_ship_through`, fall through to the normal freeze, never refuse; cast_lock's structural
> PD1 halt left intact); **Chunk 4 B1 THE SPINE** (`17d2e39`, new `nodes/_otr_dramatic_state_llm.py`
> -- derive the opposed wants+question+ending from `meta["news"]` at the writer call site via
> structured_call on the resident technical slot, post-validator requiring >=1 news term, a
> deterministic news-templated fallback opposed BY CONSTRUCTION, and a turning-slot key_terms
> floor; replaces the `_DEFAULT_A/B_WANTS` boilerplate; the pure `_otr_dramatic_state` stays
> pure); **Chunk 5 A5** (`db710a0`, new pure `build_line_dramatic_fields` adapter maps the
> news-driven slot contract -> the composer's beat_objective/obstacle/turn/subtext on the
> compose_line path; the LIVE exchange composer already consumes the same contracts); **Chunk 6
> A6** (`ee83a88`, new pure `nodes/_otr_line_hygiene.py` -- scrub parentheticals + self-vocative,
> RECOMPOSE truncation: character->spine seam, announcer open/close->the dedicated announcer
> composer; spine Stage 3.7). 36 new unit tests across the 6 chunks. **All content-only inside the
> FIXED ledger {cast,lines,meta}; ZERO JSON edits (the prod workflow is unchanged -- the new code
> runs inside the existing wired OTR_LedgerScriptWriter + OTR_LedgerFreezeCascade nodes).**
> **RE-SOAK PASS** (headless :8011, real `otr_scifi_16gb_full.json`, bark, visualizer, bypass
> OFF): leg-1 (mistral-nemo) `dramatic_state_source=llm`, wants genuinely OPPOSED and about the
> news (UCLA academic integrity vs startup commercialization at LABEST 2026), **`_DEFAULT_*`
> ABSENT**, `freeze_verdict=frozen_with_doctor_edits` (SHIPPED, no refuse), `a2_ship_through`
> stamped (A2 proven). Details: `docs/2026-06-19-story-quality-analysis/RESOAK_RESULTS.md`.
> **DEFERRED to Phase 2 (NOT done):** shape-follows-story; the use_exchange exchange composer
> enhancements; reconstruction-test validator; best-of-N; physical deletion of the neutralized QA
> modules. **NEXT:** Phase 2 (per pass02 scope) + the operator close-read of the remaining small-
> local re-soak legs (gemma-2-2b/4-E2B/4-12b). The S3 build forward order (sections 1/3) is
> UNCHANGED by this work.


> **LATEST SESSION -- 2026-06-19 (STORY-QUALITY ROUNDTABLE R1 CONVERGED; docs only, NO code; HEAD `af26492`
> == origin; build forward order sections 1/3 UNCHANGED).** Ran R1 of the story-quality campaign end to end.
> (1) Acted as the FIRST PANELIST with a fresh head -> `roundtable/pass01_opus-grounded.md` (A-E), grounding
> every claim against the REAL Windows code + the soak corpus (Desktop Commander, not the lagging mount).
> CONFIRMED: critic anti-correlation (leg_0002 best -> `needs_full_rerun`; leg_0031 worst loop ->
> `frozen_with_doctor_edits`); `_DEFAULT_A/B_WANTS` boilerplate (the news reaches only the *question*, not the
> wants); uniform 18-line mold (`act_count="auto"` -> `default_act_count`=3 for ALL >=300w; ledger has NO act
> field); lexical-only `post_assembly_keyterm_check`; ancient-DNA->aliens drift (leg_0013); 12B usable (leg_0011).
> (2) Operator supplied 6 external panel replies (`docs/story_passes/pass_1/`: chatgpt/claude/gemini-deep/grok/
> perplexity/deepseek) -- **the panels did NOT see the workflow JSON**. Read them + the private plan and synthesized
> `roundtable/pass01_architecture.md` as the grounded judge (KEEP/CUT; hallucinations cut). **Operator steers
> folded in: minimal-change/activate-not-overhaul; 'acts' are NOT a limiter; best story authentic to news; FIX
> stories not FAIL them; model-agnostic story path (per-model only for technically-unique reqs); rip-out/repurpose
> any QA-that-only-scores-or-fails (a grounded §5b inventory).** Grounded wiring the panels couldn't: story stage
> = 2 nodes (`OTR_LedgerScriptWriter` + `OTR_LedgerFreezeCascade`), critic/spine/QA/contracts all INTERNAL to the
> writer; saved `use_exchange=false` + `act_count="auto"`; `slot_drama_contract` already derives per-line
> obligations (pulls news key_terms) but is unconsumed on the single-line path; the critic `clean()`-fallback
> silently returns "strong"+no-targets = SKIPS repair (the "passes the worst" bug); `_otr_story_brief.py` is the
> VISUAL reflection brief, NOT the news brief (corrects gemini/perplexity). CONVERGED PLAN = 3 fix-moves
> (news-derived opposed wants at `derive_dramatic_state_from_meta` / call site :2780 -> deliver via the EXISTING
> contract+composer path -> critic-as-repair-driver, never a gate) + bounded shape (secondary) + the §5b
> rip-out/repurpose. **R2 DONE/CONVERGED (this session):** ran a live 2-pass roundtable on the coding plan
> (Gemini-3.1-pro + GPT-5.5 + DeepSeek-v4-pro, grounded vs the real seams; ~$0.30 total) ->
> `roundtable/R2/pass02_plan.md` is the build-ready Phase-1 plan (6 chunks A1-A6 + B1, all content-only,
> zero-JSON-targeted, audio byte-identical). Key roundtable-grounded fixes: B1's LLM call goes at the WRITER
> CALL SITE (the pure helper stays pure) + a post-validator (>=1 news term across wants/question/ending);
> CUT `stake`/`hook` (DramaticState has neither); run A4 mechanical checks UNCONDITIONALLY + union (the critic
> returns clean() identically on failure); inject the news-failure turning-slot detail INTO `key_terms`
> (validate_contract needs it); repair-then-ship ordered selection (never refuse); announcer truncation ->
> the dedicated announcer composer; neutralize the dead QA in Phase 1, defer physical deletion. **NEXT = R3
> wiring/build (operator-gated) -> R4 Comfy -> R5 polish (campaign DONE only at polish).** Build forward order
> (sections 1/3) UNCHANGED.

> **LATEST SESSION -- 2026-06-19 (ANALYSIS window; story-quality SIDE analysis, NOT a build; HEAD `af26492`
> == origin; docs only, NO code; build forward order sections 1/3 UNCHANGED).** Ran the operator-requested
> story-quality analysis off the overnight soak. Fanned out 6 READ-ONLY subagents + an Opus close-read of
> the corpus extremes, all grounded vs the REAL Windows files (Desktop Commander, not the lagging mount).
> **FINDINGS:** (1) the craft architecture (opposed wants, per-slot turn contract, exchange composer, QA)
> is BUILT but ships DORMANT -- production runs only format hygiene + a fixed 3-act/18-beat mold for any
> episode >=300 words (`_otr_episode_budget.ACT_COUNT_CONFIG`); (2) the story critic is ~ANTI-correlated
> with craft -- it gave the BEST leg (0002, blind 35/35) `needs_full_rerun` and "passed" the WORST (0031,
> 7/35, a verbatim loop) `frozen_with_doctor_edits` (verified vs `_server8011.log`); (3) word-count
> variance (34-287%) is a SYMPTOM of the fixed mold, NEVER a target; (4) 26% of legs wrote 0 words = a
> throughput/harness failure, not craft; (5) **THE NEWS IS NOT THE CRUX** -- the news brief DOES run
> (`news_briefs_required` default True, so premises are news-derived), but the news is demoted to soft
> context: the central conflict (the opposed wants in `DramaticState`) is hardcoded `_DEFAULT_A/B_WANTS`
> that IGNORE the news, the brief becomes a 15-word "theme is flavor, not structure" string, and the only
> news check anywhere is a lexical ">=2 key-terms appeared" audit (`post_assembly_keyterm_check`).
> **>>> NEW OPERATOR HARD CONSTRAINT (2026-06-19, NON-NEGOTIABLE): the news story MUST be the CRUX of the
> drama** -- folded into the deliverables as constraint 4A (content-only; exact fix seam:
> `derive_dramatic_state_from_meta` + call site `OTR_LedgerScriptWriter.py:2779`, drive the opposed wants
> from `meta["news"]`; the ledger `{cast,lines,meta}` wire format stays FROZEN; audio spine byte-identical).
> **DELIVERABLES (docs only, `docs/2026-06-19-story-quality-analysis/`):** `PROBLEM_STATEMENT.md` (code-
> illustrated, story-quality framed, news-as-crux a hard constraint) + `roundtable/PANEL_PROMPT_R1.md`
> (self-contained, paste-ready blind-panel prompt) + `_PRIVATE_OPUS_PLAN_DO_NOT_PASTE_TO_PANEL.md` (held OUT
> of the panel) + `README.md`. **CAMPAIGN MECHANICS (operator-set):** the OPERATOR runs ALL panelists himself
> (latest models, his pick; GPT-5.5 self-run) from the paste-ready file -- ZERO OpenRouter spend by Claude.
> **>>> NEW OPERATOR DIRECTIVE (2026-06-19, "surprise") -- THE NEXT WINDOW STARTS HERE:** the fresh window
> opens by acting as the **FIRST PANELIST** -- with a FRESH HEAD, read `roundtable/PANEL_PROMPT_R1.md` and
> ground every claim against BOTH the REAL code AND the REAL soak corpus (the 20+ leg scripts in
> `docs/2026-06-18-overnight-story-soak/scripts/`; it is the ONE reviewer that CAN check the repo + the data,
> unlike the external blind models), and write its own grounded panelist answer to
> `roundtable/pass01_opus-grounded.md` BEFORE opening the private plan or synthesizing. **GROUNDING BAR
> (operator): the goal is a genuinely BETTER STORY -- do NOT propose rewrites for their own sake; a lever
> ships ONLY if soak+code evidence shows it actually makes the STORY better. If it wouldn't improve the
> story, don't propose it.** (This AMENDS the brief's "Opus is never a listed panelist" line -- operator
> override.) THEN: operator runs the external blind panels -> Opus reads all + the private
> plan and synthesizes `roundtable/pass01_architecture.md` -> Rounds 2 (coding) / 3 (wiring) / 4 (Comfy
> workflow) / 5 (polish). The `/otr-handoff` for this analysis is DONE only at polish. The build forward
> order (sections 1/3) is UNCHANGED and resumes after the campaign.

> **LATEST SESSION -- 2026-06-19 (PLANNER/SOAK; HEAD `af26492` == origin; suite green, Bug Bible 16/7/3).
> SHIPPED + PUSHED earlier this session (each suite-green): bark runaway-clip cap + trailing-pad trim
> (`cc6bacc`, the caption-fit fix -- bark was over-generating ~12s for 16 words) + the `default_clean`
> commercial-clean cast voice bank (`af26492`, routes the char engine to chatterbox/dia, excludes the
> non-commercial indextts2). Also ran the voice-engine roundtable (converged) + banked the curated voice
> cast. THEN ran a 10-HOUR OVERNIGHT STORY-QUALITY SOAK (operator-requested data-gathering run, NOT a
> build): 35 legs, 13 writer LLMs (7 OpenRouter ~latest + 6 local), 320/400/500-word targets, news-driven
> premises, all-visualizer, dedicated :8011 server. Output: `docs/2026-06-18-overnight-story-soak/`
> (story_soak_results.csv/.json + 28 per-leg script .txt). Soak ended cleanly at the 10h cap.
> **>>> THIS HAND-OFF OPENS A SIDE ANALYSIS WINDOW** (does NOT change the build forward order): a
> STORY-QUALITY ANALYSIS that fans out read-only subagents over the soak corpus + the ledgers + the
> story-gen architecture, forms its own judgment, and produces a PROBLEM STATEMENT for `/roundtable`.
> **OBJECTIVE = STORY QUALITY (craft), NOT word count (operator 2026-06-19: "I don't want to chase word
> count, I want to chase story quality -- hard for a computer to decide but do your best").** The analysis
> LEADS with a qualitative close-read of the 28 scripts (+ an LLM-judge rubric + the pipeline critic).
> Supporting signals (diagnostics, NOT targets): n_lines pinned at ~18 regardless of length = one-size
> story SHAPE + the caption-fit root; story critic 0/19 clean; word-count adherence wild (34-287%).
> Full brief + the subagent plan + the HARD ledger-intact constraint (the `{cast,lines,meta}` schema is
> FROZEN downstream; audio is the first consumer):
> `docs/2026-06-19-story-quality-analysis/STORY_QUALITY_ANALYSIS_BRIEF.md`.
> **The handoff completes via a MULTI-ROUND ROUNDTABLE CAMPAIGN (operator design, brief section 5A):** a
> broad, CODE-ILLUSTRATED problem statement -> exactly 3 TOP-TIER panelists judging architecture BLIND
> (Opus holds its own best plan privately + is the sole judge/synthesizer), then a SEQUENCE of rounds --
> R1 architecture/creative approach, then coding plan, wiring, Comfy workflow, polish. The `/otr-handoff`
> is DONE only when the campaign runs through polish + the final converged plan is recorded. The S3 build
> forward order
> (3D item 5, distribution item 6) + the carried per-segment LUFS/RMS plan are UNCHANGED by this analysis.

> **LATEST SESSION -- 2026-06-18 (CODER; LUMINA-IMAGE 2.0 PROMOTED TO A VALIDATED PEER IMAGE ENGINE,
> GPU-VERIFIED; 2 commits on `v2.0-alpha`, PUSHED; HEAD `631d0b0` == origin; suite green [2 pre-existing
> test_model_catalog_scan environmental reds unchanged], Bug Bible 16/7/3, audio byte-identical green).**
> Operator ask: "code the luminous / low-VRAM image engine -- the scaffolding should already be there."
> AUDIT FINDING: the scaffolding was MORE than there -- `lumina_image` was already a COMPLETE, registered,
> cold-import-clean adapter (real `render_image`, fail-closed `assert_usable`, CAPABILITIES row, 7 CPU tests,
> a dep-pilot row, weights on disk). The ONLY gap was the tested-only dropdown gate: it was hidden pending a
> GPU smoke. So this was a verify-and-promote, not a build.
> **GPU SMOKE (RTX 5080, fresh FLOOR-lane headless server):** the EXACT engine recipe (UNETLoader
> lumina_2_model_bf16 + CLIPLoader[type=lumina2] gemma_2_2b TE + VAELoader lumina2_ae ->
> ModelSamplingAuraFlow -> KSampler -> VAEDecode) minted a real 1216x832 still in 23.5 s -- `model_type FLOW`,
> TE 4986 MB + diffusion 4977 MB staged SEQUENTIALLY, resident peak ~12.2 GB << 14.5 GB ceiling (and the
> engine's `render_image` reclaims after decode). The minted PNG is a real varied image (std 40, not flat).
> Box reset clean (GPU baseline 1.69 GB). **Commit `ed560a0`:** add `lumina_image` to `registry.VALIDATED_ENGINES`
> (now lists in the OTR_ImageDirector per-role dropdown via `validated_engine_names`); measured-VRAM note on
> the CAPABILITIES row; `test_tested_only_dropdown_gate` moves lumina out of `_HIDDEN_IMAGE` into the validated
> set. **Commit `631d0b0`:** version `scripts/_otr_lumina_image_smoke.py` (the reusable image-engine GPU smoke
> -- the counterpart to the video-only `_otr_single_engine_smoke.py`; routes the still through the `_otr_paths`
> authority so the output-tree contract stays clean). The validated IMAGE set is now flux_gen1 (default) +
> z_image_turbo + flux2_klein + lumina_image. **>>> NEXT = the S3 forward order** (3D item 5, distribution
> item 6) + the carried per-segment LUFS/RMS normalization plan (frozen-spine, roundtable-first), UNCHANGED by
> this image-lane add.

> **LATEST SESSION -- 2026-06-18 LATE (CODER; VIDEO CLIP-FILL / DYNAMIC-VRAM SHIPPED + GPU-VERIFIED;
> + the QUEUED green bottom-bars overlay SHIPPED; 4 commits on `v2.0-alpha`, NOT pushed (operator gates);
> suite 4554/0, Bug Bible 16/7/3).**
> The wan_ti2v static-episode bug is FIXED at root (no shims) -- spec was
> `docs/2026-06-18-video-clip-fill/CLIP_FILL_HANDOFF.md`; results in `GPU_VERIFY_RESULTS.md`.
> **Commit `c9bf0ab` (clip-fill, 5 pieces):** (1) `motion_common.compute_real_frame_budget` + `free_vram_mb`
> PREDICT the VRAM-affordable 4n+1 frame count from a zero-cost `torch.cuda.mem_get_info` read + a cost model
> (wan seed 7000MB + 185MB/frame @1472x832, per-frame scales with pixel area; budget = min(free,ceiling)*0.85,
> capped at the beat target, floored at the motion floor) -- NEVER react-to-OOM. (2) `eng_wan_ti2v._floor_length`
> calls it (the hard 17-frame cap is gone; `OTR_WAN_TI2V_MAX_FRAMES` is now an absolute hard cap). (3)
> `wrapper_bridge.extend_frames_to_target` seamless ping-pong-extends the short render to the beat target in
> `render_clip` (LTX boomerang untouched -> byte-identical). (4) `render_driver.persist_episode_clips` +
> `_otr_paths.otr_clips_dir` move final clips out of the swept `_shared/tmp` into durable `episodes/<ep>/clips/`
> (wired in OTR_VideoRenderBatch before the manifest). (5) `otr_silent_composite._warn_clip_underrun` LOUD-warns
> (never raises) when a real clip is far shorter than its target. **Commit `8ce9004` (pre-existing red fix):**
> `test_story_brief_c5a1::test_schema_caps_string_v2_fields` still expected reject+retry; structured_call now
> CLAMPS an over-long capped field in place (4c0e943) -- updated the assertion (this was failing at HEAD `4d8c2ba`,
> unrelated to clip-fill). **GPU SMOKE (5080, all-roles wan_ti2v 60w act=1 bark): SUCCESS.** 6/6 beats clip-filled
> with ADAPTIVE budgets (29->238, 25->290, ...; the budget shrank 29->25 across beats = live VRAM read working),
> every render peak 8610-9988 MB << 14500 ceiling (no OOM), persisted 6 clips to `episodes/<ep>/clips/`, a beat clip
> is 290 frames/11.6s with 3 distinct frames at t=2/3.5/5s (motion fills the beat, freeze GONE),
> `audio_byte_identical OK`, obs_publish OK.
> **BARS OVERLAY ALSO SHIPPED (commit `13ceeae`):** the QUEUED optional green audio-reactive bottom-bars overlay
> (operator-requested "after the gpu smoke") -- a composite-stage add to `OTR_SceneAwareScopes` (NOT a new engine):
> new `landscape_bars` widget (off DEFAULT|bottom, appended last per BUG-LOCAL-097, wired into node 94 of the real
> JSON same-commit); `bottom` paints a GREEN-ONLY freq strip (shared `scope_draw.freq_bars_green`) along the bottom
> of any LANDSCAPE clip, ABOVE the lower-15% caption safe-area (captions never occluded); `off` = byte-identical.
> Verified on the REAL persisted wan clips: off->all suppressed, bottom->all landscape clips get bars; strip ends at
> y=918 = caption-safe-area top. 7 tests. **>>> NEXT = the S3 forward order** (3D item 5, distribution item 6) +
> the carried per-segment LUFS/RMS normalization plan (frozen-spine, roundtable-first). Box reset clean (GPU baseline).
> Push is operator-gated (4 commits: `8ce9004`+`c9bf0ab`+`11d0e9a`+`13ceeae`).

> **LATEST SESSION -- 2026-06-18 EVENING (resilience + image curation; HEAD `3384418` == origin; suite green,
> Bug Bible 16/7/3). ACTIVE NEXT STEP = the VIDEO CLIP-FILL / DYNAMIC-VRAM build (spec:
> `docs/2026-06-18-video-clip-fill/CLIP_FILL_HANDOFF.md`).**
> SHIPPED + PUSHED to `v2.0-alpha` this session (each suite-green): **(a)** OpenRouter resilience -- a DELETED model
> (gone-404) now falls over to the slot's remote fallback ONCE, loudly, instead of aborting the episode
> (`OpenRouterModelGoneError`); **(b)** fixed the `KeyError 'openrouter:slot-a'` that aborted a run when the CREATIVE
> writer was a remote slot (creative_prompt_router `.get` -> modern default for non-curated ids); **(c)** OpenRouter
> A/B writer slots now show TEXT-output models ONLY (cache captures `architecture.output_modalities`; image gens like
> gemini-3-pro-image hidden); **(d)** structured_call CLAMPS an over-long capped field (e.g. outline `time_of_day`
> max 40) to its max_length instead of aborting; **(e)** the style inventor PADS a weak-model near-miss (4-of-5
> descriptors) up to 5 from a deterministic stock pool instead of aborting (operator: "no loud fail"); **(f)**
> wan_ti2v is now eligible for ALL roles (incl. announcer) + fixed flat_still's missing CAPABILITIES row; **(g)**
> DROPPED the chroma_hd image engine (`3384418`, operator values call: Chroma is a de-restricted/uncensored FLUX
> finetune -- OTR will not ship that path). All graceful-degradation "floors" follow the 3-source roundtable
> (do NOT use an LLM to repair LLM output; Python deterministic floor; LTX path byte-identical).
> **IMAGE LANE:** the 3 most-downloaded 16GB ComfyUI image models (FLUX.1-dev=flux_gen1, Z-Image-Turbo=z_image_turbo,
> FLUX.2-Klein=flux2_klein) are ALL already wired + validated. `lumina_image` is the easy SFW add NEXT -- code is
> complete (real render_image) and all its files are ALREADY on disk (`lumina_2_model_bf16` + `gemma_2_2b_fp16` +
> `lumina2_ae`), just needs `OTR_ENABLE_LUMINA=1` + a GPU smoke + promotion to `VALIDATED_ENGINES`.
> [DONE 2026-06-18: `lumina_image` GPU-smoked + promoted to `VALIDATED_ENGINES` -- commits `ed560a0`/`631d0b0`;
> see the LATEST SESSION block at the top. The validated IMAGE set is now flux_gen1 + z_image_turbo +
> flux2_klein + lumina_image.]
> **>>> ACTIVE NEXT = the video CLIP-FILL bug:** a wan_ti2v episode renders STATIC (only the procgen overlay moves).
> Root cause (code-grounded): `eng_wan_ti2v._floor_length` clamps every clip to a hard 17-frame "8GB floor"
> (`_TI2V_FLOOR_MAX_FRAMES=17`), ignoring the shot calculator's audio-derived per-beat `target_frame_count` (~280);
> the composite then holds-last-frame -> 0.68s motion + ~9s freeze. ALSO clips write to swept `_shared/tmp` (no
> persistent `clips/` folder). The roundtable-converged fix (5 pieces: dynamic-VRAM PREDICT frame budget on
> MotionEngineBase, wan honors it, loop/ping-pong-extend to target, persist clips to `episodes/<ep>/clips/`, composite
> LOUD underrun guard) is fully specified in `docs/2026-06-18-video-clip-fill/CLIP_FILL_HANDOFF.md`. BUILD THAT NEXT.
> **QUEUED FOLLOW-UP (gated on the visualizer engine being committed + pushed green):** the optional GREEN
> audio-reactive BARS overlay (old-school bottom-of-screen, over ANY engine's final video) -- a COMPOSITE-STAGE add
> to `OTR_SceneAwareScopes` + `OTR_PostUpscaleProcgenBlend` (NOT a new engine). One new `landscape_bars =
> off(DEFAULT)|bottom` widget (off = byte-identical; APPEND at end of widgets_values, BUG-LOCAL-097); landscape clips
> return a new `("bars",x,y,w,h)` bottom-strip mode instead of None; paint green-only via the shared
> `_otr_shared/scope_draw.py` (DRY); HARD: captions MUST layer ABOVE the bars (caption burn ~Node 58 is the LAST
> composite step + keep bars below the lower ~12-15% caption safe-area + a test asserting caption-after-bars); wire
> the widget into otr_scifi_16gb_full.json same-commit + re-validate. Full spec:
> `docs/2026-06-17-scope-visualizer-engine/CODER_KICKOFF_BARS_OVERLAY.md`.

> **LATEST SESSION -- 2026-06-18 (CODER; flux2_klein VERIFIED + PROMOTED + coverage-arch accepts_still SHIPPED;
> HEAD `64c14bb` == origin; suite 4500/33, Bug Bible 16/7/3). flux2_klein is DONE.**
> Two commits pushed to `v2.0-alpha`:
> **(1) `2cb25fb` coverage-arch + flux2_klein TE fix.** Root-caused the flux2_klein `mat1/mat2 (512x15360 @
> 7680x3072)` error: klein-4B uses the **Qwen-3-4B** encoder (7680-wide), NOT flux2-dev's Mistral (15360, 2x).
> Found the klein-matched TE via the official ComfyUI `image_flux2_klein_text_to_image` template; downloaded
> `qwen_3_4b.safetensors` (8 GB, `Comfy-Org/flux2-klein`) into `C:\ComfyUI-Models\text_encoders`; engine default
> TE -> qwen. Built the **coverage architecture** (operator: "all video/3D accept whatever image is selected, one
> place, no per-model whitelist"): `accepts_still` capability -- `MotionEngineBase` default True (every motion lane
> takes the selected still), `ltx_av_music`/`visualizer` opt-out False -- read centrally by `engine_consumes_still`
> in the image dispatcher (dual-read fallback to init_image-in-required_inputs; bare except -> LOUD). This makes
> silent `ltx_video` consume the flux2 still (it was skipped before). 2-model roundtable (gpt-4.1 + gemini-pro-latest
> via DIRECT APIs -- OpenRouter launcher stalled; keys in HKCU User env), grounded in
> `docs/2026-06-18-coverage-arch-wiring/`. **(2) `64c14bb` PROMOTION + creativity dial.** GPU-VERIFIED on the 5080:
> a full flux2->silent-LTX episode (all image roles=flux2_klein, all video=ltx_video, 30w, **maximum-chaos**
> creativity, bark) minted ALL 6 stills end-to-end -- `[OTR.image.flux2_klein] minted still 832x480/1472x832
> steps=20 guidance=4.00`, qwen TE staged 7671 MB, NO dim error; the LTX clips then rendered FROM those flux2 stills
> (the operator's "flux2 images on LTX"). Added `flux2_klein` to image `VALIDATED_ENGINES` (opt-in via
> `OTR_ENABLE_FLUX2_KLEIN`), out of `_HIDDEN_IMAGE`. Added `creativity` to `CREATIVE_WHITELIST` so a soak can set the
> writer dial. Box reset clean between runs.
> **>>> NEXT = the coverage-arch FOLLOW-UPS (deferred, additive; docs/2026-06-18-coverage-arch-wiring/pass01_plan.md):**
> optional_inputs for role_compat honesty (verify-at-build); optionally set accepts_still=True on the static-still
> cheap families (flux_still/station_card/still_kenburns) so they also show the SELECTED image; full Decision-3/5
> (central usable(), retire requires_mesh_portrait onto still_kind). THEN the S3 forward order (3D item 5,
> distribution item 6) + the carried per-segment LUFS/RMS plan.

> **LATEST SESSION -- 2026-06-18 (CODER; flux2_klein BUILT + image-pick DEDUP + coverage-arch roundtable;
> HEAD `c854406` == origin; suite green; Bug Bible 16/7/3). PRIORITY NEXT STEP = finish flux2_klein (klein-4B).**
> Four committed + PUSHED to `v2.0-alpha`:
> **(1) flux2_klein engine BUILT** (`36dc01a`): FLUX.2 [klein] 4B is a real image engine (official ComfyUI flux2
> recipe -- UnetLoaderGGUF + CLIPLoader[type=flux2] + FluxGuidance->BasicGuider + Flux2Scheduler +
> SamplerCustomAdvanced + VAEDecode). Apache-2.0 (commercial_clean=True). Graduated out of the stub PEERS matrix
> into its own suite `tests/test_flux2_klein_engine.py`. Stays HIDDEN/opt-in until a green GPU render promotes it.
> **(2) Image-MODEL selection collapsed to ONE place** (`b8bb388`, operator "only in one place not two"):
> removed the 3 duplicate image dropdowns from OTR_ImageDirector; it now sources picks SOLELY from
> `video_policy["image_models"]` (OTR_VideoDirector is the single home). Edited the real source-of-truth JSON node 88
> in place + widget_mapping.json (image keys -> VideoDirector only). Added `OTR_COMBO_*_IMG` env to the soak harness.
> **(3) Coverage-architecture roundtable** (`50200fd`, operator "all video accepts all images, approval in one
> place"): grounded pass01 synthesis in `docs/2026-06-18-image-video-coverage-arch/` -- add `accepts_still`/
> `still_kind` capability fields to video adapters (NOT a new type), ONE central `image_engines.registry.usable()`
> approval surface the directors+dispatcher share, dual-read migration (required_inputs wins for existing names),
> unify the 3D mesh-portrait lock onto `still_kind`. (4/6 panel models errored on token budget; gemini+grok gave
> grounded critiques -- re-run with raised max-tokens for the rest.)
> **(4) flux2_klein GPU VERIFY -- FAILED, diagnosed** (`c854406`): a full flux2+LTX episode (announcer/beats=
> ltx_av_talk, music=ltx_av_music, all image=flux2_klein, 30w act=1, bark) found two issues: (a) FIXED -- weights
> were in `Documents\ComfyUI\models` but the headless server scans `C:\ComfyUI-Models` (moved all 3 there);
> (b) OPEN -- the sampler raises `mat1/mat2 (512x15360 @ 7680x3072)`: the klein-**4B** GGUF is paired with the WRONG
> text encoder (I reused flux2-**dev**'s `mistral_3_small_flux2_fp4_mixed`, whose conditioning is 2x too wide for
> the 4B UNet). klein-4B needs its OWN ComfyUI-matched encoder. FULL RESUME RECIPE +
> the proven combo-soak invocation: `docs/2026-06-18-flux2-klein/VERIFY_ON_5080.md`. Box reset clean (GPU baseline).

> **TASK 2 PROGRESS (`4a92ed6`; HEAD afa6bf1 code): visualizer-all-roles soak via `_otr_combo_soak.py` (forces bark,
> clearing the headless-audio gap) found + fixed FOUR visualizer integration bugs -- all pushed + suite-green:
> `d460797` assert_usable no longer pre-gates audio_ref; `bad1bba` render_driver feeds b000 the master-audio slice;
> `c5c14c9` idle scopes on silent beats; `afa6bf1` 0-frame beats default to 1s. The visualizer rendered 21 real
> clips across the soaks and is now robust to every beat type (real / silent / zero-frame). A single status=success
> episode is NOT yet captured -- the confirming soak died in the WRITER's style-inventor (a transient LLM flake,
> UNRELATED to the visualizer). NEXT: one more visualizer-all-roles soak (try 30-60w to dodge the writer flake) ->
> status=success -> ADD `"visualizer"` to `registry.VALIDATED_ENGINES` (+ default-ON). Then sweep the other 8 video
> + 2 image validated engines via `_otr_combo_soak.py` (force bark). Full detail: SMOKE_SWEEP_RESULTS.md. Box reset
> clean (GPU baseline; temp visualizer extra-env deleted).**
>
> **TASK 2 EARLIER (`68b0e31`): full-episode GPU smoke sweep is BLOCKED on a headless-AUDIO env gap.** A
> visualizer-all-roles attempt aborted in the AUDIO phase (`RuntimeError: IndexTTS2 Path B not installed` -- the
> default char voice's isolated venv is absent on the ComfyUI-Installs box) BEFORE any video engine ran; this blocks
> EVERY full-episode leg. `queue_smoke.py` renders the as-saved indextts2; only the SOAK HARNESS forces
> `OTR_SOAK_CHAR_VOICE=bark`. UNBLOCK: (a) install indextts2 headless (`scripts\_otr_indextts2_install.ps1`) then
> sweep via queue_smoke, OR (b) sweep via `scripts/_otr_combo_soak.py` (forces bark). Visualizer render path is
> unit-proven (17 tests + a real ffmpeg render); promotion to VALIDATED_ENGINES stays gated on a green full-episode
> E2E + audio byte-identical. Verified sets: VIDEO 8 + IMAGE 2 (see SMOKE_SWEEP_RESULTS.md). Box reset clean; the
> temporary visualizer force-map `_marathon_extra_env.cmd` was DELETED so future boots are clean.
>
> **LATEST SESSION -- 2026-06-18 (CODER, autonomous overnight; HEAD `236db0e` == origin; suite 4496 pass / 33 skip,
> Bug Bible 16/7/3):** A long multi-thread session, all committed + PUSHED to `v2.0-alpha`:
> **(1) Still-aspect correctness + HuMo dropdown labels** (`55d9dad`). **(2) Bark whiny-voice fix S1-S4** + preset
> audition (`215738c`,`6a20c9f`). **(3) wan_ti2v promotion + render_single aspect fix** (`ca3e06c`). **(4) wan_ti2v
> 8GB-floor hardening, 3-pass roundtable-converged** (`00da690` Chunk1: 17-frame floor+clamp / euler-only sampler
> whitelist / range-checked resolver / VAE allow-list / 0.1s probe; `1587c31` Chunk2: CLIP off fp8 -> GGUF umt5 [fp8
> is Mac-MPS-broken, #9255] + tiled VAE; docs `d528084`). **(5) VISUALIZER ENGINE v1 SHIPPED** (`236db0e`, Task 1):
> the resurrected full-colour procedural CRT scope engine (engine_id `visualizer`) -- low-VRAM ffmpeg-only per-beat
> picture (ring/particles/grid/waveform/bars/CRT post), torch-free draw routines COPIED into
> `nodes/_otr_shared/scope_draw.py` (zero coupling to the floor node / SceneAwareScopes overlay), adapter
> `eng_visualizer.py` (family=abstract, has_audio=False, NO fallbacks). PLAN-PREMISE FIX: `visualizer` was already a
> cheap floor stub (`VisualizerFamily`); the new engine SUPERSEDES it (stub deleted, removed from FLOOR_NAMES +
> cheap-render tests, dep-pilot row added). 17 tests + a real CPU ffmpeg render.
> **>>> NEXT = TASK 2 (operator-requested, GPU/overnight): the 120-word verified-model smoke sweep.** Enumerate the
> validated VIDEO set (`registry.validated_engine_names()` = ltx_video, ltx_av_music, ltx_av_talk, humo, humo_1.7B,
> humo_1.7B_169, humo_14B_169, wan_ti2v) + the validated IMAGE set (OTR_ImageDirector: flux_gen1, z_image_turbo)
> PROGRAMMATICALLY; run a 120w full-pipeline smoke per engine on the REAL `otr_scifi_16gb_full.json` (random
> OS-entropy seed), RESET the box before EACH (CLAUDE.md sec 4), INCLUDE one `visualizer`-all-roles run, verify
> `test_audio_byte_identical`, write `docs/2026-06-17-scope-visualizer-engine/SMOKE_SWEEP_RESULTS.md` (engine,
> pass/fail, time, VRAM peak vs ceiling). **THEN: if the visualizer all-roles E2E is green, ADD `"visualizer"` to
> `registry.VALIDATED_ENGINES`** (+ decide default-ON) so it lists in the dropdowns. NOTE: wan_ti2v's recipe changed
> in the floor hardening (euler/17-frame/GGUF-umt5/tiled) -- its sweep leg double-checks the hardened floor renders
> + fits. Fail LOUD on any regression; reset per CLAUDE.md sec 4 between legs. Box currently free (GPU baseline).
>
> **LATEST SESSION -- 2026-06-18 (CODER; BARK WHINY-VOICE FIX S1-S4 SHIPPED; HEAD `6a20c9f`; NOT pushed yet;
> suite 4462 pass / 33 skip, Bug Bible 16/7/3):** The sanctioned upstream-TTS audio work
> (`docs/2026-06-17-bark-voice/BARK_VOICE_UPDATE_PLAN.md`) is DONE across two commits. **`215738c` (S1+S2+S4):**
> the thin/whiny Bark timbre came from a single flat `temperature` over-randomizing the acoustic stages. Bark is a
> 3-stage pipeline (semantic -> coarse -> fine); the fix sets each stage independently (semantic 0.7 warm for
> content commitment; coarse/fine 0.5 to firm up the acoustics) and routes them via the transformers
> `BarkModel.generate` PREFIXED-kwargs contract (`semantic_temperature`/`coarse_temperature`/`fine_temperature` --
> prefixed wins, routes to that sub-model; NO generation_config mutation -> no cross-voice leak; transformers 5.5.0
> probed). `_generate_single_line` gained keyword-only stage temps (back-compat: legacy `temperature=` still works,
> each None stage inherits it -> the orchestrator health probe is untouched) + a pure `_stage_temps_for_line`
> helper that applies the intl (0.55) and first-line (0.6 en / 0.5 intl) CAPS to the SEMANTIC stage only (a
> ceiling, never a floor; coarse/fine keep profile values). `eng_bark` resolves the 3 temps from char_bark_v1 via a
> precise alias ladder (honors explicit 0.0; class-attr fallback 0.7/0.5/0.5). **S4:** verified EpisodeAssembler
> already peak-normalizes per segment + a final -1.0 dBFS master limiter post-crossfade, so `normalize_bark_output`
> stays false (a Bark-side normalize would double-hit); residual "low" is RMS/LUFS, deferred. **`6a20c9f` (S3):**
> rendered the audition on the 5080 across v2/en_speaker_0..9 -- en_speaker_5 (LUFS -14.3) + en_speaker_0 (-16.8)
> are the fullest/least-thin by ~10 dB; added `recommended_speakers=[5,0,6]` to char_bark_v1 as DATA ONLY (no
> auto-filter consumer; casting enforcement is a future sprint). `scripts/bark_preset_audition.py` is the reusable
> GPU harness (CSV + manifest committed; WAVs local-only). **NOTES:** no unit-level byte-identical fixture to
> re-baseline (indextts2 is the char_voice default; a forced-Bark re-baseline is an operator GPU action). The
> render verdict ("warmer/fuller, not whiny") still wants the operator's EARS on the audition WAVs in
> `%TEMP%\bark_audition\`. Comfy Desktop was killed for the audition -- RESTART it to load the new Bark code. NOT
> pushed (operator gates). The S3 forward order (Wan smoke / 3D / distribution) is UNCHANGED.
>
> **LATEST SESSION -- 2026-06-18 (CODER; STILL-ASPECT CORRECTNESS + HUMO LABELS SHIPPED + PUSHED; HEAD `55d9dad`
> == origin; suite 4443 pass / 32 skip, Bug Bible 16/7/3):** The current-step plan
> (`docs/2026-06-17-still-aspect/STILL_ASPECT_AND_LABELS_PLAN.md`) is DONE in one commit `55d9dad`.
> **GOAL A (still-aspect):** every registered video/3D engine now declares an EXPLICIT `render_aspect` in
> {portrait, wide} -- the portrait fallback is no longer load-bearing. ROOT-CAUSE FIX: `ltx_video` declared none ->
> silently defaulted portrait -> the 16:9 render decapitated heads; now `render_aspect="wide"` (class attr only,
> the frozen sampler/sigma chain untouched). Also wide: the cheap-families base (abstract/still_kenburns/
> station_card/visualizer/flux_still + still_parallax/mesh_stage via the base), `triposr`, `wan_i2v`, `wan_ti2v`.
> Stay portrait: `humo`, `humo_1.7B`. Stay wide: `humo_1.7B_169`, `humo_14B_169`, `ltx_av_*`. The three
> `requires_mesh_portrait` 3D talkers (`triposg_talk`/`hunyuan3d_talk`/`trellis_talk`) get `render_aspect="portrait"`
> (they feed the mesher a portrait still even when the final video is 16:9 -- the one exception). **GOAL B (labels):**
> the OTR_VideoDirector dropdown shows an aspect-DERIVED label (`humo (portrait)` vs `humo_1.7B_169 (16:9)`); the
> suffix is generated from `render_aspect` so it never drifts. The saved/looked-up VALUE stays the BARE engine id:
> `direct()` parses the token before the first `" ("`, and a bare legacy value + the ADD_CUSTOM sentinel pass
> through unchanged. The applier `_is_engine_director_admissible` strips the same suffix before the registry check
> so a fresh labelled save also applies. **NO workflow-JSON change** -- the bare saved values resolve via the
> back-compat parse (a trial JSON relabel broke 4 production-pin tests; reverted -- the JSON stays bare by design).
> New `tests/test_still_aspect_and_labels.py` (13) + the tested-only dropdown gate updated to parse labels.
> **>>> NEXT STEP = the BARK "whiny/low" voice fix** (`docs/2026-06-17-bark-voice/BARK_VOICE_UPDATE_PLAN.md`, the
> sanctioned upstream-TTS audio work). The S3 forward order (Wan smoke / 3D / distribution) is unchanged.

## 1. CURRENT STEP

**>>> NO ACTIVE SPRINT (2026-06-22 hand-off). The bark-artifact, stage-direction-leak, and Story-Quality R2
sprints are ALL COMPLETE + pushed (HEAD `08010ec`); suite 4934 pass/34 skip, Bug Bible 16/7/3. NEXT = the
OPERATOR picks the next forward-order item -- the S3 forward order is UNCHANGED: 3D (item 5, detail spec
`docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`; 3D v1.5 lit/textured tier was deferred), distribution
(item 6), + the carried per-segment LUFS/RMS normalization plan -- all PARKED + untouched. One carried
verify item: the live `test_audio_byte_identical` baseline may need recapture (R2 prompt levers intentionally
change generated dialogue; the clean indextts2 fixture is a no-op). Do NOT auto-start a sprint -- wait for GO.**

**>>> STORY-QUALITY R2 -- COMPLETE (2026-06-22). All 8 craft levers + the Final-QA scan shipped + pushed
to v2.0-alpha, each suite+Bug-Bible green; suite 4934 pass/34 skip.** S1 music-text suppression (`5396c05`)
-> S2 announcer-close = concrete final image, not thesis (`118088f`) -> S3 cliche + flat stage-business
reroll gate (`1887803`) -> C0 action-verb beat intents + wants_are_default classifier (`687f766`) -> C3
contrasting speech_signatures at CastLock (`981db60`) -> C4+C5 per-act escalation + on-the-nose reroll
(`3e19906`) -> C1+C2 specificity anchors + central story-object, 3-pass roundtable-converged (`38c2c10`) ->
Final QA: `scripts/story_quality_scan.py` extended with the lever metrics (thesis/cliche/stage-business/
on-the-nose/leading-direction counts + wants_default + anchors/central-object presence + voice distinctness).
HARD held throughout: ledger schema `l3-2026-05-14` FIXED (new values ride free-form `meta`), content-only,
NO workflow-JSON change, model-agnostic (every gate is one a strong/opus line passes -> lifts the weak end),
reuse the existing reroll loop. The whole craft lift is in `_otr_line_hygiene` / `_otr_line_composer` /
`_otr_outline` / `_otr_casting` / `_otr_dramatic_state` / `_otr_specificity` + the writer wiring -- no node/
JSON change. Campaign docs in `docs/2026-06-22-story-quality-r2/` + `docs/2026-06-22-story-quality-c1c2/`.
OPERATOR VERIFY-AT-BUILD: the live `test_audio_byte_identical` baseline (indextts2) may need recapture since
the prompt levers intentionally change generated dialogue -- the clean fixture is a no-op, but a fixed-seed
baseline close that newly trips a gate would change; run the live gate + re-soak read at your convenience.

**>>> STAGE-DIRECTION LEAK SPRINT -- COMPLETE (2026-06-22). All 5 chunks shipped + pushed to v2.0-alpha;
3-pass roundtable-converged (architecture/coding/wiring, ~$0.51).** Bare undelimited stage directions
("twirls his pen nervously Look, Pinky...") were leaking into the frozen ledger text -> Bark spoke them +
SDH captions showed them (every existing scrub is delimited-only). Fix = a fallback ladder, all
content-only (ledger schema frozen, audio byte-identical, no workflow-JSON, model-agnostic): Chunk 1
`8c40182` pure `scrub_leading_stage_direction` + `detect_leading_stage_business` in `_otr_line_hygiene`
(narrow conjunction-of-guards; +62-case corpus) -> Chunk 2 `e2dd95a` `scripts/stage_direction_scan.py` +
PRECISION GATE (489 ledgers; 20 would_mutate, ALL true positives, 0 false positives) -> Chunk 4 `2278bd2`
the FREEZE floor in `_otr_ledger_scrub._strip_stage_directions` (the deterministic guarantee; preserves the
Tuple[str,bool] contract, emits CODE_STAGE_DIRECTION, restamps word_count) -> Chunk 5 `9142b2f` composer
prompt hardening + the S1 music patch (`OTR_LedgerScriptWriter` 3681: `(beat.sfx_cue or beat.intent or "")`
-> `(beat.sfx_cue or "")`, which had been silently undoing S1) -> Chunk 3 `6ce724d` the PRIMARY reroll wiring
in `_otr_line_composer.compose_line` (detect on the raw draft -> one reroll via the existing recursive-repair
pattern, hint concatenated, guard-capped, freeze floor as backstop). FINAL QA: the fixed freeze strips the
exact screenshot lines (b004/b006/b007); a fresh max-chaos render shipped a CLEAN ledger (0 leaks); suite
4861 pass/34 skip, audio byte-identical 9/0, Bug Bible 16/7/3. Campaign + judgments in
`docs/2026-06-22-stage-direction-leak/`. **>>> RESUMES: STORY-QUALITY R2 -- S1 SHIPPED (`5396c05`); NEXT =
S2 (announcer-close = concrete final image, not thesis).**

**>>> BARK ARTIFACT SPRINT -- COMPLETE (2026-06-22). All chunks shipped + pushed to v2.0-alpha; cleaned-bark
re-soak GPU-verified end-to-end + published to obs.** B0 (fixture+spine verify, `ffe5e82`) -> B1 (speech-only
dialogue mode + first-line [clears throat] gate, `467fb05`) -> B2 (thread the dropped per-line seed ->
determinism, `9e68ad5`) -> B3 (chunk-split hardening, `e1b8196`) -> QA (deterministic >4kHz high-band edge
metric + scan tool, `e93b54e`). Upstream-TTS only; audio spine FROZEN (byte-identical baseline is indextts2,
not bark -- verified); min_eos_p kept at 0.1; no per-chunk trim / no FFT reroll loop (panel cut); NO
workflow-JSON change. Suite 4782 pass/34 skip, Bug Bible 16/7/3. **Re-soak (operator config: cleaned bark +
mesh_stage 3D bookends + z_image_turbo char stills, real otr_scifi_16gb_full.json):** rendered "Spinning
Contamination" end-to-end in 21:13 -> `otr/obs/...spinning_contamination..._final.mp4` (20.3 MB). Metric
before/after on real bark lines: BEFORE (pre-B1) 1/14 char lines flagged (max 0.694); AFTER (cleaned) 0/6
flagged, all edges <= 0.055. Details: `docs/2026-06-22-bark-artifact/{B0_FIXTURE,QA_RESULTS}.md`. Box reset
clean. **>>> RESUMES: STORY-QUALITY R2 -- S1 SHIPPED (`5396c05`); NEXT = S2 (announcer-close = concrete final
image, not thesis).** (Bark sprint was a sanctioned interleave; R2 S2-C5 remain.)

**>>> STORY-QUALITY R2 -- craft lift. S1 SHIPPED (2026-06-22); NEXT = S2 (announcer-close =
concrete final image, not thesis).** Latest green: full suite 4749 pass/33 skip, Bug Bible 16/7/3, audio
byte-identical 9/0. **S1 DONE** (commit pending push to v2.0-alpha): music/non-dialogue beats no longer
bleed their placeholder intent into the spoken/caption transcript. Root cause was
`production_ledger.init_lines_from_outline` stamping a non-voiced row's `text` from `sfx_cue OR intent` ->
music_inter (no cue) leaked "Musical interlude bridging <phase>..." into the writer's `script_text`
(`assemble_script_text_from_ledger` -> `[SFX: ...]`). Fix: (a) new canonical `is_spoken_role()` in
`_otr_ledger_scrub`; (b) non-voiced rows take ONLY a genuine `sfx_cue` as text (intent fallback dropped --
music rows -> `text=""`; real sfx cues preserved; `beat_intent` still carries the intent for visual/music
consumers); (c) neutralized the `_otr_outline._assemble_outline` music_inter intent. SDH captions already
filtered non-speech roles, so this closes the transcript path. +9 tests (`tests/test_s1_music_suppression.py`
+ updated `test_phase2b_progressive_ledger.py`). Content-only; ZERO workflow-JSON change; ledger schema
`l3-2026-05-14` untouched. **>>> NEXT = S2.** (Below: the original full-sprint brief, S2-C5 unchanged.)

HEAD baseline at sprint start `90ddfca` == origin/v2.0-alpha; suite 4740 pass/33 skip, Bug Bible 16/7/3. The plan is ROUNDTABLE-HARDENED (3 passes) + Claude-creative-pass + seams
LOCATED and BUILD-READY: **`docs/2026-06-22-story-quality-r2/SPRINT_PLAN.md`** (+ `pass01_judgment.md` /
`pass01b_creative_levers.md` / `pass02_coding_judgment.md`; raw panel reviews in `roundtable/`). Goal = a
genuinely BETTER STORY (operator: NOT word count). Grounded findings: opus is genuinely good; the weak/local
end is generic + cliche + meandering stage-business + the music/non-dialogue beats BLEED their placeholder
into the spoken/caption track ("Musical interlude bridging..." in every episode). **Build = 8 ordered green
chunks, each suite+Bug-Bible -> commit AND push:** S1 suppress music/non-dialogue beats from the
spoken/caption track (`_otr_ledger_scrub._SPOKEN_ROLES` + the text-materialization point; neutralize the
`_otr_outline._assemble_outline` music_inter intent); S2 announcer-close = concrete FINAL IMAGE not thesis
(close intent + a shared banned-thesis regex scan -> reroll via the announcer composer); S3 cliche +
stage-business reject gate in `_otr_line_hygiene` (FLAGS ONLY) feeding the EXISTING Sprint-5C `reroll_hint`
loop in `_otr_line_composer.compose_line_draft`; C0 non-default-wants classifier in `_otr_dramatic_state` +
ACTION-VERB-under-pressure in the OUTLINE Stage-3 beat prompt (`_build_beat_user_prompt`, NOT the line
composer -- intents are written there); C1 specificity anchors -> `meta["specificity_anchors"]` + a
generic-line gate; C2 central story-object -> `meta["central_object"]` (derive BEFORE the S2 close); C3
CONTRASTING `speech_signature`s at CastLock (`_otr_casting`) + promote the already-wired signature clause to
a HARD per-line constraint; C4+C5 per-act escalation contract + a turn-beat subtext nudge. HARD (panel,
code-verified): ledger schema `l3-2026-05-14` FIXED -> new fields ride FREE-FORM `meta`, NEVER new Pydantic
fields; NO workflow-JSON / node / widget change; reuse the EXISTING reroll_hint loop + speech_signature (no
new reroll infra); MODEL-AGNOSTIC (every gate is one opus PASSES -> lifts the weak end, never rewrites opus);
craft-ONLY (no word/beat/budget change); audio byte-identical; UTF-8 no BOM; SFW. FINAL QA = extend
`scripts/story_quality_scan.py` (4 structural counts + craft signals) + a small re-soak (2-3 weak-local + 1
frontier leg) with the opus-no-regress gate. START WITH S1 (kills the universal music-bleed wart).

**>>> (DONE 2026-06-22) 3D IMAGE STREAMS -- all 7 chunks shipped + GPU-verified; mesh-quality v1.1/v1.2
(STUDIO-lighting gradient, not white marble) shipped + CPU-verified.** Spec was
`docs/2026-06-21-3d-image-streams/roundtable/pass01_plan.md`; results in the LATEST SESSION blocks above.
DEFERRED to 3D v1.5: Cycles + 3-point lighting + multi-view texture bake.

**>>> SHIPPED THIS SESSION (2026-06-21, all pushed, suite+Bug-Bible green per chunk):**
- **STORY-ENGINE v1 (F1-F8) + Sprint-0 harness** -- the whole story-quality sprint (`ecd0cde`..`d9b25a0`);
  see the block below. DONE.
- **3D `mesh_stage` promoted to all 3 slots** (`6b4fd97`) -- added to `VALIDATED_ENGINES`, selectable
  (announcer/music/cast), verified live; a 30w all-slots-3D episode rendered end-to-end (6 beats mesh_stage,
  2.76 GB peak). [This is what exposed the scene-still-fodder bug above.]
- **Credits: per-slot IMAGE model** (`8a7f06b`) -- dispatcher stamps `meta.image_engines.by_role`; the credits
  dossier RENDER ENGINES section now shows video AND image engine per slot. (Treatment-card `FLUX ???` is a
  separate surface, not yet wired -- optional follow-up.)
- **latentsync REMOVED from the forward order** (`156d573`) -- ripped out, no engine/JSON refs; punch-list +
  Wan 2.2 marked operator-APPROVED.

**>>> (HISTORY) STORY-ENGINE IMPROVEMENTS v1 -- ALL CODE SHIPPED + PUSHED (HEAD `d9b25a0` ==
origin/v2.0-alpha; full suite 4717 pass/33 skip, Bug Bible 16/7/3). NEXT = the GPU MEASUREMENT + SOAK, then
the 3D PoC.** Every feature F1-F8 + the Sprint-0 harness landed as its own green chunk, each suite+Bug-Bible
green, committed AND pushed (HEAD==origin verified per chunk; no 0-byte, no BOM, AST-parse):
- **Sprint 0** (`ecd0cde`): `scripts/story_quality_scan.py` (length_ratio / length_pass_fired / episode_valid /
  outro_hedge_vs_resolved / narration_self_address; reads the on-disk `*_ledger.json`) + the SHARED
  `is_resolved_ending_change` + `HEDGE_LIST` in `_otr_dramatic_state` (one source of truth with the F3 repair).
- **T1.1 F1** (`bdeccb6`): dropped the literal "about 20-30 words" hard-cap from the line rider (the 0.70
  length_ratio root cause); None/zero-safe per-line token budget.
- **T1.4 F6** (`bf5d71a`): split the rider -- indirect-performance UNCONDITIONAL on character beats,
  situation-change GATED to turn beats.
- **T1.2 F2** (`44aecd3`): costly choice bound to a CHARACTER beat (character-only candidate list +
  contract-build guard so `must_turn` never lands on announcer/music -- fixes the slot-drama audit).
- **T1.3 F3** (`f703ea4`): ending-aware outro -- threads `dramatic_state.ending_change` + the final character
  line; recompose ONCE on a resolved-then-hedged close; deterministic resolved fallback (no hedge).
- **T2.1 F4** (`79c0040`): pins speaker gender/pronouns in the line prompt from `cast[].gender` (no schema change).
- **T2.2 F5** (`feb6fa8`): `speech_signature` -- the description LLM emits a <=5-word register, backfilled to
  "plain spoken", rendered in the voice card.
- **T2.3 F7** (`958f331`): narration/self-address detector in `_otr_line_hygiene` (shared with the scan) +
  spine recompose seam (1 attempt, LOUD, fallback) + output-format constraint.
- **T3.1 F8** (`d9b25a0`): seeded `meta.arc_shape` variety (additive) + shape templates + shape-branched
  post-validator so non-confrontation shapes (investigation / slow_dread) no longer stall.
All content-only inside node 1 `OTR_LedgerScriptWriter` + its internal modules; ZERO `otr_scifi_16gb_full.json`
edits; ledger schema `l3-2026-05-14` untouched; node imports verified clean in the venv. **REMAINING (GPU):**
(1) the 12-leg/864-word measurement (baseline + after via `story_quality_scan.py`, write `SPRINT_BASELINE.md`);
(2) the operator-requested 500-word soak (indextts2 voice, LTX-audio bookends, flux2_klein char-beat stills,
max creativity, cheap OpenRouter frontier writer). KNOWN SMALL GAP: F8's macro-prompt arc_shape context was
deferred (the dramatic-state path carries arc_shape + the meta stamp -- acceptance "distribution not
single-valued / no rejections" is met). The 3D PoC (`docs/2026-06-20-mesh-stage-texturing/`) is AFTER the soak.

**>>> SUPERSEDED 2026-06-20 (DONE) -- TOP-PRIORITY VISUAL FIXES (all four shipped + GPU-verified):**
Spec: `docs/2026-06-20-visual-fixes/VISUAL_FIXES_PLAN.md` + the 2026-06-20-NIGHT top block. HEAD `ce507e8` ==
origin/v2.0-alpha; GPU FREE (35-leg 864 frontier soak done; 3D-PoC mesh_stage GPU smoke PASSED `60a2f4f`).
DIRECTIVE (hard): ONLY HuMo (portrait) + maybe the 3D lane use the VERTICAL portrait; EVERY other path uses
LANDSCAPE 16:9 images/video. Animated start/end procgen title cards + rolling credits are NON-NEGOTIABLE
(engine-independent).
- **#1 BUG 1 -- DONE 2026-06-20 (commits `7e765b7`+`9f03abd`, GPU-verified end-to-end with z_image_turbo).**
  Character beats now show the CHARACTER full-frame 16:9 (beat-aware, per-beat, distinct), not the radio booth.
  Root cause was deeper than the 8bc5381 branch: `"character"` (the canonical writer speaker_role) was missing
  from `SPEAKER_TO_VIDEO_ROLE`, so character beats were pooled as background_abstract; added the mapping +
  scene_character stills (LLM beat-aware) + render_driver gating on render_aspect. Final-obs frames show each
  character beat's still synced to its SDH caption; audio byte-identical. See the LATEST SESSION block + the
  proofs in `docs/2026-06-20-visual-fixes/`. (HuMo still uses the vertical portrait by design.)
- **#2 BUG 2 -- DONE 2026-06-20 (verify-first, NO change needed).** Eyeballed a real obs final: the animated
  START title card resolves the episode title via a signal-decode glitch (scramble@0.3s -> "GLIMPSE THROUGH
  GLASS"@4s) and the END rolling credits render the whole post-roll. Working as designed -> no code/JSON change.
  Proofs `docs/2026-06-20-visual-fixes/B2_*.png`.
- **#3 BUG 3 -- DONE 2026-06-20 (commit `9f76937`).** The forensic model/engine/system dossier now SCROLLS on the
  rolling credits (not just the `_treatment.txt` sidecar): new `_build_hud_dossier(led)` -> 5 sections
  (WRITER/LLM CONFIG, RESOLVED OPENROUTER, STORY SPINE, RENDER ENGINES, SYSTEM = CPU/RAM/GPU/VRAM/CUDA/torch),
  drawn at the top of the HUD scroll ahead of the transcript. Content-only, NO JSON change; +7 tests; verified by
  an offline pure-PIL HUD frame (`docs/2026-06-20-visual-fixes/BUG3_credits_dossier.png`).
- **#4 BUG 4 -- DONE 2026-06-20 (commit `13017ec`).** The ALWAYS-ON audio-reactive bottom bars are now a
  SEPARATE overlay layer (DEFAULT ON; manual `off`), decoupled from the scene-aware floor so they show no
  matter what clip is above/below (landscape AND portrait). New `audio_bars` widget on
  OTR_PostUpscaleProcgenBlend (appended LAST, BUG-LOCAL-097) wired IN `otr_scifi_16gb_full.json` SAME-commit
  (node 93 +1 value, validated). Implemented as a SEPARATE second pass (the fragile procgen/scopes blend
  untouched): main blend defers captions -> a bars pass PIL-renders the green `freq_bars_green` strip above
  the caption safe-area, lighten-blends at 0.60, then burns captions ON TOP. `off` = byte-identical;
  audio `-c:a copy` throughout. +6 tests (incl. a real-ffmpeg green-paint proof); verified on a real obs
  final (green bars react in the bottom strip, captions above).
- **THEN:** the 3D PoC build (`docs/2026-06-20-mesh-stage-texturing/roundtable/pass03_plan.md` + `pass04_plan.md`
  + the 4-item PIN list; chunks 6+7+3 + GPU smoke).

**>>> SUPERSEDED 2026-06-20 (historical; the story-quality roundtable is PARKED, section 8) -- 2026-06-19 STORY-QUALITY side-campaign:**
the STORY-QUALITY roundtable campaign. **R1 DONE 2026-06-19:** the grounded FIRST-PANELIST pass
(`roundtable/pass01_opus-grounded.md`) + the 5 external panels (`docs/story_passes/pass_1/`) + the converged
judge synthesis (`roundtable/pass01_architecture.md`) are written. The converged plan = 3 fix-moves
(news-derived opposed wants at `derive_dramatic_state_from_meta`/:2780 -> deliver via the EXISTING
contract+composer path -> critic-as-repair-driver, never a gate) + bounded shape (secondary) + a §5b
rip-out/repurpose of QA-that-only-scores-or-fails; bar = a genuinely BETTER STORY, FIX-not-FAIL,
model-agnostic, activate-not-overhaul, 'acts' are not a limiter. **R2 CONVERGED via a live 2-pass roundtable
(roundtable/R2/pass02_plan.md = build-ready Phase-1 plan, A1-A6 + B1); NEXT = R3 wiring/build (operator-gated)**
-> R4 Comfy -> R5 polish (campaign DONE only at polish). Materials:
`docs/2026-06-19-story-quality-analysis/` -- START with
`roundtable/PANEL_PROMPT_R1.md`; do NOT open `_PRIVATE_OPUS_PLAN_DO_NOT_PASTE_TO_PANEL.md` until AFTER the
grounded panelist pass. Carried into every round: the story must be genuinely ABOUT the news + it must work on
a SMALL LOCAL LLM + the ledger `{cast,lines,meta}` stays FIXED (content-only) + audio byte-identical + the bar
is a genuinely BETTER STORY (no rewrite-for-its-own-sake). The news-SELECTION front-end (RSS -> brief) is a
GIVEN INPUT, out of scope -- the work is everything AFTER the brief. The BUILD current-step below is UNCHANGED
and resumes after the campaign.

**>>> CURRENT STEP DONE (2026-06-18): flux2_klein VERIFIED + PROMOTED; coverage-arch accepts_still SHIPPED.**
- The TE mismatch is FIXED: klein-4B uses the **Qwen-3-4B** encoder (7680-wide; `qwen_3_4b.safetensors` from
  `Comfy-Org/flux2-klein`), NOT flux2-dev's Mistral (15360). The official ComfyUI `image_flux2_klein_text_to_image`
  template was the ground truth (CLIPLoader = qwen_3_4b, type flux2). A full flux2->silent-LTX 30w episode minted
  ALL 6 stills green (`minted still 832x480/1472x832`), no dim error; flux2_klein is now in image
  `VALIDATED_ENGINES` (opt-in via OTR_ENABLE_FLUX2_KLEIN). See `docs/2026-06-18-flux2-klein/VERIFY_ON_5080.md` +
  the LATEST SESSION block above.
- The coverage architecture is LIVE (the operator's "all video/3D accept the selected image, one place, no
  per-model whitelist"): `accepts_still` on MotionEngineBase (default True) read centrally by
  `engine_consumes_still`; silent `ltx_video` now consumes the selected image. Design + roundtable:
  `docs/2026-06-18-coverage-arch-wiring/pass01_plan.md`.
- **>>> NEXT = coverage-arch FOLLOW-UPS (deferred, additive):** (a) **IMAGE-PHASE still coverage -- DONE for
  per-beat (`3d27bcf`) + POOL_N_LOOP DONE (`838730e`):** `derive_scene_still_targets` emits a scene_beat target
  for EVERY beat (was announcer/music-only -> b002/b003/b004 hit LTX-I2V MISSING-STILL), AND under `pool_n_loop`
  the OTHER-BEATS {background_abstract, scene_broll} now SHARE N pool stills (`other_pool_0..N-1`; ShotLock stamps
  `still_pool_key=other_pool_{i mod N}`; render_driver prefers it). `pool_n_loop`+4 is now the
  otr_scifi_16gb_full.json + widget default (in sync). announcer/music/character_video stay per-beat; HuMo keeps
  scene stills (OOM-fallback insurance). Roundtable-converged plan +
  judgment: `docs/2026-06-18-pool-loop-stills/`. Suite 4502/0. (b) `optional_inputs` so role_compat sees an OPTIONAL init_image
  (verify-at-build); (c) optionally `accepts_still=True` on the static-still cheap families
  (flux_still/station_card/still_kenburns); (d) full Decision-3/5 (central `image_engines.registry.usable()`,
  retire `requires_mesh_portrait` onto `still_kind`). THEN the carried step below (per-segment LUFS/RMS) + the S3
  forward order (3D item 5, distribution item 6).

**>>> CARRIED STEP = plan per-segment LUFS/RMS voice normalization (operator-requested; FROZEN-SPINE change, needs a plan + re-baseline).**
- **`wan_ti2v` PROMOTED 2026-06-18 (`ca3e06c`):** forced-lane GPU smoke PASSED on the 5080 via the real adapter
  (render_single -> render_clip, executor thread) -- i2v rendered 33 frames from a 16:9 still at 832x480, engine
  vram_used ~8.2 GB, independent NVML peak 13,078 MiB (< the 14.5 GB cap), twice. Added to
  `registry.VALIDATED_ENGINES` (now lists in the tested-only dropdown). It's the low-VRAM tier filler for non-face
  beats -- the weakest video engine on quality but the one that fits where 14B HuMo / 22B LTX won't; NO audio.
  Same commit fixed `render_single` to render in each engine's native aspect (was always portrait -> wide engines
  letterboxed a 16:9 still = the "postage stamp"). The recorded sub-8GB tier = IMAGE `z_image_turbo` (`dcf078c`) +
  VIDEO `wan_ti2v` + characters on HuMo-2D; 3D `triposr` dark/deferred.
- **NEXT = per-segment LUFS/RMS normalization (operator instinct 2026-06-18).** The "low/thin" Bark perception is
  loudness, not peak; the assembler currently peak-normalizes each segment to 0.85 + a final -1.0 dBFS master
  limiter (no Bark-side normalize -- correct). A per-segment LUFS/RMS target would even out perceived loudness.
  **CAUTIONS (write the plan around these):** this is the FROZEN audio spine (`scene_sequencer.py`
  EpisodeAssembler `_normalize_clip`), NOT upstream TTS -> it breaks `test_audio_byte_identical` (deliberate golden
  re-baseline, operator-gated GPU render) and hits EVERY voice, not just Bark. Design = LUFS/RMS target WITH a
  max-gain clamp + a noise-floor gate (do NOT fully flatten dynamics -> pumping). Deterministic (no RNG). Recommend
  a roundtable before coding (frozen-spine + dynamics trade-off).
- **After that:** the S3 forward order -- 3D sprints (item 5) then switchable distribution (item 6). Operator-gated
  GATE-A items (punch-list audit [APPROVED 2026-06-21], coverage sweep GREEN) run in parallel on look-QA.
  (latentsync demos REMOVED 2026-06-21 -- ripped out, no live engine.)
- **Open operator eyeballs (carried):** the Bark "warmer / not-whiny" verdict (audition WAVs in
  `%TEMP%\bark_audition\`); the wan_ti2v wide smoke clip (shared this session) + the Wan 14B-vs-5B WEBM eyeball.

---

## 2. HARD RULES (invariants -- apply every session)

- **WORKFLOW SOURCE OF TRUTH (operator, hard):** `workflows/otr_scifi_16gb_full.json` IS the
  production workflow. (1) ANY node / wiring / widget change MUST be made IN that file in the SAME
  change as the code -- code that is not wired into this JSON is DORMANT and does nothing (the §4D
  miss, 2026-06-13: node + blend input shipped + tested but unwired -> ran dead in production). After
  editing, re-validate via `OTR_WorkflowValidator` + a JSON round-trip + the link/widget audit.
  (2) EVERY API / headless / soak run MUST LOAD this real JSON -- never a stale copy, a generated
  `.gen.json`, an ad-hoc graph, or the Linux-mount snapshot (the sandbox mount lags file writes; always
  read/write the Windows path + verify via Desktop Commander).
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8) -- not story-spine, not
  story-pipeline, not the broader audio stack, not other ROADMAP items.
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. Only sanctioned audio work = the upstream character-voice
  "whiny" fix.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100% local/offline; determinism
  seed-keyed (per-seed within a render, NOT run-to-run); every in-render fallback LOUD; UTF-8 no BOM;
  SFW; V-12 dep isolation; no new widgets in the static workflow shell (V-11).
- GIT (operator 2026-06-10): ONE branch `v2.0-alpha`; commit AND push together per green chunk; the
  operator eyeball gates TAGS/promotions only; after a push verify HEAD==origin / no 0-byte / no BOM /
  AST parse on touched .py. prod/`main` is GATED until operator work is done (a `v2.0-alpha-stable`
  tag on `v2.0-alpha` is fine).
- EVERY session updates this doc + the `otr-build-tracker` dashboard (content; keep the gauge+lanes
  styling).
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) only behind `OTR_C7=1`; normal runs must log
  `cast RNG seed=... (OS entropy)`. Do NOT set `OTR_C7` for normal runs.

---

## 3. FORWARD ORDER (do in sequence)

> **Two tracks, parallel.** Item 1 (punch-list audit) is OPERATOR-GATED (look-QA -- section 5); the
> ENGINE track (items 3-4, Wan + sweep GREEN) proceeds NOW. "In sequence" applies WITHIN a track, not
> across the operator gate.

1. **Punch list (GATE A) -- OPERATOR APPROVED 2026-06-21, proceed.** Captions DONE (node 86
   `OTR_CaptionBurn` in `otr_scifi_16gb_full.json`, profile resolves `burn_captions=True`). REMAINING:
   node-level audit of LTX radio-open + procgen rolling credits -- baked into the headless path but maybe
   NOT into the saved JSON; prove a render FROM the JSON has them, then operator look-QA.
2. **latentsync -- REMOVED 2026-06-21 (operator: "we ripped it out").** Verified: NO engine file under
   `nodes/_otr_video_engines/`, 0 references in `otr_scifi_16gb_full.json`, only a few stray comment/env
   strings remain (`OTR_LSYNC_BASE_ENGINE`). Not a live lane -- dropped from the forward order. (A trivial
   code-comment scrub of the stray strings can ride any future cleanup; not a roadmap item.)
3. **Wan 2.2 video engine (section 4) -- OPERATOR APPROVED 2026-06-21 ("100% approved"): proceed with the
   eyeball + acceptance.** BOTH engines BUILT + validated live (2026-06-14, `bcbe05a`):
   wan_i2v (14B, post mixin-refactor) + the new wan_ti2v (5B/GGUF 8GB tier). REMAINING = the operator
   WEBM EYEBALL (14B vs 5B) + the optional formal full-episode `--acceptance` GREEN exit (slow
   wan-music-bed leg, run attended) + the M9 CS-3 instrumented proof. Code-complete; gates are the
   operator's.
4. **Coverage sweep GREEN (GATE A acceptance).** Re-run the permutation matrix after the soak fixes.
   Matrix (additive, not cross-product): a visual-engine leg-set (varies each of music/announcer/
   other_beats), a writer-LLM leg-set (varies node-1 `creative_writing_model`/`technical_model`), and a
   curated voice-variation leg-set (2-3 refs per voice engine). Unique story per leg (OS entropy, no
   seed pins). **Wan is a CORE/BLOCKING engine** -- the sweep is NOT green until `wan_i2v` (and
   `wan_ti2v`) pass, so it stays RED until item 3 lands; that is expected. This re-run also answers the
   one open R2 question: whether `humo_1.7B` renders NATIVE char beats at 70w once its enable flag is on
   (the soak floored it only because the flag was off). **GATE-A precondition: harden the
   sweep FIRST (section 4A M1-M4) -- DONE 2026-06-13: the M1-M5 acceptance gate landed
   (`scripts/otr_coverage_sweep.py --acceptance`), so a silent fallback / empty-results
   run / missing VRAM measurement now scores RED, not GREEN.**
   **S6 harness reality:** `otr_coverage_sweep.py` enumerates ONLY the visual-engine
   leg-set today (the dropdown rotation). The writer-LLM leg-set (node-1
   `creative_writing_model`/`technical_model`) and the curated voice-variation leg-set
   are NOT yet wired into a runnable harness -- TODO: point them at a real driver
   (e.g. a `run_combo_matrix.py`) or run them as separate parametrized soak legs.
   "Coverage sweep GREEN" today means the visual-engine set only.

   **SOAK READINESS AUDIT (2026-06-13).** Walked the registry + harness. Conclusion:
   **clear to run a wan_i2v-only soak today** (no wan_ti2v hard prereq for validation).
   Verified live: `wan_i2v` enumerates `ok`/runnable under `16gb_full` (legs
   `music_visual=wan_i2v` + `other_beats_visual=wan_i2v`) -- the old "add wan_i2v to the
   enable-set" note is STALE/resolved. 27 legs enumerate; the only skips are
   `hunyuan3d_talk`/`trellis_talk` (missing cu128 toolchain, expected darks). Wan models
   on disk + `OTR_ENABLE_WAN_I2V=1` env known. **Two limitations to know:**
   (i) `--acceptance` exit is RED-by-construction until `wan_ti2v` is built (M2 requires
   BOTH Wan engines) -- expected; read the per-leg verdicts in `coverage_sweep_summary.json`,
   the wan_i2v leg PASS/FAIL is the meaningful signal.
   (ii) **The M1 no-fallback (CS-1) gate is bound to `--acceptance`** (`forbid_fallback=
   args.acceptance`); the capstone CLI does not expose it. So re-running the NON-Wan
   permutation soak (the set that originally false-greened) WITH the M1 fix active and a
   clean GREEN/RED exit needs either `wan_ti2v` built OR a small **`--strict-fallback`**
   flag that decouples M1 from the Wan-engine requirement (~10 lines; RECOMMENDED, optional
   -- operator's call). Until then: `--acceptance --only wan` exercises M1 on the wan_i2v
   legs (overall RED expected), and a non-acceptance sweep runs but with M1 OFF
   (informational). No half-built code, no missing capability rows beyond the deferred
   `wan_ti2v`, no broken tests (the 2 `test_model_catalog_scan` reds are pre-existing /
   environmental, tracked separately).
5. **3D sprints.** s2 = S-3D-0 spike + T1 template + T2a wrap smoke; then the `character_3d` family
   (image-routing must-fixes already landed). Detail in the 3D plan (pointers).
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README (closing phase).

**0-E parallel track:** `ltx_orbit`/`still_parallax`/`mesh_stage` CPU side shipped + all three GPU-green;
Phase B (E-1 probe, E-6 renders, per-engine sweep legs) HELD on the `scripts/_otr_0e_gpu_go.txt` GO file.

**Audio parallel track (own window, never blocks video):** the character-voice "whiny" fix (upstream TTS
only; frozen spine untouched). Operator note: may have self-resolved -- verify before scheduling work.

---

## 4. WAN 2.2 VIDEO -- REMAINING (active build)

Two selectable Wan 2.2 video engines, eyeball-gated, b-roll/camera motion only (lip-sync stays SEPARATE
on LatentSync/HuMo). Core Comfy Wan nodes, NOT the KJ wrapper (KJ drags in SageAttention + a numpy<2 pin
this box violates). Phase 1 + the 5 code-gap fixes are DONE (`2fbc2f3`); the full grounded spec is in
that commit + git history of this file.

- **Phase 2 -- 16GB engine leg.** Drive `eng_wan_i2v.render_clip` via the real path
  (`scripts/otr_run_leg.ps1` / `coverage_sweep --only ...`). ASSERT `wan_i2v` is the final_engine in the
  trace (FAIL LOUD on fallback, CS-1) + render-phase NVML <= 14.5 GB + byte-identical audio mux + silent
  mp4 (h264/yuv420p/bt709, fps 25, `has_audio` False). Kill/reset the Phase-1 server first.
- **8GB tier -- TI2V-5B as a SEPARATE engine.** Fetch the TI2V-5B GGUF (Q6/Q5_K_M) + the wan2.2 VAE into
  `C:\ComfyUI-Models\` (record HF repo + sha256 + license, fail-closed). Define a NEW `wan_ti2v` engine
  (own flag/model/VAE env, registry registration, `_node_candidates` incl. the 5B latent node, loader
  mode, `canonicalize`, profile hook + tests) -- do NOT alias `WanI2VEngine`.
- **Eyeball gate.** Present both webms (I2V-14B vs TI2V-5B, same still + prompt) in
  `docs/2026-06-12-ltx23-motion/wan_clips/`. Bar = real camera motion, still preserved, no warp.
  **S3 motion risk to watch:** the wired I2V-14B fp8 is a SINGLE low-noise expert (the
  two-expert HIGH/LOW MoE handoff, Path B, is NOT wired -- see `eng_wan_i2v` header). If
  the "real camera motion" bar FAILS (motion too subtle / static), the Path B two-expert
  HIGH/LOW handoff is the mitigation, not a knob tweak. Call this out at the eyeball.
- **Risk CS-3 (reframed):** sequential-residency, NOT co-residency -- see section 4A M9
  and the section-5 CS-3 entry. The supervised Wan batch proves the inter-beat reclaim,
  it does not "decide if they co-stage."

---

## 4A. WAN + GATE-A SWEEP HARDENING (roundtable 2026-06-13, grounded vs HEAD 134f8e2)

Folded from a 3-model roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4) + Claude's
grounding against the real code. Full judgment + raw reviews:
`docs/2026-06-13-goforward-wan-hardening/`. These gate item 3 (Wan) and item 4 (sweep
GREEN). MUST-FIX -- until M1-M4 land, a GREEN sweep is meaningless:

> **STATUS 2026-06-13 (autonomous build) -- LANDING LEDGER:**
> - **M1 + M4** `9b2294b` -- no-runtime-fallback gate + VRAM fail-closed (12 tests).
> - **M2 + M3 + M5** `0ab55bc` -- sweep `--acceptance`: empty/required-engine exit
>   code + Wan enable-flag / OTR_TEST_MODE / --exclude preflight (17 tests).
> - **M6** `ec91a3c` -- `assert_usable` preflights UNET + umt5 CLIP + VAE (8 tests).
> - **M7** `f71edaa` -- render_clip ffprobe-PROVES the silent-clip contract (13 tests).
> - **S1 + S5** `dfe9ab5` -- wan_i2v vram_estimate 14500 + real wan2.2-i2v asset id.
> - **S7 + S10** `f3a529f` -- per-shot/seed init staging + Pillow-required fail-loud.
> - **S3 / S6 / S8** -- folded into this doc (MoE eyeball risk, sweep-harness reality,
>   the exact acceptance invocation below).
>
> **M8 + S2 -- LANDED 2026-06-14 (`bcbe05a`).** The `wan_ti2v` engine is built: its 5B core
> node class (`Wan22ImageToVideoLatent`) was captured from a live `/object_info` first; M8 raises
> `EngineUnusable` when the resolved VAE basename is empty or is the 2.1 VAE; S2 added the
> `medium`/8000 CAPABILITIES row (registry-consistency invariant holds -- the row + the registered
> engine landed together). Validated live (5B bare-graph smoke PASS). **STILL OPEN:** **M9** (CS-3
> sequential residency) + **S4** (leg isolation/reclaim) + **S9** (post-reset verify) are live-GPU
> proof obligations -- partial evidence only. A full multi-leg `--acceptance` GREEN exit is gated on
> the slow wan-music-bed leg (run it attended/selectively), not on missing code.
>
> **S8 -- exact acceptance invocation** (ComfyUI venv python; live server on :8000;
> `OTR_TEST_MODE` UNSET; `OTR_ENABLE_WAN_I2V=1` (+ `OTR_ENABLE_WAN_TI2V=1` once built);
> Wan UNET + umt5 CLIP + VAE on disk):
> `python scripts\otr_coverage_sweep.py --acceptance --only wan`
> (`--only wan` matches the `sweep_<slot>_wan_i2v` / `_wan_ti2v` legs; drop `--only`
> for the full visual set. `--exclude` of a core Wan engine is REFUSED in acceptance.)

- **M1 -- the sweep is BLIND to silent fallback.** `otr_coverage_sweep.py` runs every
  leg with `expect_engine=""`, which `_otr_soak_capstone.py:464` treats as
  informational (no assert), so a leg that silently falls back to `still_kenburns`
  scores PASS (this is exactly CS-1). FIX (NOT per-leg `expect_engine=engine` -- that
  false-fails a slot that gets 0 beats at 30w): in acceptance mode assert ZERO runtime
  fallbacks across the whole trace -- fail any shot where `final_engine != attempts[0]`
  -- with an opt-out only for known-degrade experiment legs. (Verify the trace field is
  a stable requested-id, not an alias.)
- **M2 -- the sweep returns GREEN on EMPTY results.** `return 0 if passed ==
  len(results)` makes `0 == 0` pass when `--only`/`--exclude` filter everything out or
  `wan_ti2v` is unregistered. FIX: fail on empty results; for GATE-A, fail unless BOTH
  `wan_i2v` AND `wan_ti2v` are present in results with PASS.
- **M3 -- acceptance preflight (closes the R2 trap).** `availability()` is pure
  profile-fit and never reads `OTR_ENABLE_WAN_I2V`, so a gated-off Wan leg enumerates
  "run", `assert_usable` fails it closed, it falls back, and (pre-M1) passes -- the same
  `gated_by_flag` mechanism that floored HuMo-1.7B (commit 5231d31). FIX: the acceptance
  run preflights `OTR_ENABLE_WAN_I2V=1` (+ future `OTR_ENABLE_WAN_TI2V=1`) and the model
  files, and FORBIDS `--exclude` of the core Wan engines.
- **M4 -- the V-3 VRAM gate fails OPEN.** `driver_peak = int(report.get("vram_peak_mb")
  or -1)` then fails only if `> ceiling`, so a missing/0/negative measurement (`-1`)
  PASSES -- the `<=14.5GB` invariant can read GREEN with no measurement. FIX: fail
  closed when `vram_peak_mb` is absent or `<= 0`.
- **M5 -- the Wan render-phase VRAM assert is skipped under `OTR_TEST_MODE`** (`if not
  os.environ.get("OTR_TEST_MODE"): ... assert_peak_within_ceiling`). Phase-2 acceptance
  MUST run with `OTR_TEST_MODE` UNSET; the harness preflight fails if it is set.
- **M6 -- `assert_usable` preflights only the ckpt.** The umt5 CLIP + the VAE are
  required graph loaders. FIX: verify UNET+CLIP+VAE present + matching the sha/license
  manifest before any forward (offline / no-runtime-fetch invariant).
- **M7 -- the Phase-2 clip contract is SELF-DECLARED, not asserted.** `_clip_from_raw`
  hardcodes `has_audio=False`/h264/yuv420p/bt709/fps25 in a dict; the soak only inspects
  the obs final's audio. FIX: ffprobe the emitted silent Wan mp4 (or a real-path test)
  to PROVE those fields before mux.
- **M8 -- `wan_ti2v` VAE fail-closed.** `eng_wan_i2v` defaults the VAE to
  `wan_2.1_vae.safetensors`; the 5B needs the Wan2.2 VAE. Give `wan_ti2v` its own VAE
  env; raise `EngineUnusable` if the resolved VAE basename is empty OR equals the 2.1
  basename. Do NOT inherit `_loader_names()` unchanged.
- **M9 -- CS-3 = sequential residency (see section 5).** Prove per-beat peak <= 14.5GB +
  the inter-beat reclaim drains the prior heavy engine (incl. the retained Wan unet
  patcher) before the next loads; that is the real risk, not co-residency. Unblocks
  Phase-2 scoping.

SHOULD-FIX: **S1** raise `CAPABILITIES["wan_i2v"].vram_estimate_mb` 14000 -> the measured
Phase-2 peak (or 14500); the 14499 smoke figure was WITHOUT `free_after_use`, which is
load-bearing -- document it as mandatory. **S2** add a concrete `wan_ti2v` CAPABILITIES
row (`medium` / ~8000 DRAFT -- the 5B VAE decode may push higher, verify on the 8GB
probe / `["wan2.2-ti2v-5b"]`). **S3** surface the single-expert (low-noise) MoE motion
risk on the eyeball gate -- Path B two-expert HIGH/LOW handoff is the mitigation if the
"real camera motion" bar fails. **S4** sweep leg isolation -- reclaim/restart between
legs that swap heavy engines (one resident server, no teardown -> residue corrupts the
next leg's peak; ties to CS-2 + the CLAUDE.md reset directive). **S5** fix the stale
`["wan2.1-i2v"]` label -> the real Wan2.2 I2V asset id. **S6** point item-4's writer-LLM
+ voice-variation leg-sets at their real harness (run_combo_matrix.py?) or mark TODO --
`otr_coverage_sweep.py` enumerates ONLY the visual-engine set today. **S7** stage the
init image under a shot/seed/uuid name (`otr_wan_init_WxH.png` is fixed -> same-dim
renders overwrite; low risk, driver is sequential). **S8** spell `scripts/otr_coverage_sweep.py`
+ the exact `--only` Wan substring + required env. **S9** Phase-2 post-reset verify
(PID/start-time changed, Sage NOT active, `OTR_TEST_MODE` unset, env visible) before
submitting. **S10** `_materialize_init_image`: require Pillow + fail loud (the no-Pillow
path leans on `WanImageToVideo` cover-resize -- N9 risk).

CUTS (panel consensus -- do NOT over-engineer): no broad VRAM-budget-aware scheduler to
close CS-3 (the reclaim assertion suffices; wait for a measured failure); do NOT subclass
all of `WanI2VEngine` for `wan_ti2v` (share only pure dims/aspect/materialize/canonicalize
helpers; keep loaders + node candidates + graph SEPARATE); keep the GATE-A sweep ADDITIVE,
not a visual x writer x voice cross-product. VERIFY-AT-BUILD: capture TI2V-5B's exact core
node class from `/object_info` before coding (the "5B latent node" is underspecified).

---

## 4B. WAN PHASE 1 -- DONE (pointer)

Phase 1 PROVEN: a real Wan b-roll clip (wan_i2v 14B fp8 in-process, ~14.5 GB; commits `2fbc2f3` +
`8eaf058`). Phase 2 is the ACTIVE next step (section 1); remaining Wan work = sections 4 + 4A. The
overnight-soak companion findings (R1 GPU-proven, R2 harness fix unexercised, R3 landed) live in git +
`scripts/FABLE_SOAK_REVIEW.md`; the not-done remainder (R2 verify) is in section 5.

---

## 5. OPEN TICKETS

- **LOOK-QA BUGS (NEW 2026-06-14 eve — operator look-QA pass; all in `BUG_LOG_2026-06.md`):**
  - **BUG-408 default MUSIC sounds non-musical (SA3).** **IMPLEMENTED 2026-06-14 (`3a4f71d`).** Path B:
    SA3-shaped prompt + real negative + per-cue `seconds_start` within a 30s `seconds_total` context (latent
    stays `dur` → length+determinism unchanged), env-overridable sampler knobs. Suite 4261/0. **OPERATOR-GATED:**
    restart Desktop, A/B listen (tune `OTR_SA3_CFG/STEPS/CONTEXT_S`), then RE-BASELINE the `test_audio_byte_identical`
    golden (intended music-bytes change). Plan: `docs/2026-06-14-sa3-music-improvement/roundtable/pass01_plan.md`.
  - **BUG-409 title card scrambles the WHOLE window** — **FIXED 2026-06-14 (`9e0b658`).** New
    `_title_reveal_progress` resolves the reveal in the first ~40% of the window then holds solid (env
    `OTR_TITLE_REVEAL_FRACTION`); close card stays bounded to the main video (no credits overlap). Suite 4259/0.
  - **BUG-410 closing ROLLING CREDITS** — **CLOSED 2026-06-14 (operator-verified on flux_still).** Credits
    scroll over the held last clip to the end again (silent after the theme). Detail in `BUG_LOG_2026-06.md`
    + `docs/2026-06-14-credits-tail-fix/`. (HuMo backdrop not yet eyeballed — low risk, engine-agnostic path.)
  - **BUG-411 flux BOOKEND / image lost its "lush" cinematic tint (NEXT — HANDOFF FOCUS).** The 6/5 image
    pipeline (`visual/batch_flux_render.py` + `flux_prompt_extractor`) was WHOLLY REWRITTEN into
    `_otr_image_engines/flux_gen1.py` + `otr_meta_brief_image_prompt.py` (pure insertions after `e4cb3ac`).
    Model/steps/cfg/sampler IDENTICAL (flux1-dev-fp8, 20, 1.0, euler/simple), but the rewrite DROPPED the look
    levers: **(1) FluxGuidance = 3.5** (flux_gen1 has NO FluxGuidance node — biggest factor), **(2) the
    cinematic style suffix** `"cinematic, 35mm film, anamorphic lens, volumetric lighting, heavy vignette,
    muted color grade, sharp focus"`, **(3) the radio broadcast-distress suffix** + retrofuturistic radio
    fallback (`35mm film grain ... dim amber and cyan rim lighting`), **(4) bookend seed 4242**, **(5) portrait
    style line**. 6/5 workflow widgets inspected + confirmed (no other hidden hardcodes). FIX = restore those in
    the new pipeline (FluxGuidance node @ ~3.5 + the suffixes + seed). Full forensic in `BUG_LOG_2026-06.md`
    BUG-411. CODER-READY (the next window's task).
  These are GATE-A look-QA items (operator-gated track), parallel to the engine forward order — NOT a
  reordering of section 3.

- **IMPROVED 3D INPUT -- BLOCKED on a PATH DECISION (operator look-QA 2026-06-14; GROUNDED this session).**
  The 3D rotating output looked like a "blobby plaster-of-paris" block. GROUNDING (checked logs + disk):
  the ONLY 3D system actually installed/active is **HunyuanWorld-Mirror / WorldMirror 2.0**
  (`custom_nodes/ComfyUI-HunyuanWorld-Mirror`, model `C:\ComfyUI-Models\WorldMirror-V2\HY-WorldMirror-2.0`)
  -- NOT Blender, NOT OTR's deferred character_3d/TripoSG. Recent episode ledgers used NO 3D engine; the
  server log only shows HWM loading (no episode rendered 3D). **WorldMirror is a MULTI-VIEW SCENE
  reconstructor** (image SEQUENCE -> point cloud / Gaussian splat): per its docs 1 frame = "depth/normals
  only"; good 3D needs **8-24 FEATURE-RICH frames, orbital/forward parallax, well-lit, 50-70% overlap**. A
  single flat/low-feature image -> the plaster blob. So the earlier "clean / object-free single image"
  idea is the OPPOSITE of what WorldMirror needs -- object-free helps only single-image-to-OBJECT-mesh
  tools (TripoSG / Hunyuan3D-2 / TRELLIS), which are NOT installed. **OPEN DECISION (operator, next window)
  -- the prompt strategy is opposite per path:** (A) WorldMirror scene/world -> improved input = GENERATE
  an orbit/multi-view sequence + rich-textured scene prompts (NOT a plain bg); or (B) single-image ->
  object mesh -> INSTALL TripoSG/Hunyuan3D-2 + clean isolated-subject prompts. Do NOT draft the improved
  3D prompts (and do NOT wire character_3d) until the path is picked. A roundtable can harden the chosen
  path's prompt set. (Note: the live roundtable launcher stalled this session -- the panel blocked with no
  output; budget a retry or a smaller panel.) Example obs finals to eyeball the EPISODE look (these do NOT
  contain 3D): `output\otr\obs\signal_lost_plunging_depths_20260614_185229_silent_procgen_blended_final.mp4`
  (a pre-fix render -- shows the closing FREEZE + the skinny flux_still portrait, both now fixed at HEAD).
- **HuMo full-frame TEST (operator 2026-06-14 -- future experiment, NOT now).** Operator wants to
  eventually SEE HuMo rendered full-frame (not the 480x832 portrait pillarbox). For now portrait stays
  HuMo's REQUIREMENT -- BUG-407 shipped "full frame everything EXCEPT HuMo". Future: a HuMo full-frame /
  16:9 smoke to evaluate whether the talking-head holds at a wider aspect before changing the default.
- **Look-QA the 5 overnight 120-word episodes (NEW 2026-06-14).** The default-lane soak ran 5/5 SUCCESS
  (LTX + humo_1.7B); the episode outputs (`...\output\otr\episodes` + obs finals) are NOT yet eyeballed.
  Check audio sync, burned captions, procgen scopes/credits, character look. This is the operator's
  "analyze the soak" item; verdicts in `scripts/_otr_120word_soak_summary.json`.
- **Wan WEBM EYEBALL -- DONE 2026-06-14 (operator + Claude live smoke).** RESULT: **Wan i2v 14B
  DRIFTS** -- holds the input still ~1 frame then re-interprets the scene into its own subject (a
  generic tube close-up). NOT fixable by easy input knobs: cfg3.5->2.0 + a locked-tripod prompt STILL
  drifts; cfg1.5 COLLAPSES into incoherent abstraction. **LTX (2B v0.9) HOLDS** the composition with
  subtle motion in all 3 modes tested (ksampler 30-step, distilled, AND 1216x704 hires -- hires
  answers the "low-res" note). => **RECOMMEND: Wan i2v 14B -> BACK-BURNER for the music/announcer
  OPENER role** (keep selectable; revisit only with Path B two-expert handoff, GO_FORWARD 4A S3); LTX
  stays the opener engine; **PROMOTE LTX-REGR (below) to the active thread.** Evidence:
  `docs/2026-06-14-wan-ti2v/EYEBALL_FINDINGS.md` + `eyeball_frames/COMPARISON_montage.png`. AWAITS
  operator confirm on the re-prioritization.

- **Non-Wan soak = ENOUGH (operator call 2026-06-13).** The non-Wan permutation coverage sweep
  (`--strict-fallback --exclude wan/latentsync/triposg`) has run sufficiently; do NOT keep grinding it.
  The non-lip-sync FLOORS (`still_kenburns` / `still_parallax` / Ken-Burns / `station_card`) render fine
  and are acceptable for the 8GB tier, but they are NOT the target experience -- the operator wants real
  audio-driven lip-sync, not a still with motion. Focus the remaining runway on **getting the Wan lane
  bug-free** (section 1 + 4 + 4A). A new sweep, if ever needed, should add `--exclude-engine humo` (the
  exact-match flag added `ca10b63`: skips the 14B `humo` that TIMES OUT per CS-4, KEEPS `humo_1.7B`).
- **LTX-REGR — SUPERSEDED 2026-06-15 by the LTX 22B-GGUF splice** (`docs/2026-06-15-ltx-splice/SPLICE_PLAN.md`).
  LTX-REGR's recommended fix was to bake the **2B** v0_9 recipe into `eng_ltx_video.py`; the splice instead swaps
  `LtxVideoEngine` to the **22B GGUF** mini recipe (verified-working). **Do NOT do the 2B bake.** Original entry kept
  below for history only:
  **LTX-REGR (operator 2026-06-13; PROMOTED to active 2026-06-14 pending operator confirm)** -- LTX
  clips no longer animate like the **2026-05-30..06-05** era (motion lost / too static). `BUG-LOCAL-113b`
  (`8115c72`: ksampler 30-step euler cfg3.0 as the LTX default, distilled 8-step = the
  `OTR_LTX_SAMPLER=distilled` rollback) was the prior fix, but the operator STILL sees the regression.
  **2026-06-14 eyeball update:** the Wan-vs-LTX smoke proved LTX HOLDS the still composition cleanly
  (good) -- so the open question is narrowed to **MOTION AMOUNT** (5/30-6/5 read as more dynamic; the
  current ksampler/distilled holds are subtle). With Wan i2v back-burnered for openers, this is the
  recommended NEXT thread. Probe = an LTX **--strength / sampler-mode / step-count / cfg / frame-cap**
  sweep (otr_ltx_motion_smoke.py exposes all of them; --strength is the prime motion lever, 1.0=max
  freedom) against the 5/30 baseline + the 169 decode floor from look-QA round 5.
  **FORENSIC DONE 2026-06-14 (BUG-LOCAL-412, `BUG_LOG_2026-06.md`):** diffed the GOOD 5/09 `l001` + 5/28
  `b001` LTX bookends vs the current engine (ledgers + the DELETED `batch_ltx_render.py` @ `70d379b^` + the
  old workflow JSON widgets). The good recipe = **v0_9 / sampler `euler_cfg_pp` / 8 distilled steps / cfg
  1.0 / 832×480 / I2V strength 0.75 / `loop_via_reverse` boomerang / audio-length**; the cleanbreak
  `70d379b` DELETED that node and `eng_ltx_video.py` shipped **ksampler / `euler` / 30-step / cfg 3.0 /
  768×512-or-1472×832 / strength 1.0 / 169-cap / no boomerang** (the code comment itself admits `euler_cfg_pp`
  is the documented dynamic-motion sampler but the default was left on `euler`). The old WORKFLOW JSON baked
  in NOTHING but seed/method/cap — the recipe lived in code. **ENV-TESTABLE A/B FIRST (no code change):** at
  832×480 set `OTR_LTX_SAMPLER=distilled` + `OTR_LTX_SAMPLER_NAME=euler_cfg_pp` + `OTR_LTX_I2V_STRENGTH=0.75`,
  re-render a bookend, A/B vs `l001`/`b001`; if it matches, bake those defaults + the boomerang + audio-length
  back into `eng_ltx_video.py` (coder chunk; no JSON change implicated).
- **CS-1** -- the latentsync legs must show latentsync IN THE TRACE (a prior "PASS" was fallback-only);
  re-verify in the sweep. (Non-Wan -> deprioritized per the operator's "non-Wan soak = enough" call.)
- **CS-2** -- machine NVML pins ~16 GB per leg vs the 14.5 ceiling while driver-phase attribution reads
  ~3 GB; needs phase attribution (the 1.7B leg's 10,305 MB render-phase peak is a partial answer).
- **CS-3 (reframed 2026-06-13)** -- NOT a co-residency budget: wan_i2v (~14GB) +
  humo_1.7B (~7GB) cannot co-reside under 14.5GB by construction, so they must render
  SEQUENTIALLY. The real proof obligation = per-beat NVML peak <= 14.5GB AND the
  inter-beat reclaim (`wrapper_bridge.reclaim_idle_models`, BUG-291) fully drains the
  prior heavy engine -- incl. the retained Wan unet patcher -- before the next beat
  loads. A mixed Wan+HuMo episode is the test. This UNBLOCKS Phase-2 scoping (no
  open "decision" needed). See section 4A M9.
- **CS-4-open** (deprioritized) -- targeted post-encode umt5-TE detach for the OPT-IN 14B HuMo lane so it
  fits 14.5 GB. The default char tier is `humo_1.7B` (`955f134`); the 14B is opt-in.
- **R2 verify** -- confirm `humo_1.7B` renders native char beats at 70w with its enable flag ON (the
  soak floored it only via `gated_by_flag`); answered by the item-4 re-run.
- **README "what to expect per video model" (operator 2026-06-14).** Once the opener model bake-off
  settles (interactive render bench artifact `otr-render-bench` + `docs/2026-06-14-wan-ti2v/
  EYEBALL_FINDINGS.md`), add a user-facing "what to expect from each video engine" section to the
  README (newbie audience -- folds into S6/closing): Wan i2v 14B = drifts off the still (b-roll only,
  NOT openers); LTX = holds composition + subtle motion (opener default); TI2V-5B = 8GB tier, lower-res.
  Source the verdicts from the operator's bench ratings (export button).
- **Ship defaults (release)** -- proposed: announcer + character = `flux_still`, music = `visualizer`
  (selectable: station_card, still_parallax, abstract — `ltx_orbit` ripped 2026-06-15 in the LTX splice Phase 0). Keep HuMo/latentsync/3D
  selectable-not-default until verified. Operator eyeballs 2-3 finals/slot.
- **Harness polish** (minor) -- output-tree resolver should prefer the live server's `OTR_OUTPUT_DIR`
  (fail LOUD on mismatch); run the OH-3 janitor sweep at server boot; widen the heartbeat cadence.
- **OH-4** -- the 14-entry / ~8.2 GB live->attic migration STAGED, awaits operator "go OH-4"
  (`docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md`).
- **0-E Phase B** -- tickets E-1..E-7, gated on the sweep GO file; coder-window ready.
- **Operator gates** -- ComfyUI Desktop relaunch (look-QA), fresh-render acceptance, whiny-voice P0 matrix
  + reel, S-3D-0 green light, `v2.0-alpha-stable` tag decision. (latentsync demos REMOVED 2026-06-21.)

---

## 6. RUNWAY (remaining sprints to "done")

"Done" = platform wired into real episodes (real per-beat video + byte-identical mux + legacy procgen
path gone) + all video models verified live + the first 1-2 3D models rendering. ~s2-s9:
S-3D-0 spike -> T2b keystone GO/NO-GO (timeboxed ~1wk) -> T4 driver + LOOK gate -> W7 production wiring +
soak ("v1-usable") -> S3-S6 distribution. SHORTCUT FORK: S-3D-0 or keystone NO-GO -> `character_3d`
defers (HuMo-2D stays) -> collapses to ~2-3 sprints (0-E + closing). Done splits: "v1-usable" (one
engine, one real episode) vs "B-parity ship" (>=2 engines bind at SHIP).

---

## 7. POINTERS (evidence + tooling -- not plans)

- Tracker dashboard: `otr-build-tracker` artifact (OneDrive\Documents\Claude\Artifacts).
- Soak review (R1/R2/R3 detail + roundtable): `scripts/FABLE_SOAK_REVIEW.md`.
- Wan/sweep hardening (grounded QA + 3-model roundtable judgment, 2026-06-13):
  `docs/2026-06-13-goforward-wan-hardening/` (pass00 plan+QA, pass01/pass01b raw
  reviews, pass01_judgment.md).
- Overnight sweep: `scripts/otr_overnight_sweep_launch.ps1`; tasks `otr-overnight-sweep` +
  `otr-sweep-monitor`; digest `scripts/sweep_monitor_digest.md`; GO file `scripts/_otr_0e_gpu_go.txt`.
- 3D spec (forward item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Bug log (this repo): ACTIVE = `BUG_LOG_2026-06.md` (epoch BUG-LOCAL-400+, started 2026-06-14);
  ARCHIVE = `BUG_LOG.md` (BUG-LOCAL-001..~305, through 2026-06-12, reference only).
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; run cd-to-root + venv python + RELATIVE path).
- Full smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale;
switchable S3-S6 (closing phase, AFTER 3D); 3D GPU lanes until S-3D-0 + the operator green light.
(**LTX-AV audio-input lane MOVED OUT of PARKED 2026-06-17** -- operator revived it as the CURRENT STEP
(section 1): M1+M3 are shipped, the remaining work is recipe-align + M4 GPU smoke. It already uses the good
Q3_K_M GGUF unet; do NOT rebuild from scratch.)

(**STORY-ENGINE quality roundtable (2026-06-21) -- PARKED side campaign.** A 4-pass live roundtable converged a
sprint-ready plan for 8 content-only story-engine fixes (length tail / costly-choice binding / ending-aware outro
/ gender-pronouns / speech register / narration hygiene / arc-shape variety; F9 reorder + F10 anti-repeat list
deferred). Docs: `docs/2026-06-21-allnight-864-frontier/` -- `SPRINT_READY_PLAN.md` + `STORY_ENGINE_KICKOFF.md` +
`roundtable/pass0{1,2,3,4}_judgment.md`. All content-only inside the FIXED ledger, ZERO workflow-JSON edits
(verified vs the real consumers). NOT active -- the visual fixes (section 1) + the forward order win. Resume only
on an explicit operator green light.)
