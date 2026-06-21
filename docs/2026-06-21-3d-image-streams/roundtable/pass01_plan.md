# OTR 3D Image Streams -- hardened design (pass01, post-roundtable + Claude grounding)

> Roundtable pass01: panel = GPT-5.5, Gemini-3.1-pro, Claude-Opus-4.8, DeepSeek-v4-pro, Grok-4.3
> (Claude-Sonnet errored). Claude (grounded against the real render path) is the judge. The panel
> CONVERGED on one big correction + a set of specific seams. Every CONFIRMED item below was verified
> against `nodes/_otr_video_engines/render_driver.py` + `eng_mesh_stage.py`.

## THE CORRECTION (3-model convergence, code-VERIFIED): the mesher eats the SCENE STILL, not the portrait
pass00 was built on a false premise. `MeshStageEngine.family = "image_to_video"`, and
`render_driver.build_request_from_shot` (lines ~696-708) sets `init_image = portrait` then, because
`_family in _SCENE_INIT_FAMILIES` (`{"image_to_video","static_motion"}`, line 431), OVERRIDES it to the
per-beat SCENE STILL from `_still_index(ledger)` (rows with `kind` startswith `"scene_"`). So Hunyuan3D
is fed the whole environment -> the clay blob. **The real seam is `build_request_from_shot` + the
image-prompt mint, NOT the portrait path.** This is the root cause of the operator's observation.

## The build (re-based on the real seams)

### 1. Capability flag (the gate) -- `eng_mesh_stage.py`
`requires_mesh_portrait` does NOT exist on `MeshStageEngine` today (verified). Add an explicit
registry-visible boolean **`requires_mesh_fodder = True`** to `MeshStageEngine` (and any future single-
image-subject 3D engine). Gate ALL routing on this one field -- never an engine-name or family check,
never `required_inputs`/`uses_still` (those catch Wan/LTX/still engines too).

### 2. Init selection -- `render_driver.build_request_from_shot`
BEFORE the `_SCENE_INIT_FAMILIES` override, branch: if the resolved engine has
`requires_mesh_fodder`, set `init_image` from a NEW `mesh_fodder` index (subject still), and DO NOT
fall through to the scene-still override. The scene still becomes the BACKGROUND plate only.
- **Capability must be resolved AFTER engine overrides** (GPT, grounded): `apply_engine_override` /
  `OTR_FORCE_ENGINE_MAP` can change `engine_id` after image dispatch, so an env-forced mesh_stage
  episode must still get fodder. Resolve the final engine map BEFORE image-prompt derivation.

### 3. The image-prompt fork -- `OTR_MetaBriefImagePromptGen` (the only seam that authors prompts via the LLM)
When a beat's final engine has `requires_mesh_fodder`, emit TWO objects instead of the one cinematic
scene still, both with prompts QUITE DIFFERENT from the 2D path:
- **MESH FODDER (subject):** `kind="mesh_fodder"`. Single centered subject (character OR story object),
  plain seamless neutral background, even diffuse/studio light, full unoccluded 3/4 view, no hood, no
  hands-over-face, no hard shadows, no environment. Near-square/portrait aspect (Hunyuan needs an
  isolated subject). Positive + NEGATIVE scaffolds checked in (negatives: "busy background, multiple
  subjects, occlusion, dramatic shadow, cropped, scene, environment").
- **BACKGROUND PLATE (world):** `kind="scene_background_plate"` (the `scene_` prefix so `_still_index`
  finds it -- a bare `background_plate` would be invisible). Environment ONLY, NO subject, matched to
  the scene mood/era/palette, 16:9 scene canvas.
- A 3D beat mints ONLY these two -- NOT also a generic `scene_*` still (else `_still_index`'s
  last-write-wins returns the wrong row; Gemini). If both a `scene_*` and `scene_background_plate`
  can co-exist, `_still_index` must PRIORITIZE `scene_background_plate`.

### 4. Ledger taxonomy + indices (additive, schema l3-2026-05-14 unchanged)
- New row kinds: `mesh_fodder`, `scene_background_plate`. Additive `kind` values only.
- Add **kind-specific resolvers** so the new rows are not seen by the wrong consumer:
  `mesh_fodder_index` (kind==`mesh_fodder`, keyed by `mesh_subject_id`); the plate flows through the
  existing `_still_index` via its `scene_` prefix. `_portrait_index` (no kind filter today) must NOT
  pick up `mesh_fodder` rows for the HuMo portrait lookup -- add a kind filter there.
- The IMAGE cache is already isolated: `request_cache_key` includes `kind`, so `kind="mesh_fodder"`
  cannot collide with the cinematic portrait (Gemini -- CUT the extra image-cache-key work).

### 5. Mesh cache identity -- `eng_mesh_stage` (the REAL cache bug)
`mesh_cache_key(character_id, portrait_content_hash(still))` hashes the resolved `init_image`. Today
that is the per-beat scene still -> a DIFFERENT hash every beat -> cache MISS + full mesh rebuild every
beat (defeats per-character reuse). Once init = a STABLE per-subject mesh_fodder file, the hash is
stable and per-character reuse holds. Generalize the subject field to **`mesh_subject_id`**
(`char_id` for characters, `object_id` for story artifacts) in the cache key + the manifest
`conditioning_refs` (today non-character beats write the misleading `"uncast"`).

### 6. Subject selection policy (announcer/music have no char_id)
`MeshStageEngine.roles` includes `announcer_visual`/`music_visual`, and `_request_character_id` falls
back to `"uncast"`. Define deterministically BEFORE build: (a) character beat w/ valid `char_id` ->
mesh the character; (b) no character -> mesh a story OBJECT with a stable `object_id` (key fodder by
`object_id`/`beat_id`); (c) if neither -> reroute that slot to a 2D engine or hard-fail LOUD. Never
mesh an announcer/music beat as `uncast` on an environment.

### 7. Opaque composite (kills the ghost; the operator's frame)
The composite that consumes `bg_still_path` currently blends the mesh over the plate
(`_silent_procgen_blended_final` -> double-exposure). Change the mesh-directory-clip composite to
straight-alpha **source-over at full opacity** (the mesh already renders straight-alpha). Add a
regression check: mesh pixels with alpha==1.0 stay opaque. Keep the blended look as a NAMED opt-in
style, not the default.

## Build order
(1) capability flag -> (2) init-selection branch + override-order -> (3) prompt fork in
MetaBriefImagePromptGen + prompt scaffolds -> (4) taxonomy/indices -> (5) mesh_subject_id cache ->
(6) subject policy -> (7) opaque composite. Each its own green chunk (suite + Bug Bible), commit+push.

## Verify-at-build
- Confirm `OTR_ImageDirector` (or the prompt-gen) can actually READ the engine capability at prompt
  time (the capability must reach that seam; today it is referenced only in prose/comments).
- Confirm the final engine map (incl. `OTR_FORCE_ENGINE_MAP`) is resolvable BEFORE image-prompt mint.
- Confirm `_still_index` priority when both a `scene_*` still and a `scene_background_plate` exist.

## Invariants (unchanged)
Ledger schema `l3-2026-05-14` additive-only; audio byte-identical (image-only change); capability-gated
routing (no engine-name checks); single resident heavy <=14.5 GB; 100% local; deterministic seed-keyed;
LOUD fallbacks; UTF-8 no BOM; SFW.

## Deferred (panel SHOULD/NICE, not v1)
Cycles + 3-point lighting + multi-view texture bake (the v1.5 "lit/textured" tier) -- a separate sprint
AFTER clean fodder + opaque composite land (clean fodder is the higher-leverage fix).
