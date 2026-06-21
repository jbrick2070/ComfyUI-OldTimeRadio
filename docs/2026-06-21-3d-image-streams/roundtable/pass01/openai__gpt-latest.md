<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan leaves core routing, subject identity, ledger indexing, and composite behavior unresolved, and it contradicts the grounded render path for mesh_stage init_image.

MUST-FIX BEFORE BUILD:
1. [Problem] The stated current source is wrong against the excerpt: `build_request_from_shot()` treats every `image_to_video` engine as `_SCENE_INIT_FAMILIES`, and `MeshStageEngine.family == "image_to_video"`, so mesh_stage currently gets the per-beat scene still when one exists, not the per-character portrait. This also makes the current mesh cache per-scene-still hash, not per-character. Concrete fix: add an explicit mesh-source capability and branch before `_SCENE_INIT_FAMILIES` so mesh-capable engines resolve `asset_refs.init_image` from a mesh-fodder index, not `_still_index()`.

2. [The design / Invariants: “gate on capability field”] The capability named in the plan does not exist in the grounded engine: `MeshStageEngine` has `required_inputs = ("init_image",)`, `family = "image_to_video"`, `uses_still = True`, but no `requires_mesh_portrait`, no `character_3d`, and no mesh-specific capability. Gating on `required_inputs` would also catch Wan/LTX/still engines incorrectly. Concrete fix: add a real registry-visible boolean or enum, e.g. `requires_mesh_fodder = True` or `mesh_source = "single_image_subject"`, to mesh_stage and any future equivalent, and use only that in ImageDirector/prompt generation/render_driver/manifest stamping.

3. [Open questions: “Where does the fork live?”] This is not optional; without choosing the seam, no buildable implementation exists. The dispatcher only sees image objects and role-level policy; it does not rewrite video requests. `render_driver.apply_engine_override()` can change `engine_id` after image dispatch, so an env-forced mesh_stage episode would not get mesh_fodder/background_plate objects. Concrete fix: resolve the selected video engine capability before image prompt derivation, using the same final engine map that render will use, including `OTR_FORCE_ENGINE_MAP` or an explicit “engine overrides are applied before image policy/prompt derivation” step. Then emit the extra image objects from that seam.

4. [Open questions: Ledger `images` taxonomy] Adding `kind=mesh_fodder` with the same `object_id`/`char_id` will collide with existing indices. `_portrait_index()` currently indexes any row with `object_id` or `char_id` and uses `setdefault`, with no `kind` filter; `_still_index()` only reads `kind.startswith("scene_")`. Concrete fix: replace generic `_portrait_index()` use with kind-specific resolvers:
   - `portrait_index`: only `kind == "portrait"` or existing legacy portrait rows.
   - `mesh_fodder_index`: only `kind == "mesh_fodder"` keyed by subject id/char id.
   - `background_plate_index`: only `kind == "background_plate"` or a deliberately named scene kind.
   Do not let mesh_fodder rows be visible to HuMo portrait lookup or scene still lookup.

5. [Open questions: Ledger `images` taxonomy / The design 2] If the new plate kind is literally `background_plate`, current `build_clip_manifest()` will never find it because it uses `_still_index(led).get(bid, "")`, and `_still_index()` only accepts `kind` values starting with `scene_`. Concrete fix: either name the plate kind `scene_background_plate` and ensure it does not get fed as normal init still accidentally, or add a separate `background_plate_index()` and have mesh-capable rows stamp `bg_still_path` from that index.

6. [The design 2 / Problem] The plan says the mesh should be composited opaque and not ghosted, but the only grounded code shown merely stamps `bg_still_path`; the actual composite behavior is not specified here. Concrete fix: update the composite path that consumes `bg_still_path` to use normal straight-alpha “source over” at full opacity for mesh directory clips. Verify: `_silent_procgen_blended_final` currently ghosts/blends mesh over plate; change the alpha/opacity rule there and add a regression check that mesh pixels with alpha 1.0 remain opaque.

7. [Open questions: Subject selection] The plan admits announcer/music slots have no character but `MeshStageEngine.roles` includes `music_visual` and `announcer_visual`; current `_request_character_id()` falls back to `"uncast"`. That creates unusable subject prompts and misleading cache/manifest identity for non-character beats. Concrete fix: define a deterministic policy before build:
   - character beat with valid `char_id`: mesh the character.
   - no valid character: either select a ledger story object with a stable `object_id`, or reroute/hard-fail before render.
   - do not allow mesh_stage to render announcer/music on an environment plate as `uncast`.

8. [Open questions: Mesh cache] The current mesh cache identity function is `mesh_cache_key(character_id, portrait_sha256, ...)`; for story objects there is no character id, so all object meshes become `"uncast__<hash>..."` and the manifest writes `"character_id": "uncast"`. Content hash avoids file collision, but provenance and stable object reuse are wrong. Concrete fix: generalize the cache subject field to `mesh_subject_id` and stamp it into `conditioning_refs`, cache key, and manifest. For characters, `mesh_subject_id = char_id`; for artifacts, `mesh_subject_id = object_id`.

9. [Open questions: Prompt templates] The design requires prompt scaffolds but provides none. This is a functional dependency, not polish, because the entire fix relies on generating isolated subjects and subject-free plates. Concrete fix: check in concrete positive/negative template strings for:
   - character mesh fodder,
   - object/artifact mesh fodder,
   - background plate,
   with width/height/aspect policy and SFW/UTF-8 constraints. If engines differ in negative-prompt support, define how unsupported negatives are handled.

10. [Open questions: Aspect] The plan does not specify the actual dimensions written into image prompt objects. The dispatcher already keys cache on `kind,w,h` and passes `width/height`, but no values are defined. Concrete fix: set explicit dimensions per kind, e.g. mesh_fodder near-square/portrait and background_plate 16:9, and ensure `render_driver` passes mesh_fodder to mesher while `build_clip_manifest` passes the 16:9 plate to composite. Do not infer either from the mesh_stage render canvas.

11. [Invariants: “LOUD fallbacks” / Open questions: “fall back to a 2D engine?”] This contradicts the grounded render driver: `render_shot()` disables fallbacks and raises `RenderError` on any failure. Concrete fix: either remove “fallback to a 2D engine” from this design and make charless/invalid mesh subjects a pre-render hard fail, or reintroduce a planned reroute before render. Do not rely on runtime fallback for mesh_stage.

12. [The design / Open questions: “NEVER hardcoded engine-name check”] Grounded `build_clip_manifest()` currently uses `str(eid) == "mesh_stage"` to stamp `bg_still_path`. That directly violates the invariant if more mesh-capable engines are added. Concrete fix: replace that check with the new mesh capability lookup from the video registry, with fail-safe behavior if the engine cannot be resolved.

SHOULD-FIX:
1. [Open questions: Ledger `images` taxonomy] Define row ordering semantics. `_portrait_index()` currently uses first row wins; `_still_index()` uses newest row wins. Mesh fodder and background plates should explicitly use newest materialized episode-local row, because dispatcher cache hits append fresh rows for the current episode.

2. [Open questions: Determinism + invariants] Add test coverage for cache separation: same `char_id` with `kind=portrait` and `kind=mesh_fodder` must produce distinct `request_cache_key()` values and must not cross-feed HuMo vs mesh_stage.

3. [Open questions: Determinism + invariants] Add an integration assertion that audio sections remain byte-identical after adding mesh_fodder/background_plate rows. This should be mechanical because the dispatcher only writes `ledger["images"]` and `meta`, but it needs a regression test.

4. [The design 1] The text says “front-or-3q view”; Hunyuan single/mv reconstruction quality may be sensitive to full-body vs bust vs object scale. Define framing per subject type. Otherwise the prompt generator can produce portraits that are still face-only heads when a full object mesh is desired.

5. [Open questions: Mesh cache] The design says “per-character fodder so a stable cast reuses its mesh across beats/episodes,” but the dispatcher materializes episode-local copies and prompt hashes may include beat-specific context if not controlled. Concrete fix: ensure character mesh_fodder prompts are cast-level, not beat-level, unless explicitly intended to rebuild.

6. [Open questions: Where does the fork live?] Verify: `OTR_ImageDirector` actually reads video engine capability and emits `locked_3d_slots` as the dispatcher docstring implies. If not, do not place the fork there without adding that data flow.

7. [The design 2] Subject-free background plates need a negative/constraint that prevents the subject/object from appearing. If the prompt is “complementary environment matched to scene” but derived from a character beat, the image model may include the speaker. Add explicit “empty environment / no people / no central subject” handling if the engine supports negatives; otherwise positive-only wording like “empty room, unattended set, vacant landscape”.

OPTIONAL / NICE-TO-HAVE:
- Add manifest observability fields for `mesh_subject_id`, `mesh_fodder_path`, `background_plate_path`, `mesh_fodder_hash`, and `background_plate_hash` so failures can be audited without inspecting ledger internals.
- Add a health check that rejects mesh_fodder images whose dimensions/aspect are outside the allowed range before meshing.
- Add a one-shot debug report listing, per mesh-capable beat: selected engine, subject id, mesh fodder row, plate row, cache key, and composite mode.

CUT THESE (over-engineering):
1. [Open questions: Subject selection] Cut story-object meshing for pass00 unless a real ledger story-object register already exists. Safe minimal build: mesh only beats with a valid character subject; charless announcer/music beats are rerouted or hard-failed before render. This avoids inventing object selection, cache identity, and prompts in the same change.

2. [Open questions: Mesh cache] Cut any explicit “cache migration” tool. The grounded cache already keys on source image content hash plus mesher id/version; switching to mesh_fodder naturally invalidates old portrait/scene-still meshes. Just ensure the source image is actually the mesh_fodder row.

3. [Open questions: Prompt templates] Cut per-engine bespoke prompt templates for all four image engines in the first build. Use one shared positive scaffold plus optional negative prompt plumbing where supported; add per-engine tuning only after failures are observed.

4. [The design 2] Cut a brand-new background taxonomy if the existing scene-still system can be safely specialized. Either reuse a `scene_background_plate` kind with a separate resolver, or add `background_plate`; do not support both names in pass00.