VERDICT: no. Core contracts are missing: format engines cannot access the episode-level data the plan requires, and the Kling/cloud invocation path is not implemented yet.

MUST-FIX BEFORE BUILD:
1. [1, 3, 4] `fmt_evidence_board` / `fmt_tin_toy` cannot be added as just new files. A video engine must be registered/imported and declared in multiple places: `nodes/_otr_video_engines/__init__.py` imports registered engines, `nodes/_otr_video_engines/registry.py` has `CAPABILITIES`, `nodes/_otr_video_engines/render_driver.py` has `ENGINE_FAMILY`, and tests enforce `set(vreg.CAPABILITIES) == set(vreg.all_engine_names())` in `tests/test_capability_profiles.py:211`. Concrete fix: specify exact engine family, `required_inputs`, `@register`, import, `CAPABILITIES`, `ENGINE_FAMILY`, and tests for both rows.

2. [3-a, 3-c, 4-a, 4-b, 4-d] The current per-shot request contract does not carry the data these formats need. `VideoRequest` only has `asset_refs`, `conditioning_refs`, `audio_ref`, `base_clip_ref`, timing/canvas/etc. and forbids extras in `nodes/_otr_video_engines/schemas.py:78` and `nodes/_otr_video_engines/schemas.py:139`. `build_request_from_shot` only adds the current shot’s init image/audio and `char_id` in `conditioning_refs` at `nodes/_otr_video_engines/render_driver.py:1310` and `:1696`. It does not pass cast lists, portrait hashes, board coordinates, `episode_evidence_hash`, concept-sheet paths, GLB paths, or episode asset roots. Concrete fix: add a planning/stamping phase in ShotLock/ImageDispatcher that writes format assets and metadata into ledger/request-safe fields, then validate those fields before `render_clip`.

3. [1, 7] `visual_format` is underspecified and will not “flip all three per-role defaults” by itself. `OTR_VideoDirector.direct()` currently resolves only the three per-role model widgets plus `character_video_model` into `policy["video_models"]` in `nodes/otr_video_director.py:291` and `:307`; no format-level switch exists. Also “explicit per-role picks still win” is ambiguous because required dropdowns always have values. Concrete fix: define default-detection rules, append the widget without shifting saved values, update `direct()` to override only slots still at known defaults/sentinel, and add workflow/widget-vector tests.

4. [3-c, 4-d, 5] The Kling lipsync call path is not buildable from current code. `partner_nodes.yaml` pins `cloud_kling_lipsync` with required `audio`, `video`, `voice_language` at `nodes/_otr_shared/partner_nodes.yaml:135`, but the video cloud adapter/invocation path is not present and `canonicalize_video()` is still `NotImplementedError` in `nodes/_otr_shared/cloud_media_canonical.py:112`. Concrete fix: make F1/F2 depend on the completed S3 adapter that exposes a callable `kling_lipsync` engine/helper returning a canonical silent VIDEO asset, not just the pinned YAML row.

5. [1, 3, 4] Asset placement cannot be met from `render_clip` as described. Existing local engines commonly write temp/cache outputs, e.g. `cheap_families.render_clip` uses `otr_engine_tmp_mp4` in `nodes/_otr_video_engines/cheap_families.py:125`, and `mesh_stage` stages under `otr/episodes/_shared/mesh_cache` in `nodes/_otr_video_engines/eng_mesh_stage.py:391`. The plan requires `otr/episodes/<ep>/evidence_board` and `tin_toy`, but the render request does not carry `episode_id` or `episode_dir`. Concrete fix: pass a canonical episode asset root into the request/ledger before render, and make every generated board/crop/plate/mesh copy land there first-class.

6. [4-b] F2 mesh cache key is under-specified. The plan says `portrait hash + tin_toy profile version`, but the existing mesh cache deliberately includes subject id, portrait SHA, mesher id, and mesher version in `nodes/_otr_video_engines/eng_mesh_stage.py:106`. Concrete fix: include selected 3D row id, adapter/export version, tin-toy profile version, subject id, and source content hash to avoid cross-adapter/cache collisions.

SHOULD-FIX:
1. [5] “FACE-SIMILARITY check” is not implementable as written. No threshold, model, crop normalization, retry/fail behavior, or ledger stamp is specified. Concrete fix: name the checker, threshold, input crop source, and failure action.

2. [4-c] “Blender 4.5.10 shipped + selftested” is not enforced by current code. `mesh_stage` checks `OTR_BLENDER_EXE` and runs a cube selftest, but no version gate appears in `nodes/_otr_video_engines/eng_mesh_stage.py:338` or `:608`. Concrete fix: either add a `blender --version` gate or remove the exact version claim.

3. [3-c, 4-d] Per-line Kling jobs may be slow. Current cloud backend defaults KLING concurrency to 1 in `nodes/_otr_shared/cloud_media_backend.py:153`. Concrete fix: estimate full-episode line count latency and define concurrency/backoff/retry behavior before acceptance says “full episode”.

OPTIONAL / NICE-TO-HAVE:
- [3-a] Add a tiny board-layout manifest schema with stable IDs for polaroids, pins, strings, crop boxes, and layer hashes; it will make cache/debugging much easier.
- [4-c] Add a Blender render manifest per plate: mesh hash, camera params, frame count, fps, lighting profile, and output path.

CUT THESE (over-engineering):
1. [5] Cut the photoreal CG render from V1 for the F2 MVP. The tin-toy path is the build target; testing photoreal CG belongs to the separate Prop Shot mouth-path decision, not this format’s first build.

2. [4-e] Cut whole-episode tin-toy discussion from this build doc. It is explicitly deferred, so leaving it in the coding plan creates scope noise without changing the MVP.