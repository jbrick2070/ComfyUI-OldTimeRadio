<!-- requested_model: openai/gpt-5.5 | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan changes the image-prompt schema, dispatcher contract, path semantics, and video init selection without specifying the compatibility and ordering work needed for the grounded code to run.

MUST-FIX BEFORE BUILD:
1. [ST-2/ST-3] Schema mismatch between planned scene-still objects and current dispatcher input. `OTR_MetaBriefImagePromptGen.generate()` currently returns `char_id -> {prompt,prompt_hash,source}`; `dispatch_images()` iterates that map as `(cid, pinfo)` and hardcodes `role = "character_video"`. Planned objects `{object_id, kind, role, beat_id, w, h, ...}` will be treated as character IDs or ignored/misstamped. Concrete fix: define one versioned image prompt schema, e.g. `{"objects":[...]}` or a list of objects, and update both prompt generator and dispatcher to consume it. Preserve backward compatibility for portrait rows or migrate all portrait rows to the new object schema in the same patch.

2. [ST-2] Scene stills must skip the portrait person guard, but current prompt derivation applies `_depicts_person()` to every generated prompt inside `derive_image_prompts()`. If scene stills are added “alongside portrait rows” in that path, non-person open/scene prompts will fall back to portrait templates. Concrete fix: split portrait and scene-still derivation paths, or branch by `kind` before consistency/person/gear guards so `kind=scene_*` gets only scene-safe validation and no person guard.

3. [ST-2/ST-6] Opening still derivation depends on audio timing and fps, but the image prompt node currently has only `script_json`, `image_policy_json`, and optional `gate_in`. `derive_opening_music_beat(ledger, fps)` returns no open unless the first line has `start_s`; ShotLock currently calls `overlay_audio_timing()` after `audio_done` to fill that timing. The plan does not wire `audio_done`, does not say the image node overlays timing, and does not define the fps source. Concrete fix: either derive scene-still objects after ShotLock, or add an explicit audio gate/timing overlay to the image prompt phase and define fps/canvas source from policy or ledger.

4. [ST-2] Landscape dimensions are underspecified. “canvas-derived /32” cannot be implemented from the current `OTR_MetaBriefImagePromptGen` inputs unless `image_policy_json` is guaranteed to carry canvas dimensions; the grounded code’s video canvas currently lives in ShotLock’s `video_policy_json` / `ledger['video']['canonical_canvas']`, which may not exist yet in the image phase. Concrete fix: specify the exact field used for scene-still `w/h`, add fallback defaults, and snap both dimensions to multiples of 32 in one helper covered by tests.

5. [ST-3/ST-4] Episode-local paths will break current render-driver consumers. `_portrait_index(ledger)` returns `im["path"]` directly, and `build_request_from_shot()` passes that string as `asset_refs.init_image`. `build_clip_manifest()` also calls `os.path.exists(path)` directly. If ST-3 changes `path` to “episode-local”, engines and existence checks will fail unless the process cwd is the episode dir. Concrete fix: store both `path` as manifest-local and `abs_path`/`runtime_path`, or add path resolution in `_portrait_index()`, new `_still_index()`, and manifest building against the episode root.

6. [ST-4/ST-5] Static-motion scene-still selection will still use portrait defaults unless the request canvas/init dimensions are changed. In grounded `build_request()`, default canvas is `(480,832)`, and `build_request_from_shot()` only overrides landscape dimensions for `engine_id in ("ltx_video", "wan_i2v")`. `still_kenburns` currently remains portrait-shaped even if given a landscape scene still. Concrete fix: when init source is `scene_still` for `static_motion` or `image_to_video`, set `request["canvas"]`, `init_w`, and `init_h` from the scene-still `w/h` or ledger canonical canvas, including `still_kenburns`.

7. [ST-3] Cache-hit materialization is not implementable as stated from the current cache shape. Current `cache_index` maps `request_cache_key -> image_id`; on hit the dispatcher only reports and continues. A fresh episode ledger row and copied/link materialized file require resolving `image_id` to path/content hash, or consulting a global manifest. Concrete fix: change cache index values to include `{image_id,path/content_hash,engine_id,...}` or build a reverse lookup over existing/global image rows, then on hit copy/link into `episodes/<ep>/stills/` and append a new ledger row.

8. [ST-2] “announcer/outro via the role mapping over lines” is not a buildable algorithm. Current role mapping in `otr_shot_lock.py` maps many lines to `announcer_visual`; it does not identify “outro”, and applying it literally can emit every announcer beat, contradicting “v1 scope: open + announcer + outro beats only”. Concrete fix: define exact selection rules, e.g. synthetic open if `derive_opening_music_beat()` returns one; first `announcer_visual` line as announcer; last qualifying `announcer_visual` or `music_close` line as outro; de-dupe if same beat.

9. [ST-3] [ASSUMPTION] ImageDirector slot names are unverified. The plan names `announcer_image_model`, `music_image_model`, and `other_beats`, while grounded dispatcher currently reads `image_models.other_beats_image_model`. If actual policy keys differ, every non-current slot will resolve to no engine and skip generation. Concrete fix: verify the real ImageDirector JSON contract and implement exactly those key names; add a fixture test for all planned roles/kinds.

10. [ST-4] Request selection by “ENGINE FAMILY” must account for fallback attempts sharing one request. Grounded `render_shot()` builds one request before the fallback chain and reuses it for all candidates. If an `image_to_video` or text engine falls back to `still_kenburns`, the floor will inherit whatever init selection the original request had. Concrete fix: either define this as intentional, or rebuild/patch `asset_refs.init_image`, `canvas`, and trace `init_source` when `render_shot()` restamps to a fallback family.

SHOULD-FIX:
1. [ST-1] `era_tail_profile="still"` does not exist in grounded `_otr_story_brief_helpers.py`; only `get_era_tail()` and `finish_visual_prompt(max_chars, style_tail)` exist. Concrete fix: either add the profile parameter to shared helpers or implement `compose_still_prompt()` with its own bounded still-tail helper; ensure `prompt_hash` is computed after final trimming.

2. [ST-1/ST-4] The parity test requires driver LTX text prompt and open still prompt to share the same leading subject, but grounded `render_driver.build_request_from_shot()` currently hardcodes three open-subject branches. Concrete fix: replace those hardcoded strings with `get_open_subject(role, synthetic)` in the driver and in still prompt composition.

3. [ST-3] Content hash naming changes risk breaking existing readers. Current dispatcher writes `portrait_content_hash`; ST-3 says ledger rows carry `content_hash`. Concrete fix: write both during migration or verify and update every reader that expects `portrait_content_hash`.

4. [ST-3/ST-6] Adding `episode_id` as a required dispatcher input is a sequencing risk for existing tests/workflows. Concrete fix: make it optional with a safe fallback until `workflows/otr_scifi_16gb_full.json` is updated and tests assert the new wire.

5. [ST-3] Hardcoding `output/otr/episodes/<ep>/stills/` may bypass existing output-root configuration. Concrete fix: use the project path helper used elsewhere for episode roots, or explicitly verify `OTR_OUTPUT_DIR` behavior.

6. [ST-5] “VERIFY still_kenburns accepts external init; add if not” is not a build step. Concrete fix: make this a concrete precondition with a failing CPU/unit test or explicitly include the engine-wrapper patch in scope.

7. [ST-7] Determinism tests must include cache-hit materialization and episode-local copy/link behavior, not only prompt hashes/seeds. Otherwise ST-3 can pass direct generation but fail on the second render.

OPTIONAL / NICE-TO-HAVE:
- Add a small migration helper that normalizes both legacy `char_id -> prompt` maps and new object-list prompt schemas for one release.
- Include scene-still rows in trace/report output from the image phase so operator can inspect prompt, role slot, dimensions, cache status, and final path without opening the ledger.

CUT THESE (over-engineering):
1. [ST-3] Cut global-pool retirement discussion from this pass. It is explicitly out of scope and does not affect restoring current episode stills; leaving it in the build plan adds sequencing noise.
2. [ST-5] Cut the WAN probe workflow from v1 acceptance unless the engine wrapper patch is actually included. The core restoration goal works with `still_kenburns` scene-still drift; WAN env-gated probing is separate GPU-risk work.
3. [ST-1] Cut a broad “era-tail profile” framework if only one new still profile is needed. A single `compose_still_prompt()` helper with bounded tail behavior is enough for this pass.