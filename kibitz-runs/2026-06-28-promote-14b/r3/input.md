# Promote HuMo-14B -- r2-hardened coding plan (Codex + Claude; grounded)

r2 panel: Codex (gpt-5.5/high) + Claude. Codex VERDICT "no" -- the r1 route ("14B on
face beats only") is NOT implementable through the current single other-beats slot. All
MUST-FIX grounded + CONFIRMED. (agy offline: confirmed `agy -p` 1.0.13 needs a TTY, hangs
under headless subprocess -- Codex-only panel.)

## CENTRAL FINDING -- promotion REQUIRES per-sub-role video routing (architecture, not a flip)
`OTR_VideoDirector` validates `other_beats_video_model` with `fits_any` (NOT all-roles), so
`humo_14B_169` PASSES because `character_video` fits (otr_video_director.py:77-84,345-353) --
but `OTR_ShotLock` then stamps that ONE engine onto ALL THREE other-beat roles
(`character_video` + `scene_broll` + `background_abstract`, otr_shot_lock.py:709-780).
`scene_broll`/`background_abstract` lack audio_ref/face (role_compat.py:55-72) and
`render_shot` has NO fallbacks -- it RAISES RenderError on any engine failure
(render_driver.py:1468-1495), and the fallback chain in run_real_episode is IGNORED by
render_shot. => face-less beats would HARD-ABORT. So a clean promotion needs ONE of:
  ROUTE-A (clean, recommended): new per-sub-role data shape -- profile keys
    `character_video_model` / `scene_broll_video_model` / `background_abstract_video_model`
    (+ widget_mapping + OTR_VideoDirector widgets + ShotLock per-role stamping); set
    character->humo_14B_169, scene/background->the current compatible engine (visualizer/ltx).
  ROUTE-B (smaller, special-case): a deterministic ShotLock rewrite mapping ONLY
    `character_video`->humo_14B_169 and keeping scene/background on the configured engine.
Either way: profile JSON + widget_mapping.json + workflow JSON + tests in the SAME change.
This is an ARCHITECTURE change + an OPERATOR product call (character beats = 14B HuMo talking
head; scene/background stay on the visualizer) -- NOT the naive profile flip the task implied.

## MUST-FIX (grounded; folded)
1. Pick + build ROUTE-A or ROUTE-B so only `character_video` gets humo_14B_169; scene_broll +
   background_abstract keep a role-compatible engine. Do NOT route HuMo onto face-less beats.
2. Do NOT rely on per-beat fallback -- it is disabled (render_shot "NO FALLBACKS"). PREVENT
   incompatible assignments at routing time (Route-A/B), don't catch at render.
3. Beat-length cap AS CODE: add `HUMO_14B_SAFE_RENDER_FRAMES` (the tested-safe envelope,
   <=49-81f), render at the cap, then `wrapper_bridge.extend_frames_to_target` BEFORE encode so
   `frame_count == target_frame_count` for the manifest + the composite doesn't hold the last
   frame (eng_humo.py:53-54,339-367; wrapper_bridge.py:438-466; otr_silent_composite.py:237-262).
4. Profile<->workflow EXACT match: commit the precise saved widget string for humo_14B_169
   (verify the dropdown's exact id/aspect-suffix form) in BOTH config/profiles/16gb_full.json
   AND workflows/otr_scifi_16gb_full.json in one change; UPDATE the match fixtures to new truth
   (test_capability_profiles.py:176-205, test_workflow_apply.py:111-117) -- do not weaken.

## SHOULD-FIX (grounded)
- Acceptance asserts manifest `engine_histogram["humo_14B_169"] > 0` AND every such row is
  `role == "character_video"` (the current soak counts generic "humo", render_driver.py:
  2030-2047,2104-2107). 
- Preflight the EXACT engine: get `humo_14B_169` (HuMo14BLandscapeEngine, eng_humo.py:541-563),
  `assert_usable`, verify OTR_ENABLE_HUMO=1 + the 14B ckpt via `_ckpt_path()`.

## CUT (keep cut)
- Two-stage pre-sampler evict: post-decode `reclaim_idle_models` already runs
  (eng_humo.py:355-361) + render-loop inter-beat frees (render_driver.py:1520-1572); adding
  more eviction before the routing/cap fixes is failure surface for ~217 MB.

## Build order (after operator picks ROUTE-A/B)
per-sub-role routing (profile keys + widget_mapping + director + ShotLock) ->
HUMO_14B_SAFE_RENDER_FRAMES + extend-to-target -> profile+workflow exact-match edit + fixture
update (same change) -> OTR_WorkflowValidator + JSON round-trip + link/widget audit -> CPU test
(director rejects an aggregate engine that can't satisfy every stamped role) -> suite + Bug
Bible + B7 -> humo_14B_169 preflight -> live episode: assert histogram humo_14B_169>0 on
character rows + OBS publish + no OOM at representative AND max-cap beat -> operator eyeball.
