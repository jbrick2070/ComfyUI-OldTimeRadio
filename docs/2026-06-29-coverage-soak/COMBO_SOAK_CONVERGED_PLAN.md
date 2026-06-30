# COMBO SOAK + CLEAN-BREAK -- CONVERGED PLAN (kibitz r1-r3, 2026-06-30)

Hardened across a kibitz arc (Claude Code + Codex + Antigravity, all grounded vs the real repo).
Full design history + agent reviews: `kibitz-runs/2026-06-29-cleanbreak-soak/{r1,r2,r3}/`. The
INPUT spec is `CLEANBREAK_SOAK_REENGINEER_PLAN.md` (same folder). This is the build-ready convergence.

## THE OPERATOR ASK (2026-06-30): "modified json that starts with story ledger + audio, writes
## stills + video, upscales to obs, all 14-18 combos."

DECISIVE r3 GROUNDING: the per-beat engine_id is baked into the ledger by ShotLock (node 90), UPSTREAM
of the image dispatcher (node 91). So a bake boundary AT node 91 cannot vary engines. The correct
"start from story + audio" boundary is UPSTREAM OF THE DIRECTORS.

## THE MODIFIED COMBO-SOAK WORKFLOW (build-ready spec)
From `workflows/otr_scifi_16gb_full.json`, REPLACE the upstream generators with BAKED inputs, keep the
rest LIVE:
- BAKE (from one CLEAN reference run): node-62 `script_json` (the frozen STORY ledger) as a literal +
  the node-7 AUDIO artifacts (master wav + episode_audio) loaded from disk. Removes node-1 (writer) +
  the TTS/music front-end -> no gemma / no TTS per leg; ONE story + byte-identical audio for every
  combo (apples-to-apples, minutes not ~28 min).
- KEEP LIVE: 87 VideoDirector -> 88 ImageDirector -> 89 MetaBrief -> 90 ShotLock -> 91 ImageGenDispatcher
  -> 92 VideoRenderBatch -> 84 SilentComposite -> 86 CaptionBurn -> 94 SceneAwareScopes -> 93
  PostUpscaleProcgenBlend (the UPSCALE) -> 85 MasterAudioMux (publishes the final to output\otr\obs).
- PER-COMBO KNOB: patch node-87 video-engine dropdown(s) + node-88 image-engine selection per leg; the
  director->shotlock->prompts->image->video chain re-runs on the SAME baked story+audio so the swap is
  atomic. CLEAR `ledger["images"]["images"]` + `["cache_index"]` so stills re-mint per leg.
- AUDIO byte-identical by construction (node-85 muxes the same baked master). UPSCALE + OBS publish are
  already in the chain (nodes 93 + 85).
- MATRIX: the 15 video + 5 image legs the runner already enumerates (= the operator's "14-18").

## CAPTURE SEAMS (to enable the bake)
node-92 already writes `state/node_episode_input.json` (S-F). ADD: a node-7/62 artifact capture
(frozen script_json + master wav + episode_audio) + a dispatcher `node_image_report.json`
(image_done + made/reused + by_role) so the image engine's execution is PROVABLE (node-91 is not an
OUTPUT_NODE).

## CLEAN-BREAK RIP-OUT (ships independently; operator look-QA when awake)
KEEP (grounded production): `FamilyInputGap` + `_assert_family_inputs_satisfiable`, `engine_family`,
`classify_failure`, `RenderError`. REMOVE: `make_fallback_of`/`FLOOR_NAMES`/`UNIVERSAL_FLOOR`/
`SYNTH_FALLBACKS`, `OomSignal`+`force_oom`, the WHOLE soak (`build_soak_fixture`/`run_gpu_soak`/
`assert_soak_ok`/`_PROFILES`/`_CHAR3D`/`OOM_ENGINES`/`EXPECTED_OOM_TRAIL`/`SoakError`/`RenderFloorError`
+ mode="soak"), `nodes/_otr_shared/fallback.py`, every engine `fallback_engine` attr, the
`retry_taxonomy` fallback bits. Migrate/delete the grounded tests/scripts/profiles/fixtures in-commit
(test_video_render_driver / _survival_guide_vectors / _character_3d / _gpu_smoke / _soak_fixture /
fallback_chain_additive / retry_taxonomy(+additive); scripts otr_video_soak.py /
otr_video_gpu_smoke.demonstrate_humo_fallback / run_otr_30word_smoke allow_auto_fallback asserts;
config/profiles/8gb_lite.json + cpu_floor.json station_card refs; tests/debug_prompt.json). Retire
abstract/station_card/still_motion AFTER reassigning their scene_broll/background_abstract/
announcer_visual defaults to still_pan/still_flat. `allow_auto_fallback` clean-delete + JSON
widget-vector rebaseline. Prune the `ENGINE_FAMILY` dict with the retirement.

## STATUS
Plan BUILD-READY (r4 would be a confirm pass; no new must-fix expected). The PROVEN coverage runner is
rendering the 18 pending combos -> obs overnight as the immediate deliverable; the baked modified
workflow above is the faster apples-to-apples follow-up to build attended.
