# Widget Surface Audit -- otr_scifi_16gb_full.json
Date: 2026-07-03 | Method: static AST inventory + repo-wide consumption grep (widget_audit_raw.json) + one batched semantic subagent pass (sonnet). Read-only; no code or JSON changed in this pass.

## Headline
- 22 nodes, ~125 surfaced widgets. Zero widget-count drift between the JSON and live INPUT_TYPES.
- Zero truly DEAD (never-read) widgets, and no dropdown offers retired options (sfx / scene_broll / background_abstract / character_3d are all clean).
- The clutter is CONFUSION, not corpses: single-option placeholder dropdowns, mode-conditional widgets that go silently inert, env-var shadowing, and one duplicated toggle pair.

## Findings (verdict != KEEP)

| Node | Widget | Verdict | Evidence |
|---|---|---|---|
| 1 OTR_LedgerScriptWriter | story_scaffold | RENAME-CLARIFY | OTR_LedgerScriptWriter.py:1662-1682 -- "auto" silently defers to env OTR_ENABLE_STYLE_GRAMMAR; same saved value behaves differently across restarts. |
| 1 OTR_LedgerScriptWriter | refine_target_grade | ENV-SHADOWED | ~:2186 -- OTR_STORY_REFINE_BAR / OTR_STORY_REFINE_PASSES override it in headless runs. |
| 1 OTR_LedgerScriptWriter | openrouter_slot_a/b_model, comfy_slot_a/b_model | RENAME-CLARIFY | :2098-2170 -- inert unless creative_writing_model/technical_model is set to the magic string "openrouter:slot-a" etc. Undocumented two-widget handshake. |
| 62 OTR_LedgerFreezeCascade | protagonist_only | RENAME-CLARIFY | OTR_LedgerFreezeCascade.py:246-266 -- populated manual_line_ids silently supersedes render_selection + protagonist_only. |
| 80 OTR_CastLock | delivery_profile | HIDE | cast_lock.py:100-103 -- available_delivery_profiles() returns only "neutral". Single-option dropdown. |
| 81 OTR_BatchCharacterVoices | stereo_policy | HIDE | _otr_voice_node_common.py:235 -- single-option ["mono_safe"]. |
| 82 OTR_AnnouncerVoice | stereo_policy | HIDE | Same shared helper; untouched by the GATE B profile applier. |
| 83 OTR_StableAudioTheme | stereo_policy | HIDE | stable_audio_theme.py:119 (comments :122-125) -- placeholder for a future stereo pipeline; no second value exists. Confirms the mechanical suspect. |
| 86 OTR_CaptionBurn | burn_captions, caption_style, fps, ffmpeg | RENAME-CLARIFY / candidate HIDE | Node 86 saved burn_captions=false = pass-through no-op; node 93 (otr_post_upscale_procgen_blend.py:901,988-995) carries the LIVE burn_captions=true. Two identically named toggles in one chain; flipping node 86's does nothing visible. |
| 87 OTR_VideoDirector | announcer/music/character_video_model | RENAME-CLARIFY | _otr_video_engines/registry.py:243-365 -- legacy alias pairs (flat_still->still_flat, flux_still->still_pan, still_kenburns->still_motion, visualizer->viz_green) make duplicate-looking options collapse to one engine. |
| 92 OTR_VideoRenderBatch | engine | RENAME-CLARIFY | otr_video_render_batch.py:152-165 -- only read when mode=="single"; dead in production mode="episode". |
| 92 OTR_VideoRenderBatch | oom_index | RENAME-CLARIFY | :189-197 vs render_driver.py:2497-2524 -- only meaningful in mode=="soak". |

~45 of the ~60 deep-checked widgets are clean KEEP; the rest of the 125 passed the mechanical consumption check.

## Top 5 confusion offenders
1. stereo_policy x3 (nodes 81/82/83) -- systemic single-option placeholder; delete across _otr_voice_node_common.py + stable_audio_theme.py in ONE commit with positional widgets_values re-audit.
2. Duplicate burn_captions (node 86 vs node 93) -- only node 93's is load-bearing; node 86 is a pass-through in production.
3. VideoRenderBatch engine/oom_index -- meaning depends invisibly on the mode dropdown.
4. Writer slot-model dropdowns -- inert without the paired magic-string dropdown.
5. CastLock delivery_profile -- illusion of choice (one profile).

## Cleanup plan (separate change; NOT done in this pass)
Per CLAUDE.md section 0, any removal/hide MUST edit INPUT_TYPES and workflows/otr_scifi_16gb_full.json in the SAME change, remembering widgets_values is POSITIONAL (removing a mid-list widget shifts every later saved value -- BUG-LOCAL-097 class). Order of attack:
1. HIDE batch: remove the three stereo_policy widgets + delivery_profile (or leave as hidden constants read from code), rebuild widgets_values for nodes 80-83, run OTR_WorkflowValidator + JSON round-trip + full suite + Bug Bible.
2. Tooltip/label batch (no positional risk): add tooltips or renames for the mode-conditional (engine "single-mode only", oom_index "soak-mode only"), the env-shadowed writer widgets, the slot-model handshake, and manual_line_ids precedence.
3. Decide node 86: either drop OTR_CaptionBurn from the graph (node 93 owns caption burn) or rename its widgets to make the pass-through explicit. Operator call.
4. Dedupe the video-engine alias options in the three VideoDirector dropdowns (registry-level: stop listing both alias and canonical name).

Raw data: widget_audit_raw.json (same folder). Audit script was a temp probe (deleted, not committed).
