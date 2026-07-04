# Widget Surface Audit -- r1 hardened draft (post-codex, Claude-judged)
Base: docs/2026-07-03-widget-audit/WIDGET_SURFACE_AUDIT.md. Changes below are grounded; judgment log at end.

## Corrections to the findings table
1. NODE 86/93 CAPTION OWNERSHIP -- REVERSED from the original doc. otr_caption_burn.py:1-20 declares OTR_CaptionBurn (node 86) the dedicated CW-4 home for the SDH burn, "used to live inside the legacy OTR_PostUpscaleProcgenBlend (now being torn out)". Node 93's burn_captions=true (otr_post_upscale_procgen_blend.py:901,988-995) is the LEGACY remnant, not the load-bearing path. The saved workflow (86 OFF, 93 ON) contradicts the intended architecture. Fix direction: make node 86 the single owner (enable there or via OTR_BURN_CAPTIONS), strip the caption path out of node 93, update workflow JSON + tests in ONE change. Operator still gates whether captions default ON/OFF.
2. NODE 87 VideoDirector alias row -- RETRACTED (sonnet MISREAD). Dropdown is built from registry.all_engine_names() + aspect labels only (otr_video_director.py:137-150); _LEGACY_ENGINE_ALIASES (:110-127) exist ONLY in the pick-parse path for old saved graphs and are never displayed. No duplicate options. Cleanup batch 4 is CUT.
3. stereo_policy (81/82/83) + delivery_profile (80) -- verdict stays HIDE but redefined: remove from the WIDGET SURFACE ONLY; the values are behavior-bearing (mono conversion _otr_voice_node_common.py:362; stable_audio_theme.py:119,138,195; delivery_profile fail-closed validation + meta stamp cast_lock.py:142,175-176 and _otr_delivery_profiles.py). Preserve internal constants/defaults and output semantics; do NOT delete the feature.

## Cleanup plan v2
- Batch 1 (surface-only removal): drop stereo_policy x3 + delivery_profile from INPUT_TYPES; hard-code current defaults internally; rebuild positional widgets_values for nodes 80-83 mapped by LIVE INPUT_TYPES order (node 80 today: default | auto_registry | neutral | true), never by serialized inputs[]; validate node 80 separately; OTR_WorkflowValidator + JSON round-trip + suite + Bug Bible.
- Batch 2 (tooltip-only; NO key renames): tooltips for mode-conditional engine/oom_index, env-shadowed story_scaffold/refine_target_grade, slot-model handshake, manual_line_ids precedence. Widget KEY renames are a schema migration (_otr_workflow_apply.py addresses widgets by name, e.g. _is_engine_director_admissible :216-235 and the node-1 creative-dial carve-outs :497-503) -- CUT from this pass.
- Batch 3 (caption ownership): single-owner resolution per correction 1.
- Batch 4: CUT (retracted).
- Add per-node before/after table (visible widgets removed, constants preserved, widgets_values count before/after) to each batch PR.

## Honesty ledger
- Two-tier confidence: ~60 widgets semantically reviewed; the remaining ~65 passed only the mechanical consumption grep.
- Verify-at-build: sonnet line cites not yet independently re-read (OTR_LedgerScriptWriter.py:1662-1682, :2098-2170, :2186; OTR_LedgerFreezeCascade.py:246-266; otr_video_render_batch.py:152-165,189-197; render_driver.py:2497-2524; widget_mapping.json claim that the profile applier skips stereo_policy).

## Judgment log (r1)
- codex MUST-FIX 1 (caption ownership) CONFIRMED via otr_caption_burn.py header -- accepted, reverses the original doc.
- codex MUST-FIX 2 (hide != delete behavior) CONFIRMED via grep of consumption sites -- accepted.
- codex MUST-FIX 3 (rename = schema migration) CONFIRMED via _otr_workflow_apply.py -- accepted; renames cut.
- codex SHOULD-FIX 1 (node 80 positional recipe) CONFIRMED (delivery_profile mid-list, cast_lock.py:88-106) -- accepted.
- codex SHOULD-FIX 2 / CUT 1 (alias dedupe false premise) CONFIRMED -- batch 4 cut, node 87 row retracted.
- Claude anchor items 1-3 (coverage tiering, verify-at-build cites, node 86 rewiring context) folded in.
