# r4 CONVERGENCE + RE-VERIFY: the code changed since this review was written
HEAD is now 8c3e4911 (was 116f43ff at r1). Two production changes landed in between: (1) the credits tail-chain -- NEW node 95 OTR_CreditsRoll wired 12 SignalLostVideo -> 84 SilentComposite -> 86 CaptionBurn -> 93 PostUpscaleProcgenBlend -> 95 CreditsRoll -> 85 MasterAudioMux, with node 95 feeding a FLOAT declared credits-tail into node 85 slot 6; (2) a "no-fallback rip" touching ~10 .py files, which may have SHIFTED LINE NUMBERS cited below.

A fresh mechanical re-baseline (widget_audit_raw_v2.json, same folder as v1) already confirmed: zero widget-count drift; node 95 exposes ZERO widgets; node 3 OTR_SceneSequencer dropped default_tts (6->5, JSON in lockstep); node 87 renamed other_beats_image_model -> character_image_model. Suite 6141/0 green at HEAD.

YOUR r4 JOB: (a) verify every file:line cite in the plan below still holds at HEAD -- list any that moved/vanished; (b) re-check the Batch 3 stage-order fix against the NEW tail chain: with node 95 CreditsRoll now between 93 and 85, where must CaptionBurn sit if node 86 becomes caption owner (before or after credits)? Is 84 -> 93 -> 95 -> 86 -> 85 or 84 -> 93 -> 86 -> 95 -> 85 correct, and why -- consider that credits frames probably SHOULD be caption-free but the mux guard reads node 95's declared tail; (c) investigate the validator CLI --strict-types reports claiming node types 80-83 are "not in NODE_CLASS_MAPPINGS" (pre-existing baseline per the coder window; suite green) -- real registration gap or CLI-context artifact?; (d) confirm no NEW must-fix. Do NOT re-litigate settled r1-r3 judgments unless the new code invalidates them.

---

[PLAN UNDER REVIEW -- r3 hardened draft follows]

# Widget Surface Audit -- r3 hardened draft (wiring round; codex r3 + antigravity manual review folded)

## Batch 1 -- surface-only removal (build-ready)
- Remove INPUT_TYPES entries only; keep kwargs with defaults. Exact widget vectors:
  node 80 ["default","auto_registry","neutral",true] -> ["default","auto_registry",true]
  node 81 ["indextts2","mono_safe"] -> ["indextts2"]
  node 82 ["kokoro","mono_safe"] -> ["kokoro"]
  node 83 ["stable_audio_3","mono_safe"] -> ["stable_audio_3"]
  (validator length gate _otr_workflow_validator.py:175-179 hard-fails otherwise.)
- Update tests: test_cast_lock.py:62-64, test_announcer_voice.py:87, test_batch_character_voices.py:94-95, test_stable_audio_theme.py:80.
- Update stale docstring comments: cast_lock.py:18-20, _otr_voice_node_common.py:183-188.
- Validate: OTR_WorkflowValidator + JSON round-trip + suite + Bug Bible + test_audio_byte_identical.

## Batch 2 -- tooltip-only (NO key renames): engine ("single-mode only"), oom_index ("soak-mode only"), refine_target_grade (env override), slot-model handshake, manual_line_ids precedence. story_scaffold downgraded (already tooltipped, OTR_LedgerScriptWriter.py:2196-2213; bidirectional env resolver :1662-1682).

## Batch 3 -- caption single-owner migration (operator gates direction)
### If 86-owner:
1. STAGE ORDER: rewire so CaptionBurn is the LAST silent-video pass before MasterAudioMux (spec predates node 95 -- RE-DERIVE, see job (b)).
2. LEDGER RESOLUTION: otr_caption_burn.py:70-86 strips only _silent/_captioned/_final/_blend; node 93 outputs *_procgen_blended (otr_post_upscale_procgen_blend.py:923-930); port legacy handling (:98-108).
3. ENABLEMENT: canonical node 86 burn_captions=false, node default false (:160-198); profiles true; set canonical true OR wire OTR_BURN_CAPTIONS=1 through every launch path.
4. OUTPUT CONTRACT: _default_out writes otr/episodes/<stem>_captioned.mp4, no per-episode subdir (:183-192).
- Retarget widget_mapping.json:95-112 to OTR_CaptionBurn; update 3 profile JSONs; invert test_workflow_live_passes_validator.py:56-85 + test_post_upscale_procgen_blend.py:150-163; add test_capability_profiles.py + test_workflow_apply.py to validation set; document env overrides in tooltips.
### If 93-owner: do NOT delete otr_caption_burn.py (__init__.py:299-302 registers; test_caption_burn_cw4.py:19-20 imports); leave registered-but-unwired, fix docstring.
### Either way: full link referential integrity (links[] + input link fields + output links arrays + node order).

## Batch 4 -- CUT (alias dedupe premise retracted).
