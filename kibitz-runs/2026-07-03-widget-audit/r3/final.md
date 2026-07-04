# Widget Surface Audit -- r3 hardened draft (wiring round; codex r3 + antigravity manual review folded)
Supersedes r2/final.md. Judgment log at end.

## Batch 1 -- surface-only removal (build-ready)
- Remove INPUT_TYPES entries only; keep kwargs with defaults. Exact widget vectors:
  node 80 ["default","auto_registry","neutral",true] -> ["default","auto_registry",true]
  node 81 ["indextts2","mono_safe"] -> ["indextts2"]
  node 82 ["kokoro","mono_safe"] -> ["kokoro"]
  node 83 ["stable_audio_3","mono_safe"] -> ["stable_audio_3"]
  (agy-verified against the saved JSON; validator length gate _otr_workflow_validator.py:175-179 hard-fails otherwise.)
- Update tests: test_cast_lock.py:62-64, test_announcer_voice.py:87, test_batch_character_voices.py:94-95, test_stable_audio_theme.py:80.
- Update stale docstring comments that still list the removed widgets: cast_lock.py:18-20, _otr_voice_node_common.py:183-188 (codex r3 SF3).
- Validate: OTR_WorkflowValidator + JSON round-trip + suite + Bug Bible + test_audio_byte_identical.

## Batch 2 -- tooltip-only (unchanged from r2; story_scaffold stays downgraded)

## Batch 3 -- caption single-owner migration (operator gates direction)
### If 86-owner, the r2 spec was NOT build-ready. Four new MUST-FIXes (codex r3, judge-confirmed):
1. STAGE ORDER: chain is 84 -> 86 -> 93 -> 85 (links 247/266/250). Stripping 93's burn while 86 stays upstream burns captions BEFORE procgen/scopes/audio-bars overlays. Rewire to 84 -> 93 -> 86 -> 85 so CaptionBurn is the LAST silent-video pass before MasterAudioMux.
2. LEDGER RESOLUTION: otr_caption_burn.py:70-86 strips only _silent/_captioned/_final/_blend; node 93 outputs *_procgen_blended (otr_post_upscale_procgen_blend.py:923-930). Port the legacy resolver's _procgen_blended handling (:98-108) into CaptionBurn (correct order or loop-until-stable). CONFIRMED by judge read.
3. ENABLEMENT: canonical node 86 saves burn_captions=false and the node default is false (:160-198); profiles say true but only apply when the applier runs. Either set canonical widgets_values[0]=true or wire OTR_BURN_CAPTIONS=1 through EVERY launch path (headless launcher + desktop). Operator picks.
4. OUTPUT CONTRACT: _default_out writes <output>/otr/episodes/<stem>_captioned.mp4 with NO per-episode subdir (:183-192) -- violates the otr\episodes\<ep>\ contract if load-bearing. CONFIRMED by judge read. Write beside the input video or into the episode's own dir.
- Plus (both agents): retarget widget_mapping.json:95-112 features.burn_captions/caption_style to OTR_CaptionBurn; update 16gb_full/8gb_lite/cpu_floor profile JSONs; invert test_workflow_live_passes_validator.py:56-85 + test_post_upscale_procgen_blend.py:150-163; add test_capability_profiles.py + test_workflow_apply.py to the named validation set; document OTR_BURN_CAPTIONS/OTR_CAPTION_STYLE env overrides in node 86 tooltips (agy SF1).
### If 93-owner:
- Do NOT delete otr_caption_burn.py (registered in __init__.py:299-302; test_caption_burn_cw4.py:19-20 imports it). Leave registered-but-unwired, fix its docstring; optionally re-splice 84->93 later as a separate change.
### Either way: link edits are links[] + node input link fields + output links arrays + node order -- full referential integrity, not just "re-splice 247/266".

## Batch 4 -- stays CUT.

## Honesty ledger
- Antigravity independently verified ALL previously-unverified sonnet cites (LedgerScriptWriter :1662-1682/:2098-2170, FreezeCascade :246-266, video_render_batch :152-165/:189-197, render_driver :2497-2524) -- verify-at-build list is now EMPTY except live UI captures.
- Two-tier confidence note stands (~60 semantic-deep, ~65 mechanical-only).

## Judgment log (r3)
- codex r3 MF1 (stage order) CONFIRMED via links 247/266/250 -- accepted; this is the round's biggest catch.
- codex r3 MF2 (_procgen_blended suffix gap) CONFIRMED by direct read of :70-86 -- accepted.
- codex r3 MF3 (enablement propagation) CONFIRMED (canonical false + default false) -- accepted, operator decision.
- codex r3 MF4 (output path contract) CONFIRMED by direct read of :183-192 -- accepted.
- codex r3 MF5 / CUT1 (don't delete otr_caption_burn.py under 93-owner) CONFIRMED via __init__ registration + test import -- accepted.
- codex r3 SF1-SF3 accepted.
- antigravity MF1-MF3 all CONFIRMED (consistent with codex r2/r3; widget vectors + test cites verified against files) -- folded; its "MISREADS: none" independently ratifies the r1 retraction + honesty-ledger cites.
- Verdict trajectory: r3 codex "no" applied ONLY to the 86-owner branch as specified in r2; with the four fixes above folded in, the branch is specifiable again. No open contradictions.
