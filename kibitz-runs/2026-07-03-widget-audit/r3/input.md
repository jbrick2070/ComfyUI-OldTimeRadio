# Widget Surface Audit -- r2 hardened draft (coding plan, post-codex r2)
Supersedes r1/final.md. Judgment log at end.

## Caption ownership -- reframed (judge's correction to r1)
r1 called node 93's burn "the legacy remnant". Grounding shows the CONFLICT is deliberate-but-unfinished: tests\test_workflow_live_passes_validator.py:56-80 PINS node 93 as owner (wv93[9] True, wv93[10]=="sdh_standard") and node 86 as pass-through "so captions never double-burn"; config\profiles\widget_mapping.json:95-112 + all three profile JSONs target node 93; while otr_caption_burn.py:1-20 declares node 86 the CW-4 home. The CW-4 caption tear-out of node 93 never completed. OPERATOR CALL: which owner wins. Either way the migration is atomic across 5 surfaces: node code, workflow JSON, widget_mapping.json, profile JSONs (16gb_full/8gb_lite/cpu_floor), tests.

## Batch specs (build-ready detail)
### Batch 1 -- surface-only removal of stereo_policy x3 + delivery_profile
- Remove ONLY the INPUT_TYPES entries; KEEP the function kwargs with defaults (lock(..., delivery_profile="neutral"), generate(..., stereo_policy="mono_safe")) so direct calls/tests stay valid; behavior byte-identical.
- Exact widget vectors after: node 80 [default, auto_registry, True] (was [default, auto_registry, neutral, True]); nodes 81/82/83 [engine] (was [engine, mono_safe]).
- Update surface tests: test_cast_lock.py:55-64, test_batch_character_voices.py:83-96, test_announcer_voice.py:76-88, test_stable_audio_theme.py:70-83.
- Validate: _otr_workflow_validator widget-drift hard-fail (:140-180, :370-392) + JSON round-trip + suite + Bug Bible. Audio spine must stay byte-identical (test_audio_byte_identical).

### Batch 2 -- tooltip-only (NO key renames)
- engine ("single-mode only"), oom_index ("soak-mode only"), refine_target_grade (env override), slot-model handshake, manual_line_ids precedence.
- story_scaffold DOWNGRADED: OTR_LedgerScriptWriter.py:2196-2213 already carries a tooltip explaining the env deferral; also note the widget is bidirectional (on/off SET OTR_ENABLE_STYLE_GRAMMAR; auto restores baseline, :1662-1682). Only tweak if wording is unclear; not a defect.

### Batch 3 -- caption single-owner migration (operator gates direction)
If 86-owner: strip node 93's ENTIRE caption path (widgets :827-845, kwargs :901-902, ASS routing :993-1063); node 93 widgets go 13 -> 11 ([...green, scopes, bars], burn+style removed); retarget widget_mapping features.burn_captions/caption_style to OTR_CaptionBurn; update the three profile JSONs; invert test_workflow_live_passes_validator.py:56-85 + test_post_upscale_procgen_blend.py:150-163 (move/port _ass_filter_arg coverage to caption-burn tests); pick node 86 default (true = accessible delivery, or false + OTR_BURN_CAPTIONS=1).
If 93-owner: fix otr_caption_burn.py's docstring, consider deleting node 86 from the graph (re-splice links 247/266: 84->93), delete otr_caption_burn.py or leave registered-but-unwired.
- Optional: migration test asserting exact before/after widget slot names for nodes 80-83, 86, 93.

### Batch 4 -- CUT (alias dedupe premise retracted; do not touch OTR_VideoDirector).

## Honesty ledger
- Re-grounded this round: story_scaffold env resolver + existing tooltip; engine mode=="single" gate (otr_video_render_batch.py:152-158); oom_index widget :95; widget_mapping caption targets; validator test caption pins.
- Still verify-at-build: OTR_LedgerScriptWriter slot-model handshake exact lines (:2098-2170); OTR_LedgerFreezeCascade.py:246-266 precedence; render_driver.py:2497-2524.

## Judgment log (r2)
- codex MF1 (node 93 caption path is code+widgets+routing, not just widgets) CONFIRMED :827-845/:901-902/:993-1063 -- accepted.
- codex MF2 (profiles/widget_mapping still target node 93) CONFIRMED widget_mapping.json:95-112 -- accepted; ADDS the 5-surface atomicity requirement.
- codex MF3 (tests pin 93-owner) CONFIRMED test_workflow_live_passes_validator.py:56-85 -- accepted, and it REVERSES r1's "93 is legacy remnant" framing into an explicit operator decision (the test comment is intent, not accident).
- codex MF4 (keep kwargs, remove only surface) CONFIRMED -- accepted.
- codex MF5 (exact widget vectors + validator cites) CONFIRMED against workflow JSON -- accepted.
- codex SF1-SF3 accepted (test updates, node 86 default decision, _ass_filter_arg test relocation).
- Judge addition: story_scaffold downgraded (already tooltipped) -- codex did not flag this; found during re-grounding.
