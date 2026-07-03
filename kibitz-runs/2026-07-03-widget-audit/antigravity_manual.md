# Antigravity Independent Manual Review -- Widget Surface Audit
Date: 2026-07-03

VERDICT: yes-with-fixes

## MUST-FIX BEFORE BUILD

1. **`widget_mapping.json` Caption Re-mapping (Batch 3)**
   - **File:** [widget_mapping.json](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/config/profiles/widget_mapping.json#L95-L112)
   - **Evidence:** The configuration maps `features.burn_captions` and `features.caption_style` to `OTR_PostUpscaleProcgenBlend` (node 93). When captioning is stripped from node 93 in Batch 3, these targets must be changed to `OTR_CaptionBurn` (node 86). Without this update, profile cross-validation (`tests/test_capability_profiles.py`) and workflow application (`tests/test_workflow_apply.py`) will crash due to schema mismatch.

2. **`otr_scifi_16gb_full.json` Widget Value Array Realignment (Batch 1)**
   - **File:** [otr_scifi_16gb_full.json](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/workflows/otr_scifi_16gb_full.json)
   - **Evidence:** Dropping `stereo_policy` from nodes 81, 82, and 83 and `delivery_profile` from node 80 reduces their expected widget slot counts by 1. ComfyUI parses `widgets_values` positionally, and `OTR_WorkflowValidator` enforces a strict length gate: `len(wv) == expected` (see [_otr_workflow_validator.py:175-179](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_workflow_validator.py#L175-L179)). The saved JSON's `widgets_values` array for these nodes must be rebuilt to match the new live counts:
     - Node 80 (`OTR_CastLock`): `["default", "auto_registry", "neutral", true]` -> `["default", "auto_registry", true]`
     - Node 81 (`OTR_BatchCharacterVoices`): `["indextts2", "mono_safe"]` -> `["indextts2"]`
     - Node 82 (`OTR_AnnouncerVoice`): `["kokoro", "mono_safe"]` -> `["kokoro"]`
     - Node 83 (`OTR_StableAudioTheme`): `["stable_audio_3", "mono_safe"]` -> `["stable_audio_3"]`

3. **Audio Node & Cast Lock Unit Test Adjustments (Batch 1)**
   - **Files:**
     - [test_cast_lock.py:62-64](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_cast_lock.py#L62-L64)
     - [test_announcer_voice.py:87](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_announcer_voice.py#L87)
     - [test_batch_character_voices.py:94-95](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_batch_character_voices.py#L94-L95)
     - [test_stable_audio_theme.py:80](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_stable_audio_theme.py#L80)
   - **Evidence:** These unit tests directly assert the serialized widget lists against the legacy signatures including `"stereo_policy"` and `"delivery_profile"`. Hiding these widgets will break the assertions, causing CI to fail immediately.

## SHOULD-FIX

1. **Tooltip Documentation of Env Overrides for `OTR_CaptionBurn` (Batch 3)**
   - **File:** [otr_caption_burn.py:160-166](file:///c:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_caption_burn.py#L160-L166)
   - **Evidence:** As node 86 (`OTR_CaptionBurn`) becomes the sole owner of the caption burn, its tooltips should clearly document that env variables `OTR_BURN_CAPTIONS` and `OTR_CAPTION_STYLE` override the widgets, aligning with the Batch 2 tooltip polishing pattern.

## MISREADS IN THE DOC

None. All claims in `r1/final.md` were successfully grounded and matched:
- The retraction of the Node 87 VideoDirector duplicate alias row is correct; `_LEGACY_ENGINE_ALIASES` does not populate the combo box dropdown, which is built strictly from `registry.all_engine_names()`.
- The line citations for the Honesty Ledger (cites from `OTR_LedgerScriptWriter.py`, `OTR_LedgerFreezeCascade.py`, `otr_video_render_batch.py`, `render_driver.py`) were independently verified and matched.

## CUT THESE

None. The scope exclusions (cutting Batch 4 and omitting widget key renames) are well-reasoned since renames would trigger complex schema migrations.
