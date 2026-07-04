VERDICT: no. The 86-owner branch is not build-ready: it burns captions before later video overlays and relies on env/profile state that is not threaded through the canonical workflow.

MUST-FIX BEFORE BUILD:
1. [Batch 3 -- If 86-owner] Wrong video-stage order. Current workflow is 84 -> 86 -> 93 -> 85 (`workflows/otr_scifi_16gb_full.json`, links 247/266/250), and the pinned test documents node 93 as the final burn owner after the chain (`tests/test_workflow_live_passes_validator.py:56-85`). If node 93's caption path is stripped while node 86 stays before node 93, procgen/scopes/audio bars are applied after captions. Fix: if 86 owns captions, rewire to 84 -> 93 -> 86 -> 85 so `OTR_CaptionBurn` is the last silent-video pass before `OTR_MasterAudioMux`.

2. [Batch 3 -- If 86-owner] Ledger resolution will break after that rewire. `OTR_CaptionBurn._resolve_ledger_path` strips `_silent`, `_captioned`, `_final`, `_blend` only (`nodes/otr_caption_burn.py:70-86`), while node 93 outputs names with `_procgen_blended` (`nodes/otr_post_upscale_procgen_blend.py:923-930`). The legacy resolver already handles `_procgen_blended` before `_silent` (`nodes/otr_post_upscale_procgen_blend.py:98-108`). Fix: port that suffix handling into `OTR_CaptionBurn`, in the correct order or loop until stable.

3. [Batch 3 -- If 86-owner] Caption enablement is not reliably propagated. Canonical node 86 is saved with `burn_captions=false` (`workflows/otr_scifi_16gb_full.json`, node 86), and the node default is also false (`nodes/otr_caption_burn.py:160-198`). The profile values are true (`config/profiles/16gb_full.json:23-25`, `8gb_lite.json:23-25`, `cpu_floor.json:20-22`), but profiles only matter when the applier runs. Fix: set canonical node 86 `widgets_values[0]` true for accessible default, or explicitly wire `OTR_BURN_CAPTIONS=1` through every launch path; verify: all headless/desktop launch paths if choosing env forcing.

4. [Batch 3 -- If 86-owner] CaptionBurn's default output path violates the repo output contract for an active render stage. `_default_out` writes to `<Comfy output>/otr/episodes/<stem>_captioned.mp4` with no episode subdirectory (`nodes/otr_caption_burn.py:183-192`). If node 86 becomes load-bearing, write next to its input video or another canonical `otr/episodes/<ep>/...` path, not the root episodes directory.

5. [Batch 3 -- If 93-owner] “delete `otr_caption_burn.py`” is build-breaking unless the mapping and tests are removed in the same change. `__init__.py` still registers `OTR_CaptionBurn` (`__init__.py:299-302`), and `tests/test_caption_burn_cw4.py:19-20` imports it directly. Fix: prefer leave registered-but-unwired, or delete the class/file only with `__init__.py` and test updates.

SHOULD-FIX:
1. [Batch 3] Make the link-edit recipe explicit. For either owner choice, update `links[]`, node input `link` fields, output `links` arrays, and node `order`; do not describe only “re-splice links 247/266.” The validator checks referential integrity.

2. [Batch 3] Add `tests/test_capability_profiles.py` to the named validation set when retargeting `widget_mapping.json`; profile identity depends on mapped widget values (`tests/test_capability_profiles.py:173-201`).

3. [Batch 1] Update stale widget-surface comments after removing `delivery_profile` and `stereo_policy`; current comments still say those widgets serialize (`nodes/cast_lock.py:18-20`, `nodes/_otr_voice_node_common.py:183-188`).

OPTIONAL / NICE-TO-HAVE:
- Keep the optional migration test for exact before/after widget slot names. This is cheap insurance against positional drift.

CUT THESE (over-engineering):
1. [Batch 3 -- If 93-owner] Cut deleting `otr_caption_burn.py`. Leaving it registered-but-unwired avoids mapping/test churn and keeps the cleanup separable from the caption-owner migration.
