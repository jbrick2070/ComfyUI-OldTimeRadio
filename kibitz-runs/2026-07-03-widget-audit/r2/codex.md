VERDICT: yes-with-fixes. Direction is implementable, but the plan is not build-ready until the widget/API/profile/test migrations are specified exactly.

MUST-FIX BEFORE BUILD:
1. [Batch 3] Caption ownership migration is under-specified. `nodes\otr_post_upscale_procgen_blend.py:827-845` exposes `burn_captions`/`caption_style`, `:901-902` accepts them, and `:993-1063` still routes caption ASS paths through blend/bars commands. Concrete fix: remove the whole caption path from node 93, not just the widgets; decide whether to keep backward-compatible kwargs ignored, and update node 93 `widgets_values` from `[src,pgn,mode,opacity,ffmpeg,bypass,suffix,crush,green,burn,style,scopes,bars]` to `[src,pgn,mode,opacity,ffmpeg,bypass,suffix,crush,green,scopes,bars]`.

2. [Batch 3] Profile application will keep patching the wrong node. `config\profiles\widget_mapping.json:95-110` targets `features.burn_captions` and `features.caption_style` at `OTR_PostUpscaleProcgenBlend`; profiles still set those fields in `config\profiles\16gb_full.json:23-26`, `8gb_lite.json:23-26`, and `cpu_floor.json:20-23`. Concrete fix: retarget both mappings to `OTR_CaptionBurn` or drop them and document env-only control.

3. [Batch 3] Tests currently pin the opposite behavior. `tests\test_workflow_live_passes_validator.py:56-85` asserts node 93 owns captions and node 86 is pass-through; `tests\test_post_upscale_procgen_blend.py:157-163` asserts node 93 caption widgets exist. Concrete fix: invert these tests in the same batch: node 86 `widgets_values[0]` is the owner/default, node 93 has no caption widgets, and chain order remains valid.

4. [Batch 1] “Drop from INPUT_TYPES; hard-code defaults internally” needs explicit function-signature policy. `nodes\cast_lock.py:100-103` exposes `delivery_profile`, but `lock(... delivery_profile="neutral" ...)` still validates/stamps it at `:128-176`; `nodes\_otr_voice_node_common.py:235-242` exposes `stereo_policy`, and `generate(... stereo_policy="mono_safe")` feeds mono conversion at `:273-293` and `:362`; `nodes\stable_audio_theme.py:119-138` and `:195` do the same. Concrete fix: keep those kwargs with default constants for backward/API/test compatibility while removing only the widget surface, or explicitly update every direct call/test that passes them.

5. [Batch 1 / workflow JSON] The migration must list exact resulting widget vectors, not only “rebuild by live INPUT_TYPES order.” Current canonical workflow has node 80 `widgets_values = [default, auto_registry, neutral, True]`, nodes 81/82/83 each `[engine, mono_safe]`. Concrete fix: after removing surfaces, assert node 80 becomes `[default, auto_registry, True]`, nodes 81/82/83 become `[engine]`, and run the validator that hard-fails widget drift (`nodes\_otr_workflow_validator.py:140-180`, `:370-392`).

SHOULD-FIX:
1. [Batch 1] Update widget-surface tests explicitly. Current tests assert old slots: `tests\test_cast_lock.py:55-64`, `tests\test_batch_character_voices.py:83-96`, `tests\test_announcer_voice.py:76-88`, `tests\test_stable_audio_theme.py:70-83`. Concrete fix: change expected serialized slots to node 80 `[voice_bank, cast_voice_policy, allow_voice_reuse]` and nodes 81/82/83 `[engine]`.

2. [Batch 3] Node 86 default is still unresolved. `nodes\otr_caption_burn.py:160-164` defaults `burn_captions=False`, and the workflow currently saves node 86 as false. Concrete fix: operator decision must become an explicit implementation value: either set node 86 workflow value true for accessible delivery or keep false and require `OTR_BURN_CAPTIONS=1`.

3. [Batch 3] Removing node 93 caption helpers should include stale test cleanup. `tests\test_post_upscale_procgen_blend.py:150-154` tests `_ass_filter_arg` from node 93; after stripping caption code that helper should move to `otr_caption_burn.py` tests or be deleted.

OPTIONAL / NICE-TO-HAVE:
- Add a small migration test that loads the canonical workflow and asserts the exact before/after widget slot names for nodes 80-83, 86, and 93.

CUT THESE (over-engineering):
1. [Batch 4] Keep cut. The alias-dedupe premise was retracted, and no implementation should touch `OTR_VideoDirector` for that batch.
