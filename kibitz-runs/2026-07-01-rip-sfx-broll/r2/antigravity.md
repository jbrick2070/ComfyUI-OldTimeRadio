VERDICT: no. The plan contains a critical contradiction regarding `sfx_cue` (ordering both its total removal and its retention) and misses dozens of dependent references in engines, image directors, appliers, and unit tests that will prevent successful compilation and test runs.

MUST-FIX BEFORE BUILD:
1. [## Codex's "keep sfx_cue" -- OVERRULED by grounding] + [## Resolved decisions] + [## Invariants / verification]
   Defect: Critical internal contradiction regarding `sfx_cue`. Line 21 dictates to "RIP `sfx_cue` OUT TOO" and lists files/lines to delete, while line 61 explicitly directs "DO NOT remove `sfx_cue`", line 87 mandates "KEEP sfx_cue", and the Phase Order (lines 71-85) omits the removal.
   Concrete Fix: Resolve this contradiction by keeping `sfx_cue` as dialogue atmosphere / writer prompt nudge (matching Q1 and the invariants), and keep the field in [_otr_outline.py:L120](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_outline.py#L120). Delete the section ordering `sfx_cue` deletion (lines 10-28) from the plan.

2. [## Phase order] P4
   Defect: P4 is incomplete regarding unit test updates. It only details deleting `test_per_cue_sfx_dur.py` and updating `test_speaker_role`. However, there are over a dozen test files referencing the deleted roles `"scene_broll"` and `"background_abstract"` (including [test_video_role_compat_additive.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_video_role_compat_additive.py), [test_video_still_parallax.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_video_still_parallax.py), [test_video_platform_aseam.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_video_platform_aseam.py), and [test_video_render_driver_additive.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_video_render_driver_additive.py)) which will throw `AttributeError` or `RoleCompatError` on run.
   Concrete Fix: Expand P4 to explicitly adapt/clean up these test suites (removing or replacing references to deleted roles/attributes).

3. [## Final scope (what goes)] item 2
   Defect: In [otr_image_director.py:L61-63](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_image_director.py#L61-L63) and [otr_image_director.py:L151-152](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_image_director.py#L151-L152), `IMAGE_SLOT_ROLES` and `three_d_locked_slots` list `"scene_broll"` and `"background_abstract"`. Deleting these roles from the enum without updating these maps will cause compatibility validation and 3D lock checks to crash.
   Concrete Fix: Remove both roles from `IMAGE_SLOT_ROLES` and `three_d_locked_slots` in [otr_image_director.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_image_director.py).

4. [## Final scope (what goes)] item 2
   Defect: Multiple video engines list the deleted roles `"scene_broll"` and `"background_abstract"` in their `roles` or `default_roles` attributes: [cheap_families.py:L180-L181](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py#L180-L181) (`StillMotionFamily` has `default_roles = ("scene_broll",)`), [eng_still_parallax.py:L177-L178](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_still_parallax.py#L177-L178), and others. This will result in unreachable defaults or registration/validation failures.
   Concrete Fix: In [cheap_families.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py), change `StillMotionFamily.default_roles` to `("announcer_visual",)` (or empty), and remove the deleted roles from all `roles` lists in `nodes/_otr_video_engines/`.

5. [## Final scope (what goes)] item 2
   Defect: `_VIDEO_DIRECTOR_WIDGETS` in [_otr_workflow_apply.py:L142-L143](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_workflow_apply.py#L142-L143) contains `"scene_broll_video_model"` and `"background_abstract_video_model"`. Applying capability profiles will fail with a `KeyError`/`AttributeError` at runtime when trying to configure these removed widgets.
   Concrete Fix: Remove both keys from `_VIDEO_DIRECTOR_WIDGETS` in [_otr_workflow_apply.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_workflow_apply.py).

6. [## Final scope (what goes)] item 2
   Defect: If `sfx_cue` is deleted, references to `beat.sfx_cue` in [production_ledger.py:L860](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/production_ledger.py#L860) and [OTR_LedgerScriptWriter.py:L4811](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L4811) will crash with `AttributeError` for non-voiced music beats.
   Concrete Fix: Use `getattr(beat, "sfx_cue", None)` or ensure a default fallback of `""` is used when building these strings if `sfx_cue` is removed from the Pydantic schema. (If `sfx_cue` is kept, this defect does not occur).

7. [## Resolved decisions] Q2
   Defect: Removing the normal-path `_DEFAULT_VIDEO_ROLE` fallback in [otr_shot_lock.py:L83](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_shot_lock.py#L83) without explicitly defining a fallback raise block in `_video_role_for_line` will return `None` or cause unhandled exceptions downstream in `build_execution_plan`.
   Concrete Fix: Explicitly raise a `ValueError` inside `_video_role_for_line` in [otr_shot_lock.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_shot_lock.py) for any unmapped speaker role to ensure a clean fail-loud execution.

SHOULD-FIX:
1. [## Final scope (what goes)] item 4
   Defect: [ASSUMPTION] Removing `role_overrides.scene_broll_visual` and `role_overrides.background_abstract_visual` from `widget_mapping.json` is described, but the profile schema files themselves ([16gb_full.json](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/config/profiles/16gb_full.json), etc.) do not contain these keys.
   Concrete Fix: Clarify that profile JSON files only require updates to [widget_mapping.json](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/config/profiles/widget_mapping.json).

2. [## Phase order] P1
   Defect: In [role_compat.py:L37-L41](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_shared/role_compat.py#L37-L41), `Role` enum still retains `SCENE_BROLL` and `BACKGROUND_ABSTRACT`, and `ROLE_AVAILABLE_INPUTS` has them as keys. Removing `Role` members without updating these will leave dangling references.
   Concrete Fix: Remove `"scene_broll"` and `"background_abstract"` entries from `ROLE_AVAILABLE_INPUTS` and `ROLES` in [role_compat.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_shared/role_compat.py).

OPTIONAL / NICE-TO-HAVE:
1. [## Invariants / verification]
   A load-time check in `load_ledger` to fail loud on any legacy ledgers containing `speaker_role: "sfx"` should output a highly descriptive warning instructing the user to rewire or regenerate the ledger.

CUT THESE (over-engineering):
1. [## Codex's "keep sfx_cue" -- OVERRULED by grounding]
   Why: The deletion of `sfx_cue` from outline schemas, writers, adapters, and prompt composers is entirely safe to cut. Keeping `sfx_cue` as dialogue atmosphere / writer prompt nudge has zero runtime cost since it defaults to `None`/`""` for all standard episodes, yet preserves a valid fallback for any custom outlines that leverage it. Deleting it requires massive code churn across five different modules.
