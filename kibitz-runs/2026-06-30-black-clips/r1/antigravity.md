VERDICT: yes-with-fixes. The plan correctly describes the symptoms but fails to identify the exact code condition in the image dispatcher that skips still generation.

MUST-FIX BEFORE BUILD:
1. [mechanism] The plan notes that `init_image` is empty because the scene still is not in the ledger, but it does not specify why it was skipped.
   Defect: In [nodes/otr_image_gen_dispatcher.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_image_gen_dispatcher.py#L437-L441), the dispatcher calls `_still_needed_for_role`, which delegates to `engine_consumes_still` in [nodes/otr_image_gen_dispatcher.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_image_gen_dispatcher.py#L287-L305). Because `StillPanFamily` (`still_pan`), `StillMotionFamily` (`still_motion`), and `StationCardFamily` (`station_card`) do not declare `accepts_still = True` in [nodes/_otr_video_engines/cheap_families.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py) and their `required_inputs` is `("text_prompt",)`, `engine_consumes_still` evaluates to `False`. The dispatcher then skips generating the still, leaving `_still_index` empty for these beats.
   Concrete Fix: Add `accepts_still = True` to `StillPanFamily`, `StillMotionFamily`, and `StationCardFamily` in [nodes/_otr_video_engines/cheap_families.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py) (similar to `StillFlatFamily`).

2. [regression] The plan suspects commit `c1132196` "rename cheap still engines: flat_still->still_flat, flux_still->still_pan".
   Defect: The rename did not directly introduce the bug. The underlying defect was introduced when still-generation skipping was added in commit `b2f07e09` and refined in commit `2cb25fbe` (the coverage architecture).
   Concrete Fix: Clarify the regression history: before fallbacks were removed in commit `c8f0156c`, failed heavy engine renders (such as `ltx_audio_in` CUDA OOMs) silently fell back to `still_flat`, which had `accepts_still = True` and thus worked. Ripping out fallbacks in commit `c8f0156c` unmasked the fact that still generation was skipped for roles configured with `still_pan`, `still_motion`, or `station_card`. [ASSUMPTION] The fallback to `still_flat` previously masked the skipped still-gen.

SHOULD-FIX:
1. [mechanism] The plan references `flat_still` as the old name, but in commit `c1132196` it was renamed to `still_flat` in the python modules.
   Defect: Line 928 in [nodes/_otr_video_engines/render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L928) checks for `still_flat`, but the plan's text in `## MECHANISM` still uses `still_pan/still_flat` interchangeably with old names or explains it as "flat_still".
   Concrete Fix: Standardize all naming in the documentation to the new UX names (`still_pan`, `still_flat`, `still_motion`) and update the mechanism trace to use `still_flat` rather than references to `flat_still`.

OPTIONAL / NICE-TO-HAVE:
1. [what-to-determine] Add an integration test in `tests/test_still_aspect_and_labels.py` or similar to assert that any registered engine with `uses_still = True` evaluates to `True` under `engine_consumes_still` so this class of defect cannot re-occur.

CUT THESE (scope / over-engineering):
1. [regression] Do not perform any forensic checks on image dispatcher materialization or write-backs (Suspect 4). The write-back mechanism itself is functioning correctly when it is not skipped. It is safe to cut this investigation because the root cause lies entirely in the skipping logic of `_still_needed_for_role`.
