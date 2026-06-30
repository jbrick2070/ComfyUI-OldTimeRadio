VERDICT: yes-with-fixes. The core architectural logic is solid, but the plan contains a critical contradiction between loop-extension and lip-sync (S-A vs S-C), drops a crucial ledger key in merge-to-disk (S-E), and has discrepancies in its fallback removal sweep that will crash the soak tests (S-E).

MUST-FIX BEFORE BUILD:
1. [S-E / production_ledger.py] Ledger Merging Drops `images` Key
   - Defect: The plan correctly identifies that `production_ledger._merge_with_disk` drops the top-level `images` key. However, this is also true for other essential top-level keys like `video` (which contains the shot structure and decisions) or `audio` (which contains the frozen audio metadata) if they are not explicitly loaded or if conflict resolution occurs. If the ledger is saved without these keys, they are permanently deleted from the JSON file on disk.
   - Concrete Fix: Modify [production_ledger.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/production_ledger.py#L1220-L1226) to append `"images"`, `"video"`, and `"audio"` to `TOP_PRESERVE`. This ensures they are durably copied from disk to the in-memory dictionary during the merge.

2. [S-A / S-C / OTR_SilentComposite] Boomerang / Looping Destroys Audio Lip-Sync
   - Defect: S-A proposes looping or boomerang-extending short clips to fill beat targets at the compositor level. However, for speech beats (announcer_visual, character_video), looping or reversing the video frames completely destroys lip-sync alignment with the audio. S-C lists phrase-chunking as a solution for the same root cause. Using both without distinction is contradictory.
   - Concrete Fix: Resolve the contradiction by declaring that looping/boomerang-extend in [otr_silent_composite.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_silent_composite.py#L401-L408) is ONLY applied to non-talking roles (e.g., scene_broll, background_abstract), while talking roles (announcer_visual, character_video) must use phrase-chunking (S-C) to preserve audio-driven sync.

3. [S-E / render_driver.py] Fallback Removal Crashes Soak Tests
   - Defect: S-E states that all fallback chains must be ripped out, making failures fail loud. However, the CPU soak harness [otr_video_soak.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_video_soak.py#L221-L248) and the GPU soak harness [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L2209-L2230) explicitly force an OOM on `soak_oom_3d` and assert that the fallback chain resolves through `humo -> humo_1.7B -> still_motion`. If fallbacks are completely disabled inside `render_shot` (which has already been partially done) and we unregister/retire the fallback floor `still_motion`, both soak harnesses will raise `RenderError` and crash, breaking the regression suite and test pipelines.
   - Concrete Fix: Keep the mock/fallback machinery enabled specifically for the soak tests when running under test/soak mode (gated by `oom_shot_id` or `OTR_TEST_MODE`), or rewrite the soak tests to not assert fallback behavior. If fallbacks are completely retired, the soak tests must be refactored to assert immediate failure rather than trail resolution.

4. [S-E / cheap_families.py / render_driver.py] Retiring `still_motion` Breaks Universal Floor
   - Defect: The plan proposes retiring `still_motion`. However, `"still_motion"` is currently hardcoded in [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L46-L51) as the `UNIVERSAL_FLOOR`, and is also the default fallback and role default for `scene_broll` in [cheap_families.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py#L175-L182). Retiring it without a replacement will cause compilation and lookup errors across the registry.
   - Concrete Fix: Replace all occurrences of `"still_motion"` with `"still_pan"` (or `"still_flat"`) in [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L46-L51) and [cheap_families.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py#L175-L182) so that `still_pan` acts as the new universal floor and default broll engine.

SHOULD-FIX:
1. [S-F / workflows/otr_scifi_16gb_full.json] ComfyUI Graph Skipping Ambiguity
   - Defect: S-F proposes running the visual stages without executing the writer + audio nodes, but states this should be done "without editing the production graph." In ComfyUI, a node is executed if downstream nodes depend on it [ASSUMPTION]. If the production graph JSON `otr_scifi_16gb_full.json` connects node 1 (writer) and audio nodes to the downstream compositor, they will execute.
   - Concrete Fix: Clarify that the test harness will dynamically rewrite/prune the JSON workflow file to replace the writer and audio nodes with bypass stubs before sending the prompt to the ComfyUI server.

2. [S-B / render_driver.py] Stale Peak VRAM Comment
   - Defect: S-B refers to replacing the stale "13688" comment in `render_driver.py`. The comment actually resides at line 1166 in [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L1166).
   - Concrete Fix: Update [render_driver.py:L1166](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L1166) to point to the runtime logs/bakeoff manifest instead of hardcoding "13688 MB peak".

3. [BUG-411 / flux_gen1.py] FluxGuidance Redundancy
   - Defect: BUG-411 says "restore (1) a FluxGuidance node @ ~3.5 (flux_gen1 has none -- simplest factor)". However, the current code in [flux_gen1.py:L92](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_image_engines/flux_gen1.py#L92) and [flux_gen1.py:L133-L135](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_image_engines/flux_gen1.py#L133-L135) already implements the `FluxGuidance` node with a default value of `3.5`.
   - Concrete Fix: Modify BUG-411 to focus solely on wiring the node into the `otr_scifi_16gb_full.json` workflow file, since the Python code already supports the guidance parameter.

OPTIONAL / NICE-TO-HAVE:
- [S-E] Provide an environment variable override (e.g. `OTR_DISABLE_FALLBACKS`) rather than deleting all fallback code directly, permitting emergency override on boxes where the target engines are not fully set up.

CUT THESE (scope / over-engineering):
1. [S-E / cheap_families.py] Retiring `still_motion` class completely
   - Why it is safe to cut: While the name `"still_motion"` should be removed from dropdowns and registry defaults, completely deleting the `StillMotionFamily` class from [cheap_families.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/cheap_families.py#L175-L182) can cause compatibility errors when loading older ledger files or saved ComfyUI graphs that still refer to it. Keep the class registered as a deprecated alias pointing to `still_pan`.
