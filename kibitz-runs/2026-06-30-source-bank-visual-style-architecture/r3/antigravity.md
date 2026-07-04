VERDICT: yes-with-fixes. The plan is well-grounded, but has critical state synchronization and ComfyUI signature contract issues that will cause runtime crashes or silent data loss.

MUST-FIX BEFORE BUILD:
1. [C3] Deferring the LLM brief generation in Media Archive mode / Sequencing mismatch:
   - Defect: The plan branches before `_fetch_rss_seed_or_die` (which runs inside `_resolve_inputs` at start time, before model load) and specifies `media_archive: archive fetch + archive packet + legacy mirror`. However, the archive interpreter's packet conversion requires running an LLM ("Run the same technical LLM brief path or a thin archive-specific wrapper"). Since models are not loaded until after `_resolve_inputs` returns, the LLM-based interpretation step cannot execute during `_resolve_inputs` because no generate function is available.
   - Fix: In `_resolve_inputs`, perform only the network fetch of the raw archive item. Defer the LLM-based interpretation (the archive packet conversion and legacy mirror mapping) to the main `run()` pipeline where other structured LLM calls are executed (specifically around D.2.5 when models are resident), using `technical_generate_fn`.

2. [State Synchronization / Overwriting] VideoRenderBatch overwriting visual/images metadata in ledger save:
   - Defect: `OTR_ShotLock` (ID 90) and `OTR_ImageGenDispatcher` (ID 91) mutate the ledger in memory and pass the updated JSON string (`patched_ledger_json`) through the ComfyUI workflow links, but they never write/save the ledger back to the `production_ledger` global singleton `_CURRENT` or to disk. When `OTR_VideoRenderBatch` (ID 92) executes, it receives this `patched_ledger_json` string and uses it to render. However, its internal `_stamp_render_engines_meta` helper calls `production_ledger.get_ledger()` (which gets the old in-memory singleton `_CURRENT` left at the end of the audio stage, lacking the `video` and `images` blocks) and calls `led.save()`. Since `Ledger.save()` does not merge `video` or `images` from disk (they were never written to disk in the first place), this save operation overwrites the on-disk `<ep_root>/audio/<ep>_ledger.json` file with the stale in-memory data, erasing the `video` and `images` blocks entirely. Downstream nodes like `OTR_PostUpscaleProcgenBlend` which reload the ledger from disk then fail to find them. [ASSUMPTION] We assume there are no other out-of-band files or background processes that synchronize the workflow JSON back to the singleton `_CURRENT`.
   - Fix: In `nodes/otr_video_render_batch.py`'s `_render_episode` (or inside `_stamp_render_engines_meta`), synchronize the incoming `patched_ledger_json` data back to the global `production_ledger` singleton before saving, by doing:
     ```python
     from .production_ledger import get_ledger
     led = get_ledger()
     led.data = ledger # where ledger is the parsed dict from patched_ledger_json
     ```
     This ensures the singleton contains all visual shot-lock and image-gen metadata before writing to disk.

3. [C6] ComfyUI Execution Method Signature Mismatch for visual style policy inputs:
   - Defect: The plan appends the optional forceInput socket `visual_style_policy_json: STRING` to both `OTR_MetaBriefImagePromptGen` and `OTR_ShotLock`. However, it does not specify updating the Python signatures of their execution methods (`generate` in `otr_meta_brief_image_prompt.py:1019` and `lock` in `otr_shot_lock.py:882`). Under ComfyUI, when a node input is added, the class method designated in `FUNCTION` must accept that input as a keyword argument, otherwise it will fail at runtime with a `TypeError: got an unexpected keyword argument` when executed.
   - Fix: Update the signature of `OTR_MetaBriefImagePromptGen.generate(...)` and `OTR_ShotLock.lock(...)` to explicitly include `visual_style_policy_json="{}"` as a parameter.

SHOULD-FIX:
1. [C2] Grade Story story_form_label wiring in _otr_story_select.py:
   - Defect: In C2, the plan details converting `source_bank/source_kind` into prompt labels (`story_form_label`, `story_system_label`, etc.) and parametrizing `OutlineRequest` and `_otr_pitch_room.py`. However, `grade_story` in `_otr_story_select.py` also contains a hardcoded system prompt string referencing "grading a short science-fiction audio drama". Since `grade_story` runs during the refine loop in `OTR_LedgerScriptWriter.py` and is not passed `OutlineRequest`, it will still use the hardcoded science-fiction string in other source bank modes (like `media_archive`).
   - Fix: Add an optional `story_form_label: str = "science-fiction audio drama"` parameter to `grade_story` in `nodes/_otr_story_select.py`, and have `OTR_LedgerScriptWriter.py` pass the resolved `story_form_label` to it when executing `grade_story` inside `_refine_loop`.

2. [C0/C4] Dictionary Attribute Access syntax error in finish_visual_prompt:
   - Defect: The plan in C4 specifies: "If no `meta.visual_style`, existing output remains unchanged." and refers to `meta.visual_style` in `finish_visual_prompt`. In the project code, `meta` is normalized to a dictionary via `_meta(meta)`. Direct attribute access like `meta.visual_style` will cause an `AttributeError` at runtime since dictionaries do not support dot-notation attribute access.
   - Fix: Access the visual style policy in `nodes/_otr_story_brief_helpers.py` using dictionary get syntax: `meta.get("visual_style")` or `_meta(meta).get("visual_style")`.

3. [C5] VisualStyleDirector ComfyUI return type single-tuple wrapper:
   - Defect: The node contract for `OTR_VisualStyleDirector` specifies `RETURN_TYPES = ("STRING",)`. Under ComfyUI custom node conventions, the execution method (e.g. `direct`) must return a tuple (or list) matched to the length of `RETURN_TYPES`. Returning a raw string directly will cause ComfyUI to iterate over the string's characters, thinking each character is a separate return value, which crashes the execution thread.
   - Fix: Ensure `OTR_VisualStyleDirector.direct` returns a single-element tuple: `return (visual_style_policy_json,)`.

4. [C1/C2] Threading key terms for non-news modes:
   - Defect: In C1, `key_terms_tuple` is defined in the script writer generation pipeline. For non-news modes where `NewsBriefs` is bypassed, the plan does not explicitly specify how `key_terms_tuple` should be populated, which could leave it empty and bypass the `post_assembly_keyterm_check` validation.
   - Fix: Ensure that in non-science-news modes, `key_terms_tuple` is populated from the `StoryInputPacket`'s `key_terms` field or the mapped `meta["news"]["key_terms"]` list.

OPTIONAL / NICE-TO-HAVE:
- None.

CUT THESE (over-engineering):
1. [C5] `custom_policy_json` input widget in `OTR_VisualStyleDirector`:
   - Why it is safe to cut: The three targeted visual styles (`cinematic_35mm`, `archival_mono`, `anime`) are fully covered by the `style_id` combo selector. Removing the free-form `custom_policy_json` input keeps the UI cleaner and avoids validating unstructured visual policies in the first sprint.
