VERDICT: yes-with-fixes. Stale widget-count assertions and whitelist omissions will break CLI tests, and media archive / public-domain sources are under-specified.

MUST-FIX BEFORE BUILD:
1. [C1] The plan instructs updating the writer self-test optional-widget assertion from 16 to 17. This is incorrect: the current optional widget count is 22, and the total widgets count is 25.
   - Fix: Update `assert len(wv) == 25` to `assert len(wv) == 26` in [tests/test_workflow_json_guardrails.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_workflow_json_guardrails.py#L673), and append `"science_news"` as a default value at index 25 of `widgets_values` in `workflows/otr_scifi_16gb_full.json`.
2. [C1] Headless creative whitelist omission for the new widget. Appending `source_bank` to the writer's optional inputs without updating the whitelist will cause the CLI/API route tests to fail with a `ProfileError` when trying to patch it. [ASSUMPTION: We also observe that the recently added `story_scaffold` widget is missing from the api script's whitelist and should be aligned.]
   - Fix: Add `"source_bank"` (and `"story_scaffold"`) to `CREATIVE_WHITELIST` in both [nodes/_otr_workflow_apply.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_workflow_apply.py#L491-L506) and [scripts/otr_api.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_api.py#L753-L764).
3. [C3] Under-specified media archive database/source. The plan does not specify where media archive items are fetched from, which blocks offline testing and makes it unbuildable.
   - Fix: Create a lightweight `nodes/_otr_media_archive.py` containing a static local list of 5-10 curated retro-futuristic/historical abstracts to serve as the default database for `media_archive` in V1.
4. [C0 / C3] Missing `StoryInputPacket` schema declaration in C0.
   - Fix: Lock `StoryInputPacket` schema fields in C0: `title: str`, `raw_text: str`, `key_terms: list[str]`, and `adaptation_trace: dict[str, Any] = {}`.

SHOULD-FIX:
1. [C2] Bypass/Parameterization ambiguity for Pitch Room and Story Select. The plan leaves a choice between parameterizing the pitch-room and story-select prompts or bypassing them for non-science banks.
   - Fix: Explicitly bypass the pitch-room and story grader/refine loops (setting refine to "Off" and skipping pitch room) in `OTR_LedgerScriptWriter.run` for `media_archive` and `public_domain_story` source banks to avoid science prompt leaks.
2. [C2] Missing `source_bank` field in `OutlineRequest`. `OutlineRequest` is a frozen dataclass and needs a way to propagate the source bank to the outline prompt builders.
   - Fix: Add `source_bank: str = "science_news"` to `OutlineRequest` in [nodes/_otr_outline.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_outline.py#L351-L373) and pass it from `OTR_LedgerScriptWriter.run`.
3. [C7] Public-domain source widget choice ambiguity ("either `source_text` or `source_text_path`").
   - Fix: Lock the input strictly to `source_text_path` (string path to a txt file) as pasting long books/stories into a ComfyUI multiline text widget is highly impractical and causes browser lag/crashes.
4. [C4 / C6] Visual style in-memory stamping detail.
   - Fix: Specify that `OTR_ShotLock.lock` must parse `visual_style_policy_json` and stamp it into `meta["visual_style"]` of the returned patched ledger JSON, while `OTR_MetaBriefImagePromptGen.generate` must parse it and stamp it in-memory into the local `meta` dict before calling `derive_image_prompts`.
5. [C0] Visual style tail defaults lack explicit strings.
   - Fix: Specify explicit default strings:
     - `archival_mono`: `positive_tail="monochrome, archival 1930s photograph, vintage film grain, sepia tone, high contrast"` and `forbidden_terms=["color", "glossy", "neon", "modern", "vibrant", "digital photorealism"]`.
     - `anime`: `positive_tail="anime style, cel-shaded illustration, retro 1980s animation, hand-drawn aesthetic"` and `forbidden_terms=["photorealistic", "35mm film", "photograph", "real life", "3d render", "octane render"]`.

OPTIONAL / NICE-TO-HAVE:
1. [C5] Ensure `nodes/OTR_VisualStyleDirector.py` follows ComfyUI import isolation best practices and has zero heavy imports (e.g., numpy, torch) at the module top level.

CUT THESE:
1. [C5] Confirming that custom policy JSON validation (and the `custom_policy_json` input widget) is cut from V1 to keep the visual style director lean.

VERIFY-AT-BUILD checklist:
1. [Existing Bug Watch] Verify if `_stamp_render_engines_meta` in [nodes/otr_video_render_batch.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_video_render_batch.py#L61-L75) incorrectly overwrites updates from the incoming `patched_ledger_json` by loading the ledger via `get_ledger()`. If confirmed, synchronize the `get_ledger()` instance's `data` dict from the parsed incoming `ledger` dict before modifying and saving it.
2. [C6] Verify at run-time that `visual_style_policy_json` occupies slot index 3 on `OTR_MetaBriefImagePromptGen` and slot index 5 on `OTR_ShotLock` as expected (due to the presence of non-linkable widget inputs like BOOLEANs).
