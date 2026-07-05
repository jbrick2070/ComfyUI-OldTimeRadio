VERDICT: yes-with-fixes. The plan is solid in direction, but lacks necessary updates to self-tests, paired wiring checks, and positional widget indices, which would break the build and test suites immediately.

MUST-FIX BEFORE BUILD:
1. [Section 2 & 7.7] Broken Writer Self-Tests.
   - Defect: [OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L6036-L6091) contains an inline self-test suite (run directly to verify the module) that explicitly tests `style_custom` and `llm_auto` resolution branches. If the style picker and `style_custom` are removed, importing or running the file will crash/fail this self-test suite, breaking the build.
   - Concrete Fix: Update or remove the inline test blocks `6. _resolve_inputs 3-way style resolution` and `7. Two-pass style picker smoke` inside [OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py#L6036-L6156) to align with the new signature and catalog-based resolution flow (testing only `style_combo` and the `"let the story decide"` hash draw sentinel).
2. [Section 2 & 7.5] Broken Paired Wiring Test.
   - Defect: [test_writer_paired_wiring.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_writer_paired_wiring.py#L40-L45) explicitly expects a call site for `pick_style` in [OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py) inside its `_EXPECTED_KWARGS` mapping. If `pick_style` is removed, the test will raise an assertion error because `sites_seen["pick_style"] == 0` at [test_writer_paired_wiring.py:97-100](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_writer_paired_wiring.py#L97-L100).
   - Concrete Fix: Remove the `"pick_style"` entry from the `_EXPECTED_KWARGS` dictionary in [test_writer_paired_wiring.py:40-45](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_writer_paired_wiring.py#L40-L45) in the same change as deleting the picker.
3. [Section 2 & 7.5] Signature Checks in Test Suite.
   - Defect: [test_helper_paired_signatures.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_helper_paired_signatures.py#L118-L160) imports `_otr_style_picker` and calls `sp.pick_style(...)` on line 147. Deleting `_otr_style_picker.py` will cause this file to raise `ImportError`, halting test run collection.
   - Concrete Fix: Delete `test_pick_style_routes_inventor_creative_and_chooser_technical` in [test_helper_paired_signatures.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_helper_paired_signatures.py#L118-L160).
4. [Section 1, 1a, 4 & 8] ComfyUI Positional Widget Index Shift (BUG-LOCAL-097).
   - Defect: Deleting `style_custom` (slot 9) shifts the indices of subsequent optional widgets (`creativity` from 10 to 9, etc.) in `INPUT_TYPES` of [OTR_LedgerScriptWriter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/OTR_LedgerScriptWriter.py). Deleting `story_scaffold` (slot 24) shifts `source_bank` and `visual_style`. This breaks serialisation of existing saved user workflows.
   - Concrete Fix: Either:
     * **Option (a) (Recommended for safety):** Keep the `style_custom` and `story_scaffold` inputs in `INPUT_TYPES` as deprecated/ignored stubs to preserve index positions, and return a clean warning.
     * **Option (b) (Cleanbreak):** If complete removal is required, the plan must explicitly include migrating the canonical JSON in [otr_scifi_16gb_full.json](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/workflows/otr_scifi_16gb_full.json) (updating widget indices) and updating the positional assertions in [test_workflow_json_guardrails.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_workflow_json_guardrails.py#L644-L682) in the same commit.
5. [Section 1] Downstream `v3_validate` String Matching Break.
   - Defect: Downstream LLM output validation in [news_interpreter.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/news_interpreter.py#L472-L502) uses `resolved["style"]` as a search term to find formulaic phrasing (e.g. `"in a locked_room_suspense style"` vs `"in a locked-room suspense style"`). If `resolved["style"]` contains the raw snake_case slug, the validation regexes will fail to match natural-language text.
   - Concrete Fix: Set `resolved["style"]` to the human-readable `contract.label` string (e.g., `"locked-room suspense"`) for downstream prompt/validation consumption, and add a separate `resolved["style_slug"]` key to hold the canonical snake_case slug (e.g., `"locked_room_suspense"`) for metadata stamping and freeze validation.

SHOULD-FIX:
1. [Section 5] `meta.style_pick` Clean-up.
   - Defect: The plan specifies deleting `meta.style_pick` along with the picker but doesn't identify the reader locations.
   - Concrete Fix: Audit and clean up any reader of `meta.style_pick` inside `nodes/_otr_ledger_consistency.py` or `nodes/_otr_ledger_reviewer.py` to ensure it doesn't cause key errors or consistency failures when missing.
2. [Section 1] `_resolve_inputs` Return Type Mapping.
   - Defect: Downstream APIs (e.g. news interpreter and casting) expect `resolved["style"]` to be a string. If changed to a richer object to satisfy the contract, it will crash them.
   - Concrete Fix: Ensure `_resolve_inputs` returns `style` and `style_slug` as plain string fields, rather than a nested dictionary or object, to maintain compatibility with existing module signatures.

OPTIONAL / NICE-TO-HAVE:
- [Section 0] Run `_otr_style_catalog.validate_catalog()` at script writer module load (under `__init__.py` or startup checks) rather than relying only on tests, ensuring any catalog formatting errors fail-fast before server boot.

CUT THESE (over-engineering):
1. [Section 7.2] Separate `forced_slug` endpoint in `_otr_style_catalog`.
   - Why it is safe to cut: Instead of writing a complex new public method, we can implement `build_story_contract_from_slug(slug: str) -> StoryContract` in [_otr_style_catalog.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_style_catalog.py), which is pure, and then define `build_story_contract(...)` to call it. This avoids bloating the catalog module's public API.

ASSUMPTIONS:
- [ASSUMPTION] We assume that the operator accepts Option B for `story_scaffold` (complete deletion) and is willing to accept the breaking widget shift, or that Option A (retaining it) is selected. We have detailed fixes for both to cover all outcomes.
- [ASSUMPTION] We assume that MusicGen cue triples are completely unused by `compose_music_prompt` based on grounding in [_otr_music_prompt.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_music_prompt.py) and [stable_audio_theme.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/stable_audio_theme.py).
