VERDICT: yes-with-fixes. The plan is sound in structure but contains immediate boot-time validation crashes due to extra JSON keys, seam allowance mismatches, and search path issues.

MUST-FIX BEFORE BUILD:
1. [Section 2 Chunk 2B] Unrecognized field `"outline_rules_extra"` in adapted story packs:
   - Defect: The adapted JSON packs `faithful_radio_adaptation.json` and `media_restoration_adventure.json` contain the key `"outline_rules_extra"` (adapted from lab fixtures). However, `StoryPack` in `nodes/_otr_story_pack.py` does not define this field in its dataclass or include it in `_KNOWN_FIELDS`. Reading these packs will trigger a `StoryPackValidationError` during registry sweep.
   - Fix: Strip `"outline_rules_extra"` from both JSON files (as they are inert metadata in Stage 2) or add `outline_rules_extra` to `StoryPack` and `_KNOWN_FIELDS` in `nodes/_otr_story_pack.py`.

2. [Section 2 Chunk 2A / Section 1] Contradiction between `required_seams` in `banks.json` and strict `PRODUCTION_SEAM_ALLOWLIST` validation:
   - Defect: In `banks.json`, the newly authored non-runnable banks (`media_archive`, `public_domain_story`) list required seams such as `pitch_room_system`, `dramatic_state_system`, and `title_system`. The registry validation requires all `bank.required_seams` to be present in their default packs. However, these seams are not in the strict `PRODUCTION_SEAM_ALLOWLIST` (which cannot be expanded per Section 1), meaning they cannot be defined in the packs' `prompt_stages`. This will cause registry validation to crash at load time.
   - Fix: Ensure `required_seams` in `banks.json` for these non-runnable banks only list production-allowlisted seams that are actually defined in their default packs (i.e., only `line_composer_system` and `coda_system`, since other seams have no production consumer yet).

3. [Section 2 Chunk 2A] Sweep validation will crash on `banks.json` and `pipelines.json`:
   - Defect: The registry sweep validation scans `nodes/story_packs/` for JSON packs. A naive recursive glob search (e.g., `nodes/story_packs/**/*.json`) will match `banks.json` and `pipelines.json` at the top level and attempt to load/validate them as story packs, causing validation failures.
   - Fix: Modify the sweep validator in `nodes/_otr_story_routing.py` to explicitly ignore `banks.json` and `pipelines.json` at the top level, or only scan subdirectories matching loaded bank IDs (e.g., `nodes/story_packs/<bank_id>/*.json`).

4. [Section 2 Chunk 2A] Test regression in `test_stage1b_router_fail_loud_on_missing_pack`:
   - Defect: `test_stage1b_router_fail_loud_on_missing_pack` in `tests/test_story_pack_stage1.py` mocks `router._SCIENCE_PACK_PATH`. Since Chunk 2A drops `_SCIENCE_PACK_PATH` from `_otr_creative_prompt_router.py` in favor of dynamic routing via `resolve_story_pack`, the test will fail with an `AttributeError`.
   - Fix: Update `tests/test_story_pack_stage1.py` to mock or monkeypatch the new routing layer (e.g., mocking `_otr_story_routing.resolve_story_pack` or temporarily renaming the `science_news` directory) instead of patching the removed `_SCIENCE_PACK_PATH`.

5. [Section 2 Chunk 2A] Duplicate bank/pipeline ID detection logic missing for list structures:
   - Defect: In `banks.json` and `pipelines.json`, banks and pipelines are structured as JSON lists rather than objects. Standard duplicate key checking via `_reject_dup_keys` will not catch cases where separate objects in the list share the same `source_bank_id` or `story_pipeline_id`.
   - Fix: Implement an explicit uniqueness check in `nodes/_otr_story_routing.py` that raises a `StoryRoutingError` if duplicate `source_bank_id` or `story_pipeline_id` values are found in the parsed lists.

6. [Section 2 Chunk 2A] Import Isolation side-effect violation if registry is loaded at module import time:
   - Defect: Performing the registry sweep and file operations for `banks.json` and `pipelines.json` at module import time (top-level scope) in `_otr_story_routing.py` violates the import isolation rule. A syntax or file-missing error will crash ComfyUI startup.
   - Fix: Load the registry lazily (triggered on the first call to `get_bank`, `get_pipeline`, or `resolve_story_pack`, or inside `INPUT_TYPES` when retrieving choices).

SHOULD-FIX:
1. [Section 2 Chunk 2C] Missing unit testing for widget order and append safety:
   - Defect: Positional LiteGraph widget mapping is extremely sensitive to widget insertion/shifting (BUG-LOCAL-097). While Chunk 2C details appending `source_bank` at the end, there is no automated test proposed to guard against accidental positioning drift.
   - Fix: Add an assertion in `tests/test_creative_prompt_router.py` or similar to check the positional index of `"source_bank"` in the returned `optional` list of `OTR_LedgerScriptWriter.INPUT_TYPES` to ensure it is at the very end.

OPTIONAL / NICE-TO-HAVE:
1. [Section 2 Chunk 2A] Opaque defaults dict validation: While `defaults` is type-checked structurally as a dict, we could add a basic key validation to ensure all nested values are strings or basic scalar types rather than allowing arbitrary nested JSON structures.

CUT THESE (over-engineering):
- None. The plan is minimal and tightly scoped to routing, registries, and pack addressability.

[ASSUMPTION] `declared_seams` is expected to be a list of strings within the pipeline configuration object in `pipelines.json` and will be loaded as a frozenset during validation.
[ASSUMPTION] Standalone calls to `load_pack` for experimental custom-seam packs are expected to fail with `UnknownSeamError` unless the caller provides the custom seams list explicitly.
