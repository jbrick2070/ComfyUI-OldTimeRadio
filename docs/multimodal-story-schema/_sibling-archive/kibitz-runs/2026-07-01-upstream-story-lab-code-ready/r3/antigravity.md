VERDICT: build-ready-with-fixes. The upstream story lab compiles successfully, but the transplant plan has missing dependencies, startup-crashing file access, and prompt/bridge conflicts with downstream nodes.

MUST-FIX BEFORE BUILD:
1. [Production Transplant Targets] / [contracts.py] Missing Pydantic dependency in production pack:
   Defect: The contracts in `contracts.py` heavily rely on Pydantic (`BaseModel`, `Field`, `model_validator`), but Pydantic is not listed as a dependency in `ComfyUI-OldTimeRadio/requirements.txt` or `pyproject.toml`. [ASSUMPTION] If a user's ComfyUI environment lacks Pydantic, the transplanted code will immediately crash ComfyUI startup with an `ImportError`.
   Fix: Add `pydantic>=2.0,<3.0` to `ComfyUI-OldTimeRadio/requirements.txt` and `pyproject.toml`.

2. [nodes.py] Startup crash on missing or malformed story pack fixtures:
   Defect: `nodes.py` calls `_story_pack_choice_map()` inside `INPUT_TYPES()`, which runs on ComfyUI startup to register choices. If the fixtures folder is missing, contains duplicate keys, or has malformed JSON, a `RuntimeError` or validation error is raised during startup, taking down the entire node pack. This violates ComfyUI profile rule 5 (no side effects/file reading at module import time).
   Fix: Wrap `_story_pack_choice_map()` file-reading loops in a `try-except` block, log a warning on failure, and return a safe fallback/empty choices list (e.g., `["error_loading_fixtures"]`) to prevent boot-time crashes. Raise actual file errors only at execution time (`preview()`).

3. [Production Transplant Targets] / [catalogs.py] Transplanted fixture path resolution failure:
   Defect: `_load_visual_style_fixtures` searches for JSON files at `Path(__file__).resolve().parents[2] / "fixtures" / "visual_styles"`. The production repository `ComfyUI-OldTimeRadio` does not have a `fixtures/` folder at its root, so the path resolution will fail silently and bypass custom visual style overrides.
   Fix: Standardize the path to read visual style fixtures from a production-appropriate location (e.g., `config/visual_styles` or `assets/visual_styles`), or serialize the visual style JSONs directly into Python dictionary structures in `_BASE_VISUAL_STYLES`.

4. [Production Transplant Targets] / [_otr_line_composer.py] Hardcoded sci-fi news coda assumptions:
   Defect: `_otr_line_composer.py` has a hardcoded `compose_news_coda` method using `_NEWS_CODA_SYSTEM` and `_NEWS_CODA_ARC_BRIDGES` (containing sci-fi/news tropes). If a user runs `media_archive` or `public_domain_story`, the coda will enforce sci-fi news segues, violating the prompt profile `coda_mode` contract.
   Fix: Parameterize `compose_news_coda` to read `coda_mode` from the active `StoryPromptProfile`. If `coda_mode` is `none`, bypass the LLM generation entirely and return an empty `LineResult`.

5. [Production Transplant Targets] / [_otr_style_picker.py] Sci-fi prompt assumptions in style picker:
   Defect: `_otr_style_picker.py` uses a hardcoded prompt instructing the LLM to read a science news article. When running on `media_archive` or `public_domain_story`, style picking will fail or generate irrelevant choices.
   Fix: Parameterize `_otr_style_picker.py` to use `style_picker_inventor_system_prompt`, `style_picker_chooser_system_prompt`, and `style_picker_chooser_user_template` from `StoryPromptProfile` instead of hardcoded strings.

6. [Workflow JSON Touch Rules] / [test_workflow_canonical_baseline.py] Canonical baseline test suite break:
   Defect: `TestWriterCanonicalModelSlots.test_writer_both_slots_mistral_nemo` asserts that Node 1 in `workflows/otr_scifi_16gb_full.json` has type `OTR_LedgerScriptWriter` and pins model settings at specific widget indices (`widgets[3]` and `widgets[4]`). Injecting a bridge node at the head or changing Node 1 will break this test.
   Fix: Reconcile `test_workflow_canonical_baseline.py` and `test_workflow_json_wiring_invariants.py` to expect the new head node structure and widget indices.

SHOULD-FIX:
1. [Future Bridge Strategy] / [workflows/otr_scifi_16gb_full.json] Parallel un-gated execution of the writer node:
   Defect: `OTR_LedgerScriptWriter` (Node 1) has no input link to `OTR_WorkflowValidator` (Node 63). If validation fails, expensive LLM calls inside Node 1 will run in parallel before ComfyUI halts execution, wasting API credits.
   Fix: Append a `gate_in` optional input to `OTR_LedgerScriptWriter` and connect the validator's `validation_report` to it.

2. [Known Transplant Risk: Deep Visual Prompts] / [otr_meta_brief_image_prompt.py] Hardcoded portrait style anchors:
   Defect: `otr_meta_brief_image_prompt.py` hardcodes `STYLE_ANCHOR` and `STYLE_ANCHOR_WIDE`, bypassing the `character_portrait_style` or `character_scene_style` defined in the new `VisualStylePolicy`.
   Fix: Modify `otr_meta_brief_image_prompt.py` to read visual style anchors from the ledger's metadata block populated by the bridge node.

OPTIONAL / NICE-TO-HAVE:
- [Future Bridge Strategy] / [nodes.py] Add `gate_in` optional input socket to `OTR_StoryPackPreview` so that diagnostic/preview runs can be cleanly sequenced behind `OTR_WorkflowValidator`.

**CUT THESE (over-engineering):**
1. [Future Transplant Script] Automation script `plan_transplant.py`:
   - Why safe to cut: For a localized transplant of ~10 target files, creating a custom regex-based python patching script is complex, fragile, and prone to parsing errors. Manual code integration, verified by `OTR_WorkflowValidator` and pytest, is simpler and safer.
