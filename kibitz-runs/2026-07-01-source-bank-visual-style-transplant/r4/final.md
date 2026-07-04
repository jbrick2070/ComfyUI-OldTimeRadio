# R4 Final: Converged Build Plan

Status: final kibitz synthesis. Normal arc completed: R1-R4, two external
reviewers per round plus Codex anchor/judgment.

## Verdict

Build-ready as a staged upstream implementation plan, with explicit gates.

Do not start by editing the canonical workflow. Build the isolated upstream
source/story contracts and prompt-surgery previews first. Transplant only after
the tests and manifest are green.

## Locked Architecture

There are three independent selectors:

- `source_bank`: where material comes from.
- `story_model`: the dramatic/tonal writing lane.
- `visual_style`: still/video render language.

They fill one production ledger.

```
source_bank + story_model -> LedgerWritingSpec -> existing writer/adapter -> production ledger
visual_style -> VisualStylePolicy -> meta.visual_style -> MetaBrief / ShotLock / prompt seams
```

`media_archive` is a source bank. Its visual counterpart is
`archival_documentary`, not `media_archive`.

## Final Must-Fix Items Before Coding

1. Maintain a prompt-surgery implementation checklist.
   - Every active prompt site gets one action:
     - `SHARED`
     - `PROFILE`
     - `STORY_MODEL`
     - `VISUAL_POLICY`
     - `DEFERRED/DEAD`
   - Start from:
     - `LEDGER_PROMPT_AUDIT.md`
     - `VISUAL_PROMPT_AUDIT.md`

2. Thread `_otr_style_picker.py` overrides completely.
   - Add kwargs to `pick_style()`:
     - `inventor_system_prompt: str = ""`
     - `chooser_system_prompt: str = ""`
     - `chooser_user_template: str = ""`
   - Thread `inventor_system_prompt` into `_run_inventor`.
   - Thread `chooser_system_prompt` and `chooser_user_template` into
     `_run_chooser` / chooser prompt construction.
   - Empty strings preserve current module constants for science/default.
   - Non-science paths must never use the hardcoded sci-fi inventor prompt.

3. Define the outline system prompt injection path.
   - `StoryPromptProfile.outline_system_prompt` is copied into
     `OutlineRequest.outline_system_prompt`.
   - `_otr_outline.py` uses `req.outline_system_prompt` when non-empty.
   - If empty, it falls back to the existing
     `resolve_creative_system_prompt(creative_repo_id)` path.
   - The adapter never calls `resolve_creative_system_prompt` directly.

4. Keep profile and request responsibilities distinct.
   - `StoryPromptProfile` holds LLM persona/system prompt overrides.
   - `OutlineRequest` holds label/verb/rules fields used by outline prompts:
     `story_form_label`, `source_material_label`, `source_develop_verb`,
     `source_grounding_label`, `outline_rules_extra`,
     `forbidden_plot_patterns`.
   - `_otr_ledger_input_adapter.py` populates `OutlineRequest` from the active
     source packet, story model, and profile.

5. Resolve `story_model_id="auto"` before registry lookup.
   - Implement in `nodes/_otr_story_model_catalog.py`.
   - Deterministic default: first registered model for the source bank unless a
     seeded selector is explicitly added.

6. Preserve existing return/type contracts.
   - `compose_source_coda()` returns `_otr_line_composer.LineResult`.
   - `build_casting_seed()` returns `int`, or use a different name for text
     helpers.
   - `OutlineRequest` new fields are keyword/defaulted only; verify no
     positional construction shifts.

7. Stage visual/video transplant separately.
   - Do not touch deep visual prompt fallback code in the source/story stage.
   - `otr_meta_brief_image_prompt.py` direct `IMAGE_GRADE_TAIL` appends are V3
     visual-stage work, after policy finisher exists.
   - Preserve current shipped behavior under `sci_fi_radio`.

8. Keep `upstream_story_lab` isolated.
   - It can contain fixtures, schema drafts, preview scripts, docs, and
     transplant manifests.
   - Production nodes must not import it.
   - Add a grep/test gate before transplant.

## Transplant Gates

Before touching `workflows/otr_scifi_16gb_full.json`:

- pure schemas pass
- media archive fixture builds a `LedgerWritingSpec`
- prompt preview proves media archive lacks sci-fi/news forbidden phrases
- visual policy previews prove non-cinematic styles do not leak radio/35mm
  tails
- writer runtime default still behaves like current science flow
- transplant manifest names widget deltas and validation commands

Workflow chunk must include:

- `OTR_LedgerScriptWriter.INPUT_TYPES`
- `OTR_LedgerScriptWriter.run()`
- `OTR_LedgerScriptWriter._resolve_inputs()`
- `scripts/otr_api.py` creative whitelist
- `nodes/_otr_workflow_apply.py` creative whitelist
- `tests/test_workflow_apply.py`
- canonical workflow JSON widget append
- workflow validator / JSON round-trip / link audit / widget audit

## Verify-At-Build Checklist

- Pydantic v2 in venv.
- `_otr_style_picker.pick_style()` override messages logged for media archive
  do not contain "sci-fi radio drama showrunner."
- `OutlineRequest` additions preserve science/default system prompt.
- `compose_source_coda()` returns `LineResult`.
- `news_close_brief` stays compatibility key.
- legacy news mirror includes both `title` and `headline`.
- `source_bank` and `story_model` are added to both whitelists when exposed.
- no production import from `upstream_story_lab`.
- `archival_documentary` remains visual style id; `media_archive` remains
  source-bank id.
- canonical workflow widget values only append, never shift.

## Final Cut List

- No public-domain workflow selector in first transplant.
- No `source_text_path` until public-domain adapter exists.
- No live archive RSS/network in first build.
- No runtime read from `upstream_story_lab`.
- No deep render-driver visual fallback edits in the source/story transplant.

## External Review Count

Normal driver-aware kibitz arc completed:

- 4 rounds
- 2 external reviewers per round
- 8 external reviewer calls total

