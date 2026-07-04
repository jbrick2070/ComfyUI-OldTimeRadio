# R3 Final: Wiring And Transplant Sequence

Status: grounded wiring synthesis for R4 convergence.

## Verdict

The plan is integration-ready only if source/story transplant and visual/video
transplant remain separate stages.

The upstream lab can stay clean. The visual prompt extraction cannot be assumed
clean because current still/video prompts and repair fallbacks append style
language deep inside existing modules.

## Transplant Order

1. Keep `upstream_story_lab/` isolated.
   - Fixture JSON and manifests only.
   - Production code must not import from it.
   - Add a test/grep gate before transplant: no `nodes/` import references to
     `upstream_story_lab`.

2. Build production pure modules under `nodes/`.
   - `_otr_source_packets.py`
   - `_otr_story_model_catalog.py`
   - `_otr_story_prompt_profile.py`
   - `_otr_ledger_writing_spec.py`
   - `_otr_ledger_input_adapter.py`
   - `_otr_visual_style_policy.py`
   - `_otr_visual_render_catalog.py` only when visual policy integration begins.

3. Make prompt surgery testable before live writer wiring.
   - Render prompt previews for science and media archive.
   - Assert media archive does not emit sci-fi/news forbidden phrases.
   - Keep shared story craft; change source/story-model prompt variables.

4. Integrate source/story into writer runtime.
   - Append `source_bank` and `story_model` to writer inputs in one chunk.
   - Thread to `run()` and `_resolve_inputs()` in the same chunk.
   - Default stays current science behavior.
   - No `source_text_path` until public-domain adapter is real.

5. Update headless/API parity in the same chunk.
   - `scripts/otr_api.py` creative whitelist.
   - `nodes/_otr_workflow_apply.py` creative whitelist.
   - parity tests around workflow apply/API companion logic.

6. Only then edit canonical workflow JSON.
   - Append writer widget values only.
   - Do not insert mid-vector.
   - Validate with workflow validator, JSON round-trip, link audit, and
     widget/input audit.

7. Visual/video transplant is staged separately.
   - V1: policy classes/catalog + seam readers.
   - V2: MetaBrief/ShotLock policy sockets and ledger/meta stamping.
   - V3: deep render-driver and fallback prompt replacement.

## Must-Fix Wiring Decisions

1. `story_model_id="auto"` must resolve before registry lookup.

Add:

```python
resolve_story_model_id(source_bank_id: str, story_model_id: str, rng=None) -> str
```

`"auto"` maps to a deterministic default or a deterministic pick from the source
bank's allowed models. The chosen concrete id is what reaches
`get_story_model()` and `get_profile()`.

2. Coda mode must preserve the existing return contract.

`compose_source_coda()` must return `_otr_line_composer.LineResult`, not a new
shape. Existing writer code reads `.text` and `.compose_flags`.

`compose_news_coda()` may become a science-news wrapper around the generic
source coda.

3. `OutlineRequest` must be extended before profile injection.

Add defaulted fields:

- `outline_system_prompt`
- `story_form_label`
- `source_material_label`
- `source_develop_verb`
- `source_grounding_label`
- `outline_rules_extra`
- `forbidden_plot_patterns`

Defaults must preserve existing science behavior for old callers.

4. Profile fields must cover system prompts, not only user prompt text.

Add to `StoryPromptProfile`:

- `outline_system_prompt`
- `pitch_room_system_prompt`
- `story_select_system_prompt`
- `style_picker_inventor_system_prompt`
- `style_picker_chooser_system_prompt`
- `style_picker_chooser_user_template`

5. `_otr_style_picker.pick_style()` needs override kwargs before non-science
use.

Add optional keyword args:

- `inventor_system_prompt: str = ""`
- `chooser_system_prompt: str = ""`
- `chooser_user_template: str = ""`

The hardcoded sci-fi inventor/user prompt must never run for `media_archive`.

6. Casting seed helper must not return text.

`build_casting_seed()` should either return `int` or be renamed to
`build_casting_brief()`. Existing casting replay uses integer `cast_seed`.

7. Legacy news mirror needs both `title` and `headline`.

Populate:

- `meta.news.title`
- `meta.news.headline`
- `meta.news.script_brief`
- `meta.news.news_close_brief`
- `meta.news.casting_brief`
- `meta.news.key_terms`
- `meta.news.link`
- `meta.news.source_hash`

8. Visual style must not be bypassed by local hardcoded appends.

`otr_meta_brief_image_prompt.py` appends `IMAGE_GRADE_TAIL` directly after
calling `finish_visual_prompt()`. Those local appends must move behind the
policy-aware finisher before anime/cartoon/origami can be trusted.

9. Forbidden-term scrubbing needs one owner.

Add a helper at the visual seam:

```python
scrub_forbidden_terms(prompt: str, forbidden_terms: list[str]) -> str
```

Run it before final output/hash generation. Do not scatter ad hoc scrubs across
MetaBrief, ShotLock, and render-driver.

10. Visual style must not be conflated with writer `style`.

The existing writer `style`/`_otr_style_picker.py` path is narrative grammar.
Visual style is a separate render policy. If legacy meta needs a display field,
stamp a clear key such as:

- `meta.visual_style`
- `meta.visual_style_id`

Do not overload `resolved["style"]` unless a downstream consumer truly needs a
human-readable narrative style string.

## Public Domain

For now, omit `public_domain_story` from the workflow selector. It may be
reserved in docs, but the production registry should either omit it or raise
`UnsupportedSourceBankError` only in a test-only path. Do not expose dead UI.

## Workflow Validation Gates

The workflow transplant must run:

- `OTR_WorkflowValidator`
- JSON round-trip
- link referential integrity audit
- live `INPUT_TYPES` widget count vs JSON `widgets_values`
- forceInput sockets carry no widget slot
- `scripts/otr_api.py` and `_otr_workflow_apply.py` whitelist parity tests
- focused source/story prompt leakage tests
- focused visual style leakage tests

## R3 Judgment Log

Accepted:

- Codex anchor: split source/story and visual/video transplant stages.
- Antigravity: coda must return `LineResult`.
- Antigravity: `auto` story model needs a resolver.
- Antigravity: casting seed must stay integer.
- Antigravity: MetaBrief direct `IMAGE_GRADE_TAIL` appends bypass policy.
- Antigravity: legacy mirror needs `headline` as well as title.
- Claude: outline system prompts need profile fields.
- Claude: pitch room and story select system prompts need profile fields.
- Claude: `_otr_style_picker.pick_style()` needs prompt override kwargs.
- Claude: `OutlineRequest` must grow defaulted profile fields before adapter
  wiring.
- Claude: `SourceBrain.interpret()` should accept creative/technical functions,
  not a single vague `generate_fn`.

Rejected or modified:

- Antigravity: always run the existing style picker for media archive. Modified:
  acceptable only if the picker receives source/profile-specific prompts and a
  source-scoped candidate pool.
- Antigravity: propagate visual style by overloading `resolved["style"]`.
  Modified: keep visual style in explicit visual meta/policy keys unless a
  specific legacy consumer is migrated intentionally.
- Claude: cut visual render catalog entirely from R2. Modified: defer it until
  visual policy integration begins, but it remains part of the architecture.

