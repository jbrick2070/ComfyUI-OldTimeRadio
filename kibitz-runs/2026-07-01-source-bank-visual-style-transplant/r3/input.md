# R2 Final: Code-Ready Module Plan

Status: grounded coding-plan synthesis for R3 wiring/transplant review.

## Verdict

Code-ready after the fixes below are applied to the implementation plan.

The implementation must start with pure upstream modules and prompt-surgery
artifacts, not workflow edits. The purpose is to clone/reuse good story
machinery, surgically swap prompt variables and forbidden patterns, then
translate into the existing production ledger.

## Core Decision

`story_model_id` is a source-scoped writing lane. It does not mean visual style.
It does not blindly reuse the existing `_otr_style_catalog.py`.

V1 decision:

- `science_news` keeps current narrative-style picker behavior by default.
- `media_archive` uses a source-scoped story-model catalog first.
- `story_model_id="auto"` may pick from that source bank's allowed story models.
- An explicit media-archive story model directly selects that model's prompt
  guardrails and forbidden patterns.
- The existing `_otr_style_picker.py` can be generalized later, but it must not
  run raw sci-fi inventor/chooser prompts for `media_archive`.

This prevents the "Star Trek / Amazing Stories with archive nouns" failure.

## C0: Pure Contracts

Use Pydantic v2 only. Local venv confirms `pydantic==2.12.5`.

Create `nodes/_otr_source_packets.py`:

```python
class SourceMaterialPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    packet_version: int = 1
    source_bank_id: str
    source_mode: str = "fixture"
    source_kind: str = ""
    source_label: str = ""
    rights_status: Literal["unknown", "public_domain", "licensed", "fair_use_research"] = "unknown"
    source_title: str = ""
    source_author: str = ""
    source_url: str = ""
    source_hash: str = ""
    source_text_ref: str = ""
    source_summary: str = ""
    raw_text: str = ""

class StoryInputPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    packet_version: int = 1
    source_bank_id: str
    story_model_id: str = "auto"
    source_label: str = ""
    casting_brief: str = ""
    script_brief: str = ""
    close_brief: str = ""
    key_terms: list[str] = Field(default_factory=list)
    source_fidelity_rules: list[str] = Field(default_factory=list)
    adaptation_trace: dict[str, Any] = Field(default_factory=dict)
    source_material: SourceMaterialPacket
```

Create `nodes/_otr_story_model_catalog.py`:

```python
class StoryModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_bank_id: str
    story_model_id: str
    label: str
    tone_guardrails: list[str] = Field(default_factory=list)
    forbidden_plot_patterns: list[str] = Field(default_factory=list)
    outline_rules_extra: str = ""
```

Initial `media_archive` models:

- `media_restoration_adventure`
- `cinematic_humorous`
- `happy_archive_mystery`
- `gentle_thriller`
- `broadcast_history_comedy`

Create `nodes/_otr_story_prompt_profile.py`:

```python
class StoryPromptProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_bank_id: str
    story_model_id: str = "auto"
    story_form_label: str
    source_material_label: str
    source_develop_verb: str
    source_grounding_label: str
    key_terms_label: str = "KEY TERMS"
    close_brief_label: str = "source note"
    coda_mode: Literal["real_news_report", "archive_source_note", "source_attribution", "none"]
    title_form_label: str
    line_grounding_instruction: str
    outline_rules_extra: str = ""
    tone_guardrails: list[str] = Field(default_factory=list)
    forbidden_plot_patterns: list[str] = Field(default_factory=list)
    inventor_system_prompt: str = ""
    chooser_system_prompt: str = ""
```

Create `nodes/_otr_ledger_writing_spec.py`:

```python
class LedgerWritingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_bank_id: str
    story_model_id: str
    visual_style_id: str = "sci_fi_radio"
    source_material: SourceMaterialPacket
    story_input: StoryInputPacket
    prompt_profile: StoryPromptProfile
```

Create `nodes/_otr_visual_style_policy.py`:

```python
class VisualStylePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    style_id: str
    label: str = ""
    positive_tail: str = ""
    image_grade_tail: str = ""
    broadcast_tail: str = ""
    allow_radio_tails: bool = True
    forbidden_terms: list[str] = Field(default_factory=list)
    announcer_visual_subject: str = ""
    music_visual_subject: str = ""
    scene_open_subject: str = ""
    character_portrait_style: str = ""
    character_scene_style: str = ""
    motion_prompts: dict[str, str] = Field(default_factory=dict)
    ledger_directives: dict[str, Any] = Field(default_factory=dict)
```

Create `nodes/_otr_visual_render_catalog.py`, not
`_otr_visual_style_catalog.py`, to avoid confusion with existing narrative
`_otr_style_catalog.py`.

## C1: Fail-Closed Registries

Implement:

```python
get_story_model(source_bank_id: str, story_model_id: str) -> StoryModelSpec
get_profile(source_bank_id: str, story_model_id: str) -> StoryPromptProfile
get_visual_style_policy(style_id: str) -> VisualStylePolicy
get_source_brain(source_bank_id: str) -> SourceBrain
```

Unknown ids raise typed errors:

- `UnknownSourceBankError`
- `UnknownStoryModelError`
- `UnknownVisualStyleError`
- `UnsupportedSourceBankError`

No registry returns a science/default object for an unknown non-science id.

Visual style ids:

- `sci_fi_radio`
- `archival_documentary`
- `cinematic_35mm`
- `noir`
- `anime`
- `cartoon`
- `paper_origami`

`archival_documentary` is the visual style formerly described as
`media_archive`; `media_archive` remains only the source-bank id.

## C2: Source Brains, Fixture First

Create `nodes/_otr_source_brains.py` with a tiny protocol/base class:

```python
class SourceBrain(Protocol):
    source_bank_id: str
    def material_from_fixture(self, fixture: dict[str, Any]) -> SourceMaterialPacket: ...
    def interpret(self, material: SourceMaterialPacket, *, generate_fn=None) -> StoryInputPacket: ...
```

R2 implementation is fixture-only:

- no live archive RSS
- no public-domain file widget
- no network

`public_domain_story` may be registered as reserved, but
`get_source_brain("public_domain_story")` raises `UnsupportedSourceBankError`
until a real fixture/text adapter exists. Do not expose it in the workflow
dropdown yet.

## C3: Adapter And Field Map

Create `nodes/_otr_ledger_input_adapter.py`.

Required helpers:

```python
build_legacy_news_mirror(packet: StoryInputPacket) -> dict[str, Any]
build_outline_request_kwargs(spec: LedgerWritingSpec) -> dict[str, Any]
build_casting_seed(spec: LedgerWritingSpec) -> str
build_coda_context(spec: LedgerWritingSpec) -> dict[str, str]
```

Compatibility mirror:

| Source field | Legacy mirror |
|---|---|
| `story_input.script_brief` | `meta.news.script_brief` |
| `story_input.close_brief` | `meta.news.news_close_brief` |
| `story_input.casting_brief` | `meta.news.casting_brief` |
| `story_input.key_terms` | `meta.news.key_terms` |
| `source_material.source_title` | `meta.news.title` / existing expected title key |
| `source_material.source_url` | `meta.news.link` |
| `source_material.source_hash` | `meta.news.source_hash` |

Do not rename `news_close_brief` in the ledger now.

## C4: Prompt Surgery

The prompt audit is the worklist. Each touched prompt gets one of three labels:

- `SHARED`: reusable radio-drama craft, keep common.
- `PROFILE`: replace phrase with profile variable.
- `SOURCE_MODEL`: use source-bank/story-model guardrails.

Concrete surfaces:

- `_otr_outline.py`: source labels, develop verb, story form label,
  `outline_rules_extra`, forbidden patterns.
- `_otr_pitch_room.py`: story form label and source material language.
- `_otr_story_select.py`: story form label and source grounding label.
- `_otr_dramatic_state_llm.py`: replace `NEWS KEY TERMS`, `NEWS PREMISE`,
  "news event", "real news item".
- `_otr_line_composer.py`: grounding instruction and coda mode.
- `_otr_casting.py`: source/casting brief accessors.
- `OTR_LedgerScriptWriter.py`: title prompt and source/coda routing.
- `_otr_style_picker.py`: do not run raw sci-fi inventor/chooser for
  `media_archive`.

For `_otr_style_picker.py`, R2 decision:

- science/default may keep existing picker.
- media archive either bypasses picker or calls it only with
  profile-provided inventor/chooser system prompts and a source-scoped candidate
  pool.
- raw `_INVENTOR_SYSTEM = "You are a sci-fi radio drama showrunner."` is never
  used for non-science banks.

## C5: Coda Mode

Change coda planning, not necessarily production code in the first pure chunk:

```python
compose_source_coda(
    *,
    creative_fn,
    close_brief: str,
    premise: str,
    coda_mode: str,
    transition_pool: tuple[str, ...] = (),
    intro_text: str = "",
) -> CodaResult
```

`compose_news_coda()` can remain as a science-news wrapper.

Modes:

- `real_news_report`: existing behavior / `NEWS_CODA_POOL`.
- `archive_source_note`: archive-context transition, no "headlines" wording.
- `source_attribution`: public-domain attribution/adaptation note.
- `none`: close brief only or no coda, explicit.

## C6: Visual Style Policy

Pure helpers:

```python
parse_visual_style_policy_json(raw: str, *, required: bool) -> VisualStylePolicy | None
stamp_visual_style_meta(meta: dict[str, Any], policy: VisualStylePolicy) -> dict[str, Any]
visual_style_from_meta(meta: dict[str, Any]) -> VisualStylePolicy | None
```

Later integration targets:

- `_otr_story_brief_helpers.finish_visual_prompt`
- `_otr_story_brief_helpers.compose_still_prompt`
- `otr_meta_brief_image_prompt.py`
- `otr_shot_lock.py`
- `_otr_video_engines/render_driver.py`

Motion prompt rule:

```python
policy.motion_prompts.get(role_or_motion_key) or legacy_motion_prompt
```

Legacy fallback is allowed only when style policy is missing in old unwired
graphs or when `style_id == "sci_fi_radio"` explicitly allows radio tails.

Important transplant risk: the source/story upstream can be isolated cleanly,
but visual/video prompts are not all at clean seams. Some are deep inside still
and video repair/fallback code that is being actively fixed. R3 must stage visual
policy extraction:

1. catalog every deep hardcoded visual/video prompt before editing
2. preserve current shipped output under `sci_fi_radio`
3. wire policy reads at shared seams first
4. replace deep fallback prompts one at a time
5. add forbidden-phrase leakage tests per style

## C7: Entry Points Before Workflow Transplant

Do not edit `workflows/otr_scifi_16gb_full.json` in R2.

An isolated `upstream_story_lab/` folder may hold fixture JSON, prompt-profile
drafts, story-model drafts, and transplant manifests. It must not be imported by
production nodes until transplant.

For tests and debug harnesses, use pure entry points:

```python
build_ledger_writing_spec_from_fixture(...)
render_prompt_profile_preview(...)
render_visual_policy_preview(...)
```

Do not add workflow widgets yet. Writer `run()` / `_resolve_inputs()` signature
changes are a later code-integration chunk reviewed in R3, because once they are
wired into ComfyUI they require widget vector and workflow validation.

## Required Tests

Pure-module tests:

- import tests: no torch, no ComfyUI, no network/model imports
- Pydantic schema accepts valid fixtures and rejects extra keys
- unknown source/story/style ids raise typed errors
- media archive fixture builds `LedgerWritingSpec`
- media archive prompt preview lacks:
  - `science-fiction`
  - `sci-fi radio drama`
  - `real science`
  - `news facts`
  - `Star Trek`
  - `spaceship`
  - `mission control`
  - `lab containment`
- `_otr_style_picker.py` sci-fi inventor prompt is not used for media archive
- legacy news mirror maps exactly once through `build_legacy_news_mirror`
- `anime`, `cartoon`, and `paper_origami` policies reject hardcoded 35mm,
  film-grain, 1940s radio, and cinematic lighting tails unless explicitly
  allowed
- `archival_documentary` and `media_archive` ids cannot collide because they are
  different registries

## R2 Judgment Log

Accepted:

- Codex anchor: exact contracts, fail-closed registries, source-scoped story
  model catalog, prompt-surgery artifact as the reason for the audit.
- Antigravity: concrete schemas and typed registry signatures are required.
- Antigravity: fixture-only media archive in R2.
- Antigravity: coda mode needs an execution hook.
- Antigravity: visual motion prompts need policy overrides.
- Claude: define relationship between story_model and existing style picker.
- Claude: adapter needs an explicit field map.
- Claude: pick one source packet module name.
- Claude: raw `_otr_style_picker.py` sci-fi inventor prompt cannot run for
  media archive.
- Claude: visual render catalog should have a more distinct name.

Rejected or deferred:

- Antigravity: add `source_bank_id`, `story_model_id`, and `visual_style_id` to
  writer `run()` in the pure-module chunk. Defer to R3 code-integration /
  transplant sequencing.
- Antigravity: assume default visual style is `cinematic`. Use `sci_fi_radio`
  for current behavior unless tests prove another id is byte-stable.
- Claude: collapse visual style policy into story profile. Rejected because the
  user wants visual style independent from source/story.
- Claude: merge all packet/spec files if it obscures ownership. Accepted only
  for the two packet classes in `_otr_source_packets.py`; keep spec/profile
  modules separate.
