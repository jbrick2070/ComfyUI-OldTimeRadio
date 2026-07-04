# R2 Hardened Coding Plan

Status: coding-plan synthesis. This is the input for wiring/integration review.

## Verdict

Buildable with fixes. The first implementation must be a conservative bridge:
pure schemas + existing writer/prompt-node integration first, new standalone
source nodes later.

## Grounded Decisions

1. V1 keeps `OTR_LedgerScriptWriter` as the story owner.
   - Do not create `OTR_StoryDirector`, `OTR_SourceBankDirector`, or
     `OTR_StorySourceInterpreter` nodes in the first sprint.
   - Add pure modules and a writer bridge first.

2. V1 adds a source selector to the existing writer by appending optional
   widgets only.
   - `source_bank`: combo, default `science_news`, choices
     `science_news`, `media_archive`, `public_domain_story`.
   - Later public-domain chunk appends `source_text` or `source_text_path`;
     do not add unused public-domain widgets before the adapter exists.

3. V1 keeps the writer's current `IS_CHANGED = time.time()` behavior.
   - That behavior already exists.
   - Hash/refresh-nonce `IS_CHANGED` applies only when source loading moves to
     standalone source nodes.

4. `StoryInputPacket` becomes canonical source metadata, but `meta.news` stays
   current-shaped.
   - `meta.source`: packet/provenance fields.
   - `meta.news`: remains the legacy brief-shaped dict expected by composers.
   - In non-news modes, create a brief-shaped compatibility dict from the
     packet with exact legacy keys.

5. `VisualStylePolicy` must affect `finish_visual_prompt`.
   - Confirmed: render-driver fallback scene prompts call `finish_visual_prompt`
     after ShotLock.
   - Styling only `OTR_MetaBriefImagePromptGen` and `OTR_ShotLock` is not enough.

6. Public-domain adaptation starts with a `StoryBlueprint`, but enforcement is
   not first-chunk.
   - First PD chunk stamps blueprint/adaptation trace.
   - Hard LLM-based source-fidelity verification is a later acceptance gate, not
     required for the packet/blueprint seam.

## Code Chunks

### C0: Pure Contracts

Add:

- `nodes/_otr_source_packet.py`
- `nodes/_otr_story_blueprint.py`
- `nodes/_otr_visual_style_policy.py`

Use Pydantic v2, matching existing repo style:

- `BaseModel`
- `ConfigDict(extra="forbid")`
- explicit defaults
- helper functions to load/dump JSON strings without ComfyUI imports

`StoryInputPacket` fields:

```
packet_version: int = 1
source_bank_id: str
source_mode: str
source_kind: str
source_label: str
rights_status: str = "unknown"
source_title: str = ""
source_author: str = ""
source_url: str = ""
source_hash: str = ""
source_text_ref: str = ""
source_summary: str = ""
casting_brief: str = ""
script_brief: str = ""
close_brief: str = ""
key_terms: list[str] = []
adaptation_trace: dict = {}
```

Compatibility adapter:

```
packet_to_legacy_news_meta(packet) -> dict
packet_to_news_used(outline, packet) -> str
```

Legacy mapping:

- `script_brief` -> `meta.news.script_brief`
- `close_brief` -> `meta.news.news_close_brief`
- `casting_brief` -> `meta.news.casting_brief`
- `key_terms` -> `meta.news.key_terms`
- `source_title` -> `news_used[0].headline` fallback
- `source_summary` or `script_brief` -> `news_used[0].summary`
- `source_text_ref` or `script_brief` -> `news_used[0].full_text`
- `source_label` -> `news_used[0].source`
- `source_url` -> `news_used[0].link`

`VisualStylePolicy` defaults:

- `cinematic_35mm`: exact current tail behavior; keep existing constants.
- `archival_mono`: replace base tail with monochrome/archive language; forbid
  vibrant/color/modern glossy terms.
- `anime`: replace base tail with animated/cel-shaded language; forbid
  photorealistic/35mm/film-grain/photo terms.

Tests:

- `tests/test_source_packet_contract.py`
- `tests/test_story_blueprint_contract.py`
- `tests/test_visual_style_policy.py`

### C1: Writer Source Bridge, Default No-Op

Modify `nodes/OTR_LedgerScriptWriter.py`:

- Append optional `source_bank` combo at the end of `INPUT_TYPES`.
- Extend `_resolve_inputs` to carry `source_bank`.
- For default `science_news`, build a `StoryInputPacket` from the current
  RSS/custom-premise article and existing `NewsBriefs`.
- Stamp `meta.source`.
- Keep `meta.news = briefs.model_dump()` for the normal science-news path.
- Build `news_used` through a helper that preserves current payload shape.

Do not:

- Rename outputs.
- Move LLM/model widgets.
- Change workflow wiring except the appended widget value for the writer node.

Tests:

- Extend `tests/test_news_interpreter_wiring.py`.
- Extend `tests/test_lfc_c4_news_used_passthrough.py`.
- Add default science-news packet mirror assertions.
- Run workflow widget-vector tests.

### C2: Source Prompt Profile

Add a pure prompt-profile helper, for example:

- `nodes/_otr_story_prompt_profile.py`

Purpose:

- Convert `source_bank/source_kind` into prompt labels.
- Keep science-news defaults byte-identical or text-identical.
- Provide archive and PD labels without touching every prompt by hand.

Add optional fields to `OutlineRequest` at the end:

```
source_label: str = "Science story"
source_develop_verb: str = "extrapolates from the science story"
story_form_label: str = "science-fiction audio drama"
story_system_label: str = "short science-fiction audio dramas grounded in real science"
```

Modify:

- `_otr_outline._build_user_prompt`
- `_otr_outline._build_macro_user_prompt`
- outline system prompt construction

Audit and either parameterize or explicitly disable for non-science modes:

- `_otr_pitch_room.py`
- `_otr_story_select.py`

Tests:

- New tests verify default science prompt text remains stable.
- Archive prompt tests assert no "Science story" fallback in archive mode.
- Non-default prompt profile test asserts no "grounded in real science" system
  prompt when `source_bank=media_archive`.

### C3: Media Archive Bank

Add:

- `nodes/_otr_archive_sources.py`
- `nodes/_otr_archive_interpreter.py`

Implementation:

- Curated feed list: LOC / NFPF / ACE.
- Fetch source item using existing RSS/fetch patterns where possible.
- Convert to `StoryInputPacket`.
- Run the same technical LLM brief path or a thin archive-specific wrapper.
- Fail closed if `script_brief` is empty.

Writer integration:

- Branch before current `_fetch_rss_seed_or_die`.
- `source_bank=science_news`: current path.
- `source_bank=media_archive`: archive fetch + archive packet + legacy mirror.
- `source_bank=public_domain_story`: raise clear not-implemented until C7.

Tests:

- Archive packet unit test with fixture item.
- Archive prompt does not include science labels.
- Empty archive brief raises before outline.

### C4: Visual Style Pure Helpers

Modify:

- `nodes/_otr_story_brief_helpers.py`

Add:

- `apply_visual_style_policy(meta, prompt, *, style_tail=True, era_profile="full")`
  or equivalent helper used by `finish_visual_prompt`.

Rules:

- If no `meta.visual_style`, existing output remains unchanged.
- `base_tail_strategy=keep`: current behavior.
- `replace`: remove/suppress current `STYLE_TAIL_DEFAULT` and append policy
  positive tail.
- `suppress`: no base style tail.
- Always apply forbidden-term scrub before final hash at call sites.

Tests:

- `cinematic_35mm` exact current tail.
- `archival_mono` removes 35mm/color-vibrant terms and adds monochrome tail.
- `anime` removes photoreal/35mm/film terms and adds anime tail.
- Existing `tests/test_brief_prompt_finishing.py` remains green.

### C5: VisualStyleDirector Node

Add:

- `nodes/otr_visual_style_director.py`
- Register `OTR_VisualStyleDirector` in `__init__.py`.

Node contract:

```
CATEGORY = "OldTimeRadio/v2/video"
FUNCTION = "direct"
RETURN_TYPES = ("STRING",)
RETURN_NAMES = ("visual_style_policy_json",)
```

Inputs:

- required or optional widget `style_id` combo:
  `cinematic_35mm`, `archival_mono`, `anime`
- optional `custom_policy_json` string for later/manual experiments only if it
  does not add risk; otherwise defer.

No heavy imports. No model widgets.

Tests:

- Node registration.
- `INPUT_TYPES` shape.
- JSON output validates as `VisualStylePolicy`.

### C6: Wire Visual Style Into Prompt Producers

Modify:

- `nodes/otr_meta_brief_image_prompt.py`
- `nodes/otr_shot_lock.py`

Append optional forceInput socket:

```
visual_style_policy_json: STRING, default "{}", forceInput True
```

`OTR_MetaBriefImagePromptGen`:

- Parse policy.
- Add it to a local/meta copy as `meta.visual_style` before deriving prompts.
- Existing no-policy path is unchanged.

`OTR_ShotLock`:

- Parse policy.
- Stamp `meta.visual_style`.
- Ensure M4/directive prompt path and render-driver fallback path both see it
  through ledger meta.

Workflow:

- Add `OTR_VisualStyleDirector` node.
- Wire its output to `OTR_MetaBriefImagePromptGen` and `OTR_ShotLock`.
- Update `workflows/otr_scifi_16gb_full.json` in the same change.
- Run workflow validator, JSON round-trip, link audit, widget/input audit.

### C7: Public-Domain Text Adapter

Only after C0-C6 are green.

Append writer inputs at the end:

- `source_text` multiline string or `source_text_path` string. Prefer one first,
  not both, unless tests show both are needed.

Add:

- `nodes/_otr_pd_adapter.py`

Flow:

```
operator text -> StoryInputPacket -> StoryBlueprint -> OutlineRequest blueprint fields -> existing outline -> ledger
```

Outline integration:

- Add optional `blueprint_json` or typed blueprint field to `OutlineRequest`.
- Serialize required beats and ending into prompt context.
- Do not write ledger lines directly.

V1 fidelity:

- Stamp `meta.source.blueprint` and `meta.source.adaptation_trace`.
- Add lightweight deterministic checks first.
- LLM-based hard fidelity verifier is a later chunk.

## Test And Verification Plan

Focused tests per chunk:

- `pytest -q -p no:cacheprovider tests/test_source_packet_contract.py`
- `pytest -q -p no:cacheprovider tests/test_visual_style_policy.py`
- `pytest -q -p no:cacheprovider tests/test_news_interpreter_wiring.py`
- `pytest -q -p no:cacheprovider tests/test_lfc_c4_news_used_passthrough.py`
- `pytest -q -p no:cacheprovider tests/test_brief_prompt_finishing.py`
- `pytest -q -p no:cacheprovider tests/test_workflow_json_wiring_invariants.py`
- `pytest -q -p no:cacheprovider tests/test_default_workflow_validator.py`

After code changes:

- Run targeted tests for the chunk.
- Run the regression suite.
- Run the Bug Bible from the separate survival-guide repo.
- Validate `workflows/otr_scifi_16gb_full.json` with
  `OTR_WorkflowValidator`, JSON round-trip, widget/input audit, link integrity.
- Commit and push green chunks to `v2.0-alpha`.

## R2 Judgment Log

Accepted:

- Codex anchor: exact module/test targets needed.
- Codex/Claude: add explicit `OutlineRequest` source fields, not an ambiguous
  wrapper.
- Antigravity/Claude: visual style must reach `finish_visual_prompt`.
- Antigravity: define `OTR_VisualStyleDirector` node contract before coding.
- Antigravity: add writer `source_bank` selector appended at the end.
- Antigravity: arbitrary Gutenberg search stays cut.
- Claude: `meta.news` shape collision is real; keep `meta.source` standalone
  and mirror to a current-shaped `meta.news`.
- Claude: writer `IS_CHANGED=time()` is already production behavior; clarify
  scope instead of changing it mid-plan.

Rejected or deferred:

- Antigravity: immediate LLM-based fidelity verifier in the first PD chunk.
  Deferred; first chunk stamps blueprint/trace and adds deterministic checks.
- Claude: cutting `source_author`, `source_url`, `close_brief`,
  `adaptation_constraints` wholesale. Modified: keep provenance fields that are
  needed for rights/source audit; keep `close_brief` because it maps to legacy
  `news_close_brief`; defer complex adaptation constraints.
- Antigravity: source text widgets in the first source selector chunk. Deferred
  until C7 so unused widgets do not land early.

Verify in R3:

- Exact workflow placement for `OTR_VisualStyleDirector`.
- Whether `OTR_MetaBriefImagePromptGen` precedes `OTR_ShotLock` in the canonical
  workflow and how the style policy fan-out should be linked.
- Exact link/widget deltas for `workflows/otr_scifi_16gb_full.json`.

