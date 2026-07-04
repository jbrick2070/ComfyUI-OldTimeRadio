# Source Bank And Visual Style Code-Ready Seed Plan

Status: seed for kibitz R1-R4.

## Core Architecture

The ledger is the production bible. Its schema/contract stays intact so cast,
audio, image, video, and OBS nodes keep consuming one validated ledger.

The front end becomes multi-brain:

```
source_bank -> source-specific writer prompt pack -> same ledger contract
```

At the highest level, this should reuse the existing multi-stage story/ledger
structure wherever possible. The new abstraction is not "new ledger per
source." It is:

```
source_intent -> prompt profile -> existing multi-stage story structure -> same ledger
```

Source intent examples:

- science/news: dramatize or explain a real science/news seed.
- media RSS/archive: dramatize archive, preservation, restoration, media
  history, lost-film, broadcast, or cultural-memory material.
- public-domain text: adapt a real story text while preserving its characters,
  turns, and ending.

The prompts change because the source intent changes. The ledger contract and
downstream structure stay stable.

The visual side becomes multi-bible:

```
visual_style -> meta.visual_style + visual ledger direction -> still/video prompts
```

The two controls are independent:

```
source_bank  = science_news | media_archive | public_domain_story
visual_style = sci_fi_radio | media_archive | cinematic_35mm | noir |
               anime | cartoon | paper_origami | ...
```

Examples:

- `science_news` + `anime`: science/news story with anime visual language.
- `media_archive` + `sci_fi_radio`: archive-sourced story with classic OTR sci-fi visuals.
- `public_domain_story` + `paper_origami`: faithful adaptation with folded-paper visuals.
- `media_archive` + `media_archive`: archive-sourced story with archival/restoration visuals.

## Hard Rules

1. Same ledger contract.
   - Source banks change the data and writer prompts that fill the ledger.
   - They do not create separate downstream ledger types.

2. Different source prompt packs.
   - `science_news`: current real-science/news-driven prompt pack.
   - `media_archive`: archive/preservation/history prompt pack.
   - `public_domain_story`: source-text adaptation prompt pack preserving story, characters, turns, and ending.

3. Visual style is ledger-level direction.
   - Style is stamped into `meta.visual_style`.
   - ShotLock and MetaBrief use it to shape visual ledger fields, still prompts, and video prompts.
   - Style does not rewrite source facts or dialogue contracts.

4. No silent fallback.
   - Unknown `source_bank` raises.
   - `media_archive` never falls back to science RSS or science prompts.
   - `public_domain_story` raises a clear not-implemented error until its real adapter lands.
   - Missing/malformed visual-style policy must hard-error or visibly report, not silently become default.
   - No broad `except Exception: use old behavior` in touched code.

5. Real workflow stays authoritative.
   - Any node/widget/socket/wiring change updates `workflows/otr_scifi_16gb_full.json` in the same chunk.
   - Widget values are positional; append only.
   - Validate with workflow validator, JSON round-trip, link audit, widget/input audit.

## Initial Code Chunks

### C0: Contracts

Add pure modules:

- `nodes/_otr_source_packet.py`
- `nodes/_otr_story_blueprint.py`
- `nodes/_otr_visual_style_policy.py`
- `nodes/_otr_story_prompt_profile.py`

Use Pydantic v2, `ConfigDict(extra="forbid")`, and `Field(default_factory=...)`.

`StoryInputPacket` must carry source provenance and source-specific story intent:

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
key_terms: list[str] = Field(default_factory=list)
adaptation_trace: dict[str, Any] = Field(default_factory=dict)
```

Compatibility helpers may populate legacy-shaped `meta.news`, but only from the active packet.

`VisualStylePolicy`:

```
style_id: str
label: str = ""
base_tail_strategy: Literal["keep", "replace", "suppress"] = "keep"
positive_tail: str = ""
forbidden_terms: list[str] = Field(default_factory=list)
ledger_directives: dict[str, Any] = Field(default_factory=dict)
notes: str = ""
```

Initial style IDs:

- `sci_fi_radio`
- `media_archive`
- `cinematic_35mm`
- `noir`
- `anime`
- `cartoon`
- `paper_origami`

### C1: Source Bank Selector

Append `source_bank` to `OTR_LedgerScriptWriter.INPUT_TYPES`:

```
["science_news", "media_archive", "public_domain_story"]
default: "science_news"
```

Required deltas:

- Append `"science_news"` to writer node 1 `widgets_values`.
- Update writer inline optional-widget assertion.
- Update workflow JSON widget-vector guardrail.
- Add `source_bank` to the headless creative whitelist in both workflow apply and API.
- Keep default `science_news` output stable.

### C2: Source Prompt Profiles

All active story-generation prompts must route through source-bank profile text or prove they are test-only/dead legacy.

Audit surface from local search:

- `nodes/_otr_outline.py`
- `nodes/_otr_pitch_room.py`
- `nodes/_otr_story_select.py`
- `nodes/OTR_LedgerScriptWriter.py`
- legacy `nodes/story_orchestrator.py` must not receive new source-bank paths.

Known science wording to parameterize or isolate:

- "science-fiction"
- "science story"
- "real science"
- "grounded in real science"

For V1, non-science banks may bypass existing science-specific pitch-room/story-select/refine-grading modules, but they must use their own source-bank prompt profile to fill the same ledger contract.

### C3: Media Archive Bank

Add:

- `nodes/_otr_archive_sources.py`
- `nodes/_otr_archive_interpreter.py`

Runtime sequencing:

- `_resolve_inputs` reads source_bank and selects/fetches a raw archive item only.
- Archive LLM interpretation runs later in writer `run()` after model/generate functions exist.
- Archive prompts create a `StoryInputPacket`, `OutlineRequest`, and same ledger fields.
- Empty `script_brief` fails closed.

Sources:

- Start with LOC/NFPF/ACE-style archive/preservation feeds or curated indexes.
- Tests use network-free fixture items.

### C4: Visual Style Policy

Add `nodes/otr_visual_style_director.py` and register `OTR_VisualStyleDirector`.

Node output:

```
RETURN_TYPES = ("STRING",)
RETURN_NAMES = ("visual_style_policy_json",)
FUNCTION = "direct"
```

The node returns `(visual_style_policy_json,)`.

### C5: Visual Style Ledger/Prompt Integration

Append `visual_style_policy_json` force-input socket to:

- `OTR_MetaBriefImagePromptGen.generate(..., visual_style_policy_json="{}")`
- `OTR_ShotLock.lock(..., visual_style_policy_json="{}")`

Each method parses the policy and injects it into local `meta["visual_style"]`.

ShotLock must stamp the parsed style into the returned patched ledger. MetaBrief uses the same policy for still prompt generation.

Modify `finish_visual_prompt` so style policy is applied at the common visual prompt seam:

- no policy means unchanged only for old unwired graphs
- wired bad/missing policy reports hard
- policy positive tail is not suppressed by `style_tail=False`
- forbidden term scrub happens before final prompt output

### C6: Workflow Wiring

Update canonical workflow:

- Add `OTR_VisualStyleDirector` near visual policy nodes.
- Default style should preserve current shipped behavior, likely `sci_fi_radio` or `cinematic_35mm` after code confirms which is byte-stable.
- Fan output to MetaBrief and ShotLock.
- Preserve existing VideoDirector/ImageDirector links.
- Recompute IDs and slots live before edit.

### C7: Public Domain Adapter

After C0-C6 are green:

- Append `source_text_path`.
- Read source text.
- Build `StoryInputPacket`.
- Build `StoryBlueprint`.
- Fill same ledger through existing outline/ledger path.
- Stamp source-fidelity/adaptation trace.

Hard fidelity verifier is later, after the adapter is green.

## Verification

Required tests/gates:

- source packet/schema tests
- visual style policy tests
- prompt profile tests for no science wording in archive mode
- default science-news output/prompt stability
- malformed source/style no-fallback tests
- writer widget count and workflow widget-vector tests
- headless whitelist parity test
- workflow validator, JSON round-trip, link audit, widget/input audit after workflow edits
- full regression suite and Bug Bible after code changes
