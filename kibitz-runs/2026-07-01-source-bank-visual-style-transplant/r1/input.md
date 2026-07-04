# Consolidated Kibitz Input: Source Banks, Ledger-Writing Spec, Visual Styles

Status: current source of truth for the restarted R1-R4 kibitz arc.

This supersedes earlier R1/R2 inputs. Use this document plus the two audits as
the input for review:

- `docs/2026-07-01-source-bank-visual-style-code-ready/LEDGER_PROMPT_AUDIT.md`
- `docs/2026-07-01-source-bank-visual-style-code-ready/VISUAL_PROMPT_AUDIT.md`

Do not treat older source-bank plans as authoritative when they conflict with
this file.

## Operator Intent

The production story ledger is the bible. It should stay one stable downstream
contract consumed by cast, audio, images, video, and OBS.

The new architecture is a fresh upstream layer that decides how to write that
ledger:

```
source_bank -> source packet -> source prompt profile -> ledger_writing_spec
ledger_writing_spec -> existing multi-stage story/ledger writer -> production ledger
```

This is not a separate downstream ledger per source. It is a higher-level
ledger-writing control plane that changes prompt intent, provenance, source
fidelity, and visual bible direction while filling the same production ledger.

Build the upstream architecture from scratch where that is cleaner, then
transplant it into the existing production flow once the contracts and content
are ready.

## Hard Workflow Rule

Do not touch:

`workflows/otr_scifi_16gb_full.json`

until the upstream story-writing architecture and content are ready to
TRANSPLANT. The transplant must be its own explicit chunk with workflow
validation, JSON round-trip, link audit, and widget/input audit.

During this kibitz arc, plan the transplant but do not perform it.

## Source Banks

`source_bank` is the top-level source/story brain selector.

Required initial banks:

- `science_news`: current science/news-driven story brain.
- `media_archive`: archive, preservation, restoration, broadcast history, lost
  media, genre-film/TV artifact, or cultural-memory story brain.
- `public_domain_story`: faithful adaptation brain for public-domain source
  text, preserving source characters, turns, and ending.

The source bank changes prompts and source data, not the downstream ledger
contract.

Examples:

- `science_news`: "Create a fictional radio drama based on this real
  science/news seed."
- `media_archive`: "Create a non-violent fictional story about a media studies
  phenomenon, restoration mystery, genre-film or television artifact,
  broadcast-history oddity, or cultural-memory object."
- `public_domain_story`: "Create a BBC radio-drama / filmed puppet-show style
  adaptation of this public-domain classic while preserving its characters,
  turns, and ending."

Hard media-archive correction: the media-archive bank must not produce
Star-Trek/Amazing-Stories-style speculative TV plots with archive nouns swapped
in. It needs its own story shapes: cinematic, humorous, happy/upbeat, gentle
thriller, and media-restoration adventure. Reviewers should flag any plan that
keeps the current sci-fi anthology plot machine and merely renames the seed.

## Story Model / Tone Layer

There is a third concept besides source bank and visual style:

```
source_bank = where the material comes from
story_model = how the story is shaped tonally/dramatically
visual_style = how the ledger directs still/video imagery
```

`story_model` may be source-aware. It can live inside the selected prompt
profile at first, but the architecture should leave room for it as an explicit
control later.

Initial media-archive story models should include:

- `media_restoration_adventure`: a restoration, preservation, lost-reel,
  cataloging, rights, damaged-signal, or rediscovery adventure.
- `cinematic_humorous`: a polished, funny media-culture story with warm stakes.
- `happy_archive_mystery`: upbeat puzzle/mystery around a broadcast, object,
  record, print, performance, or cultural-memory artifact.
- `gentle_thriller`: suspense through deadlines, missing evidence, fragile
  media, institutional pressure, or a live event; non-violent and not horror.
- `broadcast_history_comedy`: character-driven comedy around production,
  reception, fandom, preservation, or media scholarship.

These are story-writing profiles, not visual styles. They can combine with
visual styles independently, e.g. `media_archive + gentle_thriller + anime`.

## Ledger-Writing Spec

The upper-level spec should tell the writer how to fill the production ledger.
It is the "super-ledger" for writing logic, not a replacement for the final
ledger.

It should include at least:

- selected `source_bank`
- selected `source_intent`
- source packet/provenance
- prompt profile id and prompt-profile variables
- story model / tonal lane and forbidden plot patterns
- adaptation mode and fidelity checks
- source material labels used in prompts
- coda/source-note mode
- selected `visual_style`
- visual ledger directives for still/video fields
- validation expectations

Compatibility helpers may mirror active packet fields into legacy
`meta.news.*`, but only from the active source packet. Do not let `meta.news`
mean "science news" forever.

## Prompt Profile Rule

All material ledger-filling prompts must be profile-aware if they contain
source-specific language. Hardcoded Python prompt strings are allowed only when
they are genuinely source-neutral radio-drama craft.

Move these behind prompt-profile variables or prove the path is dead/test-only:

- science-fiction audio drama
- sci-fi radio drama
- science story
- real science
- grounded in real science
- real news item
- news event
- NEWS PREMISE
- NEWS KEY TERMS
- real news report
- news facts
- news_close_brief

Do not rip out reusable radio-drama machinery just because its current wording
says science. Prefer parameterizing pitch-room, story-select, outline,
dramatic-state, line composer, title, and coda prompts.

Use `LEDGER_PROMPT_AUDIT.md` as the live prompt inventory and scope guide.

## Visual Style Architecture

`visual_style` is independent from `source_bank` and should be an equal
first-class selector.

Examples:

- `science_news` + `anime`
- `media_archive` + `cartoon`
- `public_domain_story` + `paper_origami`
- `media_archive` + `media_archive`
- `science_news` + `sci_fi_radio`

Initial styles:

- `sci_fi_radio`
- `media_archive`
- `cinematic_35mm`
- `noir`
- `anime`
- `cartoon`
- `paper_origami`

Visual style should stamp the ledger, not just add a suffix at the end:

```
visual_style -> VisualStylePolicy -> meta.visual_style
meta.visual_style -> MetaBrief + ShotLock + shared prompt seams
```

Style should affect:

- visual ledger directives
- MetaBrief still prompts
- ShotLock patched ledger
- still prompt finishers
- video/motion prompt fallbacks that currently assume 1940s radio/cinematic

Style must not rewrite source facts, cast contracts, dialogue, or source-story
fidelity.

Use `VISUAL_PROMPT_AUDIT.md` as the live inventory of hardcoded visual style
language and role-safety constraints.

## No Silent Fallbacks

Unknown or unsupported source/style must hard-error or produce a visible
diagnostic. No hidden "try new path then silently use science/default."

Specific rules:

- unknown `source_bank` raises
- `media_archive` never falls back to science RSS or science prompts
- `public_domain_story` can raise clear not-implemented until its adapter is
  real
- malformed `visual_style_policy_json` fails in wired new paths
- no broad `except Exception: use old behavior`
- no hidden offline archive fallback unless it is an explicit selected mode

## Proposed Pure Modules Before Transplant

These modules can be built and tested before touching the canonical workflow:

- `nodes/_otr_source_packet.py`
- `nodes/_otr_ledger_writing_spec.py`
- `nodes/_otr_story_prompt_profile.py`
- `nodes/_otr_source_brains.py` or per-bank modules
- `nodes/_otr_ledger_input_adapter.py`
- `nodes/_otr_visual_style_policy.py`
- `nodes/_otr_visual_style_catalog.py`

Likely later transplant-visible node:

- `nodes/otr_visual_style_director.py`

Later production wiring/registration must update root `__init__.py`, ComfyUI
node contracts, and the canonical workflow in the transplant chunk only.

## Expected Contracts

`StoryInputPacket` should carry source provenance and source-specific story
intent. Minimum shape:

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

`VisualStylePolicy` should include enough structured fields to replace visual
hardcoding:

```
style_id
label
positive_tail
negative_or_forbidden_terms
base_tail_strategy
image_grade_tail
radio_broadcast_tail_replacement
announcer_visual_subject
music_visual_subject
scene_open_subject
character_portrait_style
character_scene_style
motion_prompt_profile
ledger_directives
```

Contracts should be strict for canonical new objects: Pydantic v2,
`ConfigDict(extra="forbid")`, and `Field(default_factory=...)` for mutable
defaults.

## Review Questions By Round

R1: Does this upper-level architecture match the real need: source brains and
visual style policies fill the same ledger through an upstream writing spec?
What high-level risks would break the workflow later?

R2: What is the exact coding plan for pure modules, contracts, prompt profiles,
source brains, adapters, and tests before workflow transplant?

R3: What is the safe transplant/wiring sequence for writer inputs, visual style
node, MetaBrief, ShotLock, workflow JSON, widget vectors, and headless/API
whitelists?

R4: What residual defects, hidden fallbacks, or legacy prompt assumptions still
need to be removed before implementation starts?

## Non-Goals For This Kibitz Run

- Do not edit production code.
- Do not edit `workflows/otr_scifi_16gb_full.json`.
- Do not render.
- Do not run full regression suites.
- Do not choose final archive/public-domain data vendors beyond defining clean
  source packet boundaries and fixture-first tests.
