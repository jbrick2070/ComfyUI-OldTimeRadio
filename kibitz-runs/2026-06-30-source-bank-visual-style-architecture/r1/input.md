# Source-Bank and Visual-Style Architecture Scaffold

Status: scaffold for kibitz/spec work. No code is wired yet.

This doc turns the 2026-06-23 source-bank idea into an implementation-shaped
architecture. The key correction is that there are two different forks:

1. The story-source fork happens before outline/ledger generation.
2. The visual-style/model fork happens after the ledger/canon exists, before
   still and video prompts are composed.

Those forks must not be collapsed into one setting. A public-domain story source
changes the story engine. An anime or cinematic visual profile changes prompt
language and model constraints, not the written dialogue.

## Current Seams

- `OTR_LedgerScriptWriter` currently resolves one story input path:
  `custom_premise` or RSS fetch becomes a news-shaped article dict.
- `news_interpreter.build_news_briefs` turns that article into the downstream
  brief contract: `casting_brief`, `script_brief`, `news_close_brief`,
  `key_terms`.
- `_otr_outline.generate_outline` consumes `script_brief` when present, with
  `news_seed` as fallback.
- The final ledger carries `meta`, `cast`, `lines`, canon-ish fields, and
  `meta.visual_plan`; downstream audio, image, video, OBS work from that.
- `OTR_VideoDirector` already proves the director/registry pattern: one node
  emits policy JSON, downstream nodes consume explicit policy instead of
  guessing.
- `OTR_MetaBriefImagePromptGen`, `_otr_story_brief_helpers`, and the video
  render driver are the live prompt surfaces that a visual-style fork must reach.

## Axis A: Story Source and Writer Engine

The source fork owns factual/source material, rights metadata, and the shape of
the story engine. It should end by producing the same downstream story contract,
not by making every later node source-aware.

```
SourceBankDirector
  -> SourceItem
  -> SourceInterpreter
  -> StoryInputPacket
  -> StoryEngine
  -> Outline / Beat[]
  -> Ledger
```

### Source Bank Kinds

`science_news`

- Current production path.
- RSS science article -> news/source interpreter -> `script_brief`.
- Uses the seeded fiction outline engine.
- Keeps the science-fiction prompt assumptions.

`media_archive`

- Similar mechanics to science news, different subject matter.
- LOC/NFPF/ACE archive-media RSS item -> archive-seed interpreter ->
  `script_brief`.
- Still seed-based origination: the engine invents the drama from a rich archive
  hook.
- Must remove "science story" assumptions from prompts. It may be preservation,
  restoration, silent film, radio history, missing footage, recovered media, or
  cultural memory.

`public_domain_story`

- Different mechanics.
- Actual public-domain text -> adaptation interpreter -> source-fidelity plan.
- The source provides premise, characters, arc, and ending, so the engine should
  not merely "invent from a seed."
- Likely bypasses the current seed-to-outline planner, or enters through a
  stricter `StoryBlueprint` adapter that preserves source beats.

### StoryInputPacket Contract

Every enabled bank should resolve to a typed packet before the writer engine
runs:

```
{
  "packet_version": 1,
  "source_bank_id": "science_news|media_archive|public_domain_story",
  "source_mode": "seed|adapt",
  "rights_status": "unknown|public_domain|licensed|operator_supplied",
  "source_title": "",
  "source_author": "",
  "source_url": "",
  "source_hash": "",
  "source_text_ref": "",
  "source_summary": "",
  "casting_brief": "",
  "script_brief": "",
  "close_brief": "",
  "key_terms": [],
  "adaptation_constraints": {}
}
```

Rules:

- Do not keep overloading everything as `meta.news`. Add `meta.source` and keep
  any `meta.news` mirror as a legacy compatibility surface only.
- Archive news about a film is not a public-domain grant for the film.
- Public-domain text must carry source URL/hash/title/author and source-fidelity
  constraints.
- The packet is source-aware; the ledger consumer should stay mostly
  source-agnostic.

### Story Engines

`seeded_fiction_engine`

- Used by `science_news` and probably first-pass `media_archive`.
- Input: `script_brief`, `casting_brief`, `key_terms`, style, budget.
- Output: existing Outline / Beat[].

`science_explainer_engine`

- Same science-news source, different writer contract.
- Output is explanatory audio/story, not drama-first.
- Should be an engine choice, not a new RSS pipeline.

`pd_adaptation_engine`

- Used by `public_domain_story`.
- Input: source text or extracted source beats.
- Output: Outline / Beat[] plus `adaptation_trace`.
- Has a source-fidelity check: major characters, ending, core moral/twist, and
  public-domain attribution must survive.

## Axis B: Visual Style and Visual Model Fork

The visual fork should happen after the story ledger is written. By then we know
the episode title, cast, setting, beat intents, mood, and timing. The visual fork
should rewrite still/video prompts and constrain compatible visual model choices;
it should not rewrite the story ledger.

```
Ledger + Canon
  -> VisualStyleDirector
  -> visual_style_policy_json
  -> ImagePromptGen / ShotLock / render_driver prompt composers
  -> OTR_VideoDirector model policy
  -> image/video engines
```

### VisualStyleProfile Contract

```
{
  "profile_version": 1,
  "style_id": "cinematic_35mm|anime|noir|archival_mono|graphic_novel|custom",
  "prompt_tone": "",
  "style_positive_tail": "",
  "style_negative_tail": "",
  "color_language": "",
  "camera_language": "",
  "character_design_rules": "",
  "background_rules": "",
  "model_family_hints": [],
  "forbidden_terms": [],
  "allow_story_rewrite": false
}
```

Rules:

- The visual profile transforms prompts, not dialogue.
- It must reach both still prompts and video text prompts.
- It must be model-aware enough to avoid asking a model for a style it cannot
  reasonably produce.
- It should be explicit policy JSON, like `video_policy_json`, not hidden globals.
- If absent, the current cinematic/radio look remains byte-identical.

### Visual Fork Examples

`cinematic_35mm`

- Current default family.
- Keeps the restored cinematic tint, film grain, anamorphic/camera language.

`anime`

- Rewrites portrait and scene prompts toward animated character design,
  controlled linework, cel shading, simplified lighting, and anime-compatible
  motion language.
- Must scrub photoreal-only tails from both still and video prompts.

`archival_mono`

- Fits media-archive stories.
- Black-and-white, nitrate/16mm/telecine language, preservation artifacts, dust,
  gate weave, practical lighting.
- Must avoid implying the archival footage itself is licensed unless a visual
  asset bank supplies clean rights.

`graphic_novel`

- Strong composition, panels, ink, shape language.
- Useful for public-domain adaptations where source fidelity matters more than
  realism.

## Proposed Node-Level Shape

Add, eventually:

- `OTR_SourceBankDirector`
  - Emits `story_source_policy_json`.
  - Selects enabled bank and source mode.
  - Owns feed URLs, weights, rights defaults, and source selection seed.

- `OTR_StorySourceInterpreter`
  - Consumes source policy plus fetched/loaded source item.
  - Emits `story_input_packet_json`.
  - Internally dispatches `science_news`, `media_archive`, or `pd_adapt`.

- `OTR_StoryDirector`
  - Consumes `story_input_packet_json`.
  - Chooses `seeded_fiction_engine`, `science_explainer_engine`,
    `pd_adaptation_engine`, or `human_outline`.
  - Emits the existing outline/ledger path, with source metadata stamped.

- `OTR_VisualStyleDirector`
  - Consumes ledger/canon or simple style widgets.
  - Emits `visual_style_policy_json`.
  - Downstream prompt composers apply it.

Do not put all of this directly inside `OTR_LedgerScriptWriter` long-term. The
writer can host the first compatibility bridge, but the architecture wants small
director/interpreter modules with typed JSON contracts.

## Implementation Order

1. Scaffold contracts and tests with default-off behavior.
2. Split `meta.source` from `meta.news` while keeping legacy compatibility.
3. Add `science_news` as a no-op bank that reproduces current behavior.
4. Add `media_archive` as the first real alternate bank: feed swap plus prompt
   wording cleanup from "science story" to "archive/media seed."
5. Add `public_domain_story` only after the source packet and fidelity trace are
   real.
6. Add `VisualStyleDirector` with `cinematic_35mm` as byte-identical default.
7. Wire style policy into still prompts, ShotLock prompt composition, and video
   render-driver text prompts.
8. Add `anime` or `archival_mono` as the first non-default visual style.

## Acceptance Gates

- Default workflow produces the same output path and same semantics when no new
  policy node/input is wired.
- Any code change that adds a node, widget, input, or wire updates
  `workflows/otr_scifi_16gb_full.json` in the same change.
- `science_news` does not regress current RSS/news behavior.
- `media_archive` does not say "science story" in its writer prompts.
- `public_domain_story` records source title, author, URL/hash, rights status,
  and adaptation trace.
- Visual style changes prompt text and model policy only; it does not mutate
  ledger dialogue.
- Stills and video prompts receive the same visual profile, so an anime still
  does not feed a cinematic live-action motion prompt.
- Existing workflow validator, JSON round-trip, widget/input audit, regression
  suite, and Bug Bible pass before any implementation chunk is pushed.

## Kibitz Questions

1. Should `pd_adapt` produce the final Outline directly, or should it produce a
   `StoryBlueprint` that the existing outline assembler validates and stamps?
2. Should `VisualStyleDirector` be independent, or should its style profile be an
   added policy section emitted by `OTR_VideoDirector`?
3. What is the smallest first media-archive experiment: one `archive_seed` bank
   using the current interpreter, or a separate archive interpreter with only
   prompt wording changed?
4. What fields must be source-fidelity-critical for public-domain adaptations:
   ending, protagonist, antagonist, moral/twist, setting, or all of them?

