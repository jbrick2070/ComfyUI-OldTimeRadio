# Upstream Story Lab Code-Ready Brief

Date: 2026-07-01

Scope for this review: only the standalone upstream story architecture in
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab`.
Do not review the full production OldTimeRadio workflow transplant yet.

Current state:

- The lab now lives as its own sibling ComfyUI custom node folder, not nested
  inside `ComfyUI-OldTimeRadio`.
- It registers two live nodes:
  - `OTR_UpstreamStoryLabValidator`
  - `OTR_StoryPackPreview`
- The production workflow remains untouched:
  `ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json`.
- The lab has 12 fixture story packs and validates from its own folder with
  `scripts\validate_lab.py`.
- The live nodes hash `nodes.py`, `src\`, and `fixtures\` through `IS_CHANGED`
  so ComfyUI does not serve stale previews after fixture edits.

Architecture intent:

```text
source_bank -> source material -> story_model -> story_pipeline -> story pack
story pack -> ledger_writing_spec -> existing production ledger contract
visual_style -> visual policy -> ledger visual directives
```

Core product idea:

- `science_news` remains the current science/RSS sci-fi radio lane.
- `media_archive` is a separate media RSS/archive lane with distinct story
  models such as restoration adventure, cinematic humorous, happy archive
  mystery, gentle thriller, and broadcast history comedy.
- `public_domain_story` is a separate source-folder lane for books, chapters,
  comics, plays, stories, and similar public-domain source material.
- `custom_source_bank` is exposed as `+ Add Your Own`; it is not a fallback.
  It should point users to a schema guide and fail loudly until a valid schema,
  source packet, and story pack exist.
- `visual_style` is an equal partner with story/source model selection:
  examples include Sci-Fi Radio, Media Archive, Anime, Cartoon, Paper Origami.
  Visual style can influence ledger directives, still prompts, and video prompt
  language, but should not be hidden inside one source lane.

Hard invariants:

- No hidden fallbacks.
- Unknown source banks, story models, pipelines, or visual styles should fail
  loudly.
- Separate story lanes should stay independent enough that future experiments
  can change one lane's LLM scaffold without hurting the others.
- First implementation may clone the current many-pass story scaffold, but the
  cloned lanes should be structurally separable.
- The experimental `simple_4_prompt_experimental` pipeline must stay visible:
  pass 1 creative story, pass 2 ledger fill, pass 3 technical schema cleanup,
  pass 4 technical ledger audit, with optional adaptive cleanup under a hard
  deterministic cap.
- The lab may preview and validate. It must not edit the production workflow
  until a later explicit transplant chunk.

Files to inspect:

- `__init__.py`
- `nodes.py`
- `src\upstream_story_lab\contracts.py`
- `src\upstream_story_lab\catalogs.py`
- `src\upstream_story_lab\preview.py`
- `scripts\validate_lab.py`
- `fixtures\story_packs\**\*.json`
- `fixtures\source_packets\*.json`
- `fixtures\public_domain_sources\**\*`
- `fixtures\visual_styles\*.json`
- `CUSTOM_SOURCE_BANK_GUIDE.md`
- `PROMPT_SURGERY_CHECKLIST.md`
- `TRANSPLANT_MANIFEST.md`

Validation already run:

```text
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\validate_lab.py
OK upstream_story_lab
science story_model=science_news_default
archive story_model=media_restoration_adventure
archive visual_style=archival_documentary
story_packs=12
```

Import smoke already run:

```text
loaded nodes: ['OTR_StoryPackPreview', 'OTR_UpstreamStoryLabValidator']
OK Upstream Story Lab live custom node
lab_root=C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OTR-UpstreamStoryLab
story_packs=12
production_workflow=not touched by this lab node
```

Review questions:

- Is the source-bank/story-model/story-pipeline/visual-style separation clear
  enough to transplant without turning into hidden conditionals?
- Are the contracts strict enough for media archive and public-domain source
  material, including books, comics, plays, and custom source folders?
- Is the live ComfyUI wrapper valid and code-ready as a standalone custom-node
  package?
- Does the no-fallback policy show up as actual hard errors, not documentation
  wishes?
- What is the smallest code-ready next step before writing real ledger-filling
  prompt meat for media archive and public-domain lanes?
