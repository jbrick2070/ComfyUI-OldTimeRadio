# R1-R4 Rewrite - JSON Content, Python Behavior

Core law:

```text
JSON owns content and configuration.
Python owns validation, routing, execution, and fail-loud errors.
```

This replaces the fuzzier architecture discussion. The upstream rewrite should
be judged by whether it keeps prompt/story/visual content in editable JSON packs
while Python stays a strict loader, validator, router, and executor.

## R1 - Architecture Rule

The system should treat source/story/visual material as data, not hardcoded
behavior.

JSON owns:

- story prompt stage text
- source-bank examples and manifests
- story model tone rules
- forbidden leakage terms
- visual style prompt tails
- visual forbidden terms
- visual motion prompt language
- default source/story/style configuration when it is pure configuration

Python owns:

- schema contracts
- JSON loading
- source-bank routing
- story-model routing
- visual-style routing
- default resolution
- ledger-writing spec construction
- validation and clear errors
- execution of story/prompt passes

Rule of thumb:

If changing a tone, prompt, style, source pack, forbidden term, or example needs
a Python edit, the design is drifting wrong. If changing the schema or execution
flow needs only JSON edits, the design is also drifting wrong.

## R2 - Coding Plan

Keep the current standalone lab shape and make it stricter.

Content/config files:

- `fixtures/story_packs/**/*.json`
- `fixtures/visual_styles/*.json`
- `fixtures/source_packets/*.json`
- `fixtures/public_domain_sources/**/manifest.json`
- future custom source-bank schema JSON files

Python behavior files:

- `contracts.py`: Pydantic models and consistency validators
- `registry.py`: id lookup, default resolution, source/style registration
  (replaced `catalogs.py`, deleted at 7df7c80)
- `profiles.py`: pack -> resolved `StoryPromptProfile` merger; validates
  required labels + line_grounding + coda_mode; never invents prose
- `bridge.py`: `build_spec()` -> `LedgerWritingSpec` + `emit_bridge_artifact()`
  writes the frozen JSON production consumes (translator-head)
- `preview.py`: source packet -> interpreted story input -> ledger-writing spec
- `nodes.py`: ComfyUI preview/validation nodes and fail-loud UI surface
- `scripts/validate_lab.py`: command-line validation
- `tests/*.py`: automated proof that JSON content routes correctly

Concrete coding rules:

- No prompt meat should be buried inside Python unless it is genuinely generated
  by algorithm.
- No hidden fallback from media archive or public domain back to sci-fi.
- Unknown source bank, story model, story pipeline, or visual style must error.
- New JSON style packs should appear through the loader without editing node UI
  code.
- New JSON story packs should validate through the same `StoryPack` contract.
- The experimental 4-pass pipeline may be JSON-described, but Python must own
  pass sequencing, pass status, and pass failure reporting.

## R3 - Wiring And Transplant Plan

Before touching production OTR, the standalone lab should emit a bridge artifact
that production can consume.

Bridge artifact should include:

- `source_bank_id`
- `story_model_id`
- `story_pipeline_id`
- `visual_style_id`
- validated `source_material`
- validated `story_input`
- validated `prompt_profile`
- validated `visual_policy`
- compatibility mirror for current `meta.news` consumers
- explicit error if any required content/config JSON is missing or invalid

Transplant rule:

Production code should not ask, “Is this sci-fi, media archive, or public
domain?” in scattered conditionals. It should consume the validated spec and
route by declared ids and contracts.

Production edits should focus on:

- replacing hardcoded sci-fi/news prompt strings with profile fields
- replacing hardcoded cinematic/radio visual tails with visual policy fields
- proving `meta.news` compatibility against the real downstream ledger consumer
- adding any new widgets only at the end of production node widget lists
- updating the canonical workflow JSON only after code and validation are green

## R4 - Convergence Gates

The rewrite is ready to transplant only when these are true:

- JSON story packs contain the actual media archive and public-domain prompt
  content.
- JSON visual styles contain the actual archive/anime/cartoon/origami visual
  content.
- Python validates every JSON pack through strict contracts.
- Python can build a ledger-writing spec for science news, media archive, and
  public domain.
- Media archive and public domain default to non-sci-fi visual policy unless
  explicitly overridden.
- The experimental 4-pass path reports the exact failing pass.
- Tests prove media archive/public-domain do not silently fall back to sci-fi.
- Tests prove forbidden sci-fi/news terms do not leak into non-sci-fi prompt
  previews.
- Production `meta.news` compatibility is verified against the actual consumer
  code, not guessed.
- `otr_scifi_16gb_full.json` remains untouched until the transplant chunk.

## One-Sentence Version

Put the creative material in JSON, make Python enforce the contract, and let
production consume one validated ledger-writing spec instead of inheriting more
hidden sci-fi conditionals.
