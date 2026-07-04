# R2 Codex Anchor Review

- VERDICT: yes-with-fixes. R1 gives a buildable direction, but R2 must nail
  strict data shapes, registry ownership, and no-fallback semantics before any
  code is safe to write.

## MUST-FIX BEFORE BUILD

1. [R2 Coding Targets / Contract modules] CONFIRMED: define exact Pydantic v2
   fields, types, defaults, and invariants. The local venv has Pydantic 2.12.5,
   so use `ConfigDict(extra="forbid")`, typed fields, and
   `Field(default_factory=...)`. The plan cannot leave `VisualStylePolicy` as
   field names only.

2. [Must-Fix Architecture Decisions / story_model] CONFIRMED: do not reuse the
   existing `nodes/_otr_style_catalog.py` as the media-archive story-model set
   without source scoping. That file is an existing narrative grammar catalog
   and includes sci-fi/emergency shapes like deep-space distress and lab
   containment. The fix is either:
   - source-bank-specific story model catalogs, or
   - one `StoryModelCatalog` keyed by `(source_bank_id, story_model_id)`.
   Media archive must start with restoration/adventure/humor/upbeat/gentle
   thriller models, not the current sci-fi anthology catalog.

3. [Prompt surgery artifact] CONFIRMED by user requirement: the reason for the
   audit artifact is to let us clone the useful story models/machinery and then
   surgically edit prompt variables, tone guardrails, and forbidden patterns.
   R2 should not prescribe a rewrite of every story module. It should identify
   which prompts are cloned/shared, which prompt variables are replaced, and
   which phrases become source-bank/story-model forbidden patterns.

4. [Runtime registries/factories] CONFIRMED: each registry must fail closed.
   Required signatures:
   - `get_source_brain(source_bank_id: str) -> SourceBrain`
   - `get_profile(source_bank_id: str, story_model_id: str) -> StoryPromptProfile`
   - `get_visual_style_policy(style_id: str) -> VisualStylePolicy`
   Unknown ids must raise typed errors, not return science/default.

5. [Adapter] CONFIRMED: `_resolve_inputs()` currently calls
   `_fetch_rss_seed_or_die()` when `custom_premise` is empty. The first
   implementation step that touches writer runtime must pass `source_bank` into
   resolution before any fetch. For pure pre-transplant tests, build a separate
   resolver helper and do not wire it into `run()` until the transplant chunk.

6. [Prompt-profile integration plan] CONFIRMED: `_otr_style_picker.py` is an
   active prompt surface with "sci-fi radio drama" text. R2 must include it in
   the profile migration or explicitly gate it to `science_news` only. Leaving
   it out breaks the "no Star Trek/Amazing Stories drift" requirement.

7. [Visual-style integration plan] CONFIRMED: `finish_visual_prompt()` and
   `compose_still_prompt()` hardcode visual tails today. R2 needs exact helper
   APIs:
   - `parse_visual_style_policy_json(raw: str, *, required: bool) -> VisualStylePolicy`
   - `stamp_visual_style(meta_or_ledger: dict, policy: VisualStylePolicy) -> dict`
   - `finish_visual_prompt(..., visual_policy: VisualStylePolicy | None = None, ...)`
   Bad JSON must fail when `required=True`; legacy unwired callers may pass
   `required=False` and preserve old behavior.

8. [No-fallback tests] CONFIRMED: tests must check negative output, not just
   registry success. Add assertions that media-archive prompts do not include
   "science-fiction", "real science", "news facts", "Star Trek", "spaceship",
   "mission control", or "lab containment" unless the selected media-archive
   story model explicitly allows them.

## SHOULD-FIX

1. [Contract modules] Define `StoryModelSpec` instead of stuffing all tone data
   into `StoryInputPacket`:
   - `story_model_id: str`
   - `label: str`
   - `tone_guardrails: list[str]`
   - `forbidden_plot_patterns: list[str]`
   - `outline_rules_extra: str`
   The packet references the model id; the prompt profile expands it.

2. [Public domain] Treat `public_domain_story` as reserved until a fixture text
   adapter exists. A registry entry may raise `NotImplementedError`, but the
   workflow dropdown should not expose it during the upstream-only phase.

3. [Visual style id] Use `archival_documentary` as the visual style id and
   "Media Archive" as a display label. This avoids id collision with
   `source_bank="media_archive"`.

4. [Compat mirror] Implement one pure helper:
   `build_legacy_news_mirror(packet: StoryInputPacket) -> dict`.
   Do not let callers hand-assemble `meta.news` differently.

5. [Module names] Avoid near-collision with existing `_otr_style_catalog.py`:
   - narrative/story models: `_otr_story_model_catalog.py`
   - visual policies: `_otr_visual_style_catalog.py`
   Add docstrings that state the difference.

6. [Tests] Add pure import tests for every new module to ensure no torch,
   ComfyUI, network, or model imports at module load.

## OPTIONAL / NICE-TO-HAVE

1. A small fixture directory under `tests/fixtures/source_banks/` with one
   science_news packet, one media_archive packet, and one public-domain reserved
   fixture.

2. A simple prompt-rendering debug function that returns the resolved profile
   variables without making an LLM call.

## CUT THESE

1. Cut live RSS/archive fetching from the first pure-module chunk. Use fixture
   packets until the contracts and prompt profiles are proven.

2. Cut workflow registration of `OTR_VisualStyleDirector` from R2. Plan it, but
   implement only when R3 transplant sequencing is ready.

3. Cut any plan to rename `meta.news.news_close_brief` in the production ledger
   now. Keep it as compatibility mirror only.
