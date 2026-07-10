# OTR Source Banks v2 - Code-Ready Plan

Date: 2026-07-08
Scope: planning only. No production source or workflow changes made.

Kibitz: R1-R4 completed with Claude Code on Opus and Antigravity on Gemini Pro, Codex as grounded judge. Roundtable was held in reserve and not needed after convergence.

## Current Grounding

- `OTR_LedgerScriptWriter` is the execution surface. Keep it that way.
- Current canonical workflow node 1 has 25 `widgets_values` entries:
  - `story_scaffold`: zero-based slot 22
  - `source_bank`: zero-based slot 23
  - `visual_style`: zero-based slot 24
- Some writer comments say slot 25/26. Treat those as stale prose. The canonical workflow JSON and guardrail tests are authoritative.
- `source_bank` values come from `nodes/story_packs/banks.json` through `_otr_story_routing.list_bank_ids()`.
- Runnable banks today: `science_news`, `media_archive`.
- Non-runnable banks today: `public_domain_story`, `custom_source_bank`.
- `public_domain_story` currently has only `line_composer_system` and `coda_system`; it cannot run until the full writer seam/rules/fetch/interpreter lane exists.
- `SOURCE_PAYLOAD_KEYS` is exact: `headline`, `summary`, `full_text`, `source`, `date`, `link`, `seed_text`.
- Interpreter results must pass `validate_interpreter_result`: `model_dump()`, `casting_brief`, `script_brief`, `news_close_brief`, `key_terms`, `attempts`.
- `get_pipeline()` already exists in `_otr_story_routing.py`; use it for original-radio branching.

## External Source Constraints

- Project Gutenberg: use official catalog/harvest/mirror or pre-populated cache. Do not crawl `www.gutenberg.org` live. Preserve attribution/rights sidecar and strip boilerplate from prompt text.
  - https://www.gutenberg.org/ebooks/offline_catalogs.html
  - https://www.gutenberg.org/policy/robot_access.html
  - https://www.gutenberg.org/policy/license.html
- Standard Ebooks: curated supplement only when metadata supports U.S. public-domain/CC0 status.
  - https://standardebooks.org/about
  - https://standardebooks.org/feeds
- Folger: useful Shakespeare text source, but Folger Digital Texts are CC BY-NC 3.0. Stamp noncommercial rights; do not represent Folger-derived episodes as commercially releasable without permission.
  - https://www.folger.edu/explore/shakespeares-works/download/
  - https://www.folger.edu/copyright-policy/
  - https://www.folgerdigitaltexts.org/api
- Internet Archive: out of v1 runnable scope because rights/OCR evidence is uneven.

## Source Ref Widget

Append one optional string widget named `source_ref` after `visual_style` in the commit that first needs import-bank selection.

Required same-commit edits:

- `nodes/OTR_LedgerScriptWriter.py`
  - append `source_ref` in `INPUT_TYPES()["optional"]`
  - add `source_ref=""` to `run()`
  - add `source_ref=""` to `_resolve_inputs()`
  - carry `source_ref` in the resolved dict
  - ensure refine re-entry capture sees it because it is in the `run()` signature
- `workflows/otr_canonical.json`
  - append `""` to node 1 `widgets_values`
  - vector length becomes 26
  - source_bank stays at 23; visual_style stays at 24; source_ref lands at 25
  - add widget-backed input row if ComfyUI serializes it
- `nodes/_otr_workflow_apply.py` and `scripts/otr_api.py`
  - add `source_ref` to both `CREATIVE_WHITELIST` sets
- Tests
  - update existing expected writer widget length from 25 to 26
  - source_ref is last optional key
  - `widgets_values[25] == ""`
  - whitelist parity still holds
  - `patch_widget_by_name` reaches source_ref by name

Blank behavior:

- `science_news`, `media_archive`, `original_radio`: ignored.
- `public_domain_story`, `shakespeare`: use `bank.defaults.source_ref` if present; otherwise fail loud with a clear source-ref error.
- A bank must not become runnable until it has a tested default source_ref or a tested fail-loud blank behavior.

## Fetch Result Contract

Do not add source metadata keys to the exact payload.

Add to `nodes/_otr_source_payload.py`:

```python
@dataclass(frozen=True)
class SourceFetchResult:
    payload: dict
    source_meta: dict | None = None
    source_rights: dict | None = None
```

Add:

```python
def normalize_fetch_result(result, origin: str) -> tuple[dict, dict, dict]:
    ...
```

Rules:

- If `result` is a raw `dict`, validate it with `validate_source_payload(result, origin)` and return empty sidecars.
- If `result` is `SourceFetchResult`, validate `result.payload` with `validate_source_payload`.
- Normalize `None` sidecars to `{}`.
- Shallow-copy sidecars before returning.
- Unknown result type raises `SourcePayloadContractError`.
- `_resolve_inputs` must pass an explicit origin string.

Fetcher callable contract becomes:

```python
fetch(*, bank, technical_model: str, source_ref: str = "") -> dict | SourceFetchResult
```

Update existing wrappers in the same commit:

- `_fetch_science_rss(..., source_ref: str = "")`; ignore it.
- `_fetch_media_archive_rss(..., source_ref: str = "")`; ignore it.

`_resolve_inputs` fetch path:

```python
raw_fetch = _fetch_entry.fetch(
    bank=_fetch_bank,
    technical_model=technical_model,
    source_ref=source_ref,
)
news_article, source_meta, source_rights = normalize_fetch_result(
    raw_fetch,
    origin=f"_resolve_inputs fetch (bank={_fetch_bank.source_bank_id!r}, fetcher={_fetch_bank.fetcher!r})",
)
```

Every `_resolve_inputs` branch, including `custom_premise` and `original_radio`, must return `source_ref`, `source_meta`, and `source_rights`.

## Exact Payload Mappings

Public-domain payload:

- `headline`: `<title> - <unit label>`
- `summary`: required manifest/unit synopsis; fail loud if absent in v1
- `full_text`: selected unit text only
- `source`: edition/source label, e.g. `Project Gutenberg` or `Standard Ebooks`
- `date`: publication year string or empty
- `link`: source URL
- `seed_text`: title, author, unit, synopsis, and bounded excerpt

Shakespeare payload:

- `headline`: `<Play>, Act <n>, Scene <m>`
- `summary`: required curated scene synopsis
- `full_text`: scene text with speaker boundaries
- `source`: `Folger Shakespeare`
- `date`: approximate play year plus `Folger Digital Texts`
- `link`: Folger URL
- `seed_text`: play, act/scene, synopsis, speakers, and bounded excerpt

Sidecars:

- `meta["source_ref"]`
- `meta["source_meta"]`
- `meta["source_rights"]`
- `meta["adaptation_trace"]`

## Registry IDs

Do not reference ids in `banks.json` until Python registration exists.

New ids:

- fetcher `public_domain_source`
- interpreter `public_domain_interpreter`
- fetcher `shakespeare_folger`
- interpreter `shakespeare_interpreter`

Intermediate commits may register Python ids while `banks.json` still has empty ids. The commit that writes a non-empty fetcher/interpreter id must already contain the callable.

## Cache And Rights

Add a helper in the source-bank modules:

```python
def source_bank_cache_root() -> Path:
    ...
```

Resolution:

1. `OTR_SOURCE_BANK_CACHE_DIR`
2. `_otr_paths.otr_shared_cache_dir() / "source_banks"`

Write policy:

- Write a temp sibling file.
- Use `os.replace(temp, final)` for atomic same-volume replacement.
- Do not use `shutil.move` for finalization.
- No eviction in v1; warn later if cache grows too large.
- New modules stay stdlib-only at top level. No torch/transformers/pydantic/heavy imports at import time.

## Public Domain Bank

Implementation files:

- `config/source_banks/public_domain_manifest_schema.json`
- `config/source_banks/public_domain_story/manifest.sample.json`
- `nodes/_otr_public_domain_sources.py`
- `nodes/story_packs/public_domain_story/faithful_radio_adaptation.json`
- `nodes/story_rules/public_domain_story.json`
- `nodes/_otr_source_payload.py`
- `nodes/story_packs/banks.json`

Keep public-domain fetcher and interpreter in `nodes/_otr_public_domain_sources.py` for v1; split later only if the file becomes unwieldy.

Runnable `banks.json` row:

- `source_bank_id`: `public_domain_story`
- `fetcher`: `public_domain_source`
- `interpreter`: `public_domain_interpreter`
- `default_story_pipeline`: `legacy_many_pass`
- `defaults.source_ref`: required
- `required_seams`: the exact nine production seams below
- `runnable`: `true`

Production seams:

1. `outline_macro_system`
2. `outline_phase_system`
3. `outline_beat_system`
4. `line_composer_system`
5. `exchange_system`
6. `coda_system`
7. `announcer_intro_system`
8. `announcer_intro_safe_system`
9. `announcer_outro_system`

This deliberately expands the current non-runnable public-domain row from two seams to the full runnable set. Existing `science_news` and `media_archive` required_seams remain unchanged for this sprint; adding `exchange_system` to them is a housekeeping follow-up, not a source-bank v2 gate.

Interpreter:

- Use the technical slot.
- Return an object that passes `validate_interpreter_result`.
- Keep `adaptation_trace` in sidecar metadata, not in `model_dump()`.
- Preserve selected unit characters, conflict, turns, and ending.
- Fail loud if the unit cannot produce a coherent compact adaptation.

## Shakespeare Bank

Implementation files:

- `config/source_banks/shakespeare/curated_scenes.json`
- `nodes/_otr_shakespeare_sources.py`
- `nodes/story_packs/shakespeare/folger_scene_adaptation.json`
- `nodes/story_rules/shakespeare.json`
- `nodes/_otr_source_payload.py`
- `nodes/story_packs/banks.json`

Keep Shakespeare fetcher/interpreter/parser in `nodes/_otr_shakespeare_sources.py` for v1.

V1 scope: curated scene units only.

Runnable `banks.json` row:

- `source_bank_id`: `shakespeare`
- `fetcher`: `shakespeare_folger`
- `interpreter`: `shakespeare_interpreter`
- `default_story_model`: `folger_scene_adaptation`
- `default_story_pipeline`: `legacy_many_pass`
- `defaults.source_ref`: required
- `required_seams`: same exact nine production seams
- `runnable`: `true`

Parser:

```python
parse_folger_scene(xml_text: str, play_code: str, act: int, scene: int) -> FolgerScene
```

Use `xml.etree.ElementTree`, handle TEI namespaces, preserve speaker order, preserve stage directions structurally, and convert stage directions to audible radio implications during interpretation.

## Original Radio Bank

Implementation files:

- `nodes/_otr_original_radio.py`
- `nodes/story_packs/original_radio/original_radio_drama.json`
- `nodes/story_rules/original_radio.json`
- `nodes/story_packs/pipelines.json`
- `nodes/story_packs/banks.json`
- `nodes/OTR_LedgerScriptWriter.py`

`original_multi_pass`:

- `executable`: `true`
- `requires_source_contract`: `false`
- `declared_seams`: `original_concept_system`, `original_select_system`, `original_brief_system`, `original_qa_system`
- `passes`: concept fanout, candidate select, brief build, QA

Original pack prompt stages:

- nine production seams
- four original-specific seams

`banks.json` row:

- `source_bank_id`: `original_radio`
- `fetcher`: `""`
- `interpreter`: `""`
- `default_story_model`: `original_radio_drama`
- `default_story_pipeline`: `original_multi_pass`
- `required_seams`: nine production seams plus four original seams
- `runnable`: `true` only after writer branch and tests exist

`_resolve_inputs` original branch:

- Branch before `resolve_fetcher`.
- Return exact compatibility placeholder payload:
  - `headline`: `Original radio drama`
  - `summary`: `Generated from structural constraints only.`
  - `full_text`: ``
  - `source`: `original_generated`
  - `date`: ``
  - `link`: ``
  - `seed_text`: `Original radio drama generated from structural constraints only.`
- Return `seed_source="original_generated"`, `source_meta={}`, `source_rights={}`, and carried `source_ref`.

`run()` branch:

```python
pipe = _otr_story_routing.get_pipeline(_source_bank_row.default_story_pipeline)

if pipe.requires_source_contract:
    _interp = _otr_source_payload.resolve_interpreter(_source_bank_row)
    briefs = _interp(...)
else:
    if _source_bank_row.source_bank_id != "original_radio":
        raise StoryRoutingError(...)
    with slot_scheduler.helper_context("build_original_briefs"):
        briefs = _otr_original_radio.build_original_briefs(
            technical_fn=technical_generate_fn,
            technical_model=str(resolved["technical_model"]),
            target_words=resolved["target_words"],
            num_characters=resolved["num_characters"],
            visual_style_id=resolved["visual_style"],
            source_ref=resolved["source_ref"],
            max_attempts=3,
        )
```

`OriginalRadioBriefs.model_dump()` includes only compatibility fields:

- `casting_brief`
- `script_brief`
- `news_close_brief` as a string
- `key_terms`
- `attempts`

`briefs.provenance` is written explicitly to `meta["source_meta"]`.

QA:

- max three attempts
- no source attribution
- no RSS/news/Gutenberg/Folger claims
- no franchise wording or modern-story references
- no fixed seed deck leakage
- failure raises `OriginalRadioGenerationError`

## LLM Creative Visual Style

Implementation files:

- `nodes/visual_styles/llm_creative.json`
- `nodes/_otr_visual_styles.py`
- `nodes/_otr_visual_style_creative.py`

Sentinel pack:

- complete valid v2 style pack
- safe generic defaults, no placeholder text
- appears in dropdown and passes early `resolve_visual_style`

Add to `_otr_visual_styles.py`:

```python
def validate_runtime_visual_style(raw: dict, origin: str) -> VisualStyle:
    """Validate a merged runtime dict against the same v2 style schema without requiring a disk-registered id."""
```

Implementation note: call the existing row validator with a path whose stem is `llm_creative`, such as `Path("llm_creative.json")`, so the `style_id == path.stem` check remains valid.

Runtime generation:

- helper: `_otr_visual_style_creative.build_generated_visual_style(...)`
- slot: technical
- input context: source_bank, headline/title, script_brief, key_terms, target_words, source/unit mood
- output: small JSON patch over the sentinel raw pack, not a full pack from scratch
- merge patch with sentinel raw dict
- validate merged dict
- store raw merged dict under `meta["generated_visual_style"]`

Invocation point:

- after source/original briefs exist
- before any final ledger used by visual nodes is written
- before downstream render nodes call `get_visual_style(meta)`

`get_visual_style(meta)`:

- concrete style: current behavior
- `llm_creative` plus generated style: validate and return generated style
- `llm_creative` without generated style: return sentinel only for pre-generation/default compatibility

Failure:

- two repair attempts
- then `VisualStyleGenerationError`
- no silent sentinel fallback after a failed explicit creative-style request

## Metadata Compatibility

Keep `meta["news"]` for v1:

- `casting_brief`
- `script_brief`
- `news_close_brief`
- `key_terms`
- `attempts`

The name is semantically wrong for PD/Shakespeare/original, but downstream code reads it today. Rename to `meta["source_briefs"]` only in a later migration.

Sidecars:

- `meta["source_ref"]`
- `meta["source_meta"]`
- `meta["source_rights"]`
- `meta["adaptation_trace"]`
- `meta["generated_visual_style"]`

## Tests

Add:

- `tests/test_source_ref_widget.py`
- `tests/test_public_domain_sources.py`
- `tests/test_public_domain_interpreter.py`
- `tests/test_shakespeare_sources.py`
- `tests/test_shakespeare_interpreter.py`
- `tests/test_original_radio_pipeline.py`
- `tests/test_visual_style_llm_creative.py`
- `tests/test_source_bank_registry_v2.py`

Extend:

- `tests/test_workflow_apply.py`
- `tests/test_workflow_json_guardrails.py`
- `tests/test_source_payload_chunk3.py`
- `tests/test_source_bank_widget_2c.py`
- `tests/test_visual_style_widget_3c.py`

Must-cover:

- legacy dict through `normalize_fetch_result` is validated and returns empty sidecars
- `SourceFetchResult` sidecars normalize `None` to `{}`
- exact seven-key payload for PD/Shakespeare
- source_ref propagates to new fetchers and is ignored by old fetchers
- widget vector length becomes 26
- no dangling registry ids
- original_radio never calls resolve_fetcher/resolve_interpreter
- original provenance reaches `meta["source_meta"]`
- story rules files exist before runnable flip
- all nine production seams exist before PD/Shakespeare runnable flip
- llm_creative sentinel validates as a full v2 pack
- runtime generated visual style validates and never writes disk packs
- workflow validator, JSON round-trip, widget count audit, input-name audit, and link integrity after workflow edit

## First Green Chunk

Ship public-domain manifest/cache skeleton only:

1. `config/source_banks/public_domain_manifest_schema.json`
2. `config/source_banks/public_domain_story/manifest.sample.json`
3. `nodes/_otr_public_domain_sources.py` with manifest validation, cache root resolver, atomic cache helper, text canonicalizer, and fixture-only metadata parsing
4. `tests/test_public_domain_sources.py`

Do not update `banks.json` ids. Do not flip runnable. Do not edit workflow in this chunk.

## Commit Order

1. Public-domain manifest/cache skeleton, non-runnable, no `banks.json` id references.
2. Stale slot comment cleanup, then append `source_ref` widget/workflow/API surface and guardrails.
3. SourceFetchResult/normalizer and fetcher signature update; prove science/media parity.
4. Public-domain fetcher registration and exact payload tests; keep bank non-runnable until interpreter id is registered.
5. Public-domain interpreter, full pack seams, story rules, `banks.json` ids/default, then flip runnable.
6. Shakespeare parser/cache fixtures, non-runnable.
7. Shakespeare interpreter, pack seams, story rules, `banks.json` ids/default, then flip runnable.
8. `llm_creative` sentinel/resolver/generator.
9. `original_multi_pass` metadata and original helper branch, non-runnable.
10. Flip `original_radio.runnable=true` after no-source QA and 30-word smoke.

## Verify At Build

- Count node 1 `widgets_values` immediately before the source_ref commit; if another widget landed first, update the planned index.
- After any workflow edit: `OTR_WorkflowValidator`, JSON round-trip, widget-count vs live `INPUT_TYPES`, input-name audit, and link referential integrity.
- After commit 2: run source_bank/visual_style/source_ref widget tests and whitelist parity.
- After commit 3: run `tests/test_source_payload_chunk3.py` plus science/media fetcher parity.
- After each runnable flip: run story pack/rules registry tests and focused bank tests.
- Before any green commit with code: run focused regression, full suite when practical, and Bug Bible per repo rules.

## Open Risks

- Folger is noncommercial unless permission is obtained.
- Gutenberg status is U.S.-centric and trademark-sensitive.
- Cache eviction is deferred.
- Internet Archive is intentionally out of v1.
