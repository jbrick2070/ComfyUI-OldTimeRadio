# R2 Final: Coding Plan Convergence

Status: Codex-grounded R2 against the live Python prompt sites.

## Verdict

Buildable as a staged prompt/Python update, provided the first coding chunk
stays upstream/lab plus pure modules and leakage tests.

## Concrete Build Shape

1. Add story-pack contracts:
   - `source_bank_id`
   - `story_model_id`
   - `story_pipeline_id`
   - stage prompt bodies
   - examples
   - rubric text
   - forbidden leakage terms
   - coda/source-note rules
   - ledger validation expectations

2. Add source brains:
   - science keeps current science/news path
   - media archive/RSS gets its own feed normalizer/interpreter
   - public domain gets a manifest folder loader
   - custom source bank validates schema or raises

3. Update prompt sites to read selected pack:
   - `nodes/news_interpreter.py` stays science-only or becomes source-profiled
     behind a new facade.
   - `nodes/_otr_outline.py`
   - `nodes/_otr_pitch_room.py`
   - `nodes/_otr_story_select.py`
   - `nodes/_otr_dramatic_state_llm.py`
   - `nodes/_otr_line_composer.py`
   - `nodes/_otr_style_picker.py`
   - `nodes/OTR_LedgerScriptWriter.py`

4. Add compatibility mirrors, not fallbacks:
   - existing `meta.news` keys may continue to exist for downstream consumers
     while carrying source-neutral values.
   - mirror naming does not authorize science/news prompts under archive or
     public-domain lanes.

5. Add tests before workflow wiring:
   - pack validation
   - prompt preview leakage
   - resolver no-fallback checks
   - simple 4-prompt experimental ledger validation

## Must Fix

- `_resolve_inputs()` must branch on `source_bank`; only `science_news` can call
  `_fetch_rss_seed_or_die`.
- `_otr_style_picker.py` must accept prompt overrides or be bypassed by
  non-science story packs.
- `compose_news_coda()` should become a profile-backed `compose_source_coda()`
  facade while preserving current science behavior.
- New widgets, when transplanted, must append only.
