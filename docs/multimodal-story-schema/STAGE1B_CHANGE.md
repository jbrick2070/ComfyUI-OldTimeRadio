# Stage 1b change (APPLIED, under review) -- wire first live consumer

This describes a change ALREADY APPLIED to the working tree (review it against the
real code, not as a plan-to-build). Goal: the sci-fi lane now sources one creative
prompt seam from the JSON story pack instead of a Python constant, BYTE-IDENTICAL,
with zero episode-output change.

## What changed
1. `nodes/_otr_creative_prompt_router.py`
   - New import: `from ._otr_story_pack import get_pack_prompt, load_pack`.
   - New module constants `_SCIENCE_PACK_PATH` (nodes/story_packs/science_news/
     science_news_default.json) and `_PHASE_TO_PACK_SEAM = {"line_composer_system":
     "line_composer_system"}`.
   - In `resolve_creative_system_prompt`: the period branch (otr_1940s_v1 ->
     OTR_PERIOD_SYSTEM_PROMPT) is checked FIRST (unchanged). Then, if the phase is
     in `_PHASE_TO_PACK_SEAM`, it returns `get_pack_prompt(load_pack(_SCIENCE_PACK_PATH),
     seam)`. Otherwise it returns `_MODERN_BY_PHASE[phase]` (unchanged). Only
     `line_composer_system` is migrated; `outline` still returns its constant object.
2. Tests migrated from object-identity to value-equality (`is` -> `==`):
   `tests/test_creative_prompt_router.py` (2 sites) + `tests/test_audio_c7_clamp_counter.py`.
3. `tests/test_story_pack_stage1.py`: the Stage 1 dormancy guard became a
   sanctioned-consumer guard (allows `_otr_creative_prompt_router.py` only); added a
   Stage 1b equivalence test (router returns pack value == `_otr_line_composer._SYSTEM_PROMPT`;
   outline still `is` its constant).

## The invariants this must not break
- BYTE-IDENTITY: pack["line_composer_system"] == _otr_line_composer._SYSTEM_PROMPT
  (so LLM prompt + audio unchanged). test_audio_byte_identical stays green.
- The outline sentinel `_otr_outline.py:1847 resolved is _SYSTEM_PROMPT` must still
  hold (outline phase is NOT pack-sourced).
- No hidden fallback: a missing/empty pack seam RAISES (get_pack_prompt), never
  silently swaps.
- No new caller of resolve_creative_system_prompt (router-internal change).
- Period profile still wins over the pack.

## Review question
Any build-breaker, behavior change, hidden object-identity dependency on the
line_composer_system prompt, or missed test that this change violates? SHIP / NO-SHIP.
