# motion_clause spec -- roundtable pass01 judgment (grounded by Claude)

Panel: GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro. Strong convergence (all 4
"no" with the SAME core gaps). Grounded against render_driver.py + _otr_story_brief_helpers.py.

## CONFIRMED -> folded into the hardened spec
1. **Clause budget was wrong** (all 4). `finish_visual_prompt` applies max_chars to the
   WHOLE finished string (base + brief<=90 + era/style tails). A 220-char clause blows it.
   FIX: hard clause cap **~70 chars**, validated IN the generation phase; then
   finish_visual_prompt budgets the whole prompt unchanged.
2. **Dialogue must be in the cache key** (all 4 + operator). `source_hash` over
   (character, beat_id, action_summary) ignores dialogue -> stale clause when the line
   changes. FIX: hash = canonical(char_id, beat_id, normalized dialogue text,
   schema_version). DROP `action_summary` (never sourced anywhere -- Grok/DeepSeek).
3. **Subject name source** (GPT, DeepSeek). The brief prose has no guaranteed name.
   GROUNDED: shots carry `char_id` (render_driver.py:669 `shot.get("char_id")`); resolve
   the display name via the cast/character entry. Pass it as an explicit input, not via
   `get_story_brief_ltx` (which is scene context only).
4. **Dialogue field source** (Grok, DeepSeek). GROUNDED: per-beat line lives in
   `ledger['lines']` keyed by `line_id` (== beat_id == shot.source_line_ids[0]); joined per
   shot by `_line_index(ledger).get(_beat_id_for_shot(shot))` (render_driver.py:425,665).
   The generation phase reads the dialogue text off that line dict.
5. **Generation location = separate post-brief BATCH pass, NOT ShotLock** (Gemini, GPT,
   Grok, DeepSeek). Gemini's argument is sound: ShotLock would force N sequential LLM calls
   during locking. FIX: one batched LLM call for all beats, after brief+lines exist, before
   render; writes the ledger once.
6. **Render stays READ-ONLY; the generation pass writes fallbacks too** (GPT). FIX: the
   pass writes a full `motion_clause` object for EVERY shot (fallback=true + static text
   when generation is off/failed/invalid). render_driver.py:972 only READS; legacy ledgers
   with no field fall back to `_LTX_MOTION_PROMPT_BY_ROLE`. Keeps re-render deterministic.
7. **Default must preserve output** (GPT). Anti-deform negatives + any clause use only
   under the opt-in flag; flag OFF == byte-identical prompt (add a golden test).
8. **euler invariant vs OTR_LTX_SAMPLER knob** (GPT). CUT the sampler knob for v1 (or
   validate-reject euler_ancestral) so the knob can't break the identity invariant.
9. **GBNF cut for v1** (GPT, Gemini). Few-shot + a strict post-gen parser/validator
   (works on both Ollama and OpenRouter); per-shot fallback on reject.
10. **Allowed list = phrases + parser validation** (GPT), **global (not per-beat)
    anti-deform negative** (GPT, Gemini), **disposition log** generated/reused/fallback/
    invalid (GPT, Grok, DeepSeek), **schema_version for invalidation** (GPT). All folded.

## Discarded
- Grok "cut model + source_hash": keep both -- source_hash (with dialogue) is what makes
  re-renders deterministic and dialogue-aware; that's the whole point. model is cheap audit.

## Still open (verify-at-build)
- Exact dialogue text key on the `ledger['lines']` dict (field is present; confirm name).
- Cast/character table lookup for the display name from `char_id`.
- Writer-slot batch budget (one call, all beats).
