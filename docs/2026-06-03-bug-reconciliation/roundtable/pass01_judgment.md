# Fix-ideas roundtable -- pass01 judgment (Claude is the judge)

Panel: 6/6 usable (opus-4.8, sonnet-4.6, gpt-5.5, gemini-3.1-pro, grok-4.3,
deepseek-v4-pro) @ max_tokens 8000. Spend ~$0.73.

## Grounded verifications (the judge's job)
- **`news_close_brief` EXISTS with a cap** (`news_interpreter.py:159`:
  `news_close_brief: str = Field(..., max_length=_MAX_NEWS_CLOSE_BRIEF_CHARS)`,
  =250). Resolves the panel's repeated "verify news_close_brief" flag -> truncate
  it too. CONFIRMED.
- **The BUG-276 dispatch chokepoint is `_otr_voice_node_common.py`** ("Dispatch
  core shared by the v2 character + announcer voice nodes", L138). It already
  does `engine = assert_usable(engine, self.ROLE)` -> `adapter = get_engine(...)`
  -> `voice_preset = cast.get("voice_preset")` -> `adapter.generate_voice(...)`
  (L172-300). This is a PER-LINE, UNCONDITIONAL chokepoint (NOT gated on
  `bypass_freeze_halt`) -- exactly where every model said the guard belongs.
  CONFIRMED. (The deeper root cause is upstream line-batching feeding an
  announcer-tagged line into the char_voice/Bark batch.)

## Accepted (panel consensus, grounded)
- **264:** length-ONLY coercion via per-field `@field_validator(mode="before")`
  (NOT a broad model_validator that could swallow unrelated errors) -- key_terms
  -> first `_MAX_KEY_TERMS`; script_brief + news_close_brief -> truncate at a word
  boundary; leave non-list/non-str to normal validation; log a WARNING on coerce.
  Commit to FIRST-N for key_terms.
- **276:** put the guard at the `_otr_voice_node_common.py` chokepoint,
  UNCONDITIONAL; no-op when `voice_preset` starts with `v2/`; an announcer-tagged
  preset-less line REROUTES to Kokoro (preserves audio = PD1); a genuine
  character line with no `v2/*` preset FAILS LOUD (keep `eng_bark.py`'s
  `EngineUnusable` as the backstop, surfaced earlier with line/char/role).
- **295:** scope to OTHER characters' MULTI-word ALL-CAPS roster names; the
  existing own-name/whole-line filter stays.

## Rejected (panel-agreed)
- 276 **skip-the-line** -- PD1 violation (shortens audio). CUT.
- 276 **assign-a-default-v2-preset** for a character line -- masks malformed cast
  config + wrong voice. CUT (minority Gemini proposed it; majority + PD reject).
- 264 **prompt-the-model-shorter** / **source-presence re-ranking** -- second
  model call / needs source in the validator; first-N is O(1) and offline. CUT.

## Judge's split decision (264/295 retry-vs-scrub, where the panel disagreed)
- **295 location-scoped:** inside a `*...*` stage direction -> deterministic
  in-place SCRUB (strip the leaked name; cheap, no retry-ladder cost, PD1-safe;
  e.g. `*ERIN SPENDER the monkeys' enclosure*` -> `*the monkeys' enclosure*`).
  For a BARE-body leak (`safe in the ERIN SPENDER`, where a scrub leaves broken
  grammar `safe in the`) -> RETRY within the EXISTING budget, with a vocative
  exclusion (skip when the name is preceded by address punctuation / is
  sentence-initial, e.g. `Get back here, ERIN SPENDER!`), accept-on-exhaust. This
  uses the cheap deterministic tool for the common asterisk case and the safe
  tool for the grammar-fragile case -- reconciling the retry-camp (gpt/sonnet/
  opus) and the scrub-camp (gemini/grok/deepseek).

## Still verify-at-build
- 276: the exact UPSTREAM line-batching site that lets an announcer line into the
  char_voice batch (the chokepoint guard closes the crash; the upstream fix
  removes the misroute). Confirm the Gate-2 reviewer (announcer<-char block) and
  the new guard (announcer->Kokoro) cannot oscillate.
- 295: the exact retry budget + roster source at `_otr_line_composer.py` L1663.
