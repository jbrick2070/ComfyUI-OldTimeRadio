# Fix-ideas roundtable -- 3 still-live OTR bugs (post-trace, grounded)

Three open bugs survived a trace against current code. We want hardened fix
plans. (A 4th, BUG-271, was found ALREADY FIXED in code -- excluded.)

## Hard constraints (OTR prime directives -- a fix that breaks one is rejected)
- **Audio is king (PD1):** the full narrative audio output must never break,
  shorten, or degrade; audio stays byte-identical to baseline at every gate. A
  fix touching the audio/voice path must prove byte-identity (or be inert on the
  clean path).
- **PD3 wire-through:** a node `INPUT_TYPES`/widget/socket change must be
  re-wired + verified against the workflow JSON. Prefer env-var knobs (no
  re-wire).
- **PD6:** only the writer node exposes model-pick widgets; no new `model_id`
  widget anywhere.
- **No-overhaul / smallest correct change.** Run Bug Bible + full `tests/` +
  audio-byte-identical before ship.
- Local/offline; Windows; single RTX 5080 16 GB.

## BUG-264 -- news_briefs schema overrun on weak models (LIVE, non-fatal quality)
`nodes/news_interpreter.py`: the `NewsBriefs` pydantic schema hard-caps
`key_terms` at `_MAX_KEY_TERMS=7` (`max_length`) and `script_brief` at
`_MAX_SCRIPT_BRIEF_CHARS=350`. A weak writer model (e.g. `google/gemma-2-2b-it`)
returns too many key_terms (10, then 9) and an over-long script_brief; all 3
structured-call attempts fail validation -> `news_interpreter` falls back to the
raw `news_seed` with NO distilled briefs -> the announcer intro AND outro both
drop to generic deterministic fallback text. Non-fatal but a real quality hit.
**Already half-solved:** BUG-307 (2026-06-03) added a `@field_validator` that
COERCES (truncates) an over-long `key_term` *string* instead of raising. 264 is
the same family on the COUNT axis (list too long) + the `script_brief` length.
- **Proposed fix:** extend the coerce pattern -- a `@model_validator(mode="before")`
  (or field validators) that TRIM `key_terms` to the first `_MAX_KEY_TERMS` and
  truncate `script_brief`/`news_close_brief` to their caps, so the schema always
  validates on attempt 1 instead of burning the retry ladder and losing the
  briefs. (cf. BUG-303 coerce-not-reject, BUG-307.)
- **Questions:** trim vs prompt-the-model-shorter? Which key_terms to keep when
  trimming (first N? the ones present in source per the V1 check?)? Any downside
  to silently dropping terms vs the current loud-fail-then-rawseed?

## BUG-276 family -- announcer/no-preset line reaches Bark (LIVE, fail-closed stop)
Root cause: a dialogue line tagged `char_id='announcer'` (or a character line)
with no Bark `v2/*` voice preset reaches the Bark engine. `eng_bark.py:78`
fail-closes: `if not voice_preset or not startswith("v2/"): raise EngineUnusable(
MALFORMED_CONFIG)`. The gate is CORRECT (Bark needs a v2/* preset; announcer
lines belong to Kokoro). Mitigations landed: BUG-276 `needs_full_rerun` halt +
a Gate-2 reviewer guard (`_otr_ledger_reviewer.apply_deterministic_cast_repairs`)
that refuses to remap a `speaker_role='character'` line onto the announcer's row.
But the 2026-05-31 soak still hit it (remote mistral-nemo, 350w,
`bypass_freeze_halt=True`): line `b018 char_id='announcer'` reached Bark ->
crash. So a line can still be announcer-tagged with no v2/* preset and slip to
Bark on some routing paths.
- **Proposed fix direction:** a pre-Bark routing guard -- ANY line whose engine
  resolves to Bark but lacks a v2/* preset should be REROUTED (announcer-tagged
  -> the announcer/Kokoro engine) or skipped-with-warning, NOT raise. Make the
  fail-closed gate the LAST resort, with an explicit reroute step before it.
- **Questions:** reroute vs assign-a-default-v2-preset vs skip-the-line? Where is
  the safest single chokepoint (the per-line engine dispatch) to enforce "no
  character/announcer line reaches Bark without a v2/* preset"? How to keep PD1
  audio byte-identity on the clean path (the guard must be inert when every line
  already has a valid preset)? How to make it impossible to regress (a test that
  asserts no `speaker_role='character'`/announcer line reaches Bark preset-less)?

## BUG-295 -- cast name leaks into line BODY text (LIVE, content quality)
On remote mistral-nemo runs, an ALL-CAPS canonical cast name lands mid-phrase in
spoken text: `*ERIN SPENDER the monkeys' enclosure*` (stage direction missing its
verb), `safe in the ERIN SPENDER` (dangling noun). The existing leak filter
(`nodes/_otr_line_composer.py` ~L1663-1692, a BUG-279 follow-on) only retries
when the WHOLE cleaned line equals the speaker's own name or the literal
`ANNOUNCER`; it deliberately does NOT filter the broader roster (to preserve
legitimate one-word cross-character drama like `"Maeve."`). The mid-phrase /
inside-asterisk case is uncaught. Generation artifact (not wiring); local runs
were clean.
- **Proposed fix:** in the compose retry loop, flag a draft where an ALL-CAPS
  MULTI-WORD roster name appears inside a `*...*` group or as a bare mid-sentence
  token, and retry; scope strictly to multi-word ALL-CAPS so single-name drama is
  untouched.
- **Questions:** is "multi-word ALL-CAPS roster name mid-phrase" the right,
  false-positive-safe trigger? Retry vs in-place scrub (strip/replace the leaked
  name) vs both? How many retries before accepting? Any legitimate case where a
  multi-word ALL-CAPS roster name SHOULD appear in body text?

## What we want from the panel
For each of the 3: the best concrete, smallest-correct fix given the constraints;
failure modes / false-positives; the test that prevents regression; and anything
we're missing. Rank by risk. Flag any fix that endangers PD1 audio byte-identity.
