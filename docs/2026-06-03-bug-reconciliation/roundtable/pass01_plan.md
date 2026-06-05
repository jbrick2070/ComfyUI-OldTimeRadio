# Hardened fix plan -- 3 still-live OTR bugs (post-roundtable, grounded)

Order by risk: 264 (lowest, ship first) -> 295 -> 276 (highest, needs upstream
trace). Each: smallest correct change, PD1 clean-path inertness proven by test.

## BUG-264 -- news_briefs schema-overrun coercion (lowest risk, ship first)
`nodes/news_interpreter.py`, `NewsBriefs`. Mirror the BUG-307 coerce pattern with
per-field `@field_validator(mode="before")` (NOT a model_validator -- avoids
swallowing unrelated errors):
1. `key_terms`: if it's a list and len > `_MAX_KEY_TERMS` (7) -> keep the first
   `_MAX_KEY_TERMS`. Non-list -> return unchanged (let normal validation fail).
2. `script_brief` + `news_close_brief`: if it's a str and len > its cap
   (`_MAX_SCRIPT_BRIEF_CHARS`=350 / `_MAX_NEWS_CLOSE_BRIEF_CHARS`=250) -> truncate
   at the last word boundary <= cap (`s[:cap].rsplit(" ",1)[0].rstrip()`).
   Non-str -> unchanged.
3. The existing BUG-307 key_term *char-length* coerce stays (field validators run
   after the before-validators; order is fine).
4. `log.warning("NewsBriefs coerced: key_terms N->M, script_brief A->B, ...")`
   when any coercion fires (weak-model degradation stays visible).
**Tests** (`tests/test_news_interpreter.py`): 10 key_terms -> exactly 7; 360-char
script_brief -> <=350 ending on a whole word; 260-char news_close_brief -> <=250;
non-list key_terms still raises; a clean in-cap payload is `model_dump()`-identical
before/after (clean-path inertness). PD1 N/A (validator); PD3/PD6 none (constants).

## BUG-295 -- cast-name leak into line body (medium risk)
`nodes/_otr_line_composer.py` (~L1663-1692, the BUG-279 follow-on leak filter).
Build the leak set = OTHER characters' roster names that are MULTI-word + ALL-CAPS
(own-name already handled). Two location-scoped actions:
1. **Inside a `*...*` stage-direction group:** deterministic in-place SCRUB --
   remove the leaked name (collapse double spaces). Cheap, no retry, PD1-safe.
2. **Bare body (not in `*...*`):** if a multi-word ALL-CAPS roster name appears
   AND is not a vocative (not preceded by `,`/sentence-initial), trigger a RETRY
   within the EXISTING compose budget; accept-on-exhaust (PD1 length preserved).
Token-boundary match on the uppercased body only; never touch speaker labels.
**Tests:** roster `["ERIN SPENDER","MAEVE"]` -> `*ERIN SPENDER the monkeys'
enclosure*` scrubs to `*the monkeys' enclosure*`; `safe in the ERIN SPENDER`
retries; `Maeve.` allowed; `Get back here, ERIN SPENDER!` (vocative) allowed;
non-roster all-caps allowed; a clean line is unchanged (inertness). PD3/PD6 none.

## BUG-276 -- announcer/no-preset line reaches Bark (highest risk)
Two parts -- a bypass-proof chokepoint guard (closes the crash) + the upstream
root-cause fix (removes the misroute).
1. **Chokepoint guard (`nodes/_otr_voice_node_common.py`, just before the
   `adapter.generate_voice(...)` call ~L300, UNCONDITIONAL):** if the resolved
   engine is Bark (ROLE=`char_voice`) and `voice_preset` is missing / not `v2/*`:
   - line is ANNOUNCER-tagged (`char_id=='announcer'`) -> reroute to the Kokoro
     announcer engine with the episode's announcer voice (audio preserved = PD1);
   - else (genuine character line, no preset) -> fail loud with line_id/char_id/
     role context BEFORE generate_voice (don't invent a preset, don't skip).
   No-op when `voice_preset` starts with `v2/` (clean-path inert).
2. **Upstream root cause (verify-at-build):** find the line-batching site that
   put an `announcer` line into the char_voice batch and gate it on engine
   `roles` so an announcer line never enters the Bark batch. The chokepoint guard
   is the backstop; this removes the cause. Confirm no oscillation with the Gate-2
   reviewer (which blocks the opposite remap).
3. Keep `eng_bark.py`'s `EngineUnusable(MALFORMED_CONFIG)` as the final net.
**Tests:** an announcer-tagged, preset-less line -> routed to Kokoro, Bark
`generate_voice` NOT called -- asserted EVEN with `bypass_freeze_halt=True`; a
character line with no `v2/*` preset -> fails before Bark (not skip/default
voice); a clean episode -> dispatch decisions + audio byte-identical (PD1).
Emit a structured warn-event on reroute so soaks surface "preset-less line
rerouted" without a crash.

## Invariants guarded
PD1 (every fix proven inert on the clean path by test; 276 reroute preserves
length) - PD3 (no INPUT_TYPES/widget/socket; constants/env only) - PD6 (no
model-pick widget) - no-overhaul (smallest change each) - run Bug Bible + full
`tests/` + audio-byte-identical before ship.

## Also: BUG-271 -- mark FIXED (code already correct)
`_otr_ledger_reviewer.py:858-877` already validates `wrong_char_id.expected` as a
char_id (primary) + name (fallback). Update BUG_LOG to [FIXED]; add a regression
test asserting a char_id-shaped `expected` repairs.
