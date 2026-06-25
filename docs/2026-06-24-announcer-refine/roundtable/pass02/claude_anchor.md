# R2 CLAUDE ANCHOR -- coding plan / implementability (code-grounded)

VERDICT: yes-with-fixes. The seams all exist and the ordering works; the
remaining risks are (a) where `opening_status_quo` comes from, (b) keeping the
flag-off path byte-identical through new prompt text, and (c) two small lookups.

## MUST-FIX BEFORE BUILD
1. [§2 Job3] **Implement the coda as the REPURPOSED outro line, not a new beat.**
   CONFIRMED: exactly one trailing announcer beat (`last_announcer_id`), composed
   post-loop by `compose_announcer_outro` (writer :4626-4634; placeholder :4483).
   Adding a coda beat would change the outline beat sequence + validators.
   Concrete: under `story_scaffold`, add a flag-gated branch in
   `compose_announcer_outro` that (a) injects the deterministic lead-in prefix,
   (b) delivers `news_close_brief` as a plain fact, (c) passes `ending_change`
   only as "do NOT restate". When the flag is off, the existing code path runs
   verbatim (byte-identical).
2. [§2 Job1] **SafeOpenBrief is buildable from existing fields -- spec the source
   of `opening_status_quo`.** CONFIRMED: `outline.setting` + `outline.time_of_day`
   already exist (`_otr_outline.py:211-213`); cast from `led.data["cast"]`.
   `opening_status_quo` is NEW and must be OUTCOME-FREE. Do NOT add a new macro
   LLM field (weak-model risk + it could leak the outcome). Concrete: derive it
   deterministically from the SETUP beat -- the first character beat's intent (the
   status quo at the start), which structurally precedes the climax. Pass the
   discrete safe fields to `compose_announcer_intro`; SEVER `script_brief` under
   the flag (writer :4465-4471).
3. [§1] **StoryContract pre-outline ordering is feasible; note the selection-input
   change.** CONFIRMED `cast_seed` (:2878) + `script_brief` (:2785) precede
   `OutlineRequest` (:3032). But `select_style`/`premise_wants_emergency` today
   read `outline.premise` (:3224); pre-outline they must read `script_brief or
   news_seed`. That shifts which text decides the "emergency" pool -- acceptable
   and flag-gated, but state it. Concrete: `build_story_contract(script_brief or
   news_seed, meta, cast_seed)`; delete the late :3224 select_style only after
   confirming no other caller.
4. [§2 Job2] **Climax-line lookup needs `beat_id` on ledger lines.** The outro
   takes the LAST character line (`reversed(led.data["lines"])`, writer
   :4619-4623). To source the CLIMAX line instead, match `led line.beat_id ==
   _climax_beat_id` (in scope from :3271). VERIFY ledger line rows carry
   `beat_id`; if not, add it or map via the outline. Byte-identical today
   (climax==last).
5. [§5] **New fields keep byte-identity only if prompt text is flag-gated.**
   Adding fields to the frozen `OutlineRequest`/`LineRequest` with `=""` defaults
   is safe (mirrors the existing `ending_template`/`conflict_object` pattern). The
   RISK is the intro/outro PROMPT rewrites: the new system-prompt text must be
   emitted ONLY when the flag is on (like `compose_announcer_outro`'s existing
   `if resolved and ending` append). Add off-flag golden tests on the open line,
   outro line, and ledger meta.

## SHOULD-FIX
1. [§2 Job3] **Lead-in vs `validate_announcer_line`.** A lead-in like "The real
   story:" contains a colon. `validate_announcer_line` (:2581) rejects leading
   speaker labels (`_ANNOUNCER_BAD_PREFIXES`) + brackets, and the new coda band
   may differ from the current 14-34 words. Verify a mid-line colon is allowed,
   and set a coda-specific min/max so "lead-in + fact" fits.
2. [§1] **KILL-2 line-level payload is just a compact register tag.** Per R1, do
   NOT thread `sound_world`/`story_engine` into `LineRequest`. Thread only a
   short register/tone string + the existing `conflict_object`. Confirm the
   composer's DRAMATIC-FRAME render (it already renders `conflict_object`/
   `conflict_type` only when non-empty) extends cleanly to a register field.
3. [§3 KILL 4] **Truncation order fix is local + safe** (`l12:800`): compute
   `enrichment`, truncate the ORIGINAL intent to `_INTENT_MAX - len(enrichment)`,
   then append. Add a test for `len(original)+len(enrichment) > 200`.

## OPTIONAL / NICE-TO-HAVE
- A tiny `SafeOpenBrief` dataclass (frozen) keeps the intro signature clean vs.
  4-5 new kwargs.
- Telemetry flags per feature (open_spoiler_rerolls, news_coda_emitted, etc.) so
  the 3-test "baked in" check is mechanical.

## CUT THESE
- Do NOT add a separate news-coda beat (structure change; validators, beat counts,
  the first/last announcer-id logic). Repurpose the outro. (See MUST-FIX 1.)

## ASSUMPTIONS
- [ASSUMPTION] `led.data["lines"]` rows carry `beat_id` (for the climax lookup).
  verify.
- [ASSUMPTION] an `era`/period value is available (meta/news) for the open; if
  not, time_of_day + setting alone still satisfy "no spoiler". verify.
- [ASSUMPTION] the setup beat's intent is reliably outcome-free at open-compose
  time (it is authored as the opening situation). verify on a soak sample.
