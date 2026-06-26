<!-- Claude R2 anchor (coding plan / implementability). Grounded. -->

VERDICT: yes-with-fixes. The R1 architecture is sound; this is the concrete
coding plan. One grounded refinement to the news-bleed layer materially changes
its implementation (below) -- it is simpler than R1 implied (reuse the existing
Phase-0 name gate, do not build a new detector).

CODING PLAN (file -> function -> hook):

1. STAGE-DIRECTION (capitalised-participle + quote). `_otr_line_hygiene.py`: add
   `_leading_participle_quote_strip(text)` as a SIBLING to `_leading_stage_strip`
   (NOT a change to it -- its lowercase guard at line 271 is correct for its
   class). Target shape: `^["“]?[A-Z][a-z]+(ing|ed),\s+["“](.+)` -> return the
   quoted span. Idempotent, never raises. Wire into BOTH `scrub_leading_stage_
   direction` (freeze floor, after the existing strip returns unchanged) and
   `detect_leading_stage_business` (the reroll detector). The quote requirement is
   the false-positive guard: "Running to the door, I shouted" has no leading quote
   so it is untouched.

2. CAPS-CAST-VOCATIVE. `_otr_line_hygiene.py`: `scrub_self_vocative` (line 68)
   only strips the SPEAKER's own name. Add `scrub_caps_cast_vocative(text,
   roster_fullnames)` -> strip/title-case a token that EXACTLY matches a roster
   full name in ALL CAPS at a vocative position (`^NAME[,!:-]` / `[, ]NAME[.!?]?$`).
   Wire into `scrub_ledger`'s chain. (verify-at-build: confirm `scrub_ledger`'s
   current scrub list + that it has the roster.)

3. MALFORMED INTERNAL QUOTE. `_otr_line_hygiene.py`: `sanitize_transcript_text`
   (line 895) balances only one edge wrapper. Add an internal-odd-quote DETECTOR
   (quote count odd AND not a single edge wrapper) -> route to recompose, do NOT
   silently balance. Fail-closed.

4. NEWS-BLEED (grounded refinement -- the big one). DO NOT build a new
   proper-noun detector. The existing Phase-0 name gate already REJECTS
   un-allowlisted names; the leak ships because real-person news terms enter the
   allowlist via `key_terms` (grounding: `build_allowed_roster` line 302 +
   comments 233/260/329 -- "anything in news content ... must be threaded via
   key_terms", "Names that legitimately belong ... must arrive via key_terms").
   FIX: filter the `key_terms` feed to EXCLUDE a real-PERSON / political-figure
   class before it is allowlisted -- honorific prefixes (President/Senator/PM/
   Governor/Dr surname), a small living-figure stoplist, a Firstname-Lastname
   person heuristic. Org/place terms (NASA, CERN, JPL) stay allowlisted (they are
   legitimate in sci-fi). Then the EXISTING gate rejects "President Trump" with no
   new machinery. (verify-at-build: the gate's reject ACTION -- reroll vs strip --
   and where key_terms is assembled so the filter sits upstream of
   `build_allowed_roster`.)

5. LAYER 3 TYPED-REPAIR (optional A). `_otr_repair_prompts.py`: add
   `leak_clean_repair` (sibling to the existing `narration_leak_repair`); call via
   the existing `_otr_structured_call` infra (NO new node). Gate
   `OTR_ENABLE_LEAK_CLEANER` + non-offline. JSON {clean_text, removed_spans,
   reason_codes, confidence}; reject empty/over-diff/quote-malformed; only after a
   Layer-2 hit; never on clean lines.

6. FLAGS (in `_otr_config.py`, following the existing AUDIO-AFFECTING/ships-dark
   pattern at 95/107): `OTR_ENABLE_LEAK_FLOOR_V2` (rules 1-4, default-OFF/dark),
   `OTR_ENABLE_LEAK_CLEANER` (rule 5, default-OFF), `OTR_STRICT_LOCAL_CLEAN`
   (fail-closed vs best-effort+telemetry).

7. ACCEPTANCE. `tests/test_leak_floor_v2.py`: 4 positive fixtures (the real
   shipped lines) + 3 negative (legit emphatic vocative, legit in-world proper
   noun, non-stage `-ing` dialogue opening). Require 0 leak + 0 FP. Run the Bug
   Bible + regression suite per the standing rule.

SEQUENCING: rules 1-4 fire at compose (reroll detectors) and at the freeze floor
(`scrub_ledger`), BEFORE TTS, so audio derives from cleaned text. Layer 3 after a
Layer-2 hit. Everything default-OFF until a live 320w validation per lane, then
promote (the audio-affecting-flag discipline).

[ASSUMPTION] `key_terms` is assembled upstream of `build_allowed_roster` at a
single point a filter can sit (verify the news_interpreter -> roster path).
[ASSUMPTION] the freeze floor `scrub_ledger` runs before TTS synthesis (verify).
