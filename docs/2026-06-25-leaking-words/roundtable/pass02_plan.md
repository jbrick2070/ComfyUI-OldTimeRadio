# LEAKING-WORDS -- build-ready plan "leak-floor-v2" (R2-converged, 2026-06-25)

Synthesized from Claude's grounded anchor + a 3-model R2 panel (gpt-5.5,
gemini-3.1-pro, deepseek-v4-pro), every claim checked against the real
`_otr_line_hygiene.py` / `_otr_line_composer.py` / `_otr_repair_prompts.py`.
Campaign spend through R2: ~$0.42.

## The decision (what escapes the whack-a-mole)

ONE mandatory, deterministic, offline verifier over four named leak classes, using
NARROW STRUCTURAL extract-or-fail rules (not verb-whitelist widening, not broad
`-ing` scrubbing). The LLM cleaner (old Option A / Layer 3) is CUT from v1 -- all
three panelists independently called it over-engineering for four fixture classes.
The frontier writer stays a product recommendation, not the fix. This is small,
testable, model-agnostic, and roots each leak at its real cause.

## Core API (the missing interface the panel demanded)

A single entry point, called on each composed line BEFORE TTS/freeze:

```
verify_and_repair_line(text, req, policy, *, strict, repair_budget) -> VerificationResult
# VerificationResult: text, changed: bool, defects: tuple[Defect],
#   needs_recompose: bool, failed: bool, compose_flags: tuple[str]
# Defect: reason_code, target_span: (start, end)   # spans for tests + telemetry
```

`policy` is a TRANSIENT dataclass built per-episode (ledger schema stays frozen --
do NOT persist it):

```
EntityPolicy(allowed: frozenset[str], banned: frozenset[str])
# allowed  = cast + setting + fictional world nouns (incl. legit news orgs: NASA, CERN)
# banned   = real-person / political-figure source entities (President Trump, ...)
# invariant: banned & allowed == empty
```

## The four rules (file -> function -> hook), each grounded

1. **Stage-direction: capitalised participle before a quote.** New SIBLING fn in
   `_otr_line_hygiene.py` (do NOT touch `_leading_stage_strip` -- its lowercase
   guard at line 271 is correct for its own class; that guard, NOT the verb
   whitelist, is why `Gasping,` leaked). Order, reusing the existing
   `segment_double_quotes()` after curly->straight normalisation:
   normalise quotes -> if internal odd double-quote: defect `malformed_quote`,
   recompose -> else if outside-segment matches `^[A-Z][a-z]+(ing|ed),\s*$` AND
   segment 1 is a non-empty quoted span: return segment 1 (wrapper stripped).
   reason_code `capitalized_participle_before_quote`. The required quote is the
   false-positive guard ("Running to the door, I shouted" has no leading quote;
   `"Running," she said, "..."` starts WITH a quote, so the outside-segment-0 test
   fails -> untouched). Wire into `scrub_leading_stage_direction` (freeze floor)
   AND `detect_leading_stage_business` (reroll detector).

2. **Caps-cast vocative.** `scrub_self_vocative` (line 68) only strips the
   SPEAKER's own name -- confirmed it does NOT cover another character's name.
   Add `scrub_roster_vocative(text, roster_fullnames)`: a full-name PHRASE matcher
   (full names contain spaces -- not one token), sorted longest-first, boundary
   `(?<![\w'])NAME(?![\w'])`, only ALL-CAPS, only at a vocative position
   (`^NAME[,!:-]` / `[, ]NAME[.!?]?$`) -> DROP the vocative (deterministic; pick
   drop, not title-case, so the fixture is unambiguous), preserve terminal
   punctuation. Wire in the composer after `cast_strip`, before
   `detect_phantom_names`. Negative fixture must be a FIRST-name vocative
   ("YUKI!") -> untouched; the rule targets full names only.

3. **Malformed internal quote.** Predicate over DOUBLE quotes only (apostrophes /
   `don't` must not trip it -- use `segment_double_quotes` normalisation):
   `odd = norm.count('"')%2==1; edge_wrapper = (count==1 and start^end);
   internal_odd = odd and not edge_wrapper`. internal_odd -> `needs_recompose`;
   an edge wrapper still passes through `sanitize_transcript_text`.

4. **News-bleed -- fix it at the existing gate, no new detector.** Grounded:
   `build_allowed_roster` (line 302; loop at ~368-370) merges every `key_terms`
   entry into the UPPERCASE `allowed_roster`, and the comments mandate news terms
   arrive via `key_terms`. So "President Trump" ships because it is ALLOWLISTED.
   FIX: split source news entities BEFORE the roster build -- filter a real-person/
   political class out of `key_terms` (honorific prefixes President/Senator/PM/
   Governor/Dr-+surname + a small living-figure stoplist + a Firstname-Lastname
   person heuristic) and route them into `EntityPolicy.banned`; org/place terms
   (NASA/CERN/JPL) stay in `key_terms` (legit in sci-fi). Add a `banned_terms`
   param to `build_allowed_roster` that excludes them. Then the EXISTING phantom/
   roster gate rejects the name -> reroll. Needs `build_banned_source_proper_nouns
   (raw_news_brief) -> frozenset[str]`.

## Layer 1 (prompt, defect-rate only)
Change the compose-prompt line to "no real-world proper names UNLESS listed under
NAMED ENTITIES" (a bare "no real-world names" contradicts the legit `key_terms`
injection of NASA/CERN and makes the model hallucinate or refuse). Not an
enforcement layer.

## Repair budget (do not let recomposes stack)
Grounding shows MULTIPLE existing budgets (`compose_line_draft.max_attempts`,
`_stage_dir_repair_attempted`, quality reroll, `_stage3_repair_attempted`). Thread
ONE shared `_leak_repair_attempted` guard through the recursive `compose_line` so
all four defects share a single per-line recompose, then either fail-closed
(`OTR_STRICT_LOCAL_CLEAN=1`) or ship best-effort + telemetry (default).

## Flags (in `_otr_config.py`, the existing audio-affecting/ships-dark pattern @95/107)
`OTR_ENABLE_LEAK_FLOOR_V2` (rules 1-4; default-OFF/dark -- it is audio-affecting;
promote after a live 320w validation per lane). `OTR_STRICT_LOCAL_CLEAN`
(fail-closed vs best-effort). Add accessors (`strict_local_clean_enabled()`), no
ad-hoc env reads in verifier code. NO new workflow-JSON node/widget -- content-only.

## Acceptance (define + commit with the code)
`tests/test_leak_floor_v2.py`: positive fixtures = the four real shipped lines
(`Gasping, "..."`; "President Trump's orders..."; "YUKI MARTIN, no!"; the
unclosed-quote line) asserting the exact post-verifier `VerificationResult`;
negative fixtures = a first-name vocative ("YUKI!"), a legit in-world org
("NASA confirmed..."), a non-stage `-ing` opening ("Running to the door, I
shouted..."). Require 0 leak + 0 false-positive. Run the Bug Bible + full
regression after the change (standing rule).

## Resolved contradiction (panel caught it)
"Mandatory" vs "ships-dark": the verifier is the MANDATORY correctness layer in
its ENABLED/strict state (CI + release promotion assert 0 leaks under
`OTR_STRICT_LOCAL_CLEAN`); it ships DARK only for the pre-promotion telemetry
window, exactly like the existing audio-affecting flags. After live validation it
promotes to default-ON and becomes the always-on floor.

## CUT
- **Layer 3 LLM cleaner** -- deferred out of v1 (unanimous panel): Layer 2 catches
  the four classes deterministically; an online JSON repair adds cost, latency, and
  a JSON-decode/schema failure surface, and `compose_line` returns raw stripped
  text that would mangle JSON anyway (would require the `_otr_structured_call`
  path). Revisit only if a real class emerges that Layer 2 structurally cannot
  catch.
- **Option B constrained generation** (R1) and action-preservation telemetry.

## Coder verify-at-build (the internal hook points -- need live grounding)
1. The exact writer order compose -> scrub -> freeze (`scrub_ledger`) -> TTS, so
   the verifier sits upstream of audio synthesis. 2. Whether `compose_line`'s final
   deterministic strip pipeline already calls the existing scrubs (gpt: it may
   not -- wire at the writer/`_otr_ledger_scrub` level, not only composer drafts).
   3. The phantom/roster gate's reject ACTION (reroll vs strip) for rule 4.
