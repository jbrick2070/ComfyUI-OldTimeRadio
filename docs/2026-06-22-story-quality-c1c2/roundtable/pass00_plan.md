# Story-Quality R2 C1 + C2 -- specificity anchors + central object (plan to harden)

> REVIEW FOCUS THIS PASS: **ARCHITECTURE / APPROACH.** (1) Deterministic
> extraction vs one cheap LLM setup call to derive the meta values? (2) How do the
> derived values reach the line prompt + the announcer-close prompt WITHOUT a
> ledger-schema change and WITHOUT a risky LineRequest signature ripple? (3) Is the
> C1 reroll gate feasible/safe given the composer does not know the beat position?

## Context: 6 of 8 R2 chunks already shipped (this is the last 2)
SHIPPED + pushed to v2.0-alpha, each suite+Bug-Bible green: S1 (music-text
suppression), S2 (announcer close = concrete image, not thesis -- the close intent
ALREADY says "use the central object if set"), S3 (cliche + flat stage-business
reroll gate), C0 (action-verb beat intents + wants_are_default classifier), C3
(contrasting speech_signatures), C4+C5 (escalation prompt + on-the-nose reroll).
The reusable machinery they established:
- A post-draft QUALITY-REROLL block in `_otr_line_composer.compose_line` that calls
  flag_*/detect_* helpers and fires ONE guard-capped reroll via the existing
  recursive-repair pattern (no new infra).
- Pure FLAG helpers in `_otr_line_hygiene` (flag_cliche / flag_stage_business /
  flag_on_the_nose / flag_thesis_close / detect_leading_stage_business).
- A one-shot recompose in `compose_announcer_outro` (F3 hedge + S2 thesis).

## HARD INVARIANTS (a fix that breaks one is rejected)
- Ledger `{cast,lines,meta}` schema `l3-2026-05-14` FIXED -> new values ride
  FREE-FORM `meta` (meta["specificity_anchors"], meta["central_object"]); NO new
  Pydantic fields.
- Audio byte-identical SPINE invariant; deterministic + idempotent (C7-safe);
  model-AGNOSTIC (every gate is one a strong/opus line passes -> lifts the weak
  end only); reuse the EXISTING reroll loop; NO workflow-JSON change; UTF-8 no BOM.

## GROUNDED SEAMS (verified) + the open problems
- The line prompt is built by `_otr_line_composer._build_user_prompt(req)` from a
  `LineRequest`. The req carries `canon_header` (an EPISODE-CONTEXT string already
  rendered into the prompt) -- a candidate channel to inject anchors WITHOUT a
  LineRequest signature change. (VERIFY the canon_header build site in the writer.)
- The composer does NOT receive the beat position (opener / closer / music) or
  arc_phase -- so the C1 gate "a character line on a NON-opener/closer/music beat
  with no anchor + no proper noun -> reroll" cannot scope to beat position as
  written (same limitation that scoped C5 to all character lines).
- `compose_announcer_outro(script_brief, news_close_brief, intro_text, ...)` is the
  close composer; threading `central_object` means appending it to a brief or a new
  kwarg (VERIFY the call site in the writer).
- The dramatic_state is derived in the writer AFTER the outline
  (`derive_dramatic_state` ~ writer line 2828); a setup derivation for anchors /
  central_object would sit near there (the meta + resident slot exist by then).
- News + key_terms live in `meta["news"]` / the story brief (the news_briefs path).

## PROPOSED PLAN (the starting point to harden)
### C1 -- specificity anchors
- DERIVE `meta["specificity_anchors"]` (3-5 concrete anchors: place / object /
  number / named bystander). PREFERRED = DETERMINISTIC extraction from
  meta["news"] + key_terms (proper-noun + number regex; no LLM call, C7-safe). A
  cheap LLM setup call is the alternative (higher craft, adds a fail-closed path).
- INJECT into the line prompt as concrete context the writer SHOULD use (via the
  canon_header channel or a minimal additive LineRequest field).
- GATE (open): a generic character line (no anchor + no proper noun) -> reroll_hint.
  Because the composer lacks beat position, scope it to ALL character lines OR drop
  the gate and rely on injection. Proper-noun detection must EXCLUDE the speaker's
  own cast names + sentence-initial capitalization (false-positive guard).

### C2 -- central story-object
- DERIVE `meta["central_object"]` (one concrete physical object the story turns on)
  at setup -- deterministic from the brief/key_terms OR a cheap call.
- REFERENCE it: the S2 announcer close already conditions on "use the central
  object if set" -> thread central_object into `compose_announcer_outro`. Optionally
  nudge act-1 to introduce it via the outline (the outline runs before the state,
  so this is a prompt-context add, not a hard dependency).

## OPEN QUESTIONS FOR THE PANEL
1. Deterministic extraction vs one cheap LLM call -- which is the right default for
   a model-agnostic, C7-deterministic, low-risk lever?
2. canon_header injection vs a minimal additive LineRequest field -- which avoids a
   risky signature ripple while still reaching the prompt?
3. Is the C1 reroll gate worth it given no beat position, or is injection-only the
   safer, higher-precision choice?
4. Anything that breaks determinism / model-agnosticism / the frozen schema?
