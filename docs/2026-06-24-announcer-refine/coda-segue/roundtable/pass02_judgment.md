# R2 JUDGMENT -- dynamic coda segue, implementability (Claude, grounded)

Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.0961. Convergence: HIGH.

## ACCEPTED (folded into pass02)
- DROP the ">=1 news content token" bridge requirement (all 3): unimplementable
  w/o NLP AND unnecessary -- the appended payload is the news. 
- DROP the semantic "asserts no outcome" validator (all 3): unimplementable; use a
  tight length cap (~80 chars) + the prompt + a tiny outcome-verb blocklist.
- Dedicated `compose_news_coda` function (DeepSeek#9) instead of overloading
  `compose_announcer_outro` -> the outro stays byte-identical with NO new params.
- Reroll is PROMPT-based, not seed-based (GPT#3, grounded: `_announcer_generate`
  takes no seed). Corrected from "altered seed".
- Rotating-prefix fallback pool keyed by cast_seed; empty `news_close_brief` =>
  run the normal FICTIONAL outro, never fabricate a real story (Gemini#6/GPT#9).
- Exact `_NEWS_CODA_SYSTEM` text + compose_flags taxonomy + length caps + the
  punctuation join (bridge ends ':'/em-dash, capitalize the fact) -- all panel asks.
- BRIDGE_GENERIC_OPENERS is BRIDGE-ONLY; fallback prefixes are not bridge-validated
  (GPT#11).

## JUDGE CALL (GPT vs Gemini -- the one real split)
Does the bridge see `ending_change`? Gemini#3 said pass it (context to pivot away);
GPT#8 said exclude it (it carries the fictional resolution -> bleed). DECISION:
EXCLUDE `ending_change` + `final_character_line`; give the bridge the PREMISE/SETUP
(`script_brief` + `intro_text`). A pivot needs the tale's SUBJECT, not its OUTCOME --
this is specific enough to be non-generic AND removes the restate-the-fiction
temptation at the source. (Gemini's "subject extraction paradox" was partly a
grounding miss -- `key_terms` exists -- but its deeper instinct, keep the weak model
away from the news/outcome, is right, and the premise-only bridge needs neither
key_terms nor the news.)

## SUPERSEDES
This replaces the main-campaign pass04_plan.md STEP F lead-in mechanics (fixed
`NEWS_CODA_LEAD_IN` + body-lead-in validation) with `compose_news_coda`. STEP F's
other goals + the climax-line decoupling stay. The fold step removes the now-dead
STEP F lead-in pieces.

## CONVERGENCE CALL
Architecture (R1 split) + mechanics (R2) are settled with exact signatures, prompt
text, validators, and call-site branch. Run R3 to confirm the WIRING (the early
branch + byte-identity + no conflict with the main campaign); if clean, that is
convergence (no separate near-empty R4 -- per the stop-at-convergence rule).
