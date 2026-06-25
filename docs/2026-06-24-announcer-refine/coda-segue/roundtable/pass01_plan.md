# DYNAMIC NEWS-CODA SEGUE -- HARDENED (coda-segue pass01, post-R1)

R1 near-unanimous (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + anchor) pivoted the
design and killed my going-in gate. Folded.

## CORE PIVOT (the resolution)
The unsafe act was asking the weak LLM to write the WHOLE real-news coda. SPLIT it:
- the LLM writes ONLY a DYNAMIC BRIDGE -- the fiction->reality pivot, news-aware,
  asserting NO fact and NO dramatic outcome;
- the REAL FACT is the DETERMINISTICALLY-appended `news_close_brief`.
This dissolves BOTH problems at once: the weak model can't blend (it never writes
the fact), and the bridge-vs-gate contradiction disappears (the bridge SHOULD touch
the story; we no longer gate it against `ending_change`).

GROUNDING WIN this satisfies: reliability is now STRUCTURAL (the real fact is
appended, not generated), not a probabilistic gate -- the exact thing the weak-model
history demanded.

## DESIGN
1. **Dynamic bridge (LLM happy path).** One short pivot clause from tonight's drama
   toward the real-world subject, OTR host voice. References the real SUBJECT
   (news-specific) but states NO factual claim, NO dramatic outcome. Low temp.
2. **Deterministic real-fact payload.** Append cleaned `news_close_brief` after the
   bridge: `f"{bridge} {news_close_brief_clean}"`. The fact is NEVER invented by the
   weak model.
3. **Separate `_NEWS_CODA_SYSTEM` prompt** (NOT the current `_ANNOUNCER_OUTRO_SYSTEM`,
   which forbids news-summary + demands a concrete final image -- all 3 confirmed it
   fights the coda). SUPPRESS the resolved-fiction "State this outcome plainly"
   branch (:2854) under the flag. EXCLUDE `ending_change` from the coda LLM call
   (removes the blend temptation -- DeepSeek#4).
4. **Anti-generic (the operator's core want).** Forbid generic openers via a
   blacklist ("And now, the real story", "But in the real world", "In reality"...);
   REQUIRE the bridge to carry >=1 news content token so it is about THIS episode's
   subject, not a template.
5. **Validator -- BRIDGE ONLY** (`validate_news_coda_bridge`): bounded length; no
   leading bracket; not a blacklisted generic opener; >=1 news content token; does
   not assert an outcome (best-effort). **DROP the `ending_change`-overlap gate**
   (Gemini: incompatible with a real bridge). One reroll (altered seed) on failure.
6. **Fallback FLOOR = deterministic ROTATING prefix + payload** (Gemini). On twice-
   failed bridge, `fallback_news_coda_outro` emits a `cast_seed`-keyed pick from a
   small CLOSED prefix pool (e.g. "The real story:", "The true account:", "From
   tonight's headlines:") + cleaned `news_close_brief`. Varied (not one wooden
   phrase), fully deterministic, never blended. EMPTY `news_close_brief` => do NOT
   fake a real story: skip news-coda mode, use the existing outro fallback, LOUD flag
   `news_coda_no_brief` (GPT#7).

## WHAT THIS CHANGES vs the main-campaign STEP F (pass04_plan.md)
- The fixed `NEWS_CODA_LEAD_IN` on the happy path is REPLACED by the dynamic bridge;
  the fixed phrase survives ONLY inside the deterministic FALLBACK, as a rotating pool.
- STEP F's "validate the body has no lead-in, then prepend the lead-in" is REPLACED
  by "validate the bridge, then append the deterministic news payload."
- `ending_change` is EXCLUDED from the coda LLM call (was: passed as forbidden text).

## KEY OPEN QUESTION FOR R2
Does the bridge reference the real SUBJECT (pass a short news subject/noun-phrase to
the bridge prompt -> news-specific, but risk the model copying the fact, duplicating
the appended payload), OR pivot only from the DRAMA (safe, but the clause itself is
less news-specific and specificity rides on the appended payload)? Lean: pass the
news SUBJECT (a short phrase, not the full brief) + instruct "name the subject, not
its facts"; de-dup against the appended brief. R2 settles this + the exact
`_NEWS_CODA_SYSTEM` rules, the specificity threshold, the brief cleaning/length cap,
the rotating pool, and the `compose_flags` taxonomy.

## NON-NEGOTIABLE (carried)
Behind `story_scaffold` / `_style_grammar_on`, byte-identical off; the coda still
delivers the real `news_close_brief` fact; never restates the fiction as real; 100%
local; deterministic fallback; UTF-8 no BOM; SFW.
