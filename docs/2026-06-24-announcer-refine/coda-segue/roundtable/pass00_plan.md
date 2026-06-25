# DYNAMIC NEWS-CODA SEGUE -- DESIGN TO HARDEN (coda-segue pass00)

Focused roundtable that REOPENS one converged decision from the announcer campaign:
the news-coda lead-in. The main campaign (pass04_plan.md STEP F) converged on a
DETERMINISTIC fixed `NEWS_CODA_LEAD_IN` ("The real story:") because the panel found
that weak local writers (mistral-nemo / gemma) BLEND the fictional outcome into the
"real" claim unless the fiction->reality pivot is nailed down, and a post-prefix
avoids a double-lead-in stutter.

OPERATOR OVERRIDE (2026-06-24): a fixed tag feels generic/wooden. The operator wants
the segue DYNAMIC + news-specific each episode -- a crafted bridge from the
episode's fiction to the real news, different every time, but NOT generic
("And now, the real story..." every episode is exactly what to avoid).

THE TENSION (what this roundtable resolves): dynamic/crafted/news-specific
(operator) vs deterministic/reliable/teachable (the converged finding, on the exact
weak-model failure axis we have repeatedly been burned on).

OPERATOR THESIS (unchanged): the show TEACHES. Drama delivers; the NEWS is the
payload at the end. The coda must read as the REAL fact, never the fiction restated.

---

## GROUNDED CONSTRAINTS (from the real code; see ../roundtable/_grounding_excerpts.md)
- The coda IS the single trailing announcer outro line (`compose_announcer_outro`,
  `_otr_line_composer.py:2778`); `news_close_brief` (the REAL news) is already
  threaded; `ending_change` is the FICTIONAL outcome (must NOT be restated).
- We have deterministic token machinery to gate with: `_content_tokens` /
  `_TOKEN_RE` / `_strip_possessive` (`_otr_story_quality_l12.py:418/497/509`),
  reusable WRITER-side.
- A deterministic fallback already exists as the safety floor pattern
  (`fallback_announcer_outro` :2635); the coda gets `fallback_news_coda_outro`.

## PROPOSED RESOLVING DESIGN (to harden -- the panel attacks this)
A hybrid: **dynamic happy-path + deterministic guardrails + a fixed-phrase fallback
floor.**
1. **Dynamic happy path.** The LLM writes the WHOLE coda as one line: a crafted
   fiction->reality segue tailored to THIS episode's news (from `news_close_brief`),
   in OTR announcer voice. NO fixed prefix. Low temperature.
2. **Teachability via STRUCTURE, not a constant phrase.** The format is taught by
   consistent POSITION (always the final beat) + a consistent SHAPE (it turns from
   the story just heard to what really happened), with words varying per episode.
3. **Deterministic guardrails (the reliability the panel demanded):**
   `validate_news_coda_line` requires (a) >=1 real news content token from
   `news_close_brief` present, (b) NO strong content-token overlap with
   `ending_change` (catches the fiction-as-real blend), (c) the word band. One
   reroll on failure.
4. **Fixed-phrase fallback FLOOR.** When the dynamic pass fails validation (twice),
   `fallback_news_coda_outro` emits a deterministic safe coda (a fixed lead-in +
   `news_close_brief`). The floor is never generic-blended; the happy path is
   dynamic.

So: dynamic + news-specific on the happy path; deterministic safety net underneath.

## THE 4 ASKS FOR THE PANEL (attack these)
1. **Weak-model blend (attack HARDEST).** Can mistral/gemma reliably write a
   news-aware fiction->reality segue WITHOUT stating the fictional outcome as if it
   were the real news? If not, is the blend deterministically GATE-ABLE, or must the
   fallback floor carry most episodes? We already proved instruction-following alone
   fails for the body -- does it fail here too?
2. **Teachability without a fixed phrase.** Does consistent position + shape teach
   the "drama, then the real fact" format, or is a light recurring ANCHOR/cadence
   still needed? Propose the MINIMAL anchor (if any) that stays learnable without
   being generic.
3. **Is "blend" gate-able?** Is "contains a news token AND low overlap with
   ending_change" enough to deterministically separate a good segue from a blend,
   or is the semantic blend not catchable cheaply (=> lean harder on low-temp +
   the floor)?
4. **Anti-generic.** With no fixed prefix, how to stop the LLM from defaulting to a
   generic "And now, the real story..." every episode while staying in OTR voice
   and news-specific.

## NON-NEGOTIABLE
Reliability floor (the coda must never restate the fiction as real) stays; behind
`story_scaffold`, byte-identical off; 100% local; determinism for the fallback;
the coda still delivers the real `news_close_brief` fact. Output of this roundtable
folds into pass04_plan.md STEP F + CODE_MAP.md.
