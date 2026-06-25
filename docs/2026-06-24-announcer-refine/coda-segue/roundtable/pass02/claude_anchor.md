# R2 CLAUDE ANCHOR -- dynamic coda segue, implementability (grounded)

VERDICT: yes-with-fixes. The split design is highly implementable and reuses
existing infra; `key_terms` resolves R1's open question cleanly. Pin the field
threading, the bridge validator, and the assembly/length.

## GROUNDED RESOLUTION OF R1'S OPEN QUESTION
`NewsBriefs.key_terms: list[str]` (2-6 journalistic terms; `news_interpreter.py:175`)
IS the news SUBJECT anchor. Feed `key_terms` to the bridge prompt ("name one of these
subjects; do NOT state its facts/numbers"); append `news_close_brief` as the fact.
The bridge specificity validator then checks the bridge contains >=1 `key_term`
(deterministic, reuse the existing list -- no tokenizing the brief). Subject (bridge)
and fact (payload) are cleanly separated.

## MUST-FIX BEFORE BUILD
1. **Thread `key_terms` to the coda.** The outro already reads `news_close_brief`
   from `(meta.get("news") or {}).get("news_close_brief")` (writer :3949). VERIFY
   `meta["news"]` carries `key_terms`; if not, stamp `meta["news"]["key_terms"] =
   briefs.key_terms` where briefs is built (near `script_brief = briefs.script_brief`
   :2785). Then the post-loop coda reads both from `meta["news"]`.
2. **Bridge validator (`validate_news_coda_bridge`).** Reuse `key_terms` membership
   for specificity (>=1 present, casefold substring -- pass `key_terms` to the
   composer as a tuple so NO l12 import is needed; it is just strings). Plus: a
   generic-opener blacklist (casefold startswith over a small tuple: "and now",
   "but in the real world", "in reality", "the real story" on the HAPPY path),
   bridge length band (e.g. 6-22 words), no leading bracket. **DROP the
   `ending_change`-overlap gate** (R1). One reroll with an altered seed + an
   amplified "name the subject <term>" hint.
3. **Assembly + length (coda-specific, NOT the 340-char outro band).** `coda =
   clean_one_line(bridge, max~120) + " " + clean_one_line(news_close_brief, max~220)`.
   Instruct the bridge to be a PIVOT CLAUSE, not a full news sentence (avoids
   stutter/dup with the appended brief).
4. **`_NEWS_CODA_SYSTEM` (new, flag-gated).** OTR host voice; "turn from tonight's
   tale to the real world; name the subject (one of: <key_terms>); do NOT state any
   fact, number, or the drama's outcome; write only the pivot clause -- the real
   report is added after you." SUPPRESS the resolved-fiction branch (:2854) under
   the flag; EXCLUDE `ending_change` from this call.
5. **Fallback FLOOR + empty-brief.** `fallback_news_coda_outro`: `cast_seed`-keyed
   pick from a CLOSED period-lead-in tuple + `clean_one_line(news_close_brief)`.
   EMPTY `news_close_brief` => skip coda mode, use the existing outro fallback, LOUD
   flag `news_coda_no_brief` (do NOT fabricate a real story).
6. **compose_flags taxonomy** (observability -- GPT R1#SHOULD): `news_coda_dynamic`
   / `news_coda_reroll` / `news_coda_fallback_pool` / `news_coda_no_brief`.

## SHOULD-FIX
1. Do the `key_term` membership check WRITER-side OR pass the `key_terms` tuple into
   the composer -- either avoids importing l12 into `_otr_line_composer.py`.
2. Bridge prompt gets the EPISODE essence too (intro_text, already passed to the
   outro) so the pivot connects to the tale just heard, not just the subject.

## CUT
- Do NOT build a semantic "asserts an outcome" detector (fragile, false-positive
  prone). The structural split (the bridge is not the payload) + the prompt + the
  key_term/blacklist gate are sufficient; the deterministic appended fact is the
  real safety.

## ASSUMPTIONS
- [ASSUMPTION] `key_terms` is reliably 2-6 concrete subject terms (it is, by schema
  min_length=1/max=6 + the word-boundary validator). If a key_term is itself a
  generic word, the specificity check is weak -- low risk; the appended fact carries.
- [ASSUMPTION] `meta["news"]` is the right carrier for key_terms at outro time
  (the outro already reads news_close_brief from it). Verify the stamp site.
