VERDICT: yes-with-fixes -- mechanical index math is verified correct, but the
`_fetch_rss_seed_or_die`/`style` signature change ripples into a file
(`story_orchestrator.py`) and a call-site file (`_otr_source_payload.py`) the
plan never lists as touched.

MUST-FIX BEFORE BUILD:
1. [Sec 2/9] `style` param removal ripples into `story_orchestrator.py`, an
   untouched file. `_fetch_rss_seed_or_die` (OTR_LedgerScriptWriter.py:1138)
   calls `_so._fetch_science_news(max_feeds=10, style=slug, model_id=
   model_id, ...)` at :1176-1179. `_fetch_science_news` (story_orchestrator.
   py:1670-1671) forwards `style` into `_llm_rank_news_candidates(pool,
   style=style, ...)` at :1844-1849, where it becomes actual LLM prompt text:
   `genre_human = (style or "sci-fi").replace("_", " ")` (story_orchestrator.
   py:1490) feeding the rank prompt (:1492). Repeats at :1573/1605,
   :1846/1937/1963/1984. Fix: either keep style alive end-to-end fed from the
   contract's slug post-hoc, or delete story_orchestrator.py's style
   plumbing too and decide what genre text (if any) seeds the news-rank
   prompt.
2. [Sec 2/7] The plan never names the real call site of
   `_fetch_rss_seed_or_die`. It is not called by name inside
   OTR_LedgerScriptWriter.py itself -- the actual caller is
   `nodes/_otr_source_payload.py:219-230`'s `_fetch_science_rss(*, bank,
   style_slug, technical_model)`, which does `return _writer.
   _fetch_rss_seed_or_die(style_slug, technical_model)` (:230). If the
   signature drops style, this wrapper (documented "S31 B6 slot-label/id
   agreement invariant") must be updated in the same change, and
   tests/test_writer_input_resolve.py (AST-asserts the 2nd positional arg
   contract) must be re-pinned. Section 7 has no explicit step for
   _otr_source_payload.py.
3. [Sec 4] Index math VERIFIED CORRECT against the live JSON. widgets_values
   length 27; [8]='let the story decide', [9]='', [24]='auto',
   [25]='science_news', [26]='sci_fi_radio' -- matches the plan exactly.
   test_workflow_json_guardrails.py:358 independently pins
   _WRITER_STYLE_SLOT = 8; lines 673-674/732-734/770-771 contain the exact
   assertions the plan says to update. No correction needed.

SHOULD-FIX:
4. [Sec 9 ask 2] Sequencing risk is real but understated. pick_style is
   called from OTR_LedgerScriptWriter.py:2995 inside a style_pending-gated
   block; if the picker module is deleted before the call site + its import
   (line 2797) are removed, ComfyUI fails to import the node at boot. A
   SECOND import exists at line 6103 inside an embedded smoke-test helper --
   the grep sweep must catch both.
5. [Sec 1b] Doc-note landing site underspecified -- confirm whether the
   comment goes above the `if _style_grammar_on:` block (:3338) or inline in
   the try body, and that step 6 targets a stable anchor step 1/2 won't
   touch.

OPTIONAL:
- Confirm none of the listed test files also assert on story_orchestrator.
  py's style= kwarg (given finding #1) -- they may need updates too, not
  just deletion.

CUT THESE: nothing over-engineered; the gap is completeness
(story_orchestrator.py), not excess.

[ASSUMPTION] Did not read `_llm_rerank_with_bodies` in full (only its sibling
`_llm_rank_news_candidates`); verify that function's style= usage
(story_orchestrator.py:1573/1605 region) the same way before deciding fix
(a) vs (b) in MUST-FIX #1.
