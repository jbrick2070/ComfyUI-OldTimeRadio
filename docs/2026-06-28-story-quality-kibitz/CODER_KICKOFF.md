# CODER BUILD KICKOFF -- story-quality G1 (paste as message #1 of a fresh coder window)

```
You are the CODER window. BUILD the story-quality update here. Spec =
kibitz-runs/2026-06-28-story-quality/final.md (3-way kibitz CONVERGED: Codex gpt-5.5 +
Antigravity gemini-3.5-pro + Claude judge). Read final.md IN FULL first, plus the r1-r4
judgment logs alongside it + the anchors/scan in docs/2026-06-28-story-quality-kibitz/.

WHY: the 2026-06-27 anti-overstuffing gates now OVER-correct -- the reroll hint
_QUALITY_COLLAPSE_HINT (_otr_line_composer L2293, "rewrite under ~20 words, one detail")
forces rich lines into noun-salad AND, with the 14-beat skeleton x the ~22-28-word
one_breath cap, hard-caps every episode at ~210-281 voiced words regardless of
target_words. LENGTH + CRAFT are ONE root cause; G1 fixes it (tune the gates, don't
remove them). Operator-directed exception to the parked story-pipeline.

OPERATOR DECISION-GATE DEFAULTS (build to these unless I say otherwise):
- gate flag = story_quality_v2 (existing); flag-OFF must be byte-identical.
- length is a SIDE EFFECT of G1: per-line cap up to ~60 words, NEVER pad; the
  ~1363-word structural ceiling STAYS (no BEAT_WORD_HARD_MAX change this pass).
- S1 (seed fidelity) DEFERRED; S6 (phantom) CUT.

BUILD SEQUENCE -- 6 green commits; each = full suite + Bug Bible + B7 sweep green, then
commit AND push to v2.0-alpha (same session):
1. Shared leaf helpers in _otr_line_hygiene.py: derive_one_breath_cap, _hard_clauses
   (count ,;: + FANBOYS), find_cliche_phrase; is_truncated imports where needed; +
   tests/fixtures/ golden ledgers (extract plancks b03/b10, ledger_ink b04/b13, dance
   b04/b11) + tests/test_story_quality_golden.py. RUN the v2-OFF test_audio_byte_identical
   in THIS commit.
2. G1: _QUALITY_COLLAPSE_HINT_V2 (selected by req.story_quality_v2_enabled in
   _quality_reroll_hint) + v2-gated line_quality_defect_score (len(flags) + 2*is_truncated
   + (1 if _hard_clauses>3)) + words_per_beat_range threaded into the FIRST-PASS LineRequest
   (OTR_LedgerScriptWriter ~L4235) AND _otr_reroll L366 AND story_quality_scan.py L387, all
   via derive_one_breath_cap; v2-gated meta stamp.
3. S2: compose_news_coda(*, story_quality_v2_enabled=False, arc_shape="") (keyword-only);
   local copy of _NEWS_CODA_SYSTEM + premise->bridge examples (v2 only); arc_shape-keyed
   curated template pool selected by sha256(cast_seed), each validated by
   validate_news_coda_bridge; zero valid -> fall back to (KEEP) NEWS_CODA_POOL.
4. S3: score on the SHIPPED TEXT (use_exchange empties compose_flags!) via
   verify_and_repair_line + the live _episode_entity_policy; ONE total order
   score=10*grounding_failed+3*hard_leak+2*trunc+2*run_on+1*roster_caps (lower wins,
   original on tie). roster_caps = ALL-CAPS cast-FULL-NAME; a MID-CLAUSE hit =
   needs_recompose (reroll), NEVER an in-place strip; only leading/trailing vocative scrubs.
5. S4: find_cliche_phrase + exact-span replacement BEFORE every quality-gate return path
   (kept-reroll AND kept-original) + curated safe-replacement map with case-match; respect
   the single _quality_repair_attempted guard; else accept second-best + stamp.
6. S5: scan two-principals = top-2 by dialogue-line count (wants are verb phrases -- no
   name parse); update test_story_quality_scan_r2 expected register_overlap values.

OPTIONAL G1b (I'm open to it -- fold in after G1 if quick): a writer-fill sub-lever
(strengthen the per-beat length instruction in the composer prompt, v2-gated) so ONE render
measures BOTH gate-decompression AND writer-fill. Default-off.

ACCEPTANCE: the golden test passes (not is_truncated; flag_one_breath at the budget cap ==
False; word_count within budget); length_ratio rises from ~0.5 toward ~0.7+ WITHOUT padding;
gate counters (anchor_stuffing etc.) do NOT regress. AFTER G1 lands, render ONE local + ONE
frontier (grok) leg and REPORT the voiced-word counts -- that's the length experiment that
decides any follow-up (writer-fill vs the structural ceiling).

HARD RULES: CPU/content only -- NO workflow JSON, NO GPU node/widget change; reuse the
existing reroll loop (run_targeted_reroll -> compose_line is ONE path, so a fix lands once);
additive meta keys only; UTF-8 no BOM; SFW; single resident heavy <=14.5GB; determinism
(seed-keyed; first-pass cap == reroll cap == scan cap); LOUD fallbacks; commit AND push per
green chunk to v2.0-alpha ONLY; prod/main + tags GATED. Update GO_FORWARD_PLAN.md + the
otr-build-tracker as you go. Box: reset SELECTIVELY (CIM by CommandLine, never a blanket
python kill) before any headless run -- a :8000 server (FLOOR + OpenRouter) may still be
resident from the enrichment renders; the operator's Desktop :8001 is separate.
```
