# OTR Story-Quality Update Plan -- r1 hardened (arc/creative)

Panel: Codex `gpt-5.5`@high + Antigravity `gemini-3.5-pro`; both VERDICT=no -- the
draft misstated existing architecture (S1 binding, S5 threading, G1 keep-better all
ALREADY EXIST). Claude grounded every claim vs the real Windows files; survivors
folded below. The defect SET is right; several FIXES were reframed from "add X" to
"fix/enforce existing X". Judgment log at the end.

## 0. Scope / invariants (+ 2 hard additions from r1 grounding)
- Lane: writer/composer/critic/dramatic-state CONTENT only. CPU. NO workflow JSON,
  NO GPU, NO node INPUT_TYPES/widget change.
- **(NEW, grounded) Gate every shared-function change on `story_quality_v2_enabled`
  (or a sub-flag) -- never modify a shared composer/coda fn globally.**
  `compose_news_coda` runs under `_style_grammar_on` (OTR_LedgerScriptWriter L4768),
  which can be ON while v2 is OFF -> a global edit breaks byte-identity. Flag-OFF
  byte-identical, proven by `test_audio_byte_identical`.
- **(NEW, grounded) First-pass == reroll determinism.** Any value a gate reads in
  `compose_line` MUST also be reconstructed in the targeted-reroll path
  (`_otr_reroll.py` L366 rebuilds `LineRequest` from `meta`). New inputs ride
  ADDITIVE `meta` keys + a `LineRequest` field, following the existing
  `story_quality_v2_enabled` / `line_dramatic_frame` pattern (both already do this).
- Reuse the existing reroll loop; no new engine. No ledger-schema change beyond
  additive freeform `meta` keys + `compose_flags`. UTF-8 no BOM; SFW; full suite +
  Bug Bible + B7 per chunk; commit+push per green chunk to v2.0-alpha; prod/main GATED.

## G1 (LEAD) -- the line-compression gates over-correct (length + craft)
Reframed: the keep-better re-score ALREADY EXISTS (`_otr_line_composer.compose_line`
L2495-2528: scores both drafts via `_quality_flags_for_line`, keeps fewer-flag,
stamps `quality_reroll_degraded`) and ALREADY covers one_breath+anchor (L2287-2321).
Do NOT "add keep-better". The three GROUNDED root causes:
1. **The reroll hint forces compression.** `_QUALITY_COLLAPSE_HINT` (L2293) =
   *"Rewrite as one spoken beat under ~20 words, using at most one concrete detail"*
   -> grok's rich lines become 20-word noun-salad. **FIX:** rewrite the hint to
   "rephrase as natural spoken dialogue -- split into two short sentences if needed;
   keep the specifics; drop the listing/cramming; do not pad." No hard ~20-word target.
2. **"Better" = flag-count only.** `len(_after_flags) < len(_q_flags)` (L2503) -> a
   20-word fragment (0 flags) beats a clean 35-word line (1 one_breath flag).
   **FIX:** add a grammaticality term -- a draft that is a fragment / mid-clause cut
   (`is_truncated`) or a >3-hard-clause run-on is NEVER "better" than a clean draft,
   even with fewer quality flags. Keep deterministic.
3. **Static one_breath cap.** `flag_one_breath(max_words=28, soft 22)`. **FIX
   (gated, optional):** scale `max_words` by the budget's `words_per_beat_range`
   eff_hi -> `clamp(28, eff_hi, 60)` for a CLEAN line; KEEP the >3-clause run-on
   guard independent of word count. REQUIRES `words_per_beat_range` propagated to
   `compose_line` AND the reroll reconstruction (meta L3024 + `LineRequest` L735 +
   `_otr_reroll` L366) per the determinism invariant.
Length is a SIDE EFFECT -- fix the hint+metric and lines fill toward the budget;
never pad; structural ceiling (~1363w/3acts) unchanged. **Measure (Codex):** a
golden-ledger before/after set on the enrichment failures (final-text
grammaticality + `length_ratio` + per-line word-count median), NOT just counter
movement -- counters can read "clean" while scripts are bad.

## S2 -- coda bridge floor (weak-local; grok PASSES it)
Verify resolved: grok-720 got a real `news_coda_bridge` -> the 62% fallback is a
WEAK-LOCAL bridge-gen gap, not a strict validator. **FIX (gated):** add
`story_quality_v2_enabled` param to `compose_news_coda` (L3278), pass it from the
writer (L4770), gate ALL S2 changes behind it: (a) 1-2 in-context premise->bridge
examples in `_NEWS_CODA_SYSTEM`; (b) attempts 2->3; (c) replace the generic
`NEWS_CODA_POOL` with a DETERMINISTIC premise-template strategy (pick a setup phrase
from the premise via the existing `sha256(cast_seed)` rotation, validated by
`validate_news_coda_bridge`) -- NOT raw noun extraction (Codex/agy: that yields
awkward fragments). News FACT stays appended verbatim. Measure: `news_coda_fallback`
+ `news_coda_generic_bridge` down.

## S3 -- body-gate reroll: accept-criteria + roster-caps
`body_gate_reroll` accepts `_bg_res.text` on grounding-validate alone
(OTR_LedgerScriptWriter L4528), with no "no hard leak/grammar" bar (Codex). **FIX:**
accept = grounding pass AND no hard leak/grammar flags; when both drafts are
imperfect, a deterministic defect-count score (grammar + punctuation + leak-floor
flags) picks the better (agy), original on tie. Extend the roster-vocative leak rule
to strip an embedded ALL-CAPS roster FULL-NAME anywhere in the line, scoped to the
EPISODE CAST LIST (never any caps token -- NASA/UCLA safe). Measure:
`roster_name_caps_leak_lines`, `run_on_lines`.

## S4 -- cliche replacement (no naive drop)
Naive phrase-drop corrupts syntax ("Over my dead body, Lemmy." -> ", Lemmy."; all
three reviewers converge). **FIX:** targeted reroll hint ("avoid the exact phrase
'X'; say it plainly") + a 2nd attempt; if still cliche, a curated safe-replacement
map (e.g. "not on my watch"->"not while I'm here") OR accept second-best + stamp
`cliche_shipped_after_reroll` (measurement) -- never drop into a fragment.

## S1 -- seed fidelity: binding EXISTS; strengthen + window-level detector
Reframed: the dramatic frame EXISTS (compose_line L1416-1504: DRAMATIC QUESTION +
Objective/Obstacle/Subtext/Tension + beat-function + the L2 deflection contract that
withholds the objective on high-tension+subtext beats to stop command-shouting) and
survives into rerolls (`line_dramatic_frame` persisted + reconstructed, _otr_reroll
L355). So S1 is NOT "add binding" -- the L2 deflection lever UNDER-PERFORMS for weak
local models (dance_of_keys/artifacts_breath still command-shout). **FIX:** an
off-premise detector at the WINDOW level (last N character lines contain zero seed
anchors), NOT per-line (agy: per-line false-fires on "Yes."/"What?" and exhausts the
reroll budget) -- or limit to high-tension/objective-critical beats; reroll a
zero-anchor window with a seed-anchored hint. Lower priority (weak-local +
intermittent; grok stays on-premise).

## S5 -- voices: threading EXISTS; cut the line-level reroll
Reframed: `speech_signature` IS threaded (build_voice_card `speaks:` L1129 + the
system prompt), and NO generated-text register detector exists
(`speech_signature_overlap` checks cast strings only -- agy). **FIX:** CUT the
line-level register-divergence reroll (ungrounded -- would need an LLM critic or
over-constrain). Keep the existing prompt directive; add a MEASUREMENT-ONLY
`register_overlap` counter over the two principals' line sets (no reroll). An
LLM-critic register pass is a separate future lane.

## S6 -- CUT (Codex + agy + Claude). Phantom is detect-only and acronyms are common
in sci-fi (false positives); fold any invented-entity concern into S1's seed-anchor
window. (Also: the `phantom_name:Atlantic`/`SAT` flags are false positives -- a
cast/setting allowlist would help but is not worth a reroll loop.)

## Build order: **G1 -> S2 -> S3 -> S4 -> S1 -> S5(measurement only)**; S6 cut.
G1 + S3 + S4 + S5 all touch the same reroll seam, so G1-first de-risks them.

## §6 operator-questions -> R2 DEFAULTS (Codex): one_breath cap default
`clamp(28, eff_hi, 60)`; off-premise default window-level; coda fallback default
premise-template; cliche default safe-replacement-map; S5 default measurement-only.
ESCALATE only: raise `BEAT_WORD_HARD_MAX`/the structural ceiling? (DEFERRED -- out of
this pass; >1363w keeps erroring, acceptable).

## Judgment log (r1)
- ACCEPTED (grounded KEEP): S1 binding-exists reframe (Codex, code-verified);
  G1 keep-better-exists + COLLAPSE_HINT-forces-20w + flag-count-metric (Codex,
  verified L2293/L2503); S5 threading-exists + no-generated-register-detector ->
  cut line-reroll (Codex+agy); S2 needs v2 param (Codex+agy, verified L4768); G1
  budget propagation to reroll-reconstruction (agy, verified LineRequest lacks it +
  _otr_reroll L366); S4 no-naive-drop (all 3); S1 window-level detector (agy);
  S3 defect-count accept metric (Codex+agy); S6 CUT (all 3); golden-ledger
  acceptance (Codex); §6->defaults (Codex).
- REJECTED / none material -- every agent claim grounded true this round (the panel
  read the code accurately; no hallucinations to discard at r1).
- VERIFY-AT-BUILD: (a) `test_audio_byte_identical` green with each new sub-flag OFF;
  (b) first-pass cap == reroll cap once `words_per_beat_range` is propagated;
  (c) golden-ledger final-text quality improves, not just counters.
