# OTR Story Architecture -- Hardened Plan (R1 synthesis: arc/creative)

**Mission:** get OTR stories to A+ (or as close as the model ceiling allows). The quality apparatus
(critic / reroll / escalation / grouped-exchange) already exists and is wired -- see grounding in the
REV 2 kickoff. The lever is PREMISE-level variety + closing the critic->re-plan loop, NOT new gates.

**Root cause (triangulated):** beat-planner / premise sameness -- every premise collapses into a
console standoff; climax off-stage; announcer narrates the outcome.

**MVP boundary (R1 convergence):** one campaign delivers a GATE probe + 3 build candidates. Everything
else is explicitly deferred (S "Deferred"). No second architecture project rides along.

---

## Candidate 0 -- GATE: local-ceiling probe (do FIRST; cheap; decides the campaign)

Before building the pitch room, answer empirically: can the LOCAL model fill an A-pool?

- Run ~10 pitch-room sets on the LOCAL model -> compose -> `grade_story`. Record the best grade.
- If some reach >= 75 (B+): proceed local. If NONE clear 75: the local model cannot taste/diverge to
  A; STOP and put the frontier-lane-vs-accept-B decision to the operator before building Candidate 1.
- If local-only is chosen, RENAME success criteria away from "A+": target = sameness reduction +
  median/keep-best lift, not A.

Rationale: the whole campaign's value depends on the pool containing a good candidate (kickoff S5).
Resolve it with a $0 local experiment instead of assuming. (DeepSeek SF4 + GPT MF3, grounded.)

## Candidate 1 -- PRIMARY: pitch-room premise divergence + frontier-backed greenlight

Attacks the root cause directly. Nothing in the repo diverges at the premise level (best-of-N
`score_outline` diverges over outline STRUCTURES by a METRIC -- it cannot escape sameness because all
N answer the same premise).

- **Force divergence (do not just ask for it).** Seed each of 3 pitches with a DIFFERENT conflict-type
  drawn from the existing palette in `_otr_story_quality_l12.py`, a different protagonist archetype,
  and a different setting class. The local model left to "pitch 3" returns 3 console standoffs
  (Gemini + DeepSeek + my anchor converge here).
- **`PitchCandidate[]` schema** (out of the pitch step): `logline`, `protagonist`, `antagonist_or_pressure`,
  `genre_mode`, `emotional_core`, `theme_sentence`, `final_20_seconds`, `conflict_type` (the seed),
  `why_different`. (Folds in "theme & ending first" as FIELDS, not a separate step -- GPT SF1.)
- **Greenlight = explicit rubric, not vibes.** Score each candidate on surprise/freshness, human want,
  audio-stageability, ending promise, OTR fit, and console-standoff-collapse risk; force an ordinal
  ranking; REQUIRE rejecting >= 1 candidate for sameness; quote evidence per axis. Output
  `GreenlightDecision`: `selected_id`, `taste_rationale`, `risk_flags`, rewritten `script_brief`,
  `failed_premise_fingerprints`.
- **The greenlight node defaults to the FRONTIER lane even if drafting stays local** (a B-model cannot
  taste an A -- Gemini SF1 + GPT MF6). Gated by Candidate 0's outcome.
- **Handoff:** the rewritten `script_brief` feeds the EXISTING `_otr_outline.py` -> `score_outline`
  best-of-N. No change to the outline schema (verify-at-build: brief field compatibility).

## Candidate 2 -- close the critic -> re-plan loop, in TWO tiers (highest-value reuse)

The critic already emits `arc_verdict` / `flat_lines` (5B); today a STRUCTURAL verdict escalates to a
same-seed whole-episode regen (`_otr_reroll_escalation.decide_escalation_scope`, EPISODE branch),
which redraws but keeps the planner's structural BIAS -> same shape class. Split it:

- **Tier 1 -- staging failure** (flat middle, off-stage climax, weak resolution): the PREMISE is fine,
  the OUTLINE is not. Route to a divergent RE-OUTLINE on the SAME premise (re-run outline best-of-N
  with the failing axis as a penalty). (Gemini MF1 -- corrects the kickoff's single-tier route.)
- **Tier 2 -- premise unsalvageable** (`premise_clarity` / console-standoff collapse): route back to
  the PITCH ROOM (Candidate 1) with the critic report injected as a "showrunner note" and
  `failed_premise_fingerprints` excluded so the next pitch confronts the named weakness instead of
  blind-resetting (DeepSeek MF1).
- Cap divergent re-plans SEPARATELY from line rerolls; on exhaustion, keep-best per existing policy.

## Candidate 3 -- flip `use_exchange` (B2): quick win, config-only

Grouped 2-3 beat exchange is built, tested (`test_compose_exchange.py`), wired, default OFF.

- No new code. Run a GPU N=3 (VRAM <= 14.5 GB, zero slot drift, defined harness + pass/fail).
- If pass: config-only change to the canonical workflow default/link. If fail: stay OFF. (GPT SF4 --
  this is a post-validation config PR, not part of the dark build.)

## Candidate 4 -- supporting: outline-level staging critic (low cost; may fold into C2 Tier 1)

Some named symptoms are NOT premise problems and premise divergence will not fix them (GPT MF7).

- Pre-composition outline checks: every beat must TURN (power/status/knowledge/emotion change); climax
  staged ON-MIC. Implement as a `score_outline` penalty / outline-critic, BEFORE generation.
- Splits the success hypotheses: Candidate 1 fixes premise sameness; Candidate 4 fixes staging.

---

## Deferred (explicitly OUT of this campaign -- hold to 3-4 candidates)

- **B3/B4 whole-scene/whole-episode prose -> ledger parser.** Separate research spike with its own
  SPEC + fixtures. The danger is NOT a crash but SILENT mis-attribution (a line assigned to the wrong
  REAL speaker passes the cast audit, renders in the wrong voice, no error). Spike acceptance gate:
  deterministic attribution (speaker-prefixed draft, or re-derive beats and DIFF against the outline;
  any unmatched line = loud halt). Unanimous panel + anchor: cut from the first campaign.
- Multi-seed "3 headlines" assignment desk -- keep the single existing `script_brief`; generate 3
  divergent TAKES from it (GPT MF5/CUT1).
- Character interviews; listener-taste critic augmentation; distinct-character-voices -- defer (not
  the root cause).
- External-repo survey -- keep ONLY Open-Theatre prompt-mining for the pitch room; no framework import;
  no reference-library in the SPEC.
- Refine-loop hardening -- keep-best masks non-monotonicity; first DIAGNOSE grader-noise vs
  composer-regression (log grade variance on identical text) before any fix.

## Verify-at-build (carried to R2/R3)

- `_otr_outline` / `score_outline` ingest a richer rewritten `script_brief` without schema mismatch.
- The escalation EPISODE branch can accept a new premise (Tier 2) / re-outline (Tier 1) without
  breaking the freeze cascade.
- Degree of planner determinism: confirm a full regen redraws but keeps structural bias (tighten the
  kickoff's "same shape" wording to "same shape CLASS").
- `use_exchange` N=3 harness + pass/fail criteria.
