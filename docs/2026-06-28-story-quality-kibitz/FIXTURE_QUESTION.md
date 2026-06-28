# Kibitz: how to build the C1 golden fixtures (story-quality G1)

## The decision (focused)
Commit C1 of the story-quality G1 build (spec: `kibitz-runs/2026-06-28-story-quality/final.md`)
must add `tests/fixtures/` golden ledgers + `tests/test_story_quality_golden.py`, and the
**full suite must be GREEN in the same commit**. The spec contradicts itself on what the
fixtures contain. Resolve it so the golden test is green, faithful to the G1 intent, and a
real regression lock (not a tautology).

## The contradiction (both quotes from the converged spec)
1. Build-commit deliverable (c): *"the golden fixtures = the specific FAILING lines extracted
   to tests/fixtures/ (plancks b03/b10, ledger_ink b04/b13, dance b04/b11)."*
2. Acceptance: *"The golden test asserts, per fixture line: (1) not is_truncated, (2)
   flag_one_breath(text, max_words=derive_one_breath_cap(range))[0] is False at the BUDGET cap,
   (3) budget_lo <= word_count <= budget_hi."*

These cannot both hold: the real failing lines are SHORT (the over-correction compressed them),
so `word_count < budget_lo` -> assertion (3) fails on the raw text.

## Grounded data (read from the real ledgers + episode_budget)
`nodes/_otr_episode_budget.py`: `words_per_beat_range = (base_lo, eff_hi)`; configs range
base_lo in {10,18,20,22,25,28}, eff_hi in {35,40,42,45,50} (capped at BEAT_WORD_HARD_MAX=80).
Per-line `target_words` in the ledgers = the eff_hi region.

The actual extracted failing lines (text / word_count / per-line target_words / compose_flags):
- plancks b03 (i=3): "Storm the helm Quasimodo protects his paradox self to bury the merger
  disclosure." -- wc=13, tgt=50, flags=[one_breath_retry, body_gate_reroll]
- plancks b10 (i=10): "Restore the quantum beacon cells or the loan covenant buries your pension
  with those papers." -- wc=15, tgt=53, flags=[anchor_stuffing_retry, one_breath_retry, body_gate_reroll]
- ledger_ink b04 (i=4): "I need to see the actual records proving what they agreed to when
  CLARISSE GORDON claim this whole arrangement was a miracle" -- wc=21, tgt=30, flags=[body_gate_reroll]
  (note: mid-clause ALL-CAPS roster name + ungrammatical "claim" -- the S3 roster-caps case)
- ledger_ink b13 (i=13): "The ledger isn't the point, VICTOR STENDAHL; the turning of the page
  is what matters now." -- wc=15, tgt=31, flags=[]
- dance b04 (i=4): "Over my dead body, Lemmy. I've got the wrench, remember? And I won't hesitate
  to throw the breakers myself if you force my hand." -- wc=24, tgt=31, flags=[] (cliche "Over my dead body")
- dance b11 (i=11): "Not on my watch, Pim. This old girl's still got some juice left." -- wc=13,
  tgt=32, flags=[] (cliche "Not on my watch")

`derive_one_breath_cap(words_per_beat_range)` (the C1 helper) = `min(max(eff_hi,28),60)` when
eff_hi>0 else 28. So for plancks the cap is 50; ledger_ink/dance ~35-40.

## The new C1 helpers under test
- `derive_one_breath_cap(range)->int` (budget-derived one-breath cap; replaces the hardcoded 28).
- `_hard_clauses(text)->int` (count `,;:` + FANBOYS).
- `find_cliche_phrase(text)->str` (matched cliche span; "" if none).
- `flag_one_breath(text, max_words=cap)` already exists (`nodes/_otr_line_hygiene.py` L887).
- `is_truncated(text)` already exists (L1038).

## Candidate readings (panel: pick one or propose a better third)
- **A. Decompressed target fixtures.** Fixtures = budget-length (~eff_hi-word) CORRECTED versions
  of those exact beats (same cast/nouns/premise), authored by Claude. Test asserts not-truncated +
  one_breath-False at the budget cap (and WOULD flag at the legacy 28 cap -> proves the fix) +
  word_count within band. Green + encodes the G1 target. Risk: hand-authored "good" prose is
  subjective; not real model output.
- **B. Raw lines, characterization test.** Fixtures = the raw failing lines as-is. Test documents
  the DEFECT each line exhibits (wc<budget_lo; cliche span on dance b04/b11; ALL-CAPS roster name on
  ledger_ink b04) + locks the helper CONTRACT (derive_one_breath_cap return value; _hard_clauses
  count; find_cliche_phrase span). The POSITIVE acceptance bar (within budget, decompressed) moves
  to the task-7 post-G1 render. Green, real data, but inverts the spec's stated positive assertions.
- **C. Hybrid / something better?** e.g. each fixture row stores BOTH the raw failing line AND a
  decompressed target line; the test asserts the defect on the raw and the positive bar on the
  target, so one fixture file proves both the before and the after.

## Question for the panel
Which reading gives the most faithful, non-tautological, commit-1-GREEN golden test for G1
(gate-decompression + budget-derived one-breath cap)? Is there a better third option? Flag any
way the chosen approach could silently pass while the real gate regresses.
