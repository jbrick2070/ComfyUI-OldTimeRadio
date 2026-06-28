# Kibitz judgment -- C1 golden fixtures (story-quality G1)

Panel: Codex `gpt-5.5`@high (full review, grounded) + Antigravity `gemini-3.5-pro`
(BLOCKED -- zero output in ~24 min; known agy-cli hang, antigravity-cli issue #508,
recurred from 2026-06-27). Claude = code-grounded panelist + sole judge. 1 usable
agent + Claude anchor; decision converged.

## DECISION: Option C (hybrid raw+corrected fixture rows). Not A, not B.
A is under-specified (no target text -> not implementable). B is green but tests
nothing about decompression (can pass while G1 stays broken). C tests BOTH the
defect (on raw) and the G1 target (on corrected) in one fixture file.

## Fixture schema (checked-in literals, one compact JSON -- NOT copied prod ledgers)
`tests/fixtures/golden_story_quality.json`, 6 rows:
`{episode_id, beat_id, raw_text, corrected_text, words_per_beat_range,
source_target_words, expected_cliche_span}`.
(Codex CUT #1 ACCEPTED: a 6-row JSON beats copying 10k-line production ledgers;
deviation from the spec's "golden ledgers" wording, recorded here + in GO_FORWARD.)

## Assertions (test_story_quality_golden.py)
Per row, RAW (the over-correction we are fixing):
- compressed rows (plancks b03/b10, dance b11): `raw_wc < budget_lo` (documents the
  collapse); dance b04/b11: `find_cliche_phrase(raw)` == expected_cliche_span;
  ledger_ink b04: raw contains an ALL-CAPS roster full name mid-clause.
Per row, CORRECTED (the G1 target -- the regression lock):
- `not is_truncated(corrected)`.
- `flag_one_breath(corrected, max_words=derive_one_breath_cap(range))[0] is False`.
- legacy-cap proof: `flag_one_breath(corrected, max_words=28)[0] is True` (so the
  budget cap is demonstrably what permits it -- guards the cap silently reverting to 28).
- `budget_lo <= corrected_wc <= budget_hi` (range = words_per_beat_range; budget_hi
  = derive_one_breath_cap(range), NOT per-line target_words -- Codex MUST-FIX #3).
- direct helper unit tests: derive_one_breath_cap on (0,0)->28, (10,50)->50,
  (28,80)->60-cap, list/tuple/bad-input coercion; _hard_clauses on comma/semicolon/
  colon + FANBOYS; find_cliche_phrase exact spans (Codex SHOULD-FIX #1/#3).

## CRITICAL grounded finding (folds into C2, not just C1) -- Codex MUST-FIX #2 CONFIRMED
`flag_one_breath` (nodes/_otr_line_hygiene.py L887) has a SOFT tripwire: > _ONE_BREATH_
SOFT_WORDS(22) words AND (commas+semicolons + FANBOYS-ish conj) >= max_clause_markers(3)
flags TRUE *regardless of max_words*. And the gate at _otr_line_composer.py L2319 calls
`flag_one_breath(cleaned)` with DEFAULTS (cap 28, clauses 3). Consequence:
- A budget-length (~40-50w) corrected line trips the clause path even at max_words=50,
  so raising ONLY the word cap does NOT let G1 ship fuller lines.
- C1 fix: author corrected_text as 2-3 SHORT declarative sentences (periods, not commas)
  so clause markers stay < 3 AND wc is 30-45 (> 28 so the legacy-cap proof holds, <= budget_hi).
- C2 design note (verify-at-build): the v2 one-breath gate MUST thread BOTH
  derive_one_breath_cap(range) for max_words AND a relaxed max_clause_markers (scale with
  the cap), else fuller multi-clause lines keep getting rerolled -- defeating G1. The spec's
  derive_one_breath_cap (max_words only) is necessary but NOT sufficient.

## Helper placement (Codex MUST-FIX #4 -- matches spec)
All three helpers in `nodes/_otr_line_hygiene.py` (the one import path composer L39 +
scan L71 already use). find_cliche_phrase returns `m.group(0)` (exact span), not parsed reason.

## Anti-tautology (Codex SHOULD-FIX #2 ACCEPTED)
corrected_text is a hand-authored checked-in literal -- never produced by composer/repair
or the helper under test.

Agent calls this pass: 2 attempted (codex OK, agy blocked). Single focused r2 pass
(operator asked "/kibitz for ideas", not the full 4-round arc).
