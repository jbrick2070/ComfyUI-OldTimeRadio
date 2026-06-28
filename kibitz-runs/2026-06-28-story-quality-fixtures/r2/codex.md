VERDICT: no. The fixture plan is still ambiguous enough that C1 can either fail immediately or pass without locking the G1 regression.

MUST-FIX BEFORE BUILD:
1. [Candidate readings / Acceptance] Pick C explicitly and define the fixture schema. A is not implementable as written because the corrected target lines are not specified; B is green but does not test decompression. Concrete fix: each fixture row must contain `episode_id`, `beat_id`, `raw_text`, `corrected_text`, `words_per_beat_range`, `source_target_words`, and `expected_raw_defects`. Assert raw defects on `raw_text`, and assert the positive bar only on `corrected_text`. Source rows are visible in `_tmp_fixture_dump.json` lines 137, 237, 473, 607, 804, 919.

2. [Acceptance] `flag_one_breath(text, max_words=cap)` is not just a word-cap check. It still hard-fails lines over 22 words with `>=3` clause markers/conjunctions via `_CLAUSE_MARK_RE` and `_CLAUSE_CONJ` in `nodes/_otr_line_hygiene.py:880`, `nodes/_otr_line_hygiene.py:881`, `nodes/_otr_line_hygiene.py:887`, `nodes/_otr_line_hygiene.py:906`, `nodes/_otr_line_hygiene.py:907`. This contradicts “`_hard_clauses>3` stays ONLY a tie-break term.” Concrete fix: either change all budget-cap callers to pass a non-hard clause threshold for v2, or constrain the corrected fixture lines to stay below the existing soft-clause tripwire and state that constraint.

3. [Grounded data / Acceptance] Do not infer `budget_hi` from ledger `target_words`. The source dump has `meta_words_per_beat_range: null` in `_tmp_fixture_dump.json:6`, `:334`, `:676`, while plancks rows show `target_words: 53` at `_tmp_fixture_dump.json:843`, `:857`, `:875`, `:893`, `:912`. If the cap for plancks is 50, a 51-53 word “within target_words” corrected line will fail `flag_one_breath(... max_words=50)`. Concrete fix: fixture rows must store explicit `words_per_beat_range`; define `budget_hi` as that range’s hi/cap, not the per-row `target_words`.

4. [New C1 helpers / Build sequence] Place helper APIs in one real import path. `scripts/story_quality_scan.py` imports detectors from `nodes._otr_line_hygiene` at `scripts/story_quality_scan.py:71`, while `_otr_line_composer.py` imports hygiene helpers at `nodes/_otr_line_composer.py:39`. Concrete fix: add `derive_one_breath_cap`, `_hard_clauses`, and `find_cliche_phrase` to `nodes/_otr_line_hygiene.py`, then import them into composer/scan. Do not define `_hard_clauses` only in composer if the golden test and scanner need it.

5. [G1 dynamic cap / scan] The scan path will silently keep reporting legacy violations unless it is wired to the new cap. Current scan counts `one_breath` with `flag_one_breath(ln.get("text"))` at `scripts/story_quality_scan.py:387`, no range passed. Concrete fix: scanner must read `words_per_beat_range` from ledger meta/line fixture when present and call `derive_one_breath_cap`; absent range falls back to 28.

SHOULD-FIX:
1. [New C1 helpers] `_hard_clauses` as specified does not match the current one-breath clause counter. Current `flag_one_breath` counts comma/semicolon plus a broad conjunction set including `then`, `while`, `because`, `which`, `that`, `as`, etc.; the plan says comma/semicolon/colon plus FANBOYS. Concrete fix: document `_hard_clauses` as an independent scoring feature and add direct unit tests for colon, `because`, and `that`.

2. [Candidate C] Prevent tautological fixture generation. The corrected line must be a checked-in literal, not produced by composer, repair code, or the helper under test. Otherwise the test can pass after a regression because expected text moved with the implementation.

3. [Raw defect assertions] Add exact cliche-span expectations for dance rows. `flag_cliche` currently returns only `(bool, reason)` at `nodes/_otr_line_hygiene.py:666`; `find_cliche_phrase` must return exact spans like `Over my dead body` and `Not on my watch`, not parse the existing reason string.

OPTIONAL / NICE-TO-HAVE:
Include `source_ledger_path` and `source_line_index` in each fixture row for auditability, but keep the test data minimal rather than copying full production ledgers.

CUT THESE (over-engineering):
1. Full copied golden ledgers under `tests/fixtures/`. They add churn and make reviews noisy. A six-row JSON/JSONL fixture with raw and corrected text is enough for this C1 regression lock.

2. Standalone A or standalone B. A lacks exact expected target text; B only characterizes known failures and can pass while G1 decompression remains broken.