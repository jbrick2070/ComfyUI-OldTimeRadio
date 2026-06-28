VERDICT: yes-with-fixes. The plan is directionally buildable, but several fixes lack concrete interfaces and would either mis-score rerolls or drift first-pass/reroll behavior.

MUST-FIX BEFORE BUILD:
1. [G1] Dynamic one-breath cap is underspecified and will desync runtime vs metrics. `flag_one_breath(text, *, max_words=28, max_clause_markers=3)` supports a cap, but `_quality_flags_for_line` calls it with defaults and `story_quality_scan.py` also calls defaults, so changing only composer behavior makes scan output lie. Fix: add a typed `LineRequest.words_per_beat_range: tuple[int, int] = (0, 0)`, stamp `meta["words_per_beat_range"]` from `episode_budget.words_per_beat_range`, reconstruct it in `_otr_reroll.build_reroll_line_request`, compute `max_words = min(max(eff_hi, 28), 60)`, and update `story_quality_scan.py` to use the same cap. See `nodes/_otr_line_hygiene.py:887`, `nodes/_otr_line_composer.py:2319`, `nodes/OTR_LedgerScriptWriter.py:3024`, `nodes/_otr_reroll.py:366`, `scripts/story_quality_scan.py:387`.

2. [G1] The proposed grammaticality term references `is_truncated` but `_otr_line_composer.py` does not import it, and “>3-hard-clause run-on” has no named helper/API. Fix: import `is_truncated` from `_otr_line_hygiene`, define a pure `line_quality_defect_score(text, req) -> tuple[int, tuple[str,...]]` or equivalent, and use it for both original and reroll instead of only `len(_quality_flags_for_line(...))`. See `nodes/_otr_line_hygiene.py:1038`, `nodes/_otr_line_composer.py:40`, `nodes/_otr_line_composer.py:2502`.

3. [S3] “No hard leak/grammar flags” is not an implementable accept criterion as written. Body-gate reroll currently accepts solely on `validate_composed_grounding`; leak-floor output lives in `compose_flags` and verification result internals, while grammar/cliche flags are in `_quality_flags_for_line`. Fix: define the exact hard-fail predicate, e.g. reject if `compose_flags` contains `leak_floor:malformed_quote` / `leak_floor:banned_source_entity` / `quality_residual:<hard-code>` or if `_quality_flags_for_line(text, line_req)` contains selected hard codes, then use one deterministic score for original vs reroll. See `nodes/OTR_LedgerScriptWriter.py:4506`, `nodes/OTR_LedgerScriptWriter.py:4528`, `nodes/_otr_line_composer.py:2299`, `nodes/_otr_line_hygiene.py:1333`.

4. [S2] Adding `story_quality_v2_enabled` to `compose_news_coda` will break current callers/tests if it is required. Existing tests call `compose_news_coda(...)` without that param. Fix: make it keyword-only with default `False`, preserve byte-identical behavior when false, and add a writer call-site pass-through from `meta["story_quality_v2_enabled"]`. See `nodes/_otr_line_composer.py:3278`, `nodes/OTR_LedgerScriptWriter.py:4768`, `tests/test_announcer_kill2_c3.py:42`.

5. [S2 / 0] The plan says v2-off must be byte-identical, but coda currently runs under `_style_grammar_on`; style grammar can be enabled independently. Fix: test the specific matrix `OTR_ENABLE_STYLE_GRAMMAR=1` + `OTR_STORY_QUALITY_V2=0` and assert the old coda prompt/fallback path is unchanged. See `nodes/OTR_LedgerScriptWriter.py:4768`, `nodes/_otr_config.py:67`, `nodes/_otr_config.py:156`.

6. [S1] The “zero seed anchors in last N character lines” detector has no defined data source or target line selection. “Seed anchors” could mean `grounded_nouns`, `central_object`, `canon_header` anchors, title tokens, or premise nouns, and the plan does not say which line(s) get rerolled when a window fails. Fix: define `seed_anchor_set` source, window size, character-only filtering, pass/fail function, and reroll target policy before implementation. Relevant existing sources: `LineRequest.grounded_nouns` at `nodes/_otr_line_composer.py:913`, `meta["central_object"]` use at `nodes/OTR_LedgerScriptWriter.py:4089`, dramatic frame stamping at `nodes/OTR_LedgerScriptWriter.py:4222`.

SHOULD-FIX:
1. [S4] “Targeted reroll hint + a 2nd attempt” conflicts with the current single quality-reroll guard. `compose_line` has `_quality_repair_attempted` and keeps recursion capped. Fix: either use the existing one reroll and deterministic replacement map after failure, or add a cliche-specific bounded loop outside the generic quality gate. See `nodes/_otr_line_composer.py:2375`, `nodes/_otr_line_composer.py:2471`.

2. [S4] The safe-replacement map needs exact matching semantics. `flag_cliche` returns only a reason string, not the matched phrase as structured data. Fix: return `(flagged, reason, phrase)` or add `find_cliche_phrase(text) -> str`, then map only exact phrase spans. See `nodes/_otr_line_hygiene.py:666`.

3. [S3] Roster-vocative expansion says “embedded ALL-CAPS roster FULL-NAME anywhere,” but current scrub only handles leading/trailing vocative positions. Fix: add tests for mid-line patterns and avoid removing acronyms by matching only episode cast full names. See `nodes/_otr_line_hygiene.py:1231`.

4. [G1] “Golden-ledger before/after set” is not named. Fix: list the fixture ledgers or add a test harness path; otherwise implementors can only run broad scans. verify: exact golden-ledger corpus location.

OPTIONAL / NICE-TO-HAVE:
- [S2] Keep `NEWS_CODA_POOL` as legacy false-path data and add the premise-template fallback beside it; deleting/replacing it risks breaking existing deterministic fallback tests. See `tests/test_announcer_kill2_c3.py:67`.

CUT THESE (over-engineering):
1. [S5] Cut adding a new runtime `register_overlap` counter. `scripts/story_quality_scan.py` already emits `register_overlap_ratio`, and tests already cover it. Adding ledger/runtime metadata buys little and risks byte drift. See `scripts/story_quality_scan.py:456`, `tests/test_story_quality_scan_r2.py:179`.

2. [S1] Defer the off-premise window reroll from this build unless exact anchors and target policy are defined. It is weak-local/intermittent per the plan, and G1/S2/S3/S4 already touch the same reroll path.