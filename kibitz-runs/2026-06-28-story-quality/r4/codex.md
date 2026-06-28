VERDICT: yes-with-fixes. The plan is close, but S3 still has build-blocking ambiguity/gaps, and the golden-ledger acceptance corpus is still not concretely locatable.

MUST-FIX BEFORE BUILD:
1. [S3] The accept rule is contradictory: “accept iff grounding passes” conflicts with “when both imperfect, score lower wins,” and the score omits grounding failure. Current body-gate only ships a reroll when `_bg_res_ok` is true (`nodes/OTR_LedgerScriptWriter.py:4528-4534`). Fix: define one total ordering for original vs reroll, e.g. include `grounding_failed` as the highest-weight score term, or explicitly preserve “reroll must pass grounding” and use the defect score only among grounding-passing rerolls.

2. [S3] `hard_leak = any(f in compose_flags...)` misses the real workflow’s exchange path. The workflow has `use_exchange` enabled (`workflows/otr_scifi_16gb_full.json:1`), and exchange text sets `beat_compose_flags = ()` without running `compose_line` hygiene flags (`nodes/OTR_LedgerScriptWriter.py:4431-4433`). Fix: compute hard leaks/roster caps from the shipped text for both original and reroll using the same deterministic hygiene checks, not only existing `compose_flags`.

3. [G1 Acceptance] The “golden” fixtures are still not concrete. The plan names `plancks`, `ledger_ink`, and `dance`, but no matching fixture files are visible under the repo paths searched; `tests/golden/` currently contains cast baseline artifacts, not these ledgers (`tests/golden/cast_pool_baseline.json`, `tests/golden/capture_cast_baseline.py`). Fix: add checked-in fixture paths or specify the exact existing ledger files the new `tests/test_story_quality_golden.py` must load.

SHOULD-FIX:
1. [S5] “parse the name prefix” from `character_a_wants` / `character_b_wants` is not grounded in the schema: those fields are documented as verb phrases (`nodes/_otr_dramatic_state.py:143-152`), and tests use values like “hide the falsified ledger...” (`tests/test_dramatic_state.py:40`). Fix: use top-2 by dialogue-line count, or verify a separate source that actually maps wants to speakers.

2. [S2 / Verify] The v2-off coda regression matrix should be explicit because coda currently runs under `_style_grammar_on` (`nodes/OTR_LedgerScriptWriter.py:4768`) and `compose_news_coda` is the function being changed (`nodes/_otr_line_composer.py:3278`). Fix: add a verify item for `OTR_ENABLE_STYLE_GRAMMAR=1` + `OTR_STORY_QUALITY_V2=0` proving old coda prompt/fallback behavior remains unchanged.

3. [S2] “arc_shape-keyed CURATED template pool” is still content-free. That is buildable but not deterministic across implementors. Fix: either list the template strings in the plan or require the build commit to add tests asserting the exact pool validates through `validate_news_coda_bridge` (`nodes/_otr_line_composer.py:3239`).

OPTIONAL / NICE-TO-HAVE:
- [G1.2] Make `_hard_clauses` public enough for tests, or test only through `line_quality_defect_score`.

CUT THESE:
1. [S5] Cut “parse speakers named in wants.” It is safe to cut because the real wants fields are verb phrases, and top-2 dialogue-line count already gives a deterministic scan-only principal set.

VERIFY-AT-BUILD checklist:
1. Exact golden-ledger fixture paths for `plancks`, `ledger_ink`, and `dance`; `tests/test_story_quality_golden.py` must load those exact files.
2. First-pass cap == reroll cap == scan cap through one shared `derive_one_breath_cap` helper.
3. `OTR_STORY_QUALITY_V2=0` byte-identical per sub-flag, including no `meta["words_per_beat_range"]`.
4. Explicit coda matrix: `OTR_ENABLE_STYLE_GRAMMAR=1` + `OTR_STORY_QUALITY_V2=0` remains legacy behavior.
5. Golden assertions cover `not is_truncated`, `_hard_clauses <= 3`, and word count within the episode budget.
6. [ASSUMPTION] Pre-existing EntityPolicy/speaker_gender reroll gaps remain out of scope; verify no new S3 logic depends on those being reconstructed.