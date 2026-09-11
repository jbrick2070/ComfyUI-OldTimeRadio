VERDICT: yes-with-fixes. The core OldTimeRadio boundary correction is mathematically sound and verified by 72 tests, but survival-guide test docstring syntax leaves promoted BUG-11.62 statically unregistered in regression test coverage.

MUST-FIX BEFORE BUILD:
1. [survival-guide tests/bug_bible_regression.py:L1595] Defect: The test method docstring uses `"""BUG-11.48/11.62/12.58: current owners need executable coverage."""`. The static test coverage collector `_collect_test_coverage()` at L913-924 extracts bug IDs exclusively via `re.finditer(r'BUG-(\d+\.\d+)', line)`. Because `11.62` and `12.58` are preceded by `/` instead of `BUG-`, `11.62` is not registered in `test_coverage`, leaving it reported as an uncovered entry in `TestThreeFileContract.test_all_bible_ids_covered_in_tests`.
   Concrete fix: Change line 1595 to repeat the prefix:
   `"""BUG-11.48 / BUG-11.62 / BUG-12.58: current owners need executable coverage."""`

SHOULD-FIX:
1. [survival-guide README.md:L29] Defect: Section 1 of `README.md` at line 29 originally cited `a 338-entry, machine-readable bug bible` while `BUG_BIBLE.yaml` has 339 entries, causing `TestThreeFileContract.test_entry_count_matches_readme` to fail (`assert 339 == 338`) because `_extract_readme_count()` parses `r'(\d+)[- ]entr(?:y|ies)'`. While line 29 was updated in the working tree to `339-entry`, this modification must be verified and committed alongside `BUG_BIBLE.yaml`.
   Concrete fix: Ensure the working tree modification updating line 29 from `338-entry` to `339-entry` is staged and committed with the promotion of 11.62.

OPTIONAL / NICE-TO-HAVE:
1. [OTR tests/test_my_story_runner.py:L851] In `test_zero_boundaries_preserves_all_unused_cue_proposals`, test coverage handles `(3, False, "act_breaks_disabled")` and `(1, True, "unused_surplus")`. Adding a multi-act surplus case `(3, True, 4 proposals -> 2 used, 2 unused_surplus)` directly to this parametrized test would make surplus partition verification explicit in a single test function (currently covered indirectly via `test_missing_metadata_optional_frame_and_numbering_reach_readonly_freeze`).

CUT THESE:
1. None. The diff across both repositories is strictly minimal: one shared 2-line pure helper `_interstitial_count()`, five updated call sites in `nodes/_otr_my_story.py`, targeted parametrized assertions in `tests/test_my_story_runner.py`, and standard Bible promotion artifacts in `comfyui-custom-node-survival-guide`. No extraneous changes or over-engineering detected.

VERIFY-AT-BUILD checklist:
1. [OTR tests/test_my_story_runner.py] Confirm all 72 focused runner tests pass (`pytest tests/test_my_story_runner.py`). Verified: 72 passed in 6.39s.
2. [OTR tests/test_my_story_*.py] Confirm all 6 my_story suite modules pass. Verified: 169 passed in 7.68s.
3. [survival-guide tests/bug_bible_regression.py] Confirm `pytest tests/bug_bible_regression.py -k "test_entry_count_matches_readme or TestBibleIsActuallyParseable or test_otr_my_story"` passes without errors. Verified: all pass when `--pack-dir` points to OTR.
4. [survival-guide tests/bug_bible_regression.py] Confirm `TestThreeFileContract.test_all_bible_ids_covered_in_tests` recognizes `11.62` after applying MUST-FIX 1.
5. [OTR full suite] Run full test suite to confirm zero regressions against the 54-known-failure baseline [ASSUMPTION].
6. [OTR production hardware run] Exercise live 4060 run with 1 act and enabled act breaks to confirm zero interstitial prompts in P0/P1/frame, 0 interstitial cues in assembled ledger, and all proposed cues classified as `unused_surplus` in metadata.

[ASSUMPTION] checklist:
- [ASSUMPTION]: Untracked files in the survival-guide repository (`tests/interface_integrity_check.py`, log files, `reddit_post_v2_1.md`) and in the OTR repository (`diff.txt`, `diff_utf8.txt`) are pre-existing artifacts from concurrent or prior diagnostic sessions, ignored per `input.md` line 15.
- [ASSUMPTION]: The full test suite baseline is 54 known failures as cited in `driver_anchor.md`.
