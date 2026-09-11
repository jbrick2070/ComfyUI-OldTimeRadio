VERDICT: build-ready as-is? yes-with-fixes. The core music sentinel null parent fix is sound, but the restored LLM-slot tag comment uses invalid syntax that fails CI slot-sweep tests with a regression exit.

MUST-FIX BEFORE BUILD:
1. [lines 26-30] Invalid LLM-slot tag syntax causes CI regression failure.
   - Defect: In nodes/_otr_my_story.py, line 376 restores the LLM-slot comment as:
     `# LLM slot: caller supplies the creative or technical slot for this pass.`
     The slot sweep test in tests/_s28_llm_slot_sweep.py:42 enforces:
     `TAG_RE = re.compile(r"#\s*LLM\s+slot:\s*(creative|technical|per-sub-pass)")`
     Because the word immediately following `# LLM slot:` is `caller` instead of `creative`, `technical`, or `per-sub-pass`, `TAG_RE.search()` returns `None`. Consequently, `find_untagged_call_sites()` flags `nodes/_otr_my_story.py:377` as an untagged `structured_call()`. Running tests/test_llm_slot_sweep.py fails and triggers `KNOWN-FAIL-GUARD` in tests/conftest.py:286-301 with `SystemExit(2)` (REGRESSION), blocking the build gate.
   - Concrete fix: Change nodes/_otr_my_story.py:376 to:
     `    # LLM slot: per-sub-pass -- caller supplies the creative or technical slot for this pass.`

SHOULD-FIX:
1. [lines 18-25] Test in `test_music_cue_duration_reaches_the_beat.py` does not assert `beat_id is None`.
   - Defect: In tests/test_music_cue_duration_reaches_the_beat.py:135-152, `test_my_story_null_parent_music_rows_preserve_timeline_identity()` inserts `_music_sentinel()` rows into `ledger["lines"]` and verifies `extract_beats(ledger)`. However, `extract_beats()` in nodes/otr_shot_lock.py:789 generates downstream beat dicts by keying `beat_id` to `str(ln.get("line_id") or ...)`, completely ignoring `ln.get("beat_id")`. While tests/test_my_story_runner.py:301 checks `by_id[cue["anchor_line_id"]]["beat_id"] is None`, this unit test never directly verifies that the inserted sentinels have `beat_id is None`, creating an indirect test that could pass even if the sentinel beat_id was reverted.
   - Concrete fix: Add an explicit assertion inside `test_my_story_null_parent_music_rows_preserve_timeline_identity()`:
     `assert all(ln.get("beat_id") is None for ln in ledger["lines"] if str(ln.get("speaker_role", "")).startswith("music_"))`

2. [survival-guide / tests/bug_bible_regression.py:85-87] Directory exclusion in survival-guide regression test misses `.claude` worktrees.
   - Defect: In comfyui-custom-node-survival-guide/tests/bug_bible_regression.py:85-87, `py_files()` excludes `("__pycache__", ".git", ".venv", "venv", "tests", "llm_round_robin")`. When running against a repository containing agent worktrees under `.claude/worktrees/`, `os.walk()` collects uncommitted or detached `.py` files inside `.claude`, causing AST and formatting assertion failures during full Bug Bible runs against `--pack-dir`.
   - Concrete fix: In comfyui-custom-node-survival-guide/tests/bug_bible_regression.py:86, add `".claude"` to `excluded_dirs`.

OPTIONAL / NICE-TO-HAVE:
1. Note in nodes/_otr_scifi_news_pro.py:4209 that SciFi news pro's `_music_sentinel()` still stamps `"beat_id": lid`, matching legacy published ledgers, and can adopt `beat_id: None` in a future bank modernization pass if desired.

CUT THESE:
1. None — plan converged. The diff contains only the 1-line sentinel fix, 1-line slot tag, targeted unit test assertions, and Bug Bible documentation/coverage links. No extraneous code or scope creep exists.

VERIFY-AT-BUILD checklist:
1. CI slot-sweep green check: Run `pytest tests/test_llm_slot_sweep.py -k test_every_llm_call_site_has_slot_tag` after applying the Must-Fix to ensure `nodes/_otr_my_story.py` is tagged and no `KNOWN-FAIL-GUARD` regressions occur.
2. Live canonical freeze qualification [ASSUMPTION: requires active GPU environment with Qwen 4B weights]: Execute the real canonical graph (partial execution target 62: validator63 -> writer1 -> freeze62) with the post-fix commit on the 5080/RunPod, and confirm the resulting ledger records `meta.freeze_verdict == "frozen_clean"` with zero beat-reference warnings.
3. Full suite regression baseline: Run the full `pytest` suite and confirm the failure count matches the pre-change baseline (53 pre-existing failures in S4) with 0 new failing node IDs.
4. Survival-guide coverage verification: Run `pytest tests/bug_bible_regression.py -k test_otr_my_story_current_repair_and_parent_identity_coverage --pack-dir="C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"` and confirm 1 passed.
