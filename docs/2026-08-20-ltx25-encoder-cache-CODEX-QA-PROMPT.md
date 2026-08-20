## My request

**Do a read-only QA review of the uncommitted diff on branch `v2.0-alpha` in
`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`, then WRITE
YOUR REVIEW TO
`docs\2026-08-20-ltx25-encoder-cache-CODEX-QA.md` and tell me what you found.**

Read the real files and cite `file:line`. Do not modify any source file (the
review markdown is the one file you should create). **Do not run the full test
suite -- a GPU render is live on this box**; targeted pytest files are fine:

```
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
$env:PYTHONUTF8=1
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests\test_ltx25_encoder_cache.py tests\test_episode_encoder_scope_wiring.py
```

Rank findings by what would actually break, each with a concrete failure
scenario. If a category is clean, say so plainly rather than padding.

---

## Context: what the change does

The LTX 2.5 video lane re-read an **8.86 GiB** Gemma-4 12B Q5 GGUF text encoder
from disk on **every shot** -- measured on a live leg: 13 encoder loads for 13
shot renders, ~63 s each, on top of a 54.2 s CPU encode.

The change adds an **episode-scoped** cache of (a) the loaded CLIP and (b) the
empty negative conditioning, injected through `run_graph`'s pre-existing
`external_results` contract, with the corresponding node dropped from the graph.

Ownership is in the DRIVER, not the engine, and that was the design argument:
the registry builds each adapter once at import and returns the same instance
forever (`engine_registry_base.py:148-155`), so an engine-owned cache would
never be released. Every engine-level hook runs too often to be the release
point -- `free_otr_pipeline_residue` is the per-SHOT preflight,
`teardown`/`unload` are per BEAT. So `render_driver.run_episode` opens the scope
and closes it in a `finally`.

## Files changed

* `nodes/_otr_video_engines/eng_ltx25.py` -- `_encoder_cache_enabled`,
  `_copy_conditioning`, `_encoder_cache_key`, `_cached_clip_is_live`,
  `begin_encoder_scope` / `end_encoder_scope`, the cache branch in
  `render_clip`, and the atomic publish after the graph succeeds.
* `nodes/_otr_video_engines/render_driver.py` -- `_begin_engine_scope` /
  `_end_engine_scope` closures, and a `try/finally` around the shot loop.
  **The loop body was mechanically re-indented by 4 spaces.**
* `tests/test_ltx25_recipe_matches_lab_golden.py` and
  `tests/test_ltx25_video_lane.py` -- stale-comment corrections only.
* NEW `tests/test_ltx25_encoder_cache.py` and
  `tests/test_episode_encoder_scope_wiring.py`.

A copy of the diff is at
`kibitz-runs\2026-08-20-ltx25-encoder-cache\THE_DIFF.patch` -- but review the
REAL FILES; the patch is only a convenience and may lag the tree.

## Hunt these specifically

1. **The re-indent.** ~86 lines of `run_episode`'s loop body moved 4 spaces
   right inside a new `try:`. Verify independently that only `try:` and the
   `finally:` block were ADDED and no line content changed. Then check control
   flow: does any `continue` / `break` / early `return` inside that loop still
   behave, and can the new `finally` mask or reorder the existing inter-engine
   reclaim?

2. **`external` vs `graph` desync in `render_clip`.** `run_graph` raises a NAMED
   error if an id appears in BOTH the graph and `external_results`. Find any
   path where an id is added to `external` but not popped from `graph`, or
   popped but not added.

3. **Binding and lifetime.** Is `key` bound everywhere it is later read
   (including the publish block after the `try/finally`)? Is `pending` always
   defined, including when the graph raises? Is `scope` re-read after any
   branch that reassigns or clears `self._encoder_scope`?

4. **Cache release.** A stale or unusable entry must be RELEASED, not merely
   skipped -- otherwise the old 8.86 GiB CLIP and its replacement are resident
   at once. Verify that holds on every miss path: unresolvable key, changed
   key, failed placement check, and cache disabled mid-process.

5. **Tautological tests.** Would any new test pass with the feature removed?
   Check `test_a_FAILED_episode_still_closes_the_scope` in particular: if the
   hooks were never invoked at all, does it still pass?

6. **What only breaks on the GPU box**, at real render time, that CPU tests
   cannot see. This is the one I most want your eyes on.
