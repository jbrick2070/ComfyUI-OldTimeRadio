<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan misses active legacy fallback paths in the code that will immediately trigger the new fail-LOUD guards.

MUST-FIX BEFORE BUILD:
1. [Section 1/2] Unhandled legacy fallbacks. `_otr_paths.py` explicitly returns `otr/_legacy_audio`, `otr/_legacy_stills`, and `otr/_legacy_portraits` when `episode_id` is empty (lines 145, 161, 169). The plan ignores these. If the 2-folder contract is enforced, these will crash the run. Fix: Route empty `episode_id` fallbacks to `otr_episodes_root() / "_shared" / "legacy_..."`.
2. [Section 3] Contradictory deletion rules. The plan states "nothing auto-deletes" but immediately requires "a janitor (stale-tmp sweep) with age threshold." Fix: Explicitly exempt the new `tmp` directory from the no-auto-delete rule so the janitor is actually allowed to unlink stale files.
3. [Section 4] Mid-flight state corruption. If you execute this migration FIRST (interrupting the 7-leg sweep), moving `otr/state/news_history.json` or `otr/tmp/` will crash the running render when it attempts to read state or perform atomic `os.replace` across old/new paths. Fix: Mandate the OH-1 sequencing (wait for queue to drain).

SHOULD-FIX:
1. [Section 1] System tier placement. Pick Option B (`episodes/_shared/{cache,tmp,state}`). It is the only option that strictly satisfies the Operator's literal "episodes or obs ONLY" directive, keeps `tmp` on the same volume for Windows `os.replace`, and preserves the cross-episode stills cache reuse.
2. [Section 2] AST sweep bypass. Grepping for hardcoded `"otr/<dir>"` strings will fail to catch `comfy_output_dir() / "otr" / "stills"` because the path is constructed via division, not a single string. Fix: Audit for unauthorized imports of `comfy_output_dir()` instead, ensuring all callers use the specific `otr_*` helpers.

OPTIONAL / NICE-TO-HAVE:
- [Section 2] Add a single `_validate_contract(path)` helper in `_otr_paths.py` that asserts `path.is_relative_to(otr_episodes_root()) or path.is_relative_to(otr_obs_dir())`, and wrap all return statements with it. This guarantees the fail-LOUD constraint at the Python level before disk touch.

CUT THESE (over-engineering):
1. [Section 2] "prove no writer bypasses _otr_paths.py (grep/AST sweep)". Cut the static analysis requirement entirely. It is brittle and redundant. The post-run hygiene gate already scans the actual filesystem — if a rogue writer bypasses the helpers and writes outside `episodes/` or `obs/`, the hygiene gate will catch the physical file and fail the run. Ground truth beats static analysis here.