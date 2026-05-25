# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-24

## Core goal
Execute the **Story Pipeline Ledger Writer Hardening Plan v4** (`story_pipeline_sprint_plan_v4_audited.md`) -- decompose the overloaded LLM calls in the story-writing + cleanup pipeline so the writer produces good scripts regardless of which small model holds a slot. Sprint 0 and Sprint 1 are done; Sprint 2A's shared helper module is built. The next session runs **Wave 2**: convert the structured-call sites onto the new helper, dispatched as parallel file-disjoint subagents.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md + BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **Build Tracking Protocol.** `story_pipeline_sprint_plan_v4_audited.md` is the single source of truth for build PROGRESS (Sprint Status Board, checkboxes, dated Build Progress Log at the bottom). `BUG_LOG.md` is the single source of truth for BUGS. Link by pointer only. After every work session: append a dated Build Progress Log entry + update the Status Board in the same edit. ROADMAP untouched until the whole build lands.
- **Subagent build pattern (this is what "agents optimized" means).** Partition work by **file ownership** -- never by fix. Two agents must NEVER write the same file. Brief each agent with exact file paths, line numbers, and the precise change; do not delegate understanding. After agents return, the lead verifies every diff on the REAL files with the Read tool, then runs regression + commits itself.
- **Stale workspace-bash mount.** `mcp__workspace__bash` serves stale / null-padded copies of files. Trust the **Read tool** for real file content. Run tests, git, and byte-checks through **Desktop Commander**. `py_compile` via the bash sandbox throws false "unterminated string" errors far from edits -- ignore those, re-verify on the real file.
- **Tests + git via Desktop Commander**, `shell: "cmd"` (NOT powershell -- powershell can't find `cmd`, and `%` chars break `python -c`). Venv python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Bug Bible regression: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide\tests\bug_bible_regression.py`. Commit message via the file tool to `.git\COMMIT_EDITMSG`, then `git commit -F`. Per work batch: code commit(s) then a separate plan-tracking commit that references the code hash. Verify `local HEAD == origin HEAD` after push.
- `docs/s28_diff_tmp.txt` is parked-dirty -- never commit it. `session_handoff.md` also shows `M` (this file).
- Run Bug Bible + core + audio-byte-identical regression after every code change, unprompted (CLAUDE.md).
- Round-robin consultation is WAIVED (Jeffrey, 2026-05-24) -- no sprint is gated; the build runs sprint-after-sprint. (CLAUDE.md still carries a general "Round-Robin Consultation" section; it is not a gate on this build's sprint sequencing.)

## What's done & decided
- **HEAD `cde08b9` on `v2.0-alpha`, local == origin.** 7 commits this session.
- **Sprint 0 -- 4 of 5 items done** (`e7a8eb6`, `2dd8d1b`, `51f7226`, `8d9064e`):
  - `json_str` -> `raw` at `_otr_story_brief.py:648` -- the schema-repair `NameError` that silently killed every repair pass. Logged + fixed as **BUG-LOCAL-268**.
  - Three `helper_context(...)` wraps at the `OTR_LedgerScriptWriter.py` call sites (generate_outline / title regen / run_story_brief_reflection).
  - Two stale `pick_style` routing comments corrected.
  - New `tests/test_story_brief_repair_pass.py` (4 tests).
  - Refreshed the stale `test_pick_style_internally_uses_creative_fn_default` -> `test_pick_style_routes_inventor_creative_and_chooser_technical` in `tests/test_helper_paired_signatures.py`.
  - NOT done: the 5th item, a CI AST-level `# LLM slot:` sweep -- deferred, out of scope. Sprint 0 cannot formally close until it lands or is dropped.
- **Sprint 1 -- COMPLETE** (`df7f9b1`): HuMo tier renamed `high_quality` -> `high_quality_unsafe_on_16gb` in `_otr_humo_tier_loader.py`, `tests/test_humo_tier_loader.py`, one `__init__.py` comment. No workflow JSON wires the tier value, so no JSON re-wire.
- **Sprint 2A step 1 -- LANDED** (`61f8cfa`): new `nodes/_otr_structured_call.py` -- the shared 4-attempt structured-JSON retry ladder. `tests/test_structured_call.py` (11 tests). The module is NOT yet imported by any node. Converting the 6 call sites is Wave 2.
- **Decisions resolved this session:**
  - D1 `humo_max_lines_per_process` -> stays `0` (no change).
  - D2 `clip_length` -> stays an editable widget, default `7.0` (no lock, no code change).
  - D3 `experimental_gguf` tier -> NOT renamed.
  - D6 `technical_fn` on `compose_line` -> **KEEP it.** It is test-enforced paired-contract surface (`tests/test_helper_paired_signatures.py::test_compose_line_accepts_paired_generators`), not dead weight. The Sprint 0 "drop technical_fn" item is PULLED -- **do not reopen this.**
- **Round-robin consultation WAIVED** (Jeffrey, 2026-05-24): these changes were round-robined in earlier sessions. The v4 plan's round-robin gating is fully removed -- the build runs sprint-after-sprint, no consultation gate.
- **Rejected:** dropping `technical_fn`; renaming `experimental_gguf`; `humo_max_lines_per_process = 6`; locking `clip_length`; re-auditing the v4 plan.

## State of the art
- `nodes/_otr_structured_call.py` -- complete + tested. Public API: `structured_call(*, prompt, schema, slot_fn, base_temperature, structural_retry_temperature, repair_prompt_factory, grammar_path, max_attempts, helper_name) -> T`; raises `StructuredCallFailedError`; `RepairPromptFactory` Protocol + `default_repair_prompt_factory`. 4-attempt ladder; the 2B principle (structural retry temperature strictly LOWER than base) is asserted at entry. This is what Wave 2 call sites adopt.
- `story_pipeline_sprint_plan_v4_audited.md` -- committed, tracking current. Status Board: Sprint 0 IN PROGRESS (CI sweep only), Sprint 1 COMPLETE, Sprint 2A IN PROGRESS. Decisions 1-3 + 6 marked RESOLVED. Build Progress Log has dated entries through Wave 1.
- Working tree clean except parked `docs/s28_diff_tmp.txt` + `session_handoff.md`. No code mid-edit.

## Immediate next steps
**Wave 2 -- convert the 6 structured-call sites onto `structured_call`, folding in 2C (typed repair factories) + 2D (cleanup-pass retries). Five file-disjoint subagents, safe to run in parallel:**
1. **Agent 1 -- `nodes/_otr_story_brief.py`:** convert `run_story_brief_reflection` to use `structured_call`; add the 2C typed repair factory for the story brief. (2B repair-temp behaviour is already embodied in `structured_call`.)
2. **Agent 2 -- `nodes/_otr_ledger_reviewer.py`:** convert `audit_cast_contract` + `run_script_doctor` (both currently single-shot) to `structured_call` with `max_attempts=4` (Sprint 2D) + 2C typed factory. REUSE the existing `_levenshtein` + `auto_remap_phantom` (threshold 3) for cast-membership repair -- do not write a second matcher.
3. **Agent 3 -- `nodes/news_interpreter.py`:** convert `build_news_briefs` (currently 3 attempts incl. repair).
4. **Agent 4 -- `nodes/_otr_casting.py`:** convert `cast_one_character` (3 attempts; attempt 3 repair routes technical slot).
5. **Agent 5 -- `nodes/_otr_outline.py`:** convert the 3 inline outline stages (macro / phase / beat, currently via `_run_call_with_retry` inside `generate_outline()`). **AUDIT WARNING:** do NOT flatten Stage 2 (phase) -- it has a falling-temp schedule `(0.35, 0.25, 0.15)`, a deterministic `_deterministic_phase_skeleton` fallback, and a singleton-cast skip. Preserve all three.
- Each agent: convert + run regression. Lead verifies diffs on real files, runs full regression (Bug Bible + core + audio-byte-identical + the touched suites), commits per the two-commit pattern, updates the plan Status Board + Build Progress Log.
- After Wave 2: Sprints 2E (GBNF -- wire) and 3A-3G run straight through -- no consultation gate. **File collision to sequence:** 3C and 3F BOTH touch `_otr_ledger_reviewer.py` -- never put them in the same parallel wave.

## Open questions
- Sprint 0's CI AST `# LLM slot:` sweep -- deferred. Decide: implement it, or formally close Sprint 0 without it.
- Carried from the prior handoff, still unaddressed: the Gemma-4 / 90-word test episode (operator-gated run); Bible-promotion of BUG-LOCAL-265 / -266 / -267 via the Three-File Contract.
- All six v4 Open Decisions are resolved and round-robin gating is removed (2026-05-24) -- nothing in the plan is gated; Wave 2 onward runs continuously.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
