# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-24

## Core goal
Execute **Wave 2** of the Story Pipeline Ledger Writer Hardening Plan v4
(`story_pipeline_sprint_plan_v4_audited.md`) -- convert the structured-JSON
LLM call sites onto the shared `structured_call` retry ladder so the writer
produces good scripts regardless of which small model holds a slot. Sprint
0/1 done; Sprint 2A helper landed; this session extended the helper and
converted the first call-site file. **Next session converts the four
remaining files.**

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md +
BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the single
  source of truth for build PROGRESS (Sprint Status Board, checkboxes, dated
  Build Progress Log at the bottom). `BUG_LOG.md` owns BUGS. After every work
  session: append a dated Build Progress Log entry + update the Status Board
  in the same edit. ROADMAP untouched until the whole build lands.
- **`structured_call` now has two extra params (commit `fed6327`)** -- use
  them: `post_validator(instance) -> str|None` (content check beyond the
  pydantic schema; a non-None return raises `PostValidationError` and advances
  the ladder like a schema failure) and `max_new_tokens` (per-caller token
  budget, default 512).
- **Tests + git via Desktop Commander**, `shell: "cmd"`. Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR suite:
  `cd /d <repo> && <venv python> -m pytest tests -q` (~30s, ~2645 tests). Bug
  Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **cmd path gotcha.** `dir` / bare path commands with Windows backslashes
  mangle under Desktop Commander (a literal `\t` etc. corrupts the path, and
  forward-slash `dir` also failed). What works reliably: `cd /d <backslash
  repo path>` followed by **forward-slash relative** paths (`tests/foo.py`).
  Use that form for every pytest/git command. `git status`/`git log`/`git
  commit`/`git push` after `cd /d` are fine.
- Commit message via the file tool to `.git\COMMIT_EDITMSG`, then `git commit
  -F .git\COMMIT_EDITMSG` in cmd. Per work batch: code commit(s) then a
  separate plan-tracking commit. Verify `local HEAD == origin HEAD` after push.
- `docs/s28_diff_tmp.txt` is parked-dirty -- never commit it. Stage files
  explicitly by name (`git add nodes/<file>`), never `git add -A`.
- Run Bug Bible + core + audio-byte-identical regression after every code
  change, unprompted (CLAUDE.md).

## What's done & decided
- **Code commits this session, on `v2.0-alpha`:** `fed6327` (structured_call
  extensions, 2 files +271/-16) and `7f3b65f` (`_otr_ledger_reviewer`
  conversion, 1 file +73/-48). A plan-tracking commit sits on top. Predecessor
  HEAD was `cc0e85a`. Local == origin after this session's push.
- **`structured_call` extended (`fed6327`).** Auditing the six call sites
  found the shipped helper (`61f8cfa`) could not host them: (a) four of five
  target files run CONTENT validation beyond the pydantic schema that drives a
  retry; (b) it hardcoded `max_new_tokens=512` but real passes need 160-3500.
  Added `post_validator` + `max_new_tokens` (both keyword-only). New
  `PostValidationError(ValueError)`. 7 new tests (11-17 in
  `tests/test_structured_call.py`; file now 18 passed). Sprint 2B (repair-temp
  inversion) is baked in -- structural retry < base, asserted at entry.
- **`_otr_ledger_reviewer.py` converted (`7f3b65f`).** `audit_cast_contract`
  and `run_script_doctor` each replaced their single-shot call + 3 hand-rolled
  failure arms with `structured_call(max_attempts=4)`. **Conversion pattern
  (reuse for the other 4 files):** replace call+parse+validate with one
  `structured_call(...)`; wrap it in `except StructuredCallFailedError` AND a
  broad `except Exception` -- structured_call does NOT catch slot-fn (LLM
  loader) exceptions -- and map BOTH to the function's existing failure
  contract (sentinel / specific exception). Never change the caller-visible
  return/raise contract: audio is king.
- **Decided:** content validation rides `structured_call.post_validator`, not
  a separate gate (Jeffrey, 2026-05-24). 2C bespoke per-failure-class repair
  factories are deferred -- the ledger conversion uses
  `default_repair_prompt_factory`.
- **Rejected:** converting call sites against the un-extended helper (would
  drop content-validation retries and truncate output at 512 tokens).

## State of the art
- `nodes/_otr_structured_call.py` -- extended + tested. Signature:
  `structured_call(*, prompt, schema, slot_fn, base_temperature,
  structural_retry_temperature, repair_prompt_factory=None,
  post_validator=None, max_new_tokens=512, grammar_path=None,
  max_attempts=4, helper_name="structured_call") -> T`. Raises
  `StructuredCallFailedError`; `PostValidationError(ValueError)` is the
  content-rejection type. 4-attempt ladder; structural retry temp must be
  strictly < base (asserted).
- `nodes/_otr_ledger_reviewer.py` -- converted, committed, regression green.
  The reference example for the remaining conversions.
- `story_pipeline_sprint_plan_v4_audited.md` -- Status Board current: Sprint
  0 IN PROGRESS (CI AST sweep only), 1 COMPLETE, 2A IN PROGRESS, 2B COMPLETE,
  2D IN PROGRESS. Build Progress Log has the dated entry for this session.
- Working tree clean except parked `docs/s28_diff_tmp.txt`. No code mid-edit.

## Immediate next steps
**Convert the 4 remaining Wave 2 files. File-disjoint -- safe in parallel
subagents OR sequential. For each: convert, lead verifies the diff on the
real file, run full regression, commit, update the plan Status Board + Build
Progress Log.**

1. **`nodes/_otr_story_brief.py` -- `run_story_brief_reflection` (def ~L564).**
   Currently TWO repair arms (schema-validation arm + content-validation arm
   via `_validate_brief`). Collapse to ONE `structured_call`: `schema=
   StoryBriefModel`, `slot_fn=technical_fn`, `base_temperature=
   _REFLECTION_TEMPERATURE` (0.3), `structural_retry_temperature` ~0.15,
   `max_new_tokens=_REFLECTION_MAX_NEW_TOKENS` (160). `post_validator` = a
   closure that calls `_validate_brief(instance.story_brief, ledger)` and
   returns the joined reasons (or None). `StructuredCallFailedError` ->
   `_failure_sentinel(...)`; keep never-raises. `_repair_pass` /
   `_build_repair_messages` / `_REPAIR_TEMPERATURE_BUMP` / `_CEILING` become
   dead -- remove, or repurpose `_build_repair_messages` as a 2C typed
   `RepairPromptFactory`. (The old `_REPAIR_TEMPERATURE_BUMP=0.15` RAISED
   repair temp -- the 2B bug; converting fixes it.)

2. **`nodes/news_interpreter.py` -- `build_news_briefs` (def ~L498).**
   Currently a 3-attempt loop with a final repair branch. `post_validator` =
   a closure running `v1_validate` / `v2_validate` / `v3_validate` + the
   `_MIN_KEY_TERMS` count check (closes over `source_text_full`, `style`).
   `base_temperature=0.7` -> `structural_retry` must be < 0.7.
   `NewsInterpreterError` is the failure type -> map `StructuredCallFailedError`
   to it. **WRINKLE:** the current code builds `NewsBriefs(**content_only)`
   from a 4-key SUBSET of the parsed dict, then Python-stamps `source_hash` /
   `source_chars` / `prompt_version` / `schema_version`. `structured_call`
   does `NewsBriefs.model_validate(full_parsed_dict)`. CHECK `NewsBriefs`
   `model_config` `extra=`: if `ignore` (pydantic default) `model_validate`
   on the full dict is equivalent and the Python stamping still runs after;
   if `forbid`, pass a lenient wrapper schema or strip extras first.

3. **`nodes/_otr_casting.py` -- `cast_one_character` (def ~L286).** 3-attempt
   loop. `post_validator` = voice-pool check (`response.voice_preset in
   available_presets`; closes over `available_presets`). `base_temperature=
   0.7`. `CastingFailedError` is the failure type. **WRINKLE (open question
   below):** the current code routes attempts 1..N-1 to `generate_fn`
   (creative slot) and the repair attempt to `validation_fn` (technical slot,
   S32 B3). `structured_call` calls ONE `slot_fn` for all attempts -- it
   cannot switch slots per attempt. Resolve the open question before
   converting.

4. **`nodes/_otr_outline.py` -- 3 inline stages in `generate_outline` (def
   ~L1429), currently via `_run_call_with_retry` (def ~L1183).** Replace each
   stage's `_run_call_with_retry` call with `structured_call`. Map
   `_run_call_with_retry`'s `extra_check` (`(parsed, raw) -> str|None`) onto
   `post_validator`. Map the Stage 2 `temperature_schedule` (falling
   0.35/0.25/0.15) onto `base_temperature` + `structural_retry_temperature`.
   **AUDIT WARNING:** do NOT flatten Stage 2 -- preserve `_deterministic_
   phase_skeleton` fallback and the singleton-cast skip. Stages 1+3 currently
   RAISE temp on retry (the 2B bug) -- converting fixes it. `_run_call_with_
   retry` may be dead once all 3 stages convert -- remove if so.

## Open questions
- **Casting `validation_fn` repair-slot routing.** `structured_call` has one
  `slot_fn`; it cannot route the repair attempt to the technical slot the way
  `cast_one_character` does today (S32 B3). Decide: (a) pass
  `slot_fn=generate_fn` and drop the repair->technical routing -- check
  `tests/test_otr_casting.py` for a test enforcing `validation_fn`-on-repair;
  if one exists it must be updated, which reverses S32 B3; or (b) keep
  `cast_one_character` partially hand-rolled around `structured_call`.
- 2C bespoke typed repair factories per failure class -- deferred. Decide
  whether to add them per-conversion or as a single 2C pass afterward.
- Sprint 0's CI AST `# LLM slot:` sweep -- still deferred.
- 2E GBNF wiring into `structured_call` Attempt 4 -- after Wave 2.
- Carried, still unaddressed: Gemma-4 / 90-word test episode (operator-gated);
  Bible-promotion of BUG-LOCAL-265/-266/-267 via the Three-File Contract.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
