# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. This session closed
**Sprint 2C** (typed repair prompts by failure class). The build is
NOT finished: next is Sprint 0's last item (the deferred CI AST
`# LLM slot:` sweep), then Sprints 3A-3G, 4, 5, 6 in plan order.
ROADMAP stays parked until the whole build lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md
+ BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the
  single source of truth for build PROGRESS (Sprint Status Board +
  dated Build Progress Log at the bottom). `BUG_LOG.md` owns BUGS.
  After every work session: append a dated Build Progress Log entry +
  update the Status Board in the same edit. Two commits per session:
  one code commit, one `docs:` commit for the plan update.
- **LLM backend is HF Transformers 5.5.0** (`nodes/_otr_model_loader.py`
  `make_generate_fn` -> `model.generate()`). No llama.cpp, no GBNF /
  grammar-constrained decoding. Sprint 2E deleted that scaffolding --
  do not reopen it.
- **Tests + git via Desktop Commander**, `shell: "cmd"`. Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~25s,
  2656 tests). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **cmd path gotcha.** `cd /d <backslash repo path>` then forward-slash
  relative paths for pytest/git. Commit message via the file tool to
  `.git\COMMIT_EDITMSG`, then `git commit -F` in cmd.
- `docs/s28_diff_tmp.txt` is parked-dirty -- never commit it. Stage
  files explicitly by name, never `git add -A`.
- Run Bug Bible + full OTR suite after every code change, unprompted.
- **Operator-gated work cannot run unattended.** Live ComfyUI episode
  runs need Jeffrey to start the run and paste console output (AI has
  no real-time ComfyUI log access). Sprint 4 (VRAM) must be confirmed
  against live hardware. Plan accordingly.

## What's done & decided
- **Sprint 2C COMPLETE.** Commits on `v2.0-alpha`: `1fa6b40` (code,
  9 files, +950/-14) + `4742d8c` (plan update). Pushed; local == origin.
  - New pure module `nodes/_otr_repair_prompts.py`: six typed
    `RepairPromptFactory` builders (`json_syntax_repair`,
    `schema_field_repair`, `cast_membership_repair`,
    `too_many_words_repair`, `narration_leak_repair`,
    `forbidden_name_repair`) + `make_dispatching_repair_factory`.
  - The dispatcher routes a `structured_call` Attempt 3 failure by
    error: `json.JSONDecodeError` -> json_syntax;
    `pydantic.ValidationError` -> schema_field; `PostValidationError`
    -> classified by message substring (`locked cast` ->
    cast_membership, `named_character` -> forbidden_name,
    `dialogue_verb`/`plot_verb` -> narration_leak, `too_long` ->
    too_many_words); anything else -> `default_repair_prompt_factory`.
  - **`structured_call` extension (decided + shipped):** a repair
    factory may now return a finished `schema` instance instead of a
    repair prompt. The Attempt 3 block detects it, runs it through
    `post_validator`, and returns it with NO LLM repair call. This was
    the enabling mechanism for the v4 plan's "cast-membership repair
    never calls the LLM if Levenshtein resolves" requirement -- the
    3-rung ladder could not express it otherwise. The change is
    additive: factories returning messages (all pre-2C behaviour) are
    untouched.
  - **No second Levenshtein.** The outline phase stage supplies a
    `_phase_cast_phantom_repair` callback reusing the existing
    `auto_remap_phantom` (threshold 3) from `_otr_ledger_reviewer.py`
    via a lazy import.
  - All 8 `structured_call` sites wired to
    `make_dispatching_repair_factory()`. Only the outline phase site
    passes a `deterministic_repair` callback (the one site with a
    locked cast).
  - No node surface touched -- no workflow JSON re-wire. No new LLM
    call -- the typed factories reshape the existing Attempt 3 prompt;
    the deterministic path removes a call. (Prime Directives 3 + 6
    N/A.)
- **HEAD = `4742d8c`, local == origin. Working tree clean** except the
  parked `docs/s28_diff_tmp.txt`. No code mid-edit.

## State of the art
- `nodes/_otr_repair_prompts.py` -- the six builders + dispatcher.
  Pure: stdlib + pydantic + the sibling `_otr_structured_call` only.
- `nodes/_otr_structured_call.py` -- the 3-rung ladder, now with the
  Attempt-3 "factory may return a finished `schema` instance" branch.
- Sprint Status Board (`story_pipeline_sprint_plan_v4_audited.md`):
  Sprint 0 IN PROGRESS (only the deferred CI AST `# LLM slot:` sweep
  left), 1 COMPLETE, 2A-2E COMPLETE, 3A-3G NOT STARTED, 4 NOT STARTED,
  5 NOT STARTED, 6 NOT STARTED.
- Last full regression (2026-05-25, post-2C): OTR suite 2656 passed /
  21 skipped; Bug Bible 16 passed / 7 skipped / 3 xfailed;
  audio-byte-identical 9 passed / 1 skipped. 0 failed.

## Immediate next steps
1. **Sprint 0 -- the deferred CI AST `# LLM slot:` sweep** (plan
   Sprint 0). A CI/audit script that AST-parses the codebase and
   verifies every LLM call site carries a `# LLM slot: creative` or
   `# LLM slot: technical` tag (Prime Directive 6). Read the Sprint 0
   section + `docs/_s28_forbidden_sweep.py` for the existing
   forbidden-pattern sweep to model it on. Add a regression test;
   commit code + `docs:` plan update.
2. Then **Sprint 3A-3G** in plan order. NOTE: 3A is a rewrite of the
   ~1660-line `_otr_line_composer.py` (2-3 days); 3B/3C/3D/3E touch
   node surfaces and DO require workflow JSON re-wiring + a live
   ComfyUI episode test to validate (operator-gated). 3F + 3G are
   "Hours" each and safer for unattended work.
3. Then Sprint 4 (VRAM -- verify against live hardware), 5, 6.

## Open questions
- Carried, still unaddressed: Gemma-4 / 90-word test episode
  (operator-gated -- needs a live ComfyUI run); Bible-promotion of
  BUG-LOCAL-265/-266/-267 via the Three-File Contract.
- `CLAUDE.md` still carries a "Round-Robin Consultation" section as a
  general project rule; the round-robin waiver was scoped to the
  sprint plan only. Flag for Jeffrey if he wants CLAUDE.md amended.
- Sprints 3A-6 cannot be safely completed unattended in one session:
  several need a live ComfyUI episode run to validate and several
  touch node surfaces under the audio-is-king reversion gate. They
  need Jeffrey in the loop.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
