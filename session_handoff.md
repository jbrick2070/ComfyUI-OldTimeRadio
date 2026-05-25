# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. This session completed
**Sprint 0** (CI AST `# LLM slot:` sweep). The build is NOT finished:
next is Sprints 3A-3G, then 4, 5, 6 in plan order. ROADMAP stays
parked until the whole build lands.

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
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~26s,
  2660 tests). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
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
- **Sprint 0 COMPLETE.** Commit `c99fdfb` on `v2.0-alpha` (6 files,
  +290). NOT YET PUSHED -- needs Jeffrey to push.
  - **10 new `# LLM slot:` tags** across 4 files:
    - `_otr_style_picker.py`: creative (inventor), technical (chooser)
    - `_otr_outline.py`: creative x3 (macro, phase, beat)
    - `_otr_ledger_reviewer.py`: technical x2 (auditor, Script Doctor)
    - `_otr_line_composer.py`: creative x5 (compose, polish, announcer,
      + 2 TypeError fallback branches)
  - **New CI sweep script** `docs/_s28_llm_slot_sweep.py`:
    AST-walking audit that finds all LLM call sites (structured_call,
    generate_fn, creative_fn, technical_fn, polish_generate_fn,
    request_slot) and verifies each has a `# LLM slot:` tag within
    ±8 lines. Exempt files: internal plumbing (`_otr_structured_call`,
    `_otr_model_loader`, etc.) + writer scheduler + vram_context_test.
    20 call sites found, all 20 tagged.
  - **New regression test** `tests/test_llm_slot_sweep.py` (4 tests):
    zero-untagged assertion, floor count ≥12, synthetic catch, exempt-
    file guard.
- **Sprint 2C COMPLETE** (prior session). Commits `1fa6b40` + `4742d8c`.
  Pushed.
- **Sprints 1, 2A-2E COMPLETE** (earlier sessions).
- **HEAD = `c99fdfb`, NOT pushed.** Working tree clean except the
  parked `docs/s28_diff_tmp.txt`. No code mid-edit.

## State of the art
- `docs/_s28_llm_slot_sweep.py` -- the CI AST sweep. Standalone
  runnable: `python docs/_s28_llm_slot_sweep.py`. Exposes
  `find_llm_call_sites()` and `find_untagged_call_sites()` for test
  import.
- `tests/test_llm_slot_sweep.py` -- 4-test regression suite.
- `nodes/_otr_repair_prompts.py` -- typed repair prompt factories (2C).
- `nodes/_otr_structured_call.py` -- the 3-rung ladder (2A/2B/2C).
- Sprint Status Board (`story_pipeline_sprint_plan_v4_audited.md`):
  Sprint 0 COMPLETE, 1 COMPLETE, 2A-2E COMPLETE, 3A-3G NOT STARTED,
  4 NOT STARTED, 5 NOT STARTED, 6 NOT STARTED.
- Last full regression (2026-05-25, post-Sprint 0): OTR suite 2660
  passed / 21 skipped; Bug Bible 23 passed / 1 skipped / 2 xfailed.
  0 failed.

## Immediate next steps
1. **PUSH `c99fdfb`.** Sprint 0 code is committed but not pushed.
   Jeffrey needs to push: `cd /d <repo> && git push origin v2.0-alpha`.
2. **`docs:` commit** -- update the sprint plan Status Board and append
   a Build Progress Log entry for the Sprint 0 sweep completion.
3. Then **Sprint 3A-3G** in plan order. NOTE: 3A is a rewrite of the
   ~1660-line `_otr_line_composer.py` (2-3 days); 3B/3C/3D/3E touch
   node surfaces and DO require workflow JSON re-wiring + a live
   ComfyUI episode test to validate (operator-gated). 3F + 3G are
   "Hours" each and safer for unattended work.
4. Then Sprint 4 (VRAM -- verify against live hardware), 5, 6.

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
