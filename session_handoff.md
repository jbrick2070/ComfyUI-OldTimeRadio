# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. Decompose every LLM call in
the script-writing + cleanup pipeline into single-job passes with
deterministic guards, so a structurally valid script can no longer ship
dramatically empty. This session **closed Sprint 0** and **completed
Sprints 3F + 3G**. The build is NOT finished: 3A, 3B, 3C, 3D, 3E, 4, 5,
6 remain. ROADMAP stays parked until the whole build lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md
+ BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the
  single source of truth for PROGRESS (Sprint Status Board + dated
  Build Progress Log). `BUG_LOG.md` owns BUGS. After every work
  session: append a dated Build Progress Log entry + update the Status
  Board in the same edit. Two commits per session: one code, one
  `docs:` for the plan update.
- **LLM backend is HF Transformers 5.5.0** (`model.generate()`). No
  llama.cpp, no GBNF / grammar-constrained decoding. Sprint 2E deleted
  that scaffolding -- do not reopen it.
- **Tests + git via Desktop Commander, `shell: "cmd"`.** Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~30s, ~2713
  tests). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **The Cowork Linux sandbox `Bash` mount is STALE/unreliable for this
  repo** -- it served partial / old file contents this session. Use
  Desktop Commander (real Windows FS) for all tests + git. The
  Read/Write/Edit file tools DO write the real Windows file correctly.
- **cmd quoting gotchas.** `git log --format="%h %s"` breaks (cmd
  splits on the space) -- use `git log --oneline`. No inline
  `python -c "..."` through cmd. Commit messages: write via the file
  tool to `.git\COMMIT_EDITMSG`, then `git commit -F`.
- `docs/s28_diff_tmp.txt` is parked-dirty -- never commit it. Stage
  files explicitly by name, never `git add -A`.
- Run Bug Bible + full OTR suite after every code change, unprompted.
- **Operator-gated work.** Sprints 3B-3E + 4 need a live ComfyUI
  episode run to validate (AI has no real-time ComfyUI log access);
  Jeffrey starts the run and pastes console output. 3A is a 2-3 day
  rewrite. Plan accordingly.

## What's done & decided
- **Sprint 0 COMPLETE + pushed.** CI AST `# LLM slot:` sweep.
  - `c99fdfb` -- `docs/_s28_llm_slot_sweep.py` (AST-walks `nodes/`,
    verifies a `# LLM slot:` tag within +/-8 lines of every LLM call
    site; 20 sites, all tagged) + 10 logical tags + the regression
    `tests/test_llm_slot_sweep.py`.
  - `b3d6355` -- QA hardening: `find_parse_failures` surfaces a node
    file that fails to AST-parse instead of the sweep silently
    swallowing it (a swallowed file = invisible call sites). +2 tests;
    suite now 6 tests.
  - `bdb2333` -- plan close (Status Board + Build Progress Log).
- **Sprint 3F COMPLETE + pushed (`e14d364`).** Removed the
  `confidence` field from `CastViolation`; the cast auditor now does
  pure anomaly extraction. New `_resolve_cast_member` decides repairs
  deterministically -- exact case-fold, then the EXISTING
  `auto_remap_phantom`/`_levenshtein` -- and escalates ambiguous ties
  to the Script Doctor.
- **Sprint 3G COMPLETE + pushed (`088aba1`).** `_build_reflection_input`
  now anonymizes cast names + proper nouns to neutral tokens
  (`character_a`, `source_entity`) BEFORE the LLM sees the text;
  `_REFLECTION_PROMPT` suppression list collapsed to one positive
  instruction; the OUTPUT reject-list safety net is untouched.
- `docs:` commit `bb53870` -- plan board + progress log for 3F + 3G.
- Sprints 1, 2A-2E COMPLETE (earlier sessions).
- **HEAD = `bb53870`, pushed; `origin/v2.0-alpha` == HEAD.** Working
  tree clean except the parked `docs/s28_diff_tmp.txt`.
- **Build method: parallel subagents on disjoint file sets.** Validated
  again this session -- 3F + 3G ran as two parallel subagents (3F on
  `_otr_ledger_reviewer.py`, 3G on `_otr_story_brief.py`), the lead ran
  the authoritative combined-tree regression and committed each
  separately. Subagents do NOT commit/push or edit `BUG_LOG.md` / the
  plan -- the lead integrates serially.

## State of the art
- Build tracker `story_pipeline_sprint_plan_v4_audited.md` Status Board:
  Sprints 0, 1, 2A-2E, 3F, 3G COMPLETE; 3A, 3B, 3C, 3D, 3E, 4, 5, 6
  NOT STARTED.
- The **multi-agent execution plan for the remaining sprints** is
  written into `docs/2026-05-24-story-pipeline-llm-audit.md` (new
  "Multi-agent execution plan" section at the end) -- file-to-subagent
  lane map, dependencies, per-lane gating.
- Last regression (2026-05-25, post-3F/3G combined tree): OTR suite
  2692 passed / 21 skipped (2713 collected); Bug Bible 16 passed / 7
  skipped / 3 xfailed; audio-byte-identical green; LLM-slot sweep 6
  passed. 0 failed.

## Immediate next steps
1. Read `docs/2026-05-24-story-pipeline-llm-audit.md` -> "Multi-agent
   execution plan" section. It maps the remaining Sprint 3 work to
   parallel subagent lanes on disjoint files.
2. Launch the parallel wave: **4 subagents** for **3B**
   (`nodes/_otr_outline.py`), **3C** (`nodes/_otr_ledger_reviewer.py`),
   **3D** (`nodes/_otr_casting.py`), **3E** (`OTR_LedgerScriptWriter.py`
   + title path) -- four disjoint files, four lanes.
3. Hold **3A** (`nodes/_otr_line_composer.py` -- 2-3 day rewrite of a
   ~1660-line file) as a dedicated lead-driven effort; do NOT one-shot
   it as a parallel subagent.
4. Lead integrates: full OTR + Bug Bible regression on the combined
   tree, one commit per sprint, plan Status Board + Build Progress Log
   update, `docs:` commit, push, verify HEAD == origin.
5. After 3B-3E land, Jeffrey runs ONE live ComfyUI episode to validate
   the batch (audio-is-king reversion gate).
6. Then 3A, then Sprint 4 (VRAM -- verify-only, live-hardware gated),
   then 5 + 6.

## Open questions
- **3C adds +1 LLM call per episode** (split Script Doctor -> diagnosis
  + edit). Prime Directive 6: the new call site needs a `# LLM slot:`
  tag, the model id wired from the writer's broadcast socket, and the
  routing table updated. The Sprint 0 CI sweep WILL enforce the tag.
- **3E edits `OTR_LedgerScriptWriter.py`, which is EXEMPT from the CI
  sweep.** A new untagged LLM call there will NOT be auto-caught -- the
  3E subagent must tag it manually.
- `CLAUDE.md` still carries a "Round-Robin Consultation" section as a
  general project rule; the round-robin waiver was scoped to the sprint
  plan only. Amend `CLAUDE.md` if you want them consistent.
- Carried: BUG-LOCAL-265/-266/-267 Bible-promotion via the Three-File
  Contract; Gemma-4 / 90-word test episode (operator-gated live run).
- Minor 3G nit: `_PROPER_NOUN_STOPWORDS` in `_otr_story_brief.py` lists
  `"interior"` twice -- harmless (frozenset dedupes), cosmetic cleanup
  only if that file is touched again.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
