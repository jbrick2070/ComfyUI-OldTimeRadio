# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25 (Sprints 5C + 6)

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`, is **FEATURE-COMPLETE**. This
session landed Sprint 5C (targeted reroll loop) and Sprint 6 (critic ->
render coupling). The pipeline now has the full continuity-ledger +
story-quality critic + targeted reroll + render coupling chain -- the
direct fix for the `ozempics_glitch`-class "structurally valid but
dramatically empty" failure the v4 audit flagged. **The ONLY remaining
gate is the operator RTX-5080 live-run** -- one ComfyUI episode on
current HEAD validates 3A + 5A + 5B + 5C + 6 + the carried
cast/style/seed batch in one shot (Prime Directive 1 -- audio is king).

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md +
BUG_LOG.md auto-load -- not repeated here. Live operational notes:

- **HEAD = `244e9fd`** (Sprint 6: critic -> render coupling). Commits
  this session in order: `2387cef` (Sprint 5C), `7a36f5e` (5C plan
  update), `244e9fd` (Sprint 6 + plan update). All three pushed;
  `local HEAD == origin/v2.0-alpha`.
- Working tree clean except the parked `docs/s28_diff_tmp.txt` (never
  commit it).
- **Cowork Linux sandbox `Bash` mount is STALE for this repo** -- its
  `git status` shows phantom modified files. Use Desktop Commander
  (`shell: "cmd"`) for git + tests. The Read/Write/Edit file tools DO
  write the real Windows file correctly.
- Venv python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- Full OTR suite: `cd /d <repo> && <venv python> -m pytest tests -q`
  (~34s, 2842 tests). Bug Bible: `cd /d
  C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
  && <venv python> -m pytest tests/bug_bible_regression.py -q`.
  LLM-slot sweep: `<venv python> docs\_s28_llm_slot_sweep.py` (must
  exit 0 with `23 tagged, 0 untagged, 0 parse failures`).
- Redirect long runs to a log file (`> %TEMP%\x.log 2>&1`) and read the
  file -- a single long `interact_with_process` call drops the cmd
  session at the MCP timeout.
- `cmd` gotchas: commit messages via the file tool to
  `.git\COMMIT_EDITMSG`, then `git commit -F`. Never `git commit -m`.
- LLM backend is HF Transformers 5.5.0 (`model.generate()`). No
  llama.cpp, no GBNF.
- Run Bug Bible + full OTR suite after every code change, unprompted.

## What's done & decided

- **Sprint 5C COMPLETE (`2387cef`, pushed).** Fork RESOLVED **Option A**
  -- technical-slot in-cascade reroll, no creative-slot VRAM swap, no
  node-surface change. PD2 (VRAM ceiling) and PD3 (no JSON re-wire)
  both favour Option A; deviation from the plan's literal "creative
  slot" wording is deliberate and documented.
  - New module `nodes/_otr_reroll.py`: `RerollDisposition` +
    `build_reroll_line_request` (from `meta` + the ledger row) +
    `run_targeted_reroll` (the loop). NEVER raises (PD1).
  - Loop cap: `MAX_REROLL_CYCLES = 2`. Cycle 3 -> `needs_full_rerun`;
    the cascade restores pre-reroll lines from a snapshot and skips
    Phase 7/8/10 via the new shared `_build_terminal_skip_disposition`
    helper (the reviewer-terminal block was refactored onto the same
    helper).
  - Re-composition deliberately uses `compose_line` with
    `enable_polish_pass=False` (NOT bare `compose_line_draft`) so the
    deterministic strip pipeline (`cast_strip` / phantom gate /
    `vocative_strip`) still guards the technical-model output.
  - Composer: `LineRequest.reroll_hint` field; `_build_user_prompt`
    renders a REVISE block at the WRITE LINE tail, gated on non-empty
    value (Sprint 5A `continuity_slice` pattern -- existing callers
    unaffected). `compose_line` / `compose_line_draft` gain a
    `reroll_hint` kwarg that overlays via `dataclasses.replace`.
  - Writer stamps `meta.canon_header` / `outline_spine` / `theme` /
    `style_descriptor` so the cascade can rebuild a `LineRequest` from
    `meta` + the ledger row.
  - 11 new tests (`tests/test_otr_reroll.py`).

- **Sprint 6 COMPLETE (`244e9fd`, pushed).** Fork RESOLVED **full v1
  coupling** (Jeffrey 2026-05-25) -- widgets on the cascade AND HuMo
  consumes the plan today. Cascade-only stamping was rejected.
  - 4 new widgets on `OTR_LedgerFreezeCascade.INPUT_TYPES` after
    `vram_ceiling_gb`: `render_selection` (COMBO "all" |
    "dramatic_peaks_only", default "all"), `render_max_n` (INT default
    6, 0 disables the cap), `protagonist_only` (BOOLEAN default False),
    `manual_line_ids` (STRING default ""). All default to
    behave-as-before so unmodified workflows render unchanged.
  - New pure module `nodes/_otr_render_plan.py`:
    `build_render_plan(ledger_data, *, render_selection, render_max_n,
    protagonist_only, manual_line_ids) -> dict | None`. Pure dict /
    list arithmetic; no LLM call (sweep stays at 23/23). NEVER raises
    (PD1) -- a degraded computation returns None and HuMo falls back
    to render-all.
  - Selection rules (in `build_render_plan`):
    - **Arc-verdict gate:** `cycle_count >= MAX_REROLL_CYCLES (2) AND
      arc_verdict in (mid_collapse, flat)` -> stamp `blocked=True,
      line_ids=[]`. `uneven` and `strong` do NOT block.
    - **manual_line_ids:** highest priority; LIFTS the arc-verdict
      gate; unknown ids dropped; preserve operator order + dedupe.
    - **protagonist_only:** restricts to the most-spoken character
      (ties broken by cast-roster order).
    - **dramatic_peaks_only:** reorders by critic.`render_priority`;
      unprioritized lines fall to the tail in ledger order.
    - **flat_lines exclusion:** drop UNLESS the line is in
      `meta.reroll_history` with `status="rerolled"`.
    - **render_max_n cap:** applied last.
  - Cascade: `run_freeze_cascade` accepts the 4 kwargs; calls
    `build_render_plan` AFTER the 5C reroll on the non-terminal path,
    BEFORE Phase 7/8/10; stamps `meta.render_plan`.
  - HuMo: `OTR_BatchHumoRender`'s per-line loop reads
    `meta.render_plan` once at the top and SKIPs character lines not
    in the plan (announcer / music / sfx untouched). Blocked plan
    skips every character line. Skip runs BEFORE timing / chunking /
    portrait resolution -- excluded line costs zero.
  - Workflow JSON re-wired: `OTR_LedgerFreezeCascade`'s
    `widgets_values` extended from 3 entries `[true, true, 14.0]` to
    7 entries `[true, true, 14.0, "all", 6, false, ""]` matching the
    INPUT_TYPES order. Guardrail test
    (`test_workflow_json_guardrails.py::test_cascade_widget_vector_trimmed_in_canonical_json`)
    updated 3 -> 7 with per-slot type assertions; original "the
    deleted B3 widgets stay deleted" intent preserved.
  - 19 new tests (`tests/test_otr_render_plan.py`).

- **Sprint plan updated and pushed.** Status Board: Sprint 5 +
  Sprint 6 both COMPLETE. Build Progress Log carries both sessions'
  entries with the full fork-resolution rationale + regression
  numbers.

- **Decisions explicitly made + rejected (do NOT reopen):**
  - **5C Option B** (creative-slot reroll with VRAM swap) -- REJECTED.
    PD2 + PD3 both favour Option A.
  - **5C continuity_slice on rerolls** -- DEFERRED to v2 (the critic
    hint carries the actionable signal; reconstructing the slice from
    `meta.continuity` + beat index is straightforward but deliberately
    out of 5C v1 scope).
  - **5C reroll calls `compose_line` (polish OFF)**, not bare
    `compose_line_draft` -- keeps deterministic strip safeguards on
    technical-model output. Documented deviation from the plan's
    literal "compose_line_draft" wording.
  - **Sprint 6 cascade-only stamp** -- REJECTED in favour of full v1
    coupling (operator pick).

## State of the art

- **Status Board (`story_pipeline_sprint_plan_v4_audited.md`):**
  Sprints 0, 1, 2A-2E, 3A-3G, 4, 5A-5C, 6 ALL COMPLETE. The whole v4
  plan is now landed pending the operator live-run.
- **HEAD `244e9fd`** == `origin/v2.0-alpha`. Working tree clean except
  parked `docs/s28_diff_tmp.txt`.
- **New files (5C + 6):** `nodes/_otr_reroll.py`,
  `nodes/_otr_render_plan.py`, `tests/test_otr_reroll.py` (11 tests),
  `tests/test_otr_render_plan.py` (19 tests).
- **Modified files (5C + 6):** `nodes/_otr_line_composer.py`
  (`reroll_hint` field + REVISE block + kwarg on
  `compose_line`/`compose_line_draft`), `nodes/_otr_freeze_cascade.py`
  (reroll call + render-plan stamp + shared
  `_build_terminal_skip_disposition` helper + reviewer-terminal
  refactor), `nodes/OTR_LedgerScriptWriter.py` (stamps
  `meta.canon_header`/`outline_spine`/`theme`/`style_descriptor`),
  `nodes/OTR_LedgerFreezeCascade.py` (4 new widgets +
  `run()` kwargs + pass-through to `run_freeze_cascade`),
  `nodes/batch_humo_render.py` (`meta.render_plan` filter in the
  per-line plan loop), `workflows/otr_scifi_16gb_full.json`
  (cascade `widgets_values` 3 -> 7),
  `tests/test_workflow_json_guardrails.py` (cascade widget-vector
  guard 3 -> 7 with per-slot type assertions),
  `story_pipeline_sprint_plan_v4_audited.md` (Status Board + two
  Build Progress Log entries).
- **Regression baseline at HEAD `244e9fd`:** full OTR suite **2842
  passed / 21 skipped / 0 failed**; Bug Bible 16 passed / 7 skipped /
  3 xfailed; LLM-slot sweep **23/23 tagged, 0 parse failures**. All
  touched files UTF-8 no BOM, AST + JSON parse OK.

## Immediate next steps

1. **Operator RTX-5080 live-run -- the Prime Directive 1 gate.** ONE
   ComfyUI episode on current HEAD `244e9fd` validates, in a single
   run: **Sprint 3A** (`compose_line` split + `cast_strip` remap), the
   carried cast/style/seed batch (`61dda9c` / `906a57f` / `d0ea595`),
   **Sprint 5A** (continuity ledger), **5B** (whole-script critic),
   **5C** (targeted reroll loop), and **Sprint 6** (critic -> render
   coupling). Operator-gated: Jeffrey starts the run from ComfyUI
   Desktop (localhost:8000) and pastes the full console on
   completion / error. The AI has no real-time ComfyUI log access.
   What to watch for in the console:
     - `[OTR_StoryCritic] critic complete: arc_verdict=...` (5B fires).
     - `[OTR_Reroll] cycle N/2: M reroll target(s)` (5C rerolls some
       lines) OR `[OTR_Reroll] critic named no reroll targets --
       nothing to do` (no-op -- clean critic).
     - `[LFC] Sprint 6 render plan: mode=..., N line(s) (blocked=...,
       applied_max_n=..., arc_verdict=..., cycle_count=...)` (6 stamps
       the plan).
     - `[BatchHumoRender] Sprint 6 render plan ACTIVE: N line(s)
       selected ...` (HuMo honours it) OR no Sprint 6 log line (no
       plan stamped -> HuMo's pre-Sprint-6 render-all fallback).
     - `freeze_verdict` anything BUT `needs_full_rerun`.
     - Final mp4 produced (audio is king).

2. **If the run goes RED:** paste the full console output (from
   `Exec` node start through completion / error). Log every new bug to
   `BUG_LOG.md` immediately (do not batch). The triage path is
   `BUG-LOCAL-NNN` per CLAUDE.md; Bible-promotion follows the
   Three-File Contract.

3. **If the run goes GREEN:** the v4 plan is officially shipped. Mark
   the carried operator-live-validation items (3A, 5A, 5B, 5C, 6,
   cast/style/seed batch, 14B VRAM cap) as `LIVE-VALIDATED` in the
   Status Board. Open `BUG-LOCAL-265/266/267/271` for Three-File
   Contract Bible-promotion.

4. **Optional v2 follow-ups (no blockers, not required for ship):**
   - **continuity_slice on rerolls.**
     `_otr_reroll.build_reroll_line_request` leaves `continuity_slice`
     empty today. Reconstructing it from
     `ContinuityState.model_validate(meta["continuity"])` +
     `render_continuity_slice(state, speaker, beat_index)` (beat_index
     = line row position in `ledger_data["lines"]`) would give a
     rerolled line the same per-speaker hard constraints the
     first-pass compose got. Deliberately deferred -- the critic hint
     already carries the actionable signal.
   - **Scoped critic re-invocation.** The 5C loop re-runs the WHOLE
     `run_story_critic` each cycle (handoff Q2 recommended default).
     A per-affected-line re-critique is the v2 optimization.
   - **Per-line v1/v2 versioning on the ledger.** Today
     `update_line_text` overwrites in place; `meta.reroll_history` is
     the audit trail (5C Q3 default). Full ledger versioning is the
     other option.
   - **`build_render_plan` `protagonist_only` definition.** Currently
     "most-spoken CHARACTER-role char_id, cast-roster tiebreak". An
     explicit `meta.protagonist_char_id` writer stamp would replace
     the heuristic if a future sprint wants determinism over
     line-count.

## Open questions

None blocking. The live-run is the only gate.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
