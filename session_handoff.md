# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25 (Sprint 5A + 5B)

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. This session closed Sprint 4
and landed Sprint 5A (continuity ledger) + Sprint 5B (whole-script story
critic). Remaining: Sprint 5C (targeted reroll) and Sprint 6 (critic ->
render coupling). **5C is blocked on one architecture decision** -- see
"Sprint 5C open fork" below; that decision is the first thing the next
session needs. ROADMAP stays parked until the whole build lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md +
BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **HEAD = `4b7db99`** (Sprint 5B). Substantive commits this session:
  `056fca1` (Sprint 4 docs close-out), `8fef3c5` (Sprint 5A), `4b7db99`
  (Sprint 5B). All three pushed; `local HEAD == origin/v2.0-alpha`.
- Working tree clean except the parked `docs/s28_diff_tmp.txt` (never
  commit it).
- **The Cowork Linux sandbox `Bash` mount is STALE for this repo** -- its
  `git status` shows phantom modified files. Use Desktop Commander
  (`shell: "cmd"`) for git + tests. The Read/Write/Edit file tools DO
  write the real Windows file correctly.
- Tests + git via Desktop Commander, `shell: "cmd"`. Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~29s). Bug
  Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
  Redirect long runs to a log file (`> %TEMP%\x.log 2>&1`) and read the
  file -- a single long `interact_with_process` call drops the cmd
  session at the MCP timeout.
- `cmd` gotchas: commit messages via the file tool to
  `.git\COMMIT_EDITMSG`, then `git commit -F`. Never `git commit -m`.
  `%errorlevel%` after `&` on one line is parsed pre-run -- read the log
  to get the real pytest result, not the inline echo.
- LLM backend is HF Transformers 5.5.0 (`model.generate()`). No
  llama.cpp, no GBNF.
- Run Bug Bible + full OTR suite after every code change, unprompted.
- Console-output analysis: Jeffrey must paste it -- no real-time
  ComfyUI log access.

## What's done & decided
- **Sprint 4 CLOSED (`056fca1`, docs-only).** 14B VRAM cap resolved
  no-change (keep the Sovereignty branch -- Jeffrey's call). Prompt-cache
  bullet resolved N/A (`cache_prompt` / `n_cache_reuse` are llama.cpp
  params with no HF equivalent; HF `generate()` runs the within-call KV
  cache by default). Operator live-RTX-5080 VRAM confirm stays parked
  (non-blocking).
- **Sprint 5A COMPLETE (`8fef3c5`).** New module `nodes/_otr_continuity.py`
  -- `ContinuityFact` / `ContinuityState` models, `build_continuity_ledger`
  (one technical-slot LLM call after the outline lands; never raises,
  degrades to `ContinuityState.neutral()`), and the pure
  `render_continuity_slice` projector. Wired: `OTR_LedgerScriptWriter.py`
  section H.5 builds the continuity ledger + stamps `meta["continuity"]`;
  per-beat closure threads `render_continuity_slice` into a new
  `LineRequest.continuity_slice` field; `_otr_line_composer._build_user_prompt`
  renders a CONTINUITY CONSTRAINTS block in the per-beat tail.
- **Sprint 5B COMPLETE (`4b7db99`).** New module `nodes/_otr_story_critic.py`
  -- 6-section `StoryCriticReport` (continuity_issues, voice_drift,
  flat_lines, arc_verdict, reroll_targets, render_priority), `run_story_critic`
  (one technical-slot LLM call via `structured_call`; never raises,
  returns `StoryCriticReport.clean()`). Wired: `_otr_freeze_cascade.run_freeze_cascade`
  calls `run_story_critic` on the non-terminal path (after the
  terminal-verdict short-circuit, before Phase 7) and stamps
  `meta["story_critic_report"]`. For 5B the report is ADVISORY -- it
  changes no line text.
- **Regression green** after both: full OTR suite 2812 passed / 21
  skipped (0 failed); Bug Bible 16 passed / 7 skipped / 3 xfailed;
  LLM-slot sweep 23/23 tagged.
- No node `INPUT_TYPES` / widget / socket touched by 5A or 5B -> no
  workflow JSON re-wire. Both new LLM calls are technical-slot,
  PD6-compliant (no `model_id` widget).
- **Build method:** two parallel subagents on disjoint NEW files; lead
  did all shared-file integration serially.

## State of the art
- Status Board: Sprints 0, 1, 2A-2E, 3A-3G, 4, 5A, 5B COMPLETE; 5C + 6
  NOT STARTED (blocked -- see the fork below).
- **5A + 5B are code-complete + regression-green but NOT live-validated.**
  Both change pipeline behaviour (5A adds an LLM call + injects
  continuity constraints into every compose prompt; 5B adds a critic
  call in the cascade). Prime Directive 1 wants ONE operator live
  ComfyUI episode run to confirm audio is unbroken. That live run now
  also still covers Sprint 3A and the carried cast/style/seed-removal
  batch (`61dda9c` / `906a57f` / `d0ea595`), none yet live-validated.
- New files: `nodes/_otr_continuity.py`, `tests/test_otr_continuity.py`,
  `nodes/_otr_story_critic.py`, `tests/test_otr_story_critic.py`.
- Integration touch points already in place: `OTR_LedgerScriptWriter.py`
  (section H.5 + `_build_line_request_for_beat`), `_otr_line_composer.py`
  (`LineRequest.continuity_slice` + `_build_user_prompt`),
  `_otr_freeze_cascade.py` (critic call on the non-terminal path).

## Sprint 5C open fork (DECIDE THIS FIRST)
5C re-composes the critic's `reroll_targets` lines. The blocker: the
Script Doctor + the 5B critic both run inside the `OTR_LedgerFreezeCascade`
node, which keeps only the **technical** model resident and does NOT
persist the outline context (`outline_spine`, `canon_header`, `theme`,
`style_descriptor`) that `compose_line_draft` needs to rebuild a
`LineRequest`. The plan says reroll hooks `compose_line_draft` (a
creative-slot job) -- the cascade physically cannot do that as-is.

**The fork -- which model slot the reroll re-composition runs on:**
- **Option A -- technical slot, in-cascade (lower risk).** Reroll
  re-composes flagged lines on the technical model already resident in
  the cascade, like the Script Doctor's existing post-cleanup edit pass.
  No node-surface change, no workflow JSON re-wire, no VRAM swap. Tagged
  `# LLM slot: technical`. Deviates from the plan's literal "creative
  slot" wording but matches how the Doctor already rewrites lines
  post-cleanup.
- **Option B -- creative slot, plan-faithful.** Add a
  `creative_writing_model` input socket to the `OTR_LedgerFreezeCascade`
  node + re-wire the workflow JSON; reroll re-composes via
  `compose_line_draft` on the creative model. Plan-faithful but adds a
  technical->creative->technical VRAM swap inside the cascade and a
  node-surface change that needs an RTX 5080 live-run to verify.

**Either option also requires:** the writer must stamp `outline_spine` /
`canon_header` / `theme` / `style_descriptor` onto `meta` (today they are
writer-local and lost by the time the cascade runs), so the cascade can
reconstruct a `LineRequest` for re-composition.

**Sprint 6 is gated behind the same decision** -- it adds
`render_selection` / `render_max_n` / `protagonist_only` /
`manual_line_ids` widgets to a node surface and re-wires the workflow
JSON regardless.

## Five reroll-loop design questions (recommended defaults, confirm or override)
1. **Re-entry point** -- reroll loop INSIDE the cascade (post-Doctor /
   post-critic, before Phase 7 / the final freeze). Recommended: inside
   the cascade. An outer loop wrapping the writer re-runs the whole
   expensive writer; the plan wants a *targeted* reroll.
2. **Critic re-invocation scope** -- re-run the FULL whole-script critic
   each cycle. Recommended: full re-run -- `run_story_critic` already
   exists, is cheap relative to a full episode rerun, and a rerolled
   line shifts its neighbours' continuity. Scoped re-critique is a later
   optimization, not v1.
3. **Line versioning** -- in-place overwrite of `line.text` via the
   ledger's existing `update_line_text`, with the prior text preserved
   in a `meta["reroll_history"]` audit list. Recommended: in-place +
   meta audit trail (a per-line `v1`/`v2` field is a ledger schema
   change -- avoid).
4. **Reroll context** -- the critic's `RerollTarget.hint` is passed as a
   HARD constraint via a new `LineRequest.reroll_hint` field, rendered as
   a REVISE block at the WRITE LINE tail of `_build_user_prompt` (the
   composer was already mapped for exactly this). Recommended: hint as a
   hard constraint, not a bare fresh prompt.
5. **Cycle counter granularity** -- per-batch: one critic pass + its
   whole `reroll_targets[]` batch = 1 cycle. `meta["cycle_count"]`
   increments once per round. Cap at 2; cycle 3 -> `needs_full_rerun`
   (follow the existing `review_ledger` verdict-stamp pattern -- restore
   snapshot, stamp `meta["reviewer_verdict"]`, build a disposition).
   Recommended: per-batch (the plan says "2 critic->reroll cycles").

## Immediate next steps
1. **Decide the Sprint 5C fork** (Option A vs B above) and confirm/override
   the five reroll-loop defaults.
2. **Build 5C.** Writer: stamp `outline_spine` / `canon_header` / `theme`
   / `style_descriptor` onto `meta`. Composer: add `LineRequest.reroll_hint`
   + render it in `_build_user_prompt`; add a `reroll_hint` keyword param
   to `compose_line` / `compose_line_draft`. Cascade: the reroll loop in
   `_otr_freeze_cascade.run_freeze_cascade` -- read
   `meta["story_critic_report"].reroll_targets`, rebuild a `LineRequest`
   per target from `meta` + the ledger line row, re-compose, write back
   via `update_line_text`, re-run `run_story_critic`, cap at 2 cycles via
   `meta["cycle_count"]`, cycle 3 -> `needs_full_rerun`. Option B also
   adds the `creative_writing_model` cascade input socket + workflow JSON
   re-wire. Regress, commit.
3. **Build Sprint 6** -- critic -> render coupling. `render_selection:
   dramatic_peaks_only` reads `render_priority[]`; `flat_lines[]` excluded
   unless rerolled; `arc_verdict in (mid_collapse, flat)` blocks render
   until critic cycle 2 clears; `render_max_n` default 6 + `protagonist_only`
   / `manual_line_ids` overrides. Adds widgets to a node surface ->
   workflow JSON re-wire (Prime Directive 3). Regress, commit, push.
4. **Operator live-run** -- ONE ComfyUI episode on current HEAD validates
   3A + 5A + 5B + the carried cast/style/seed batch (Prime Directive 1).
   Operator-gated: Jeffrey starts the run and pastes the console.
5. Update the plan Status Board + Build Progress Log per commit; final
   verification (suite + Bug Bible green, no BOM / 0-byte, HEAD==origin).

## Open questions
- The Sprint 5C fork (Option A vs B) -- Jeffrey's call. This is the one
  hard blocker.
- The five reroll-loop design questions above -- recommended defaults
  given; confirm or override.
- Carried: BUG-LOCAL-265/266/267/271 Bible-promotion via the Three-File
  Contract (all await the operator live verification).
- Pre-existing nits (not blocking, not introduced this session): the
  `_otr_line_composer.py` `__main__` self-test calls `compose_line`
  positionally (stale since the signature went keyword-only);
  `OTR_LedgerScriptWriter.py` `__main__` asserts 14 optional widgets vs
  15 live. Neither is pytest-collected.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
