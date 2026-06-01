# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-06-01 (go-forward refactor SHIPPED)

## Core goal
The `docs/otr-go-forward-final.md` refactor is **DONE and pushed** -- all of
Sprints 0-4 shipped in 4 atomic commits on `v2.0-alpha`. The post-script story
spine now runs: conditional length pass -> 3-verdict QA defect router ->
beat-local micro-repair -> REJECT abort at the writer -> mechanical scrub. Every
headless gate is green. **The only work left is the operator's GPU validation**
(no AI session can run it). HEAD = `05ee2a2`, local == origin.

## Tech stack & constraints
Windows, RTX 5080 16GB. Venv python (full path): `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
Standard rules live in CLAUDE.md (branch policy, git-via-DC, VRAM ceiling, no-BOM,
no "dummy", LLM-slot tagging, run-regression-after-every-change) -- not repeated here.

Cowork-environment facts learned this session (NOT in CLAUDE.md):
- **Bug Bible repo is NOT checked out here** (`comfyui-custom-node-survival-guide`
  absent). Its regression can only run on the host. Headless gate here = full
  `tests/` + module self-tests + forbidden sweep.
- **File tools (Read/Write/Edit) DO reach the real Windows FS** (same view git/DC
  see) -- edits land, verified. **Glob has a path-param quirk (returns nothing);
  use Grep instead.** Subagent sandboxes were the stale-mount ones -- author on the
  main thread via file tools + Desktop Commander, not subagents.
- Full `tests/` runs in ~30s headless; **baseline now 3327 passed / 12 skipped /
  0 failed**. cmd mangles inline `python -c "..."` (quoting) -- write a script file
  or use `findstr`; `tail` is not a Windows command (read pytest tails via the
  Read tool on a redirected file, or DC `read_process_output -offset`).

## What's done & decided (this session)
- **4 commits, b1abbc9 -> 05ee2a2, all headless-green + pushed:**
  - `36cbb6b` Sprint 2: editor two-entry. Added `SMOOTH_DIALOGUE`/`CLARIFY_TURN`
    Tier-1 actions (+ `_FRESH_PROSE_ACTIONS`; Guard 3 now fences ALL fresh prose),
    `needs_length_normalization()`, `normalize_length()`, `micro_repair()`,
    `_make_scoped_validator` (action-whitelist + flagged-beat fence). Wired
    `normalize_length` as spine Stage 2.5.
  - `3267531` Sprint 1: QA grader -> router. `CreativeQAVerdict`/`run_creative_qa`
    REPLACED by `StoryQAVerdict`/`run_story_qa` (PASS/MICRO_REPAIR_NEEDED/REJECT +
    flagged_beats + reason + 6 evidence flags). Full spine reorchestration.
  - `e524c64` Sprint 3: writer raises `RuntimeError` on `meta["story_verdict"]=="REJECT"`
    right after the spine call. New `tests/test_reject_gate_sprint3.py`.
  - `05ee2a2` Sprint 4: scrub verified (already mechanical/fail-closed/never-rewrites);
    new `tests/test_scrub_handoff_sprint4.py`.
- **Key decisions:**
  - QA fails **OPEN to PASS** on any crash (never falsely REJECT/abort; audio is king).
  - QA judges the **VOICED view (character + announcer)** to match
    `_otr_radio_editor.voiced_beats`, so `flagged_beats` map 1:1 to `micro_repair`.
    (This was a real bug avoided -- the old grader was character-only.)
  - REJECT: **spine sets meta + unloads + skips scrub + returns, NEVER raises (PD1);
    the WRITER raises.** A QA crash (fail-open PASS) never reaches the raise.
  - Scrub always called `repair_available=False` (micro-repair runs upstream).
  - The **B-cap correction holds**: do NOT add a `target_words*2` per-line word cap
    (that `*4` is a token budget); real ceiling is `_MAX_OVERSIZE_RATIO=3.0`.
  - Executed serially on the main thread via Desktop Commander (NOT parallel
    subagents -- their mounts were stale); collapsed author+wire into 4 atomic
    per-sprint commits.

## State of the art
- `nodes/_otr_radio_editor.py` -- entries: `run_radio_editor` (general, now UNUSED
  by the spine), `normalize_length`, `micro_repair`, `needs_length_normalization`.
  Self-test 58/58.
- `nodes/_otr_creative_qa.py` -- `StoryQAVerdict` + `run_story_qa` (router; cold
  context, skeptical, high REJECT bar, fail-open). Old grader symbols GONE.
  Self-test 7/7.
- `nodes/_otr_story_spine.py` -- `run_post_script_spine` is the live flow above;
  helpers `_make_recompose_fn`, `_map_arc_indices`, `_verdict_summary` (router fields).
  meta key is now `story_qa_verdict` (was `creative_qa_verdict`). Self-test 10/10.
- `nodes/_otr_ledger_scrub.py` -- unchanged (already correct). Self-test 8/8.
- `nodes/OTR_LedgerScriptWriter.py` -- REJECT raise added after the spine call
  (~L3779), before the `meta["creative_model"]` stamp.
- New tests: `tests/test_reject_gate_sprint3.py` (4), `tests/test_scrub_handoff_sprint4.py` (4).
- No node-surface or workflow-JSON change anywhere (in-process helpers, D4/D5).

## Immediate next steps
1. **Operator GPU gate (host, not headless):** render an episode with the spine
   default-on, confirm **audio byte-identical** to the 2026-05-31 baseline on a
   clean (non-reject) run, and run a **scored A/B** vs that baseline. Tail logs
   with `scripts/otr_tail_logs.py`.
2. **Run the Bug Bible regression on the host** (repo absent in cowork):
   `python -m pytest C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -q`
3. Once a 2nd technical model is bound, **measure micro-repair / reject / dud
   rates** (go-forward Sec 7) to confirm QA earns its per-run call and tune the
   REJECT bar.
4. Decide: **remove `run_radio_editor`?** It is now unused by the spine (a general
   editor entry kept after the two-entry split). Cleanbreak would delete it +
   its self-test Test 4; harmless to keep as a general API.

## Open questions
- Does the default-on spine hold byte-identical audio on the real render? (Only
  the GPU A/B can answer; all headless gates pass.)
- Keep or delete the now-unused `run_radio_editor` (see step 4).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
