# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-31 (story spine LIVE -> go-forward refactor)

## Core goal
The story spine shipped and is ON by default. The NEXT work is
`docs/otr-go-forward-final.md` -- it REFACTORS the spine into a refined
architecture (defect-router QA, conditional length pass + fenced
micro-repair, REJECT abort). Execute its Sprints 0-4 with the loop
`REVIEW -> CODE -> WIRE -> REGRESS -> COMMIT`, parallel subagents for the
disjoint module authoring, serial wiring into the spine + writer.

**All work committed to `v2.0-alpha`; HEAD = `0d16620`.**

## What shipped this session (8 commits, all headless-green + pushed)
`4bb382a` spine doc rewritten to live tree (Part 0 = 9 drift corrections) ->
`9a04cfe` Stream B1 news-grounding rider -> `c9fe300` Stream A outline arc
gate (+13 tests) -> `46270ed`/`f829d58`/`3343de7` Streams C/D/E modules ->
`b73f57a` Wave 2 wiring (in-process, was env-gated) -> `0d16620` spine
DEFAULT-ON + full-power editor seam.

The live pipeline now (env `OTR_ENABLE_STORY_SPINE`, DEFAULT ON, =0 to
opt out): writer -> Stage 3 critic -> Stage 3.5 editor repair (on
REPAIR_ONCE) -> writer-LLM unload -> Stage 4 scrub. Orchestrated in
`nodes/_otr_story_spine.py` (`run_post_script_spine`), called from
`OTR_LedgerScriptWriter.run()` right after the reflection (the gated
if/else that replaced the L3754 unload block). Never raises (PD1).

## Current module state (what the go-forward refactors)
- `nodes/_otr_creative_qa.py` -- `CreativeQAVerdict` GRADER (PASS/
  REPAIR_ONCE/FAIL + leak/SFW fields), `run_creative_qa(led, fn, *,
  critic_model_id)`. Sprint 1 -> convert to `StoryQAVerdict` ROUTER
  (PASS/MICRO_REPAIR_NEEDED/REJECT + evidence flags dead_ending/
  broken_turn/flat_contrast/unclear_grounding/chopped_dialogue/
  pacing_failure); DROP the leak/SFW/JSON checks (scrub owns them).
- `nodes/_otr_radio_editor.py` -- single `run_radio_editor(ledger, *,
  editor_model, slot_fn, recompose_fn, turn/button_beat_index, apply)`,
  Guards 1/2/3, apply_plan. Sprint 2 -> two entries: `normalize_length(led)`
  (runs only when total>350+/-20% OR any line over spoken cap) +
  `micro_repair(led, flagged_beats)` (beat-local, flagged only, 1 cycle);
  add actions SMOOTH_DIALOGUE / CLARIFY_TURN; keep the 3 guards.
- `nodes/_otr_ledger_scrub.py` -- `scrub_ledger(led, *, repair_available)`.
  Sprint 4 -> wire with `repair_available=False` (micro-repair runs
  upstream now); confirm mechanical-only, fail-closed, LAST.
- `nodes/_otr_story_spine.py` -- orchestrator. Rework for: conditional
  `normalize_length`, the 3-verdict router, `micro_repair` on flagged
  beats, and the REJECT signal (Sprint 3: set meta["story_verdict"]=
  "REJECT" + meta["story_reject_reason"], skip scrub, return normally).
  Full-power seams already here: `_make_recompose_fn` (wraps compose_line),
  `_map_arc_indices` (outline turn/button -> voiced-view index by beat_id).
- `OTR_LedgerScriptWriter.run()` -- Sprint 3: after the spine call, before
  the `meta["creative_model"]` stamp (~L3764) and the return, add
  `if meta.get("story_verdict")=="REJECT": raise RuntimeError(...)` --
  matches the existing fail-loud pattern; only new raise.

## REVIEW findings vs the go-forward doc (carry these in)
1. **Sprint 0 B is the recurring D1/D2 drift.** The doc's
   `min(cap, target_words*2)` per-line cap is WRONG: that `*4` in
   `_otr_line_composer.compose_line_draft` (~L1606) is a `max_new_tokens`
   TOKEN budget, not a word cap. Pulling it to x2 truncates lines ->
   audio hazard. The real word ceiling is `_word_bands` `word_cap =
   target_words * _MAX_OVERSIZE_RATIO` (3.0). DO NOT add a x2 word cap.
   The B rider is already shipped (9a04cfe) -- that is the right B lever.
2. A (turning_point/button on Outline) + E (scrub) are shipped and pass.
3. The QA "always runs, model-agnostic" rule already holds -- the critic
   routes to `resolved["technical_model"]`, no model-class skip.

## Tech stack & how to run (verified this session)
Windows, RTX 5080 16GB. Venv python (full path, no spaces, no quotes
needed in cmd): `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- Headless gates (any session, fast, no GPU): Bug Bible
  `python -m pytest C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py -q`
  + full `pytest tests/` (3342 passed / 0 failed baseline) + each module's
  `python nodes\_otr_*.py` self-test + `python docs/_s28_forbidden_sweep.py`.
- GPU-only gates (operator): audio byte-identical render + scored A/B vs
  the 2026-05-31 baseline. Streams change the writer, not the audio path.
- Git: Desktop Commander **cmd** shell only. `del .git\index.lock 2>nul`
  first (subagent git leaves stale locks). `git add <files> && git commit
  -F .git\COMMIT_EDITMSG` (write the message via the file tool first) &&
  `git push origin v2.0-alpha` && verify `git rev-parse HEAD` ==
  `git rev-parse origin/v2.0-alpha`. cmd mangles inline `python -c "..."`
  and quoted paths -> use script files / py_compile on bare filenames.
- Stale-mount caution: subagent sandboxes showed phantom file/git state
  this session; the venv-python self-tests on the real FS (via DC) are
  authoritative. Verify any "broken file" claim with py_compile, not a
  subagent's mount view.

## Immediate next steps (go-forward §7 wave plan)
1. Sprint 0: confirm A/B/E (above) -- no code; reaffirm the B-cap
   correction. No commit.
2. Wave 1 (3 parallel subagents, rework dormant + `__main__` self-test,
   report diff, no commit, no wiring): Sprint 1 `_otr_creative_qa.py`
   router; Sprint 2 `_otr_radio_editor.py` two-entry; Sprint 4
   `_otr_ledger_scrub.py` verify/adjust. Brief each with the go-forward
   §2 rules + §9 constraints + their sprint section.
3. Wave 2 (one thread, regress + commit each, flow order): editor length
   pass -> QA router -> micro-repair -> REJECT abort (Sprint 3) -> scrub
   repair_available=False. Re-run full `tests/` + Bug Bible after each.
4. Each module self-test green + full suite 0-fail before commit. GPU
   audio/A-B is the operator's final gate (spine is default-on now).

## Resume instructions
Open a fresh window, attach this file + `docs/otr-go-forward-final.md`,
and say: "Read this handoff + the go-forward doc, run Sprint 0 verify,
then dispatch the Wave-1 subagents. Acknowledge when ready."
