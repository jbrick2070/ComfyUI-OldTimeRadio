# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-25 (Sprint 3A + Sprint 4)

## Core goal
The Story Pipeline Ledger Writer Hardening build, tracked in
`story_pipeline_sprint_plan_v4_audited.md`. Decompose every LLM call in
the script-writing + cleanup pipeline into single-job passes with
deterministic guards, so a structurally valid script can no longer ship
dramatically empty. This session landed Sprint 3A (the `compose_line`
split) and the Sprint 4 code-side VRAM verify. Remaining: Sprint 4
close-out, Sprints 5 + 6. ROADMAP stays parked until the whole build
lands.

## Tech stack & constraints
OTR `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`. CLAUDE.md + ROADMAP.md
+ BUG_LOG.md auto-load -- not repeated here. Live operational notes:
- **HEAD after this session = the `docs:` commit that carries this
  file.** Substantive code commits: `e24b327` (Sprint 3A), `6b9300e`
  (Sprint 4 + BUG-LOCAL-272). Predecessor HEAD `263d9cd`. Working tree
  clean except the parked `docs/s28_diff_tmp.txt` (never commit it).
- **Build tracking.** `story_pipeline_sprint_plan_v4_audited.md` is the
  single source of truth for PROGRESS; `BUG_LOG.md` owns BUGS. After a
  work session: append a dated Build Progress Log entry + update the
  Status Board in the same edit.
- **LLM backend is HF Transformers 5.5.0** (`model.generate()`). No
  llama.cpp, no GBNF.
- **Tests + git via Desktop Commander, `shell: "cmd"`.** Venv python:
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Full OTR
  suite: `cd /d <repo> && <venv python> -m pytest tests -q` (~29s, 2808
  collected). Bug Bible: `cd /d C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide && <venv python> -m pytest tests/bug_bible_regression.py -q`.
- **The Cowork Linux sandbox `Bash` mount is STALE for this repo.** Use
  Desktop Commander for tests + git. The Read/Write/Edit file tools DO
  write the real Windows file correctly.
- `cmd` gotchas: commit messages via the file tool to
  `.git\COMMIT_EDITMSG`, then `git commit -F`. For a long pytest run,
  redirect to a log file with `start_process` and read the file --
  a single `interact_with_process` call long enough to span the run
  drops the cmd session at the MCP timeout.
- Run Bug Bible + full OTR suite after every code change, unprompted.
- For console-output analysis, Jeffrey must paste it -- the AI has no
  real-time ComfyUI log access.

## What's done & decided
- **Sprint 3A COMPLETE (`e24b327`).** `compose_line` split into
  `compose_line_draft` (the creative job -- generate / format-strip /
  named-prefix strip / size-band / retry ladder; returns the draft
  string, raises `LineCompositionFailedError` on exhaustion) + a thin
  `compose_line` orchestrator (draft -> optional polish ->
  deterministic strip pipeline -> `LineResult`). New `cast_strip`
  strip step wraps `auto_remap_phantom` to remap a near-miss phantom
  name to its cast spelling at compose time, before the line enters
  the rolling `last_lines` window. `cast_strip` uses `threshold=1`
  (tighter than the reviewer's default 3) -- the regression guard
  caught a distance-3 false match ("CARLA" -> the news term "CERN");
  compose-time mutation with no story context must fire on slam-dunk
  typos only, and the reviewer keeps the full threshold-3 pass.
  `_word_bands` + `_strip_named_prefix` extracted as shared helpers.
  `_build_user_prompt` left intact (the audited "New design" only
  specified the `compose_line` split).
- **Sprint 4 IN PROGRESS (`6b9300e`).** Code-side verify: Zero-Prime
  Wash / Sovereignty Buffer (2.5 GB) / 2B-12B VRAM caps / bf16+tf32 --
  all confirmed already present; the VRAM gate fires for the renamed
  `high_quality_unsafe_on_16gb` tier. BUG-LOCAL-272 fixed (dead
  attention selector: `common_kwargs` hardcoded `sdpa` instead of
  consuming the computed `attn_impl`).
- **Regression green** (authoritative combined-tree run by the lead):
  full OTR suite 2787 passed / 21 skipped (2808 collected); Bug Bible
  16 passed / 7 skipped / 3 xfailed. 0 failed.
- No node `INPUT_TYPES` / widget / socket touched -> no workflow JSON
  re-wire. No LLM call added or removed.
- **Sprints 0, 1, 2A-2E, 3A, 3B-3G COMPLETE.**
- **Build method this session: two concurrent lanes on disjoint files**
  -- 3A lead-driven, Sprint 4 a parallel subagent.

## State of the art
- Build tracker Status Board: Sprints 0, 1, 2A-2E, 3A, 3B-3G COMPLETE;
  Sprint 4 IN PROGRESS; 5, 6 NOT STARTED.
- **3A is code-complete + regression-green but NOT live-validated.**
  3A changes script-pipeline behaviour (the `cast_strip` near-miss
  remap) -- Prime Directive 1 (audio is king) requires ONE operator
  live ComfyUI episode run before 3A is fully signed off, the same
  gate the 3B-3E wave used.
- **BUG-LOCAL-271 / 272** are FIXED in code; 271 still needs the live
  verification noted in `BUG_LOG.md`; 272 is test-verified.
- **Finding C still open.** The pipeline can still ship a structurally
  clean but THIN episode -- the quality critic is Sprint 5, not built.
- The cast/style/seed-removal commits (`61dda9c` / `906a57f` /
  `d0ea595`) are still NOT live-validated -- the next live run covers
  them too.

## Sprint 4 open items (before it can close)
1. **14B VRAM cap decision.** The plan's Sprint 4 asks for a
   14B -> 10.1 GiB cap. The 14B class currently has NO explicit cap --
   it falls into the `total_vram >= 12.0` Sovereignty branch (13.5 GiB
   budget on a 16 GB card). Jeffrey decides: keep the Sovereignty
   branch governing 14B, or add the explicit cap (touches the live
   load path, needs RTX 5080 verification).
2. **Prompt-cache bullet** (`cache_prompt=True`) -- not yet verified
   against the loader; outside the verify wave's four-bullet scope.
3. Operator live-RTX-5080 confirmation of the VRAM behaviour.

## Immediate next steps
1. **Live-validate Sprint 3A + the carried batch in ONE ComfyUI run on
   current HEAD.** Confirm: (a) the episode composes + freezes clean
   end-to-end and the final mp4 is produced (Prime Directive 1 -- audio
   must not break); (b) dialogue is sane via the new
   `compose_line_draft` path; (c) if a `cast_strip` remap fires, the
   console shows `cast_strip remapped N phantom(s)` and the remapped
   name is a real cast member; (d) the carried items -- BUG-271
   `wrong_char_id` repairs, the cast varies, no `seed` widget,
   WORD_BUDGET_DRIFT no longer false-fires. Operator-gated -- Jeffrey
   starts the run and pastes the console.
2. **Sprint 4 close-out** -- resolve the three open items above.
3. **Sprint 5** -- continuity ledger + story-quality critic + targeted
   reroll (new module `nodes/_otr_continuity.py`; the targeted reroll
   hooks `compose_line_draft`, which 3A now provides). The direct fix
   for Finding C. New LLM calls -> Prime Directive 6 each.
4. **Sprint 6** -- critic -> render coupling (ships with Sprint 5).

## Open questions
- Sprint 4's 14B cap (see "Sprint 4 open items" #1) -- Jeffrey's call.
- Carried: BUG-LOCAL-265/266/267/271 Bible-promotion via the
  Three-File Contract (all await live verification).
- The `_otr_line_composer.py` `__main__` self-test calls `compose_line`
  positionally -- stale since the signature became keyword-only. Not
  pytest-collected, so no suite is affected; a follow-up nit.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
