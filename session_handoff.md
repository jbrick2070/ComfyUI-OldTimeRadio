# Session Handoff -- ComfyUI-OldTimeRadio (OTR) -- 2026-06-06 (episode-budget scaling fix + parked caption/audio ship)

## Core goal
Fix a live full-episode crash and make episode length scale cleanly. A 780-word
run on `act_count=auto` crashed in outline generation with an
`OutlineBudgetViolation` AFTER ~28 OpenRouter (Opus 4.8) calls. Root cause: the
Phase-2A episode budget could allocate per-phase word targets that the fixed
beat structure physically can't hold. Fixed in two commits (widen per-beat cap +
fail-fast guard; then auto-scale the act count), and shipped the two parked
output-polish fixes (caption name-border, louder master) from the prior session.
THREAD NOTE: this is the engine / output-polish / budget thread, NOT the OTR
video-engine platform build (that lives in VIDEO_BUILD_HANDOFF.md + the
otr-video-handoff skill -- do not conflate).

## Tech stack & constraints (session-specific; CLAUDE.md + memory auto-load the rest)
- ComfyUI Desktop on :8000, Windows, RTX 5080 16GB. Main venv
  `C:\Users\jeffr\Documents\ComfyUI\.venv`. `.py` edits need a ComfyUI RESTART
  (module cache); JSON/voice-bank hot-reload.
- **Verify on the real machine via Desktop Commander, NOT sandbox bash.** The
  Cowork sandbox mount serves STALE/TORN reads of files just edited with the
  file tools -> it reported a phantom `SyntaxError` on an untouched line this
  session. Run py_compile / pytest through DC + venv python against the real
  files. (Related: memory `feedback_cowork_sandbox_corruption`.)
- DC pytest recipe: `shell:"cmd"`, `set HF_HOME=C:\ComfyUI-Models\huggingface`,
  venv python `-m pytest -q ... > scripts\_otr_*.txt 2>&1 & type scripts\_otr_*.txt`
  (DC won't reliably capture external-exe stdout; redirect + type). DC PowerShell
  strips `$`. Bug Bible lives in the separate repo
  `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`, run with
  a RELATIVE backslash path from its root (memory `reference_bug_bible_location`).
- Git per CLAUDE.md + memory: stage/commit/push via Desktop Commander (cmd),
  UTF-8 no BOM, one push attempt then verify HEAD-match + no-BOM + AST. Branch
  `v2.0-alpha`, remote `https://github.com/jbrick2070/ComfyUI-OldTimeRadio.git`.

## What's done & decided (all committed + pushed to origin/v2.0-alpha)
- **`9e7db41` fix(budget): per-beat ceiling widening + fail-fast guard.** In
  `compute_episode_budget` (`nodes/_otr_episode_budget.py`): each phase now gets
  `eff_hi = min(BEAT_WORD_HARD_MAX, max(base_hi, ceil(per_phase_words/n_beats)))`
  so the fixed beat count can actually hold the requested length. `base_lo`
  untouched; short episodes are byte-identical (350/3-act stays `(20,35)`).
  `BEAT_WORD_HARD_MAX = 80` matches the Stage-3 `Beat.target_words` schema
  (`ge=3, le=80`). A guard raises `InvalidEpisodeBudgetError` BEFORE any LLM call
  if a phase still can't fit (saving the ~28-call burn). Old bug mechanism:
  `_allocate_phase_target_words` hard-caps every beat at `hi`, so 3 acts x 14
  beats x 35 = 490-word ceiling < 780. Default `target_words` is 350; 780 was
  user-typed.
- **`bbdb21a` fix(captions)** (parked, prior session): name label override
  `{\b1\bord1\3c{color}}NAME` -> `{\b1}NAME` so the speaker name shares the one
  caption box with no border/color. `nodes/_otr_captions.py` + test.
- **`7758b4b` feat(audio)** (parked, prior session): louder final master via
  `_master_loudness` (makeup gain + tanh soft-knee limiter + re-trim to -1 dBFS,
  peak-safe, deterministic). Env `OTR_MASTER_MAKEUP_DB` default 4.0 (0 = legacy
  pure peak-normalize). `nodes/scene_sequencer.py` (EpisodeAssembler).
- **`4d77153` feat(budget): auto-scale act_count with length.** New
  `auto_act_count(target_words)` in `_otr_episode_budget.py` (exported); the
  writer's `act_count='auto'` path (`OTR_LedgerScriptWriter.py`) now calls it
  instead of `default_act_count`. Picks the FEWEST acts >= the narrative
  `default_act_count` floor whose widened budget fits (so beats stretch toward 80
  within a low act count before adding an act); climbs only when forced; beyond
  the ceiling returns the max-capacity act so `compute_episode_budget`'s guard
  reports the true max. Feasibility is decided by a throwaway
  `compute_episode_budget` build per candidate (one source of truth).
- **Length envelope (both fixes live):** auto stays **3 acts <= ~1364 words**,
  climbs to **5 acts ~1365-1502**, **6 acts ~1503-1820**; **> ~1820 = clean
  fail-fast** (engine structural ceiling from the 80-words/beat x 32-beats
  schema caps). Per-act feasible ranges: act3 238..1364, act4 273..1335, act5
  417..1502, act6 481..1820, act7 580..1502. Manual act_count 1-7 still works and
  stays strict (infeasible pick -> same clear guard error, not a crash).
- **OpenRouter dropdowns refreshed** (not a crash cause; all 28 Opus calls had
  succeeded). Ran `scripts/otr_openrouter_refresh.py` -> 344 models live;
  `anthropic/claude-opus-4.8` now in `models/openrouter_models.json` (gitignored,
  per-machine), so the "not in local catalog cache" warning clears next run.
  Node never auto-fetches by design; user will refresh manually when new models
  appear (declined scheduling).
- **Regression GREEN:** 369 (core+budget+outline+writer+openrouter) + 239
  (budget+auto-scale+core+writer+outline) + 117 targeted, plus the budget module
  self-test, plus Bug Bible 16 passed / 7 skip / 3 xfail. All four commits
  verified: HEAD local==origin `4d77153`, no BOM, AST-parse clean.
- Rejected this session: "auto-add acts" as the primary lever (Option 1) and a
  static per-beat cap bump -- chose dynamic widen-then-climb (Option A) per the
  operator's "fewer acts, longer beats" preference.

## State of the art
- HEAD `4d77153` on `v2.0-alpha`, local == origin. Working tree clean except:
  `M session_handoff.md` (this file), `?? docs/2026-06-05-video-planning__carryover-problem-statement.md`
  (other thread), `?? scripts/_otr_matrix_out/` + `scripts/_otr_*.txt` (this
  session's scratch test-capture files).
- Files changed this session, all committed: `nodes/_otr_episode_budget.py`
  (widening + guard + `auto_act_count` + `BEAT_WORD_HARD_MAX` +
  `_max_target_words_for_act_count`), `nodes/OTR_LedgerScriptWriter.py` (auto
  path -> `auto_act_count`), `tests/test_phase2a_episode_budget.py`
  (`TestAutoActCount` + widening/guard cases), `nodes/_otr_captions.py`,
  `tests/test_otr_captions.py`, `nodes/scene_sequencer.py`.
- **Operator restarted Comfy Desktop** (all four fixes now loaded) and queued a
  live 740-word episode on `act_count=auto` (-> 3 acts, feasible). Run in
  progress at handoff time ("1. Story Writer (LPL v2.0): 0%"). This is the live
  validation that the 780-class crash is fixed.

## Immediate next steps
1. **Watch the running 740-word episode finish.** Confirm it clears outline
   generation (no `OutlineBudgetViolation`) and renders to a fresh
   `output/otr/obs/<ep>_procgen_blended.mp4`. Then eyeball the three shipped
   fixes together: (a) captions have NO box/border behind the speaker name,
   (b) final audio is audibly louder -- tune `OTR_MASTER_MAKEUP_DB` (6 = hotter,
   0 = old) if +4 dB isn't right, (c) the 740-word script reads full-length.
2. **Spot-check auto-scale on a long run** if desired: set `target_words` ~1500
   on auto and confirm it auto-selects 5 acts and completes; ~1900 should
   fail-fast instantly with the "lower target_words / raise act_count" message.
3. **Stale baseline:** `tests/fixtures/baseline_v1.5.wav` is stale vs the louder
   master. Re-capture (`python tests/test_audio_byte_identical.py
   --capture-baseline`) or retire that GPU gate. Structural tests pass; the GPU
   byte-identical test already skips headless.
4. Optional docs: note the new length envelope (auto scales acts to ~1820) +
   `OTR_MASTER_MAKEUP_DB` in README/openrouter-setup as part of the pending
   newbie README refresh.

## Open questions
- Is +4 dB (`OTR_MASTER_MAKEUP_DB`) the master loudness the operator wants, after
  hearing it live?
- Re-capture the v1.5 audio byte-identical baseline, or retire that gate?
- Pursue "Option B" true long-form (> ~1820 words) -- scale beat COUNT with
  length + lift the 80-word/32-beat schema caps? That's a roundtable-worthy
  architectural change, not yet scoped.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
