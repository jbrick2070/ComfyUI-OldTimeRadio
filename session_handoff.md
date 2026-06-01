# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-31 (story-quality + Opus)

## Core goal
Establish a story-quality baseline for the OTR writer across local vs OpenRouter models,
fix any bugs the testing surfaces, and hand the next sprint a single consolidated plan
(story-quality + bugs) that can be round-robined across parallel sessions. This session
proved the OpenRouter wiring, scored 6 + Opus runs, fixed one real bug, and wrote the
sprint-feed docs. **All work is committed to `v2.0-alpha`; HEAD = `e85db02`.** (Supersedes
the prior "OpenRouter Remote LLM" handoff -- that feature shipped; this went further.)

## Tech stack & constraints
Windows, RTX 5080 16GB, ComfyUI Desktop on `localhost:8000`. Venv python:
`C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. Branch `v2.0-alpha` only.
Hard rules in CLAUDE.md (auto-loaded, not repeated): audio byte-identical (PD1), 14.5GB
VRAM ceiling, wire every change into the workflow JSON, SFW/non-violent, never "dummy",
every LLM call tagged creative/technical + routed via the writer's model widgets (a new
editor/critic model = a new *broadcast output*, not a per-node widget), Bug Bible regression
after every code change. **Git: Desktop Commander `cmd` only, `.git\COMMIT_EDITMSG` + `-F`,
one push attempt, verify `local HEAD == origin HEAD`.**

## What's done & decided
- **OpenRouter local<->remote parity is CLOSED.** Same model completes the whole pipeline on
  both transports; the quality gap is *content*, not transport.
- **Opus is materially the best writer: 31/35 (89%)** vs best local mistral 27/35 (77%),
  remote mistral 67%. ~$0.47/episode. Wins on news-grounding (+2) and payoff (+1).
  **Recommended as default creative slot but NOT auto-flipped -- operator's call** (paid
  default + real "too dense for radio" caveat: Opus overshot 350->829 words).
- **Length sweet spot ~350 words** (60w too cramped, 500w no better/undershot).
- **Three-pass architecture is the agreed go-forward:** Pass 1 Opus writes -> Pass 2 tactical
  editor (radio length/cadence; = old C1) -> Pass 3 creative-QA critic (re-aim the existing
  `run_story_critic`, premium non-Opus model, rides the existing reroll loop).
- **BUG-LOCAL-296 FIXED + committed** (`d982821`): OpenRouter per-run cost budget never reset
  (`reset_run_budget()` defined+exported, never called) -> accumulated per-process. Wired into
  `OTR_LedgerScriptWriter.run()` top, defensively. +3 tests; full suite **3306 passed / 0
  failed**; live-confirmed (~43k tok one-episode budget).
- **Rejected:** overhauling the local writer (solid B+); shipping the F3 name-leak fix blind
  (operator review / maybe obsoleted by Pass 3); restart without verifying it came back up.
- Logged-not-fixed: **F3/BUG-295** (remote cast-name leak into stage directions), **F4**
  (inventor fail-closed), **F5** ("damn" slips SFW), **BUG-276/271** (cast-routing -> Bark
  crash on one remote run). All fail-closed/safe.

## State of the art
**All committed (`v2.0-alpha`, HEAD `e85db02`).** Session commits: `c2c1955` baseline ->
`d982821` BUG-296 -> `0224821` T6 -> `7526b2d` length sweep -> `3696627` Opus comparison +
plan -> `670636a` three-pass -> `e85db02` consolidated problem statement.

Read these docs, not the transcript:
- `docs/2026-05-31-otr-consolidated-problem-statement.md` -- **THE SPRINT FEED.** P0-P10
  (story+bugs), dependency table, 4 round-robin streams (A creative core / B robustness /
  C hygiene+safety / D stability+cleanup), open questions.
- `docs/2026-05-31-otr-story-quality-comparison.md` -- all-runs scorecard + the Opus 89%
  "The Green Book of Nights" full script + routing audit + cost.
- `docs/2026-05-31-otr-story-quality-baseline.md` -- 6 scored runs, F1-F5, length curve.
- `docs/openrouter-llm-call-improvement-plan.md` -- C1-C6 + three-pass architecture (model
  table w/ live pricing, control flow, effort, open questions).

Source changed this session (only this): `nodes/OTR_LedgerScriptWriter.py` `run()` calls
`_otr_openrouter_backend.reset_run_budget()` at the top (BUG-296);
`tests/test_openrouter_budget_reset.py` is the new regression.

**Live machine state (NOT in any tracked file):**
- ComfyUI server **PID 57540** up + idle, **still Opus-on-slot-A** from the Opus test. The
  launcher bat is reverted to mistral-nemo, so the **11:59 PM daily `OTR_API` task trigger
  relaunches on mistral-nemo** (cheap default). No queued runs.
- Headless launcher (OUTSIDE repo): `C:\Users\jeffr\Documents\ComfyUI\_otr_headless_launch.bat`
  -- `OTR_ENABLE_OPENROUTER=1`, `OPENROUTER_MODEL_A=mistralai/mistral-nemo` (reverted), seeds
  11/11, cost caps. Backup `..._launch.bat.premopus`. Opus run = edit slot A to
  `anthropic/claude-opus-4.8:nitro`, restart task, revert after.
- Scratch harness in repo ROOT (untracked; relocate to `tools/` under P10): `_otr_soak2.py`
  (submit/poll/full), `_otr_dump_scripts.py`, `_otr_show_episode.py`, `_otr_routing_audit.py`,
  `_otr_interrupt.py`, `_otr_or_anthropic.py`, `_otr_or_price*.py`, `_otr_or_mistral.py`,
  older `_otr_or_diag*.py`, `_otr_soak_ids.json`, `_otr_last_prompt.json`.

**Operational gotchas (save rework):**
- ComfyUI runs via Scheduled Task `OTR_API` -> `_otr_headless_launch.bat`. Console -> 
  `_otr_headless_soak.log` (NOT `%APPDATA%\ComfyUI\logs\comfyui.log`). OTR phase log:
  `otr_runtime.log`.
- Restart: `taskkill /F /T` the python whose cmdline has `main.py --port 8000`, then
  `schtasks /End /TN OTR_API` + `/Run /TN OTR_API`. **`/End` orphans the python** -- kill it
  first; confirm `netstat -ano | findstr 8000` empty before relaunch; poll
  `_otr_soak2.py poll --pid none` until `/queue` answers.
- `wmic` gone (Win11) -> PowerShell `Get-CimInstance Win32_Process`. cmd mangles inline
  `python -c "..."` -> use a script file. OpenRouter key in `HKCU\Environment` (winreg); never log it.
- Score from the writer/cascade **ledger** (`output\otr\episodes\<id>\...\*_ledger.json`);
  7 axes x 0-5; audio irrelevant to the score.

## Immediate next steps
1. **Dispatch Stream B first** (cheap, isolated, unblocks valid runs): P4 relax the
   `news_interpreter` key-term gate (`news_interpreter.py:803` -- accept LLM-judge-supported
   paraphrase, not just verbatim) + P5 harden the inventor pass (dedup keys + one repair).
   Two valid runs died here (T7; the BUG-296 verify run).
2. **Then Stream A** (headline): build the three-pass rig per the improvement-plan doc --
   `editor_model`/`critic_model` broadcast outputs (PD6), Pass-2 editor prompt, Pass-3
   creative-QA critic, per-model prompt riders + temps.
3. **Streams C (F3/C5, F5 SFW, C6) and D (BUG-276 crash + dead-code cleanup)** in parallel;
   re-check F3 after Stream A.
4. Each stream exit gate: Bug Bible + core + audio byte-identical green **and** a scored A/B
   vs the 2026-05-31 baseline.
5. (If asked) draft the **round-robin kickoff-prompt template** Jeffrey requested.

## Open questions
- Pass 2 editor model (local / Haiku 4.5 / GPT-4o-mini)? Pass 3 critic model (Sonnet 4.6 rec /
  Haiku / local; not Opus-self-judge)?
- Flip production default to Opus? Per-episode budget cap for a paid default?
- Dead-code pointer (P10): confirm whether a specific committed "dead-code synthesis" doc
  exists, else P10 uses ROADMAP staleness audit + forbidden-sweep + BUG_LOG cleanbreak +
  the scratch-file cleanup above.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps. Acknowledge when you're ready to start."
