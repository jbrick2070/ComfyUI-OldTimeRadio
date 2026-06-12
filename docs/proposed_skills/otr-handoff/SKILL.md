---
name: otr-handoff
description: >-
  Bidirectional OTR build baton -- RESUME at the start of a fresh window, HAND
  OFF at the end, with anti-drift guardrails. On a fresh window it reads the
  single source of truth (docs/GO_FORWARD_PLAN.md) + the otr-build-tracker
  dashboard + git, states the CURRENT STEP, and waits for go. At wrap-up it
  refreshes GO_FORWARD_PLAN.md, updates the tracker, and prints a kickoff. Use
  when starting a new chat, "resume/continue the OTR build", "pick up where we
  left off", "what's the current step", OR "hand off the build", "save the
  build context", "session handoff", "wrap up for a new chat". Replaces the old
  otr-build-handoff and otr-video-handoff skills.
---

# OTR Build Baton (bidirectional, anti-drift)

**Why:** across many windows the build drifts -- a fresh chat reads a stale doc or
an OLD/parked sprint, and the handoff gets rewritten around whatever was last in
focus. This skill makes the baton-pass deterministic. There is now ONE source of
truth: the git-tracked **`docs/GO_FORWARD_PLAN.md`** (forward order + runway +
open tickets + current step + hard rules). The `otr-build-tracker` artifact is the
visual DASHBOARD that mirrors it. `docs/VIDEO_BUILD_HANDOFF.md` and
`docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` section 0 are thin pointers to
GO_FORWARD_PLAN.md (the 3D plan is the detail spec for forward-order item 5).

**Pick the direction by context:** fresh window / "resume" / "what's the current
step" -> **RESUME**. "hand off" / "save state" / "wrapping up" / running low on
context -> **HAND OFF**. If genuinely unsure, ask the operator one line. Write docs
only -- do NOT commit unless the operator says to (the operator gates pushes).

## HARD RULES (apply in BOTH directions; they live verbatim in GO_FORWARD_PLAN.md section 2)
- The forward order is GO_FORWARD_PLAN.md section 3. Do NOT start/resume/"continue"
  any OTHER sprint -- NOT story-spine, NOT story-pipeline, NOT the broader audio
  stack, NOT any other ROADMAP item. PARKED.
- The audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no
  `-shortest`); `test_audio_byte_identical` stays GREEN. The ONLY sanctioned audio
  work is the character-voice "whiny" fix -- UPSTREAM TTS only.
- EVERY session (planner AND coder) UPDATES GO_FORWARD_PLAN.md + the
  `otr-build-tracker` dashboard (content; preserve the gauge + lanes styling).
  Never tell a window "don't touch the tracker".
- Ignore any stale `session_handoff.md` / memory / ROADMAP "active" entry.
  GO_FORWARD_PLAN.md is the source of truth until the operator says otherwise.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100%
  local/offline; determinism (seed-keyed); every in-render fallback LOUD; UTF-8 no
  BOM; SFW; V-12 dependency isolation.
- v2.0 PRODUCTION / main is GATED until the operator's work is done; a
  `v2.0-alpha-stable` tag on `v2.0-alpha` is fine; prod/main is NOT.

## RESUME (start of a fresh window) -- orient, then STOP
Do NOT write code or docs yet.
1. Read `docs/GO_FORWARD_PLAN.md` IN FULL (current step + hard rules + forward
   order + runway + open tickets); skim the `otr-build-tracker` dashboard; run
   `git log --oneline -12` + `git status` on `v2.0-alpha`.
2. The 3D detail spec (forward-order item 5) is
   `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` -- read it only when the current
   step is in the 3D sprints.
3. State the CURRENT STEP (GO_FORWARD_PLAN.md section 1, cross-checked vs the
   tracker + git) + a <=5-line summary of its build-vs-stub list + pass/fail
   (acceptance) assertions, to prove comprehension.
4. If GO_FORWARD_PLAN.md and the tracker disagree on the current step, say so and
   ask which is right -- do NOT guess.
5. STOP and wait for the operator's GO. No code until they confirm.

## HAND OFF (end of a session) -- refresh state, then print the kickoff
1. Refresh `docs/GO_FORWARD_PLAN.md`: update the "Last updated / HEAD" line, the
   CURRENT STEP (section 1), WHERE WE ARE (section 6, only from THIS conversation:
   done / in progress / next; open verify items; last commit + branch), and tick
   any advanced item in the runway (section 4) + open tickets (section 5). Keep it
   lean and forward-only; other work -> PARKED (section 8).
2. UPDATE the `otr-build-tracker` dashboard to match (gauge + lanes + a session
   row; preserve styling).
3. Print this kickoff for the operator to paste as message #1 of the next window
   (fill <current step>):

```
Run the otr-handoff skill to resume. (Manual fallback:) read docs/GO_FORWARD_PLAN.md IN FULL + skim the otr-build-tracker dashboard + git log/status. ACTIVE step: <current step>. Tell me the current step + a 5-line summary to prove you've got it, then STOP -- no code until I confirm. Rules: forward order = GO_FORWARD_PLAN.md section 3; no other sprints (story-spine/story-pipeline/broader-audio PARKED); audio spine FROZEN (only the whiny-voice fix is sanctioned, upstream TTS only); EVERY session updates GO_FORWARD_PLAN.md + the tracker; 100% local; single resident heavy <=14.5GB; determinism; LOUD fallbacks; UTF-8 no BOM; SFW; commit per chunk, do NOT push unprompted; prod/main GATED.
```

## Guardrails for the agent running this skill
- `docs/GO_FORWARD_PLAN.md` is the DURABLE source of truth -- read it on RESUME,
  refresh it on HAND OFF. The tracker mirrors it.
- The forward order is GO_FORWARD_PLAN.md section 3; never fold other ROADMAP
  sprints in as active.
- Fill "WHERE WE ARE" only from the actual conversation; no invented status.
- Docs only; do not git-commit unless the operator says to.
