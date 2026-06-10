---
name: otr-3d-handoff
description: >-
  Hand off the OTR 3D TOOLKIT sprint (character_3d build) to a fresh conversation
  WITHOUT it drifting into old or shipped sprints (the 2D video build is SHIPPED at
  B-ship; story-spine, story-pipeline, and the audio refactor are PARKED/FROZEN).
  Use when wrapping up a 3D coding session, starting a new 3D window, saving 3D build
  context, or saying "hand off the 3D build", "3D session handoff", "wrap up the 3D
  sprint for a new chat", "save the 3D context", "next 3D window". Produces (1) a
  canonical 3D_BUILD_HANDOFF.md pinning the active ticket + the hard no-other-sprints
  rules + the SPRINT_STATUS.json discipline, and (2) a copy-paste kickoff prompt for
  the next session. This is session-handoff specialized for the 3D toolkit sprint,
  with the anti-drift pin and the live-tracker update policy baked in.
---

# OTR 3D-Toolkit Handoff (anti-drift)

**Why this exists:** a fresh conversation reading ROADMAP / memory / a stale
`session_handoff.md` tends to resume the most salient OLD work — the (SHIPPED) 2D
video build, story-pipeline, or an audio sprint — instead of the 3D toolkit sprint.
This skill removes that ambiguity, and it keeps the live sprint tracker honest: the
tracker artifact renders `SPRINT_STATUS.json`, so a handoff that skips the status
update silently breaks the operator's board.

When invoked, do BOTH steps below. Write docs only — do NOT commit code (the operator
or the coder window commits per its ticket; the handoff DOC itself may be committed
docs-only if the operator asks).

## The fixed facts (bake these into every handoff; do not re-derive)

- **Plan of record:** `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md` (the CONTRACT)
  + `docs/2026-06-09-3d-toolkit/SUBAGENT_SPRINT_PLAN.md` (the ticket DAG: lanes
  R repo / T template / G GPU-spikes / W live-build; one gate per window).
- **Status of record:** `docs/2026-06-09-3d-toolkit/SPRINT_STATUS.json` — every
  ticket completion updates it (ticket status/note/updated + top-level
  updated/updated_by/next_action) IN THE SAME COMMIT as the code. The operator's
  live tracker reads this file; stale JSON = lying board.
- **Gates:** S-3D-0 (no-compile sidecar probe) gates Lane G; T2b keystone GO is
  failures <= 4/25 (5/25 = NO-GO, `keystone_gate` is already strict); Lane W opens
  only after R+T+G green AND operator go. Lane R/T are CPU no-regret and may run
  before/without G.
- **Invariants:** frozen master audio + mux-LAST + byte-identical
  (`test_audio_byte_identical` green); 14.0 GB 3D sidecar sub-ceiling (14.5 machine);
  protected cu130 main venv never touched (3D = cu128 sidecars, prebuilt wheels only
  in v1 — any source build is a gate failure, not a workaround); V-6/V-11/V-12;
  fail-closed, every fallback LOUD; UTF-8 no BOM; SFW (use "placeholder", never the
  d-word); commit per ticket; do NOT push unprompted (docs-only pushes excepted).
- **Bug Bible:** separate repo `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`
  (`BUG_BIBLE.yaml` at root). Run its regression by cd-ing to THAT repo root and
  invoking the venv python with the RELATIVE backslash path
  `tests\bug_bible_regression.py` — an absolute forward-slash path through cmd fails
  to collect (the recurring "can't find it"). Run Bug Bible + core + dropdown
  regression after every code change, without being asked.
- **ComfyUI gotchas that bite every 3D session:** .py edits need a ComfyUI RESTART
  (module cache) before V-6 dropdown checks; heavy in-process fallback forwards run
  on the EXECUTOR thread via `OTR_VideoRenderBatch` (/prompt), never a background
  thread; VRAM gates are machine-wide NVML, never `get_free_memory`; never
  `unload_all_models` — patcher detach only.

## Step 1 — write/refresh the canonical handoff doc

Overwrite `docs/2026-06-09-3d-toolkit/3D_BUILD_HANDOFF.md` with, in this order:

- **ACTIVE MISSION (the only active build):** the OTR 3D Toolkit sprint, per the two
  plan docs above. Current ticket: <FILL from this conversation — e.g. "R3
  request-builder migration", "G0 S-3D-0 probe (operator present)"> and its LANE +
  the gate it ends at.
- **HARD RULES (copy verbatim):**
  - Do NOT start / resume / "continue" any other sprint — NOT the 2D video build
    (SHIPPED at B-ship; only touch its files where a ticket explicitly names them),
    NOT story-spine / story-pipeline, NOT any audio sprint, NOT other ROADMAP items.
    They are PARKED or SHIPPED.
  - The audio pipeline is SHIPPED; the audio script ledger is FROZEN (read-only).
    `ledger['audio']` is never written by 3D code.
  - Ignore any stale `session_handoff.md` / memory / ROADMAP entry implying other
    "active" work. The 3D toolkit sprint is the ONLY active build until the operator
    says otherwise.
  - Respect the ticket DAG: do not open a ticket whose dependencies are not `done`
    in SPRINT_STATUS.json; do not edit files owned by another lane's open ticket.
  - End every ticket by updating SPRINT_STATUS.json in the same commit as the code.
- **WHERE WE ARE:** a tight, factual list filled ONLY from THIS conversation — tickets
  finished (with commit hashes), the ticket in progress + exactly where it stopped,
  gate outcomes (S-3D-0 / T2b verdicts if run), open blockers (also mirrored into
  SPRINT_STATUS.json `blockers`), last commit / branch / HEAD-vs-origin. Do not
  invent progress.
- **PARKED — not now:** anything else the conversation touched, listed so the next
  window knows it is intentionally untouched.
- **FIRST ACTIONS for the next session:** (1) read both plan docs + SPRINT_STATUS.json;
  (2) summarize the active ticket's goal, files, and gate in <=6 lines to prove
  comprehension; (3) verify repo state (HEAD==origin, suite green) before any edit;
  (4) wait for operator go before writing code.

Keep it lean and forward-only. Update SPRINT_STATUS.json's `next_action` +
`updated`/`updated_by` to match the handoff while you are at it (docs-only change).

## Step 2 — print the kickoff prompt

Output this block for the operator to paste as message #1 of the new conversation
(fill <ticket> and <lane/gate>):

```
You are starting a fresh session with ONE job: the OTR 3D Toolkit sprint. Nothing else.
ACTIVE = 3D toolkit, ticket <ticket> (lane <lane>, ends at gate <gate>).
READ FIRST (in order): docs/2026-06-09-3d-toolkit/3D_BUILD_HANDOFF.md, SUBAGENT_SPRINT_PLAN.md,
SPRINT_STATUS.json, then the cited sections of 3D_TOOLKIT_PLAN.md. The plan is the contract.
HARD RULES:
- Do NOT start/resume any other sprint — the 2D video build is SHIPPED (B-ship), story-* and
  audio are PARKED/FROZEN. Ignore stale session_handoff.md / memory "active" entries.
- Audio is FROZEN: byte-identical master, mux-LAST, ledger['audio'] read-only, test_audio_byte_identical green.
- Respect the ticket DAG + lane file ownership; one ticket, one gate, one commit (code + SPRINT_STATUS.json together).
- 14.0 GB 3D sub-ceiling; cu128 sidecars, prebuilt wheels only (a source build = gate failure); protected cu130 venv untouched.
- Bug Bible + core + dropdown regression after every change (survival-guide repo, cd to ITS root, relative path tests\bug_bible_regression.py).
FIRST ACTIONS (then STOP and wait for my go): (1) read the docs above; (2) give me a <=6-line summary
of <ticket> (goal, files, gate) to prove you've got it; (3) verify HEAD==origin + suite green; (4) no code until I confirm.
```

## Guardrails for the agent running this skill

- Never fold other ROADMAP sprints into the handoff; never mark the video build active.
- Fill "WHERE WE ARE" only from the actual conversation; no invented status, no
  invented gate verdicts.
- If the session ended mid-ticket, say so precisely (file + the next concrete edit),
  and set the ticket `in_progress` with a note in SPRINT_STATUS.json — never `done`.
- Docs only; never commit code from this skill.
