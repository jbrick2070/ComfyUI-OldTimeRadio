# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-28 (overnight refresh)

This file supersedes the 2026-05-27 handoff. The overnight autonomous
session refined the Sprint 1-5 build plan against live source, captured
the Sprint 0 baseline scaffold, and surfaced a Sprint 1 adapter-layer
finding. No code changes were made overnight by design (no Desktop
Commander push, no live ComfyUI tests, no Windows .venv pytest run).

The morning runbook is `docs/2026-05-28-morning-runbook.md`. Read it
first.

## Core goal

OTR v2.0 Visual Drama Engine. Sprint 10B Waves 1-3 (multi-agent
writers' room) are SHIPPED and live. Story-quality ceiling is ~3.70
(Stage 7 critic rubric mean, ship threshold 3.5). Sprint 1-5 in
`docs/OTR_story_quality_build_plan.md` push past the ceiling --
starting with delivery integrity (`dialogue_slot_id` keystone +
Extract scope reduction).

## Tech stack & constraints

Unchanged from 2026-05-27 handoff. Highlights:
- Python 3.12 via `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`.
- ComfyUI Desktop at localhost:8000.
- Mistral-Nemo-Instruct-2407 NF4 4-bit, single-GPU RTX 5080 Laptop 16 GB
  VRAM. VRAM ceiling 14.5 GB peak.
- Two-slot LLM architecture (PD6): no `model_id` widget on consumer
  nodes; forbidden-pattern sweep enforces.
- Branch: **v2.0-alpha ONLY**.
- Git push: Desktop Commander `cmd` shell only. Commit messages via
  `.git\COMMIT_EDITMSG` + `git commit -F`. cmd-style `( echo & echo )`
  blocks crash; PowerShell mangles quotes -- never use it for git.
- Prime directives (CLAUDE.md): audio byte-identity at every gate; SFW
  always; never use the word "dummy"; wire every code change into
  `workflows/otr_scifi_16gb_full.json`.
- Bug Bible regression after EVERY code change. Baseline:
  **3597 passed / 21 skipped / 0 failed**.

## What changed overnight (2026-05-27 -> 2026-05-28)

### Refined build plan (`docs/OTR_story_quality_build_plan.md`)

Re-audited against live source at `bcfe8a5`. The 2026-05-27 plan had
two field-name drifts:

1. The handoff said Sprint 1 edits `_otr_outline.py` to add
   `dialogue_slot_id`. **Wrong file.** Path B (Story Room) consumes
   `_otr_stage1_plan.Stage1Beat`, not `_otr_outline.Beat`. Slot ids
   land on `Stage1Beat`. Path A's `_otr_outline.Beat` gets a mirrored
   field only when Sprint 4 needs it.
2. The handoff said voicedness was "speaker_role in {character,
   announcer}". **Stage1Beat has no `speaker_role` field.** Voicedness
   is `speaker != "MUSIC"` (covers cast-name speakers + ANNOUNCER
   bookends; only `MUSIC`-speaker beats are non-voiced).

Also: field-name drift between Stage1Beat (`length_target_words`,
`emotional_register`) and `init_lines_from_outline`'s expected attrs
(`target_words`, `mood`). The init method uses
`getattr(beat, X, default)` so it tolerates the gap, but Sprint 1's
`dialogue_slot_id` wire-through needs an explicit one-line add in
`production_ledger.py`. Plan documents two adapter options; Sprint 1
owner picks.

The plan now reads top-to-bottom as a sequence of subagent contracts:
each sprint section is self-contained enough that a fresh subagent
with no prior context can execute it end to end.

### Sprint 0 baseline (`docs/2026-05-27-otr-quality-baseline.md`)

Scaffolded -- captures the three numbers operators must paste in
during the Sprint 1 5-episode soak (rows_skipped sum,
fallback_to_legacy count, editor-rubber-stamp rate). The autonomous
session couldn't read the live `pending_*` ledgers (not in the
mounted workspace; only the 2026-05-19 pre-Wave-3 fixture was
present), so the live numbers are an operator deliverable.

### Morning runbook (`docs/2026-05-28-morning-runbook.md`)

Step-by-step `cd` + commands for the morning Jeffrey or a focused
subagent to execute Sprint 1.

## State of the art

**HEAD: `bcfe8a5` on `v2.0-alpha`** (origin matches; no overnight
commits pushed). Last shipped commit:
`docs: Sprint 1 -- Extract dialogue-only scope reduction`.

## Immediate next steps (in order)

1. **Read** `docs/2026-05-28-morning-runbook.md` -- top-to-bottom.
2. **Verify** the pre-Sprint-1 regression baseline is green
   (3597 / 21 / 0).
3. **Commit + push** the three new/refined doc files (plan, baseline,
   runbook) via Desktop Commander cmd. Commit message stamped in the
   runbook.
4. **Execute Sprint 1** per the plan's Sprint 1 section. Pick
   adapter Option A vs B (documented in the plan); record the
   decision in the commit body.
5. **5-episode soak** with `use_story_room=true` + `commit=true`.
   Paste the three numbers into the baseline doc. If any episode
   shows `rows_skipped > 0` or `fallback_to_legacy == true`, revert
   Sprint 1 and open a Bug Bible candidate. Do NOT proceed to
   Sprint 2 until 5/5 clean.
6. **Sprints 2-5** -- each sprint is a self-contained subagent
   contract in the plan. Dispatch one at a time; order is
   non-negotiable.

## Open questions

Unchanged from 2026-05-27 handoff plus the adapter-option question
above.

- **Writer halt on news-brief exhaustion** -- Jeffrey 2026-05-27
  direction. Add as Sprint 2 sub-section: writer raises when
  `build_news_briefs` exhausts retries; queue retry re-rolls news.
- **DramaticState storage** -- attach to Stage1Plan as top-level
  `dramatic_state` field (decided in the refined plan).
- **Sprint 5 cap-to-1 timing** -- ship as written in Sprint 5; do
  not pre-empt during Sprints 2-3.

---
## Resume instructions
Open a fresh window with the OTR-OldTimeRadio folder selected, attach
`docs/2026-05-28-morning-runbook.md` and `session_handoff.md`, and say:

"Read the runbook then the handoff. Run the pre-Sprint-1 regression
baseline. If green, commit the docs, then start Sprint 1 per the plan.
Stop after Sprint 1 commit + push and hand me the 5-episode soak
checklist."
