# Story-Engine Sprint -- CODER-WINDOW KICKOFF

Paste this into a fresh coder window to begin. It is self-contained; the full plan is the source of truth below.

## You are
The coder window for the OTR story-engine quality sprint. You run everything yourself (Desktop Commander first; never hand the operator a command). Reset the box before every headless run (selective CIM kill, never a blanket python kill). File I/O via the file tools on the real Windows files; DC/PowerShell for git, the venv python, tests (chain with `;`, no `python -c` nested quotes -- write a temp `.py`, run, delete).

## Source of truth (read first, in order)
1. `docs/2026-06-21-allnight-864-frontier/SPRINT_READY_PLAN.md` -- the FINAL plan (scope, invariants, measurement contract, Sprints 0-3, acceptance, verify-at-build). **This governs.**
2. `docs/2026-06-21-allnight-864-frontier/STORY_ENGINE_IMPROVEMENT_PLAN.md` -- the per-fix grounding (file:line, before->after, why).
3. `docs/2026-06-21-allnight-864-frontier/WIRING_PLAN.md` -- why v1 is ZERO workflow-JSON edits + the freeze invariants you must not regress.
4. The four `roundtable/pass0{1,2,3,4}_judgment.md` -- what was accepted/rejected and why (don't re-litigate).

## What this is
Eight grounded fixes to the story written by node 1 `OTR_LedgerScriptWriter` (all internal Python; ZERO workflow-JSON edits in v1). The corpus problem: every episode is the same arc, lands ~70% of target length, the costly-choice beat isn't wired to a line 76% of the time, the announcer outro hedges over endings that already resolved, voices are interchangeable, and some lines leak third-person stage directions. Fixes F1-F8; F9/F10 are DEFERRED -- do not build them.

## Hard rules (from SPRINT_READY_PLAN -- do not violate)
- **C2** ledger schema unchanged; only additive `meta.*`/`cast[].*`; `lines[]` order preserved; `test_audio_byte_identical` holds where text unchanged.
- **C3** no scoring/reject/reroll PASS. A deterministic detector -> the EXISTING single-line recompose seam (max 1 attempt, distinct log marker, fallback) is allowed.
- **C4** stay inside node 1 + its internal modules. No new node/DB/training. ZERO workflow-JSON edits.
- Freeze CRITICAL invariants (7 top-level lists; unique `line_id`; `speaker_role` enum; voiced lines keep `char_id`; skipped line `text==""`+`tts_skip_reason`) must not regress.
- Each `.py` task = its own green chunk: suite + Bug Bible, then commit AND push to `v2.0-alpha`, verify HEAD==origin / no BOM / AST parse.

## Step 1 -- Sprint 0 (do this first)
1. Reset before headless (kill ONLY Comfy/headless-server + soak-harness pythons by CommandLine via CIM, plus port 8000/8011 owners by PID; confirm :8000/:8011 empty and VRAM ~1.5GB).
2. Build `scripts/story_quality_scan.py` reporting: `length_ratio` (voiced words / target), `length_pass_fired` (count), `episode_valid` (= freeze_valid AND `validate_episode_contracts`), `outro_hedge_vs_resolved` (HEDGE_LIST + shared `is_resolved_ending_change()`), `narration_self_address_lines`. Commit it as tooling.
3. Pin a fixed 12-leg smoke (exact 12 news inputs + `OTR_CAST_SEED`/`OTR_STYLE_SEED` per leg) and run it on CURRENT code at `target_words=864`. Write the before-numbers + the 12 inputs + seeds to `docs/2026-06-21-allnight-864-frontier/SPRINT_BASELINE.md`. Commit.
4. Resolve the four verify-at-build items (SPRINT_READY_PLAN bottom) and record results in SPRINT_BASELINE.md; gate Sprint 1 on them.

## Step 2 -- Sprint 1 ship-first (order: T1.1 F1 -> T1.4 F6 -> T1.2 F2 -> T1.3 F3)
Exactly as specified in SPRINT_READY_PLAN Sprint 1, each its own green commit+push. Note the bug-pass fixes baked in: F1 defaults to DROPPING the "20-30 words" literal (no `beat_lo/hi` interpolation unless confirmed in scope) + the exact None-guard; F6 is SPLIT (indirect-performance unconditional, situation-must-change only on turn beats); F2 zero-eligible -> do NOT emit the contract (no crash); F3 -> deterministic fallback outro template guarantees 0 hedges.
**Exit / STOP point:** `length_ratio`>=0.85, `episode_valid`>=11/12, `outro_hedge_vs_resolved`=0/12, suite+Bug Bible green, `test_audio_byte_identical` green. This is the ship-first milestone -- stop here unless continuing to Sprint 2 (F4/F5/F7) and Sprint 3 (F8).

## Note on GO_FORWARD_PLAN.md
As of 2026-06-20 night, `docs/GO_FORWARD_PLAN.md` tracks the SEPARATE 3D textured-hero PoC build (HEAD `8de1057`). This story-engine sprint is a different workstream. Do NOT overwrite that baton -- run this sprint from THIS kickoff + SPRINT_READY_PLAN, and only fold it into GO_FORWARD_PLAN.md if the operator says the story sprint is now the active build.
