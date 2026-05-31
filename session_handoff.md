# Session Handoff — OTR Cast-System fix (name↔gender↔voice) — 2026-05-30 (Phase 1 SHIPPED)

## Core goal
Fix cast incoherence: characters got a name from a gender-blind pool while gender was
drawn separately, so male-coded names landed on female slots/voices (`MALIK HIBBERT`→female,
`PHYLLIS OKAFOR`→male). The voice always followed gender correctly — **only the NAME was wrong.**
**Phase 1 (deterministic name-repair) is DONE, verified, and committed.** Phase 2 (the optional
schema-locked LLM naming layer) and S3 (a VRAM-timing change) remain.

## THE execution doc
`docs/2026-05-30-cast-system__go-forward-sprint-plan.md` — full sprint grid S0–S9, frozen S0
contracts, R1–R9 bar, waves. Phase 1 = S1+S0+S2 (shipped). Phase 2 = S4–S9. S3 = VRAM.
Round-robin reasoning: `docs/2026-05-30-cast-name-gender-voice-coherence__*` (still untracked).

## Tech stack & constraints (live; CLAUDE.md prime directives still apply, not repeated)
- **Execution reality this session:** file edits via the harness hit the real Windows FS; **all
  python/pytest/git run through Desktop Commander (cmd) on the Windows venv**
  `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` (Python 3.12.11). The Linux sandbox
  was NOT used (stale-mount risk). DC stdout isn't captured reliably → scratch scripts write
  results to `C:\Users\jeffr\AppData\Local\Temp\otr_s0\*_out.txt`, read back with DC read_file.
- Tests run green via the venv; full `tests/` walk exits 2 by design (the conftest known-fail
  guard) — parse the output file, don't trust the exit code. **Baseline failure set = 5 known
  pre-existing:** test_b7_forbidden_sweep (x2), test_llm_slot_sweep, test_bark_freeze_halt_bypass,
  test_workflow_validator_empty_path_fallback. Green bar = "these 5, no more."
- No GPU / no Bark here — audio + LLM end-to-end + VRAM smoke are operator-only.
- Branch `v2.0-alpha`. **Tagging is Jeffrey's (CLAUDE.md): do NOT `git tag v2.0` — recommend it.**

## What's done & decided (this session)
- **Phase 1 shipped — 3 commits on `v2.0-alpha`:**
  - **S1 `acc091a`** `config/cast_pools.py`: `FIRST_NAMES_BY_GENDER` (98 M / 24 F / 31 unisex —
    partitions the SAME names; `FIRST_NAMES` left byte-identical, order is load-bearing for C7),
    `gender_of_first_name()` (case-insensitive; bare or "FIRST LAST"; cross-cultural→unisex so it
    never forces a false repair), `FIRST_NAMES_BY_GENRE`+`names_for_genre()`, `pick_first_last`
    gains `genre="auto"` (auto = unchanged draw), import-time `_verify_name_buckets()` drift guard.
  - **S0 `a0bd03b`** `nodes/_otr_cast_env.py` (frozen env contract), `tests/golden/` capture script +
    `cast_pool_baseline.json` (coherent seed=1, known-incoherent seed=0 = NEMO-on-female),
    `tests/test_cast_invariants.py` (R1/R2/R4/R5/R6), `test_cast.sh`.
  - **S2 `30c666d`** `nodes/_otr_casting.py`: `_repair_ensemble_names` + `_pick_same_gender_first_name`
    — swaps ONLY a mismatched first name for a same-gender one via ISOLATED rng
    `random.Random(f"{cast_seed}:{char_id}")` (main cast_rng never perturbed → byte-identical for
    coherent seeds, coherent otherwise). `cast_seed` threaded writer→`lock_cast`→`precompute_ensemble_slots`;
    `lock_cast` now carries repaired `ens.name` downstream. Standalone `cast_one_character` passes
    `repair_names=False`. R3 + cross-gender-rate tests added.
- **Key decisions:** repair is a gender-blind-POOL concern → only in lock_cast's ensemble, NOT the
  standalone explicit-name path. Pool mode (default) stays byte-identical (PD1). Every knob is an
  env var → no workflow-JSON change. `OTR_NAME_CROSS_GENDER_RATE` default 0.0 = strict repair.
- **NOT done (deliberate):** S3 + Phase 2 — see Immediate next steps. Bug Bible regression couldn't
  run (`comfyui-custom-node-survival-guide/tests/bug_bible_regression.py` absent in this checkout).

## State of the art
- HEAD `30c666d` on `v2.0-alpha`. Chain: 30c666d (S2) → a0bd03b (S0) → acc091a (S1) → e04c80b.
- Verified: full `tests/` walk = only the 5 known failures, 0 new. R1–R6 + R3 green
  (177-item cast bar: 171→ green). BOM/AST/0-byte sweep clean on all 8 touched files.
- **Untracked, intentionally left for Jeffrey to decide:** the consult docs + sprint plan in
  `docs/2026-05-30-cast-*`. (Handoff open question — Jeffrey's lean is "don't hoard docs".)
- The cast pipeline ground truth (still accurate): `OTR_LedgerScriptWriter.execute` →
  `lock_cast` (`_otr_casting.py:997`) → `assemble_pre_locked_rows` (names via
  `cast_pools.pick_first_last`) → `precompute_ensemble_slots` (gender + **now repair**) →
  `python_assign_voice_preset` (one `rng.choice` per slot at `:785`).

## Immediate next steps
1. **S3 (VRAM) — needs the GPU, do WITH an operator smoke.** Move the writer-LLM unload to after
   the script, before the TTS phase (not order 17). Use `_flush_vram_keep_llm()` semantics; never
   `force_vram_offload()` between LLM phases (CLAUDE.md). **Reconcile with Phase 2 first** — S6/S8
   add a *second* writer LLM pass, so "unload after script" must not strand Pass-2. Validate by a
   VRAM-envelope episode run (operator pastes console). Do not blind-commit.
2. **Phase 2 (S4–S9) — optional `OTR_NAME_MODE=llm_slot_fill`.** Build behind the default-off flag
   so pool mode stays byte-identical (C7). S4 `nodes/_otr_castplanner.py` (immutable slot schema,
   frozen-contract jsonc in the sprint plan), S7 `nodes/_otr_cast_validator.py` (rejects
   out-of-contract keys / dup names / wrong count / gender-carryback; fallback → the S2 repair, no
   LLM retry), S6 writer Pass-1 (name+texture, schema-locked, model id via STRING socket — NO new
   model_id widget, PD6), S5 voice (gender×age) keeping ONE `rng.choice`/slot (`:785`, R6) and NOT
   changing pool-mode picks, S8 writer Pass-2 vs frozen cast (R9), S9 mode-matrix regress. All
   unit-testable headless with stub generate_fns (mirror `tests/test_otr_casting.py`); real-LLM
   validation is an operator run.
3. Run `./test_cast.sh` (or the equivalent venv pytest selection) after every change; then the full
   `tests/` walk — green bar = the 5 known failures only.
4. **Phase 1 ship:** recommend Jeffrey run `git tag v2.0` (tagging is his per CLAUDE.md). Push is
   done if origin accepted; otherwise the cmd block is in the final report.

## Open questions
- **Jeffrey owns, set before S6:** `OTR_NAME_CROSS_GENDER_RATE` lane — strict (0.0, ships clean,
  recommended default, already the default) vs LLM-owns-intent (>0, heavier validator contract).
  Phase 1 ships strict; flip later without structural change.
- Commit the untracked `docs/2026-05-30-cast-*` consult/sprint docs, or treat as ephemeral?
- S3 vs Phase 2 ordering: Phase 2 changes the writer LLM lifecycle, so do S3 AFTER S8 (or design
  S3 against the final two-pass lifecycle) to avoid reworking it.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
