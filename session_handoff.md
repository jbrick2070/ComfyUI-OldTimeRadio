# Session Handoff — OTR Cast-System sprint (name↔gender↔voice) — 2026-05-30 — ALL SPRINTS SHIPPED

## Core goal (DONE)
Fix cast incoherence: male-coded names landed on female slots/voices
(`MALIK HIBBERT`→female, `PHYLLIS OKAFOR`→male) because names were rolled from a
gender-blind pool while gender was bound separately. The voice always followed
gender — only the NAME was wrong. **Phase 1 (deterministic repair), Phase 2
(optional LLM naming layer), and S3 (VRAM) are all implemented, tested, and
committed on `v2.0-alpha`.** Only operator-gated validation remains.

## State of the art
- HEAD chain on `v2.0-alpha`: S1 `acc091a` → S0 `a0bd03b` → S2 `30c666d` →
  docs `6479d27` → S4–S8 `7e04dc4` → S3 `31ea332` (+ this docs commit).
- Full `tests/` walk: green except the 5 known pre-existing failures
  (test_b7_forbidden_sweep ×2, test_llm_slot_sweep, test_bark_freeze_halt_bypass,
  test_workflow_validator_empty_path_fallback), **0 new**. BOM/AST/0-byte sweep
  clean. Pool mode byte-identical (C7).
- Mode-matrix: `test_cast_invariants.py` (pool, R1–R6+R3, byte-identical),
  `test_cast_llm_naming.py` (llm, 17), `test_writer_llm_unload.py` (4).
  `./test_cast.sh` wraps all of it.

## What shipped
- **S1** `config/cast_pools.py`: `FIRST_NAMES_BY_GENDER` (98/24/31), genre
  buckets, `gender_of_first_name()`. `FIRST_NAMES` byte-identical.
- **S0** `nodes/_otr_cast_env.py` (env contract), `tests/golden/` (coherent
  seed=1, incoherent seed=0), `tests/test_cast_invariants.py`, `test_cast.sh`.
- **S2** `nodes/_otr_casting.py`: `_repair_ensemble_names` — isolated-rng
  same-gender swap; main cast_rng never perturbed. `cast_seed` threaded.
- **S4** `nodes/_otr_castplanner.py`: immutable `CastPlanSlot`.
- **S7** `nodes/_otr_cast_validator.py`: `validate_pass1()`.
- **S5** `python_assign_voice_preset(age_band=...)` — inert in pool mode, one
  rng.choice/slot.
- **S6** `_apply_llm_slot_fill` in `lock_cast` — gated by
  `OTR_NAME_MODE=llm_slot_fill`; deterministic cast built first as the coherent
  backstop, one creative-slot LLM call renames+textures, validated, S2
  gender-repair backstop, full fallback on failure (no retry). Reuses
  `creative_fn` (PD6: no new model_id widget).
- **S8/R9** names final at `lock_cast` return (frozen before dialogue).
- **S3** `nodes/_otr_writer_vram.py` + writer call: post-script LLM unload
  before TTS. Never raises; `OTR_WRITER_UNLOAD_AFTER_SCRIPT` (default on).

## Immediate next steps (OPERATOR — cannot be done headless)
1. **GPU smoke for S3:** run an episode, watch peak VRAM across the writer→TTS
   boundary, confirm the LLM is evicted before Bark/render. Set
   `OTR_WRITER_UNLOAD_AFTER_SCRIPT=0` to A/B it.
2. **Real-LLM llm_slot_fill validation:** run an episode with
   `OTR_NAME_MODE=llm_slot_fill` (a real creative model in the slot), confirm
   `meta.llm_naming_applied` / `meta.cast_texture` and coherent, non-duplicate
   names. Headless tests stub the LLM.
3. **`git tag v2.0`** — Jeffrey's call (CLAUDE.md reserves release tags).
4. Decide `OTR_NAME_CROSS_GENDER_RATE` lane (default strict 0.0 in place).
5. Decide the untracked `docs/2026-05-30-cast-*` consult docs (commit or drop).

## Live log monitoring (during operator runs)
`scripts/otr_tail_logs.py` tails both logs + queries the ComfyUI history/queue API.
Run with the venv python (NOT type/tail — the logs sit under OneDrive-virtualized
paths cmd can't read, and `tail` doesn't exist on Windows):

    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\otr_tail_logs.py --lines 40

Logs: `C:\Users\jeffr\AppData\Roaming\ComfyUI\logs\comfyui.log` (s/it, VRAM, FFmpeg)
and `…\ComfyUI-OldTimeRadio\otr_runtime.log` (OTR phases). The script forces UTF-8
on BOTH the read AND stdout (the logs carry box/emoji chars that crash cp1252). An
`[error]` on an advancing `NN%|…| s/it` line is ComfyUI's stderr progress bar —
NORMAL, not a failure. Re-run every ~2 min until `/history` shows `completed`.

## Open questions
- `OTR_NAME_CROSS_GENDER_RATE`: strict (0.0, in place) vs LLM-owns-intent (>0).
- Bug Bible regression RAN against the OTR pack (operator-uploaded
  `bug_bible_regression.py`): **23/23 applicable static checks pass**, incl. all
  new cast files. Only the guide's internal Three-File Contract fails (its own
  `BUG_BIBLE.yaml`/`README.md` aren't reachable here). The
  `comfyui-custom-node-survival-guide` dir is an inaccessible reparse point in
  this environment (cmd → "syntax incorrect"); a session that can reach it
  should do the Three-File promotion if this fix earns a Bible entry.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
