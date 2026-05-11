# ADR — Script Writing Architecture (Phases 0-3)

**Date:** 2026-05-11
**Branch:** v2.0-alpha
**Status:** code shipped; soak handoff to Jeffrey
**Sources:** `docs/2026-05-10-script-writing-architecture__00_question.md` + the 04 synthesis MD (round-robin between ChatGPT + Gemini + Claude).

## Decisions

| # | Decision | Why |
|---|---|---|
| D1 | Single load-bearing failure is phantom-name leakage from the composer into the ledger. Three-way consensus across the brief, ChatGPT, and Gemini converged on this. | Composer sees `beat.speaker` but never the full cast roster; Mistral-Nemo invents names for beats whose intent references another character. Phantoms propagate silently. |
| D2 | Build day-one robustness across Phase 0 → Phase 3; do not defer the reviewer. | This is unsoaked greenfield. Deferring the reviewer leaves the phantom-leak vector open and is what produced the original soak symptom. |
| D3 | Cast data model lives at `led.data["cast"]` (canonical), with `character_cast` as a trivial tuple-of-names view. `_otr_cast_contract.CastContract` is greenfield infra and stays untouched. | Single-source verdict per synthesis §10 acceptance #1. Pre-Phase-0 read pass confirmed two surfaces but only one is wired. |
| D4 | Cast is LOCKED early; no LLM reroll on name violations anywhere downstream. Phantoms are flagged, then deterministically repaired (Levenshtein auto-remap) or muted (Step 2.5 phantom-skip). | Per synthesis §6.A. A reroll cannot invent the *right* name when the LLM didn't know the cast. |
| D5 | KV cache reuse in `_otr_model_loader.make_generate_fn` is NOT implemented today. Decision: pay the full prefill cost per composer call. ~800-tok hot path is acceptable on RTX 5080 / Mistral-Nemo 12B. | Pre-flight read pass verified each `model.generate()` rebuilds prompts from scratch. The §6.D plot-leak mitigation (trim spine to current arc_phase + previous-phase summary) also serves as the no-KV fallback if soak shows wall-clock degradation. |
| D6 | Outline `arc_phase` field is Optional. Beat schema accepts None for back-compat with pre-Phase-2A outlines. Phase 2A budget validator requires arc_phase on every voiced beat when `req.budget` is non-None. | Back-compat with existing tests + early-stage callers. New writer always sets budget, so production runs enforce arc_phase. |
| D7 | `act_count` widget appended at END of optional INPUT_TYPES. | Legacy `widgets_values` arrays (17 entries) preserve positional mapping. ComfyUI's loader maps widgets positionally; inserting mid-list shifts every subsequent widget's value. |
| D8 | `target_length` widget retained as deprecated (smoke presets still force `target_words`). `self_critique` retained as deprecated no-op. | Saved workflows load cleanly. Synthesis allows silent ignore. |
| D9 | `compose_line` returns `LineResult(text, compose_flags)` instead of bare string. | Side-channel-free way to carry phantom-name detections from composer to writer to ledger. |
| D10 | Progressive ledger writes (Phase 2B) — `init_lines_from_outline` + per-iteration `update_line_text` + `save()`. | Reviewer (Phase 3) reads the durable ledger, not in-memory state. Mid-loop crash leaves a coherent partial ledger on disk. |
| D11 | Reviewer is always-on (no widget gate). Bypass available via `meta.skip_reviewer=True` for unit tests only (synthesis G9). | Per synthesis §6.B; `self_critique` widget removed from semantic surface. |
| D12 | Three reviewer LLM calls implemented as a SINGLE `audit_cast_contract(ledger, label)` function called twice (pre + post), plus `run_script_doctor` once. | Synthesis G4 — keeps the two audits structurally identical and impossible to drift out of sync. |
| D13 | Reviewer `edit_cap = min(8, max(3, voiced_beats // 3))`. | Synthesis G1 — scales with episode size so a 7-act episode can accommodate 6 plausible rewrites without flipping `too_many_edits`. |
| D14 | Reviewer Levenshtein threshold = 3 for names ≥ 5 chars; substring-containment fast-path; ties drop to None. | Synthesis G8 test table pinned: "alice"→ALICE, "Allice"→ALICE, "Alyce"→ALICE, "BOBB"→BOB, "Robert"/"Patel"/"Dr. Patel"/"the council" → None. |
| D15 | Step 2.5 deterministic phantom-skip fallback runs between Pass 2 and Pass 3. Sets `line.skip=True` + `tts_skip_reason` on any titled phantom that survived Pass 1's auto-remap AND the Script Doctor. | Synthesis M2. Closes the exact failure mode the round-robin opened on: "Dr. Patel" surviving all three LLM passes. |
| D16 | Workflow JSON wiring of `OTR_LedgerScriptReviewer` deferred to Jeffrey (drag connection in ComfyUI UI). | Bash sandbox mount went stale around the workflow file; safer to let Jeffrey wire writer→reviewer→director manually in the GUI than risk hand-editing 47kB of JSON with link-id math from a stale view. |

## What ships in each phase

| Phase | Module(s) touched | New tests |
|---|---|---|
| 0 | `_otr_line_composer.py` (name roster + phantom detect + `LineResult` + `aggregate_compose_flags`) + writer wiring. | `tests/test_phase0_name_roster.py` |
| 1 | `_otr_line_composer.py` (`render_outline_spine`, `build_voice_card`, prompt restructure, LineRequest fields) + writer; `LAST_LINES_WINDOW` 3→5. | `tests/test_phase1_composer_prompt.py` |
| 2A | NEW `_otr_episode_budget.py`; `_otr_outline.py` (Beat.arc_phase, beats cap 24→32, `validate_outline_against_budget`); composer arc_phase block; writer act_count widget + plumbing; NEW `web/js/otr_act_count_widget.js` + `WEB_DIRECTORY` in `__init__.py`. | `tests/test_phase2a_episode_budget.py` |
| 2B | `production_ledger.py` (`init_lines_from_outline` + `update_line_text`); writer per-beat save loop. | `tests/test_phase2b_progressive_ledger.py` |
| 3 | NEW `_otr_ledger_reviewer.py` (Pass 1+3 auditor, deterministic repairs, Script Doctor, Step 2.5, dispositions); NEW `OTR_LedgerScriptReviewer.py` node; `__init__.py` registration. | `tests/test_phase3_ledger_reviewer.py` |

## Invariants held

- Local-only; no cloud. VRAM ≤14.5 GB unchanged (no new loaded models — only LLM calls already in budget).
- C7 byte-identity unchanged for the writer path; reviewer LLM calls use deterministic-leaning temperatures (0.2 / 0.5 / 0.2) but C7 doesn't extend to reviewer rewrites (cast is the same; only dialogue may shift).
- Schema additivity only: every new field on `lines[]` / `meta` is additive; existing consumers (SceneSequencer, Bark, Kokoro, HuMo, RTXUpscale) untouched. `skip=True` + `tts_skip_reason` are honored by Bark per synthesis §11.4 — needs verification post-soak.
- No `dummy` placeholders anywhere; SFW; non-violent; arc structured.
- One commit per logical change (Phase 0 / 1 / 2A / 2B / 3a / 3b+3c). Each phase's commit message captures the synthesis section + spec items it lands.

## Workflow JSON wiring (Jeffrey to do)

`OTR_LedgerScriptReviewer` is registered in `__init__.py` and discoverable in the ComfyUI node menu. To wire into `workflows/otr_scifi_16gb_full.json`:

1. Open the workflow in ComfyUI Desktop.
2. Drop `OTR_LedgerScriptReviewer` between `OTR_LedgerScriptWriter` and `OTR_LLMDirector`.
3. Connect `OTR_LedgerScriptWriter.script_text` → `OTR_LedgerScriptReviewer.script_text`.
4. Connect `OTR_LedgerScriptReviewer.script_text` → `OTR_LLMDirector.script_text`.
5. Save the workflow.

The reviewer reads the production ledger directly via `get_ledger()`; the `script_text` socket is purely for graph-layout continuity.

## Known watch items

- **W1 — KV cache reuse (D5).** If soak shows composer wall-clock per call > 1.5s consistently, revisit the model loader to thread `past_key_values=`. Until then, the lean prompt is sufficient.
- **W2 — Plot-leak from full outline_spine (synthesis §6.D ADR watch item).** If Jeffrey observes setup-phase ALICE telegraphing the resolution, switch `render_outline_spine` to current-arc-phase + previous-phase-summary only. The helper already accepts a slice; the writer would pass `outline.beats[:current_idx+1]` instead of `outline`.
- **W3 — JS/Python breakpoint duplication.** `_DEFAULT_ACT_BREAKPOINTS` lives in `_otr_episode_budget.py` (Python authoritative) and `web/js/otr_act_count_widget.js` (UI mirror). Update both when changing.
- **W4 — Reviewer LLM cost.** Three additional Mistral-Nemo calls per episode (~2k tok / 3.5k tok / 2k tok prompts). Soak measurement pending.

## Round-robin attribution

Synthesis sourced from `docs/2026-05-10-script-writing-architecture__00_question.md` plus external opinions from ChatGPT (gpt-4.1) and Gemini (gemini-2.5-pro) per Jeffrey's standing process. Final design is the synthesis MD (`04_synthesis (1).md` uploaded 2026-05-11) which integrates all three perspectives and locks the seven §6 decisions.
