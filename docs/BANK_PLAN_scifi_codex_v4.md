# Bank Plan -- scifi_codex_v4 (v4 campaign, bank #1)

**Date:** 2026-07-17. **Branch:** v2.0-alpha. **Plan of record:** `docs/2026-07-17-v4-campaign/final.md` + `LESSONS_GATE_BRIEF.md`. This is the per-bank build record + the lessons the remaining four v4 banks inherit. Operator directive (2026-07-17 evening): "don't forget past PBUGs + lessons -- add a bank plan."

## 1. What shipped

`scifi_codex_v4` -- a fully INDEPENDENT bank seeded from `scifi_codex` v1. The proof-pressure creative delta lives ENTIRELY in the pack seams (no code): a person with a WANT, the science as the gating PROOF, a mandatory COST beat, one REVERSAL. Atomic chunk (commit `1fd7743d`, then the P3 cap-restatement follow-up):

- `nodes/story_packs/banks.json` -- row inserted before `custom_source_bank` (custom stays last). `default_story_model=scifi_codex_v4`, `default_story_pipeline=scifi_codex_circuit_v4`.
- `nodes/story_packs/scifi_codex_v4/scifi_codex_v4.json` -- pack, exactly the 11 codex seams (three-way parity: pack keys == pipeline declared_seams == pass seam_refs).
- `nodes/story_rules/scifi_codex_v4.json` -- rules keyed by exact id.
- `nodes/story_packs/pipelines.json` -- `scifi_codex_circuit_v4` (executable:true, requires_source_contract:false, same P0-P12 passes as v3).
- `nodes/OTR_LedgerScriptWriter.py` -- `scifi_codex_circuit_v4 -> _run_scifi_codex_lane` mapped **DIRECTLY** (NOT the `_make_v3_runner` advisory wrapper; final.md "v4 pipelines don't invoke the v3 wrapper").
- `tests/test_bank_variants.py` + `tests/test_fable2_registry.py` -- roster counts 11->12 visible / 10->11 runnable; new `TestScifiCodexV4` coverage. Bijection test is dynamic (auto-satisfied).

Gates: full suite 8139 / Bug Bible 17 / AST+JSON+BOM+zero-byte clean / `otr_canonical.json` byte-unchanged (dropdown auto-enumerates -- no graph edit) / HEAD==origin.

## 2. Gate posture (opt-in Phase-1 gates) -- vetoable

ON: `require_science_floor` (core codex contract, same as v1), `placeholder_guard` (G13), `scene_coherence_check` (G15).

DEFERRED (OFF): `genre_guard_spoken` (G10), `require_outro_cast_complete` (G12).

**Why deferred -- grounded, load-bearing lesson:** the two ON gates are pure DETERMINISTIC terminals that the codex fail-closed compiler structurally satisfies (unique `scene_id` `^scene_\d{3}$`; no bare placeholder tokens). The two DEFERRED gates have an authored "improve" repair that lives in the writer's INLINE composer body (`OTR_LedgerScriptWriter.py` I.7 genre repair ~:6839, I.8 outro repair ~:6859). **The dedicated codex runner `_run_scifi_codex_lane` does NOT cross that inline boundary** (grep-confirmed: no genre/outro guard call in `_otr_scifi_codex.py`). The writer stamps the gate meta flags in `run()` BEFORE lane dispatch, so the Phase-10 terminals WOULD fire for codex -- but with no upstream repair, they would be no-repair hard gates that can fail a good story. Lawful under THE LAW (a deterministic validator may end an episode), but risky and against the "improve, never fail" spirit. Enable them for codex only AFTER wiring the authored genre/outro repair into the codex finalizer (before assembly/seal) -- final.md specifies that boundary; Phase 1 only wired the inline path.

Note: PBUG-20260710-07 retire rides the per-lane **announcer-sentinel mint** on a live codex leg, NOT the outro gate -- so deferring `require_outro_cast_complete` does not block the retire.

## 3. Lessons learned THIS build (carry forward)

1. **P3 RadioScoreV4 `string_too_long` is MODEL-INDEPENDENT (the unstated-cap class, PBUG-20260713-11/12).** First live legs failed at codex P3 with BOTH `mistralai/Mistral-Nemo-Instruct-2407` AND `google/gemma-4-E4B-it` creative: the model wrote `title`/`premise`/`setting`/scene/shot/beat fields over their Pydantic caps, and the 3-attempt repair ladder (base -> structural retry -> typed repair) copied the over-cap values (PBUG-20260713-04/05/06). **A model swap is NOT the fix -- it only masks it with a more concise model.** ROOT FIX = restate the EXACT caps in the model-visible seam (`codex_radio_score_system`), per the lesson: "every field carrying a schema BOUND must be restated in the model-visible contract; the auto `schema_shape_instruction` emits paths but NO min/max." This is also what makes the bank model-agnostic (operator `feedback_no_model_gating_per_slot`).
   - Exact RadioScoreV4 caps restated: title<=64, premise<=144, setting<=80; per scene env<=56, description<=72; per shot description<=72, visual_prompt<=120; per beat intent<=64, arc_phase<=28. (Source of truth: the Pydantic models in `nodes/_otr_scifi_codex.py`.)
   - The P5 `ScriptArtifactV4` play caps are a candidate for the SAME restatement if a live P5 cap failure surfaces -- but do NOT pre-emptively invent it (admission rule: fix proven failures). Watch the live P5 audit.
2. **Model-id strings drift.** The `creative_writing_model`/`technical_model` combo no longer carries the ` [LOCAL HF]` suffix. Valid ids at HEAD: `mistralai/Mistral-Nemo-Instruct-2407`, `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `unsloth/gemma-4-12b-it-GGUF`, `unsloth/Qwen3-8B-GGUF`, `google/gemma-2-2b-it`, `Qwen/Qwen2.5-14B-Instruct`, `Nitral-AI/Captain-Eris_Violet-V0.420-12B`, plus cloud `openrouter:slot-a/b` + `google_api:slot-a/b`. `patch_creative` validates against the live combo and fails loud -- a stale id dies at pre-flight (fast, cheap catch).
3. **Live-leg harness (proven autonomous path):** `scripts/otr_headless_canonical.ps1 -Profile none -Words 30 -Set OTR_LedgerScriptWriter.source_bank=<id> -Set ...model...` does selective CIM reset (only OTR server/runner -- preserves the MCP pythons) + free-port boot + loads the REAL `otr_canonical.json` + `--set` name-patch (no canonical edit). Exit 0 = RESULT SUCCESS. Then Test-Path the asset under `C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\<ep>\` + `...\otr\obs\`. Codex 30w local leg ~= 3-6 min (13-pass circuit + audio + floor video + obs_publish).

## 4. Go-forward -- the remaining four v4 banks

Order (final.md): `shakespeare_v4` -> `public_domain_story_v4` -> `media_archive_v4` -> `original_radio_v4`. Each an atomic per-bank chunk gated on a live leg; `runnable:true` LAST.

- **They write their OWN idiom, never sci-fi** (operator hard constraint): Shakespearean drama, public-domain literary adaptation, archival-media. Genre-bleed is a defect -- hardened `banned_phrases` + a positive register contract in each pack.
- **The three adaptation/inline lanes use `legacy_many_pass_v4`; original_radio uses `original_multi_pass_v4`** (both inline, added to `_LEGACY_INLINE_PIPELINES` -- runner map returns None, byte-identical inline body). **These inline lanes DO cross the I.7/I.8 authored-repair boundary**, so `genre_guard_spoken` + `require_outro_cast_complete` ARE safe to enable for them (improve-then-gate). This is the opposite of the codex-lane constraint above.
- **Apply the cap-restatement lesson PRE-EMPTIVELY** where a lane authors into a capped schema. The inline lanes are less schema-strict than codex, but any structured field with a bound must restate it in the seam.
- Per-bank config (final.md matrix): shakespeare = folger fetcher + `style_pool_class=adaptation` + `propagate_adaptation_cast=true` + Folger CC-BY-NC provenance; public_domain = Wells-first + adaptation + `research_only` BLOCKS publish; media_archive = `style_pool_class=media` + own `drama_seeds` sidecar (needs a `_PACK_SIDECAR_FILENAMES_BY_BANK` skip entry) + fix the truncated template outro; original_radio = source-contract-free + shares `original_radio/spark_deck.json` (hardcoded path -- add a sidecar skip entry for the v4 id).

## 5. Reusable v4-bank recipe (the atomic chunk)

1. banks.json row (before custom); pack `story_packs/<id>/<id>.json` (exact declared_seams); `story_rules/<id>.json` (exact id); pipelines.json entry (executable flag per registry law); runner map OR inline set; sidecar skip entry if the lane ships a sidecar.
2. Restate every model-visible schema bound in the authoring seam.
3. Roster/bijection test updates (counts + a per-bank coverage test).
4. `runnable:true` LAST.
5. Green gate: focused + full suite + Bible + AST/JSON/BOM/zero-byte + canonical hash unchanged.
6. Commit AND push; HEAD==origin.
7. Live leg (reset -> boot -> `otr_headless_canonical.ps1` with source_bank + local models) -> RESULT SUCCESS + obs_publish + Test-Path asset.
8. Post-build blind A/B (separate; may use cloud) = the "strictly better than seed" evidence, NOT the ship gate.
