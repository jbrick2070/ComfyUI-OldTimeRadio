# Bank Plan -- scifi_codex_v4 (v4 campaign, bank #1)

**Date:** 2026-07-17. **Branch:** v2.0-alpha. **Plan of record:** `docs/2026-07-17-v4-campaign/final.md` + `LESSONS_GATE_BRIEF.md`. This is the per-bank build record + the lessons the remaining four v4 banks inherit. Operator directive (2026-07-17 evening): "don't forget past PBUGs + lessons -- add a bank plan."

## 1. What shipped

`scifi_codex_v4` -- a fully INDEPENDENT bank seeded from `scifi_codex` v1. The proof-pressure creative delta lives ENTIRELY in the pack seams (no code): a person with a WANT, the science as the gating PROOF, a mandatory COST beat, one REVERSAL. Atomic chunk (commit `1fd7743d`, then the P3 contract-visibility follow-up -- kibitz r2 reverted the cap-restatement and made the whole compiler contract model-visible; see section 3):

- `nodes/story_packs/banks.json` -- row inserted before `custom_source_bank` (custom stays last). `default_story_model=scifi_codex_v4`, `default_story_pipeline=scifi_codex_circuit_v4`.
- `nodes/story_packs/scifi_codex_v4/scifi_codex_v4.json` -- pack, exactly the 11 codex seams (three-way parity: pack keys == pipeline declared_seams == pass seam_refs).
- `nodes/story_rules/scifi_codex_v4.json` -- rules keyed by exact id.
- `nodes/story_packs/pipelines.json` -- `scifi_codex_circuit_v4` (executable:true, requires_source_contract:false, same P0-P12 passes as v3).
- `nodes/OTR_LedgerScriptWriter.py` -- `scifi_codex_circuit_v4 -> _run_scifi_codex_lane` mapped **DIRECTLY** (NOT the `_make_v3_runner` advisory wrapper; final.md "v4 pipelines don't invoke the v3 wrapper").
- `tests/test_bank_variants.py` + `tests/test_fable2_registry.py` -- roster counts 11->12 visible / 10->11 runnable; new `TestScifiCodexV4` coverage. Bijection test is dynamic (auto-satisfied).

Gates: full suite 8139 / Bug Bible 17 / AST+JSON+BOM+zero-byte clean / `otr_canonical.json` byte-unchanged (dropdown auto-enumerates -- no graph edit) / HEAD==origin.

**LIVE-PROVEN (2026-07-17 night6):** leg `c1f3891f` RESULT SUCCESS + obs_publish -- "The Whisker Effect" (56.6 MB, obs + episode dirs Test-Path OK), Mistral-Nemo both slots, 30w. Reaching green required TWO pre-existing codex-lane fixes the live legs surfaced (both grounded via the operator cross-check window): **P0** source-whitespace normalization at admission (PBUG-20260717-01 @ `26ba8e1d`) and **P3** non-spoken metadata cap raise (premise 144->240, scene/shot description 72->144; load-bearing on the 8192 budget so the output reservation was resized 1647->1829 + all exact-token guards updated) @ `9730e2dc`. Full suite 8144 / Bible 17. See section 3 lessons + PROD_BUG_LOG PBUG-20260717-01.

## 2. Gate posture (opt-in Phase-1 gates) -- vetoable

ON: `require_science_floor` (core codex contract, same as v1), `placeholder_guard` (G13), `scene_coherence_check` (G15).

DEFERRED (OFF): `genre_guard_spoken` (G10), `require_outro_cast_complete` (G12).

**Why deferred -- grounded, load-bearing lesson:** the two ON gates are pure DETERMINISTIC terminals that the codex fail-closed compiler structurally satisfies (unique `scene_id` `^scene_\d{3}$`; no bare placeholder tokens). The two DEFERRED gates have an authored "improve" repair that lives in the writer's INLINE composer body (`OTR_LedgerScriptWriter.py` I.7 genre repair ~:6839, I.8 outro repair ~:6859). **The dedicated codex runner `_run_scifi_codex_lane` does NOT cross that inline boundary** (grep-confirmed: no genre/outro guard call in `_otr_scifi_codex.py`). The writer stamps the gate meta flags in `run()` BEFORE lane dispatch, so the Phase-10 terminals WOULD fire for codex -- but with no upstream repair, they would be no-repair hard gates that can fail a good story. Lawful under THE LAW (a deterministic validator may end an episode), but risky and against the "improve, never fail" spirit. Enable them for codex only AFTER wiring the authored genre/outro repair into the codex finalizer (before assembly/seal) -- final.md specifies that boundary; Phase 1 only wired the inline path.

Note: PBUG-20260710-07 retire rides the per-lane **announcer-sentinel mint** on a live codex leg, NOT the outro gate -- so deferring `require_outro_cast_complete` does not block the retire.

## 3. Lessons learned THIS build (carry forward)

1. **P3 draft failures are the UNSTATED DETERMINISTIC-CONTRACT class (PBUG-20260713-02 prose overflow + -06 beats-by-scene-overflow) -- NOT a simple "unstated cap".** First live legs failed at codex P3 (`string_too_long`, then `beat_count`) with BOTH `mistralai/Mistral-Nemo-Instruct-2407` AND `google/gemma-4-E4B-it` creative -> model-independent. The two-strikes kibitz r2 (Codex `gpt-5.6-sol` + Antigravity Gemini 3.1 Pro, both grounded + Claude-grounded) BROKE the naive "unstated cap" framing: `_RADIO_SCORE_DRAFT_SURFACE_INSTRUCTION` (`nodes/_otr_scifi_codex.py:563-586`) ALREADY injects TIGHTER safe ceilings (title<=48, premise<=108, ...) into base + restart + repair prompts, so the panel judged the seam cap-restatement redundant and it was first reverted. **A LIVE leg then OVERTURNED that**: reverting REGRESSED `string_too_long` on `premise` (the text-patch deliberately never clips prose -- `_otr_scifi_codex.py:1748` -- so model-visible caps are the ONLY compliance lever), so the cap-restatement was **RE-ADDED** as load-bearing salience (the PBUG-20260713-02 lesson stands: restate every schema bound in the model-visible contract; judge takeaway = weight live evidence over a static "redundant" argument). **ALSO make the WHOLE deterministic contract model-visible**: the compiler also enforces `unused_shot` (`:958`), `cast_coverage` (`:972`, announcer included), `cue_id` uniqueness (`:982`), and `cue_anchor < beat_count` (`:988`) -- none were in the prompt. All four were appended to the shared surface instruction, plus a 12-beat fully-constrained distribution clause (`MAX_SCENES` 3 * `PER_SCENE` 4). The `beat_count` fix that survived = the v4 seam harmonization ("produce exactly as many beats as the advisory plan lists" -- COST/REVERSAL are ROLES on existing beats, not additions), reinforcing the dynamic topology instruction. `string_too_long` recovery stays the `codex_radio_score_text_patch` seam. The `beat_count` compile error now reports observed-vs-expected for diagnosability. **Lesson: the auto `schema_shape_instruction` emits field PATHS but no min/max and no cross-field invariant -- EVERY deterministic gate must be hand-written into the model-visible contract** (operator `feedback_no_model_gating_per_slot`; kibitz run under `kibitz-runs/2026-07-17-p3-beatcount/`).
   - Advisory beat count = `max(3, min(12, len(cast)*3))` with cast 2..4 rows incl announcer (`:3245`, `:215`) -> reachable 6/9/12 (the floor-3 is unreachable).
   - `ScriptArtifactV4` (P5) does NOT cap authored prose (`:647-652`) -- do NOT invent a P5 cap project (grounded false, kibitz r2).
2. **Model-id strings drift.** The `creative_writing_model`/`technical_model` combo no longer carries the ` [LOCAL HF]` suffix. Valid ids at HEAD: `mistralai/Mistral-Nemo-Instruct-2407`, `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-12b-it`, `unsloth/gemma-4-12b-it-GGUF`, `unsloth/Qwen3-8B-GGUF`, `google/gemma-2-2b-it`, `Qwen/Qwen2.5-14B-Instruct`, `Nitral-AI/Captain-Eris_Violet-V0.420-12B`, plus cloud `openrouter:slot-a/b` + `google_api:slot-a/b`. `patch_creative` validates against the live combo and fails loud -- a stale id dies at pre-flight (fast, cheap catch).
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
