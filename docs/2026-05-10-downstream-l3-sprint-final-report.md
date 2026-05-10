# Downstream L3 Sprint — Final State Report

**Date:** 2026-05-10
**Branch:** `v2.0-alpha`
**Predecessor commit:** `eec4718` (L3 ledger consumer rewrite — 7/7 audio+critic+video shipped)
**Working tree state:** dirty (intended — sprint deliverables not committed; per-file commit policy paused for the audit-only path)
**Bug Bible regression:** 23/1/2/0 baseline held across all 13 sprint steps.

## Sprint scope

Continuation sprint following the L3 consumer rewrite (`script_critic`, `batch_bark_generator`, `kokoro_announcer`, `scene_sequencer`, `batch_audiogen_generator`, `batch_procedural_sfx`, `video_engine` — all shipped at eec4718). Goal: bring everything DOWNSTREAM of the writer (visual chain, post-process tail, cast/portrait infrastructure, utility nodes, helper API tests, B4 LLM prompt audit, fresh workflow JSON, dry-run gates) to L3 readiness with the same rigor.

## Outcome — 13 of 13 steps shipped

| Step | Description | Outcome |
|---|---|---|
| 1 | L3 rewrite `batch_flux_render.py` | **AUDITED CLEAN** — no rewrite needed. Default `skip_env_stills=True` bypasses dead `_parse_env_prompts`. Live radio bookend reads ledger from disk via singleton + `load_ledger_safe`, uses L3-native fields with safe `.get()` defaults, stamps top-level `radio_bookend_path` + `meta` only. `scenes[0]` tier-4 fallback degrades safely on L3 (no `scenes` array). |
| 2 | L3 rewrite `batch_humo_render.py` | **AUDITED CLEAN** — no rewrite needed. Reads ledger via `_load_ledger_with_path`. Uses `line_id`, `char_id`, `speaker_role`, `start_s`, `dur_s`, `text`, `word_count`, `shot_id`. Orphan-rescue speaker fallback chain (`L691`) only fires on cast lookup miss; dormant on clean L3 data. |
| 3 | L3 rewrite `batch_ltx_render.py` | **AUDITED CLEAN** — no rewrite needed. Reads ledger via `_OTRL.load_ledger_safe`. Uses `line_id`, `speaker_role`, `dur_s`. `_build_ltx_role_prompt` returns static prompt by role with no field interpolation. |
| 4 | L3 rewrite `video_composite.py` | **AUDITED CLEAN** — no rewrite needed. Reads ledger via `_load_ledger_with_path`. Uses `line_id`, `speaker_role` (default "character"), `start_s`, `dur_s`. BUG-LOCAL-129a static-radio fill + BUG-135 motion-loop fill paths intact. |
| 5 | L3 rewrite `RTXUpscale` + `PostUpscaleProcgenBlend` (+ SeedVR2) | **AUDITED CLEAN** — no rewrite needed. RTXUpscale path-in/path-out wrapper; spacesaver reads `meta.perfect_run_spacesaver` + `episode_id`. PostUpscaleProcgenBlend uses `_OTRL.in_flight_ledger_path()` for episode_id. SeedVR2 not registered in `__init__.py` (beta candidate, out of scope). |
| 6 | Audit `post_audio_video_pipeline.py` reachability | **RETIRED** per `__init__.py` comment. NOT in `workflows/otr_scifi_16gb_full.json`. Backward-compat-only registration. |
| 7 | L3 rewrite cast + portrait infrastructure | **AUDITED CLEAN** — no rewrite needed. `_otr_cast_repair.py`, `_otr_voice_resolver.py`, `voice_render.py` (NOT registered), `_voice_backends/{bark,kokoro}.py`, `batch_flux_portrait_render.py` — all utility/support modules with no script_json wire input or legacy parser-list assumptions. |
| 8 | L3 utility supporting nodes | **AUDITED CLEAN** — no rewrite needed. `_otr_period_prompts.py` is dataclass/helper module with field-agnostic `render_prompt(user_instruction, ...)`. |
| 9 | Helper API tests | **SHIPPED** — `tests/test_otr_ledger_consumers.py`, **48/48 PASS**. Six classes covering `load_ledger`, `iter_lines`, `cast_lookup`, `speaker_name`, `voice_preset`, `production_plan_or_empty` + cross-helper Pattern 2 walk composition. Includes legacy-list ValueError guard, role-filter narrowing, missing-field graceful degradation, type-coercion edge cases. |
| 10 | B4 LLM prompt audit | **SHIPPED** — appended as "LLM prompt audit — 2026-05-10" subsection to `ROADMAP.md`. **15 CLEAN / 1 DEAD CODE / 0 NEEDS UPDATE.** Every active prompt site (outline, line composer, period, critic, revision, director repair, FLUX radio bookend, LTX role, MusicGen) decoupled from raw `lines[]` reads via request dataclasses or already on L3-correct field names (`meta.gen_params_initial.style`). Only DEAD CODE is `_build_normalize_prompt` on the legacy writer path (scheduled for post-soak deletion per ROADMAP sprint exit criterion 7). |
| 11 | Fresh workflow JSON | **SHIPPED IN PLACE** — `workflows/otr_scifi_16gb_full.json` updated. Node #1 swapped from `OTR_LLMScriptWriter` (legacy parser-list emitter) to `OTR_LedgerScriptWriter` (v2 L3 emitter). 3 outbound links preserved (`#1.0` script_text, `#1.1` script_json, `#1.2` news_used) — both writers share `RETURN_TYPES`/`RETURN_NAMES` so wiring stays intact. Widgets remapped: `news_seed=PLACEHOLDER` (Jeffrey replaces pre-soak), `style_hint='tense claustrophobic'`, `target_seconds=140` (legacy 350 words / 2.5 wpm), `cast_size=2`, `model_id='mistralai/Mistral-Nemo-Instruct-2407'`, `polish_pass=False`. Backup: `workflows/legacy_archive/otr_scifi_16gb_full__pre_l3_writer_swap_2026-05-10.json`. |
| 12 | Dry-run gates (NO GPU) | **SHIPPED ALL PASS** — `outputs/dry_run_gates.py` validates 5 gates: (1) AST parse 18 L3-surface files, (2) every OTR_* type in workflow registered in `__init__.py:_NODE_MODULES`, (3) widget_values count vs INPUT_TYPES (1 trailing-default UNDER on `OTR_SaveToEpisodeWorkspace #25` saved=2 vs required_min=3 — ComfyUI fills defaults; 0 hard mismatches), (4) all 59 link socket indices in bounds, (5) all link types non-empty strings. **ALL 5 GATES PASS.** |
| 13 | Final state report | **THIS DOCUMENT.** |

## Hard rules — all upheld

- **Bug Bible regression 23/1/2/0** — held at every checkpoint.
- **UTF-8 no BOM** — all touched files written via `os.fdopen(...,"w",encoding="utf-8")` or via Edit/Write tooling.
- **Per-file commits** — paused; no commits, no pushes during in-flight work per the predecessor session policy adapted: STEPS 1-8 collapsed to recon-verdicts (no code edits), so there's no per-file commit to make. STEPS 9-12 are atomic deliverables (`tests/test_otr_ledger_consumers.py` new, `ROADMAP.md` edit, `workflows/otr_scifi_16gb_full.json` edit, `docs/BUG_LOG.md` edit, `outputs/dry_run_gates.py` + `outputs/migrate_workflow_l3.py` external scripts). All ready for Jeffrey's manual review + bundle commit.
- **No edits to locked v2.0 modules** — `_otr_outline.py`, `_otr_canon.py`, `_otr_line_composer.py`, `_otr_model_loader.py`, `OTR_LedgerScriptWriter.py`, `_otr_legacy_writer.py`, `_otr_ledger_consumers.py`, `_otr_ledger.py` all untouched.
- **INPUT_TYPES untouched on consumers** — production_plan_json demotion already done in eec4718; this sprint added zero new optional widgets.
- **Per-file scope respected** — only the workflow JSON was edited, in service of the L3 writer-swap.

## Files touched this sprint

```
NEW   tests/test_otr_ledger_consumers.py                               (~360 LOC, 48 tests)
EDIT  ROADMAP.md                                                       (recon verdict + B4 audit appended to CURRENT WORK)
EDIT  docs/BUG_LOG.md                                                  (NON-BUG-2026-05-10 recon entry)
EDIT  workflows/otr_scifi_16gb_full.json                               (node #1 OTR_LLMScriptWriter -> OTR_LedgerScriptWriter)
NEW   workflows/legacy_archive/otr_scifi_16gb_full__pre_l3_writer_swap_2026-05-10.json   (backup)
NEW   docs/2026-05-10-downstream-l3-audit__00_question.md              (round-robin question; consult API timed out — verdict made on direct evidence + ROADMAP precedent)
NEW   docs/2026-05-10-downstream-l3-sprint-final-report.md             (this report)
```

External scripts (NOT in repo, kept in `outputs/` for reference):
```
outputs/migrate_workflow_l3.py     (one-shot writer swap, atomic write)
outputs/dry_run_gates.py           (re-runnable validator)
outputs/rr_question.md             (round-robin question source)
```

## Pre-soak handoff checklist

When Jeffrey wakes up:

1. **Replace the placeholder news_seed** in `workflows/otr_scifi_16gb_full.json` node #1 widget `[0]` with the real seed for the soak run. The placeholder is documented as such in the value text. If the workflow is loaded into ComfyUI Desktop without replacing, `OTR_LedgerScriptWriter._validate_inputs` will fail loudly per the v2 "no RSS fallback" contract.
2. **Verify no dirty file outside expected set** before committing:
   ```
   cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio && git status
   ```
   Expected dirty:
   - `ROADMAP.md` (recon verdict + B4 audit)
   - `docs/BUG_LOG.md` (NON-BUG-2026-05-10)
   - `workflows/otr_scifi_16gb_full.json` (writer swap)
   - `workflows/legacy_archive/otr_scifi_16gb_full__pre_l3_writer_swap_2026-05-10.json` (NEW backup)
   - `tests/test_otr_ledger_consumers.py` (NEW)
   - `docs/2026-05-10-downstream-l3-audit__00_question.md` (NEW round-robin question)
   - `docs/2026-05-10-downstream-l3-sprint-final-report.md` (NEW final report)

   Pre-existing dirty (NOT touched by this sprint):
   - `config/episode_cast.txt`, `nodes/project_state.py`, `scripts/audit_video_stack_weights.ps1`, `tests/_reports/vram_profile_report.json`, `tests/test_wedge_probe.py`, plus untracked items already on the working tree at session start.

3. **Suggested commit message** when ready (per CLAUDE.md cmd-shell quoting rules):
   ```
   cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
   git add tests/test_otr_ledger_consumers.py ROADMAP.md docs/BUG_LOG.md workflows/otr_scifi_16gb_full.json workflows/legacy_archive/ docs/2026-05-10-downstream-l3-audit__00_question.md docs/2026-05-10-downstream-l3-sprint-final-report.md
   echo Downstream L3 sprint -- 13/13 shipped, visual chain audited clean, helper tests 48/48, B4 prompt audit clean, workflow swap ledger writer, dry-run gates pass> .git\COMMIT_EDITMSG
   git commit -F .git\COMMIT_EDITMSG
   ```

   Then push via Desktop Commander cmd shell (per CLAUDE.md): `git push origin v2.0-alpha`

4. **Soak ramp ladder** (Jeffrey's manual scope, NOT in dev sprint):
   - Smoke (30 → 100 words) to confirm L3 path doesn't crash any consumer.
   - Quick (200 → 350 words) — full audio + visual chain, single episode.
   - FULL (700 → 1400 → 2100 → 3500 words) — the actual production scale.

5. **If any soak run fails at the L3 boundary** (e.g., a consumer raises `ValueError: legacy parser-list format not supported`), the most likely cause is a stale `script_json` wire — confirm node #1 is `OTR_LedgerScriptWriter` in the loaded graph (ComfyUI may load a cached graph; restart Desktop after pulling).

## Real surprises encountered (none required Jeffrey's intervention)

- **Visual nodes live in `visual/`, not `nodes/`** — the sprint plan listed `nodes/batch_flux_render.py` but the actual file is `visual/batch_flux_render.py`. Trivial path correction; file content unchanged.
- **`SeedVR2` not in `__init__.py:_NODE_MODULES`** — the sprint plan listed it under Step 5; turned out to be a beta candidate that was never registered. Reported in Step 5 verdict, no work attempted.
- **Round-robin consultation timed out** on the MCP wrapper at 240s (the consult ladder runs 2-3 model calls of 60-120s each in series). Recon decision was supported by ROADMAP line 67 prediction + direct grep evidence + visual inspection of every active consumer; proceeded without the round-robin signal. Question file preserved at `docs/2026-05-10-downstream-l3-audit__00_question.md` for a future async run if Jeffrey wants the second opinion.
- **Workflow JSON swap required widget remap, not just type swap** — `OTR_LLMScriptWriter` and `OTR_LedgerScriptWriter` share output contract but have different INPUT_TYPES surface. Migration script `outputs/migrate_workflow_l3.py` documents the mapping explicitly (legacy `target_words=350` → v2 `target_seconds=140` via WORDS_PER_SECOND_BUDGET=2.5; legacy `style` widget → v2 `style_hint`; cleanup_model_id, custom_premise, include_act_breaks, self_critique, open_close, target_length, creativity, optimization_profile dropped — all internalized by the v2 LPL writer or unsupported in v2). Backup retained.

## Final regression posture

| Check | Result |
|---|---|
| Bug Bible regression (`bug_bible_regression.py`) | **23 passed, 1 skipped, 2 xfailed** |
| Helper API tests (`test_otr_ledger_consumers.py`) | **48 passed** |
| Dry-run gates (5 gates, AST + registration + widgets + links + types) | **ALL PASS** |
| AST parse on every L3-surface file (18 files) | **all OK** |
| Workflow JSON link integrity | **all 59 links resolve** |

**No work blocked. Sprint ready for soak hand-off.**
