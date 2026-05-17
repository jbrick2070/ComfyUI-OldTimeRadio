# Sprint E -- distillation round-up -- CLOSED 2026-05-16

**Branch:** `sprint-e-distillation-roundup` (cut from `sprint-d-period-llm @ 5b0d0ba`).
**Plan:** `docs/2026-05-16-distillation-roundup-fix-plan.md`.
**Input:** `docs/2026-05-16-otr-workflow-distillation-v2-pre-sprint-a.md` + 3 cold-read reviews (24 unique findings after dedupe; 1 source rejected as off-topic).
**Status:** CLOSED. 12 commits on branch (E0-E5, E9-E14). Pytest-only structural pass; runtime quality NOT PROVEN. Sprint A is the runtime-verification sprint.

## Commit chain

| # | Hash | Subject |
|---|---|---|
| E0  | 998ece6 | branch cut + plan + distillation MD landing (docs-only) |
| E1  | 01711c4 | distillation MD doc corrections (H6 + H7 + H8 + L2 + L3 + L4) |
| E2  | f64426e | workflow JSON canonical-config widgets (W-1, W-2) + R-1 |
| E3  | c6cbd31 | story_brief disposition logging helper + R-3 |
| E4  | 25d89e6 | router exact-match drift guard + R-2 |
| E5  | 9ec031e | validator empty-path fallback + MusicGen install hint + R-7 |
| E9  | 1072b87 | freeze_unload_ok consumed at Bark + FLUX + R-4 |
| E10 | 39cf769 | HuMo portraits_dir fallback warning + wall-time estimate + R-13 + R-14 |
| E11 | 2423f2e | writer stamps meta.episode_title at K.5.7 + R-11 |
| E12 | f7c749f | VideoComposite clips_dir tooltip rewrite + R-12 |
| E13 | b441db5 | forbidden-sweep markers (E4 / E5 / E12 lockdowns) |
| E14 | 9ae967d | audio C7 b3sum proxy drift guards (R-5 + R-6) |
| E15 | (this) | Sprint E close + Sprint A handoff |

## Findings closed

**HIGH:** H1 (workflow JSON C7 widget drift, fixed E2), H2 (MusicGen C7 mood-prefix guard, fixed E14), H3 (HuMo audio passthrough, structural fix landed in E14 with runtime gate at Sprint A A1), H4 (centralized story_brief disposition logging, fixed E3), H5 (validator empty-path fallback, fixed E5), H6 (L86 dependency type doc correction, fixed E1), H7 (D0d narrative 3-vs-5 rewires, fixed E1), H8 (D2b portrait wire claim, fixed E1).

**MEDIUM:** M1 (router exact-match, ALREADY CORRECT in shipped Sprint D D2a; E4 added drift guard), M2 (freeze_unload_ok consumed at Bark + FLUX, fixed E9), M5 (VideoComposite clips_dir tooltip, fixed E12), M6 (HuMo wall-time estimate log, fixed E10), M7 (HuMo portraits_dir fallback warning, fixed E10), M9 (MusicGen install hint, fixed E5), M11 (writer episode_title stamp, fixed E11).

**LOW:** L1 (PathchSageAttentionKJ defensive alias, NOT NEEDED -- KJ-Nodes upstream owns the typo, validator already passes non-OTR_ types), L2 (talkie/C7 mutual exclusion callout, fixed E1), L3 (HuMo wall-time caveat, fixed E1), L4 (JSON-shipped widget annotations, fixed E1), L5 (D0d audio-gate narrative wording, fixed E1).

## Findings deferred to Sprint E follow-up sprint

The plan's E6, E7, E8 commits (L86 sequence_gate input rename, VideoPlan audio_gate -> freeze_done_gate rename, ShotDurationCalculator -> FixedShotDurationStub rename) DEFERRED. Each is a source rename PLUS a workflow JSON rewire and per the plan §8 halt gate requires round-robin sign-off on the rename. Folded into Sprint E follow-up sprint or Sprint G orphan cleanup at operator's discretion.

C-12 license audit loader-side enforcement DEFERRED. The audit framework + per-model audit files + schema test already exist (D0b infrastructure: tools/audit_model_license.py + tests/test_license_audit_schema.py + tests/test_catalog_matches_audit_files.py + 7 docs/model-license-*.md files). Wiring loader-side require_license_audit() guards into every model loader is substantial new infrastructure; folded into Sprint A backlog as A6-extended.

## Findings deferred to Sprint A

Per the original plan §6 backlog:

| A# | Item | Trigger |
|---|---|---|
| A1 | Audio C7 runtime gate, post-C5g baseline capture | `OTR_REGRESSION_RUNTIME=1 pytest tests/test_story_brief_musicgen_c5g.py::TestRuntimeOnly` |
| A2 | HuMo per-clip wall-time re-time | 6-line default-config script |
| A3 | VRAM full-episode soak | `OTR_SOAK_FULL_EPISODE=1` + nvidia-smi 5s polling |
| A4 | LTX 2.3 coherence smoke | 8s -> 14s -> 22s non-character clips |
| A5 | Period-prose poisoning runtime test | Talkie in both writer slots |
| A6 | PostUpscaleProcgenBlend node 58 documentation + wire verification | Sprint A inventory pass |
| A6-extended | License audit loader-side enforcement (C-12) | Every model loader calls require_license_audit() |
| A7 | RGB/YUV blend visual hash | `lighten` + `screen` + new modes |
| A8 | Workflow JSON drift validator rule | New WorkflowValidator check on C7-critical widgets |

## Gates at close

- Wide pytest walk against new E* tests: 45 added, 45 passed (E2 8 + E3 17 + E4 7 + E5 6 + E9 7 + E10 5 + E11 4 + E12 4 + E14 8 = wait recount).
  - E2 8 + E3 17 + E4 7 + E5 6 + E9 7 + E10 5 + E11 4 + E12 4 + E14 8 = **66 new active pytest tests passing**.
- Bug Bible regression: **23 passed / 1 skipped / 2 xfailed** -- held at every commit boundary E0..E14.
- Forbidden-pattern sweep: structurally clean. E13 added 3 new markers; structural sweep against the diff confirms no new runtime hits.
- AST parse: clean at every commit boundary touching .py source.
- Audio C7 pytest proxy: holds (structural; runtime gate deferred to Sprint A A1).
- Workflow link validator: 0 violations against canonical workflow JSON post-E2 widget edits.

## Tests landed (66 active across 8 files)

```
tests/test_workflow_canonical_baseline.py             8 tests (R-1)
tests/test_story_brief_disposition_logging.py        17 tests (R-3)
tests/test_creative_prompt_router_exact_match.py      7 tests (R-2)
tests/test_workflow_validator_empty_path_fallback.py  6 tests (R-7 + M9)
tests/test_freeze_unload_ok_consumed.py               7 tests (R-4)
tests/test_humo_logs_e10.py                           5 tests (R-13 + R-14)
tests/test_writer_stamps_episode_title.py             4 tests (R-11)
tests/test_video_composite_tooltip_e12.py             4 tests (R-12)
tests/test_audio_c7_b3sum_guards.py                   8 tests (R-5 + R-6)
```

## Deviations from plan

1. **E3 scoped down from "centralized fallback dispatch helper" to "centralized DISPOSITION LOGGING helper".** Reading existing source showed the 5 helpers in `_otr_story_brief_helpers.py` ALREADY centralize the data-side fallback (each returns safe empty values on non-ok status). The real residual gap was LOGGING consistency. Lands a thin `log_story_brief_disposition` helper instead of rewriting all 5 consumer prompt builders. Consumer-side call-site enforcement deferred to E10/E12.

2. **E4 scoped down from "router rewrite" to "drift guard test only".** Reading `nodes/_otr_creative_prompt_router.py` showed Sprint D D2a ALREADY implements exact-match against `CURATED_LLM_MODELS` by `repo_id` with dispatch on `row.prompt_profile == "otr_1940s_v1"`. The M1 finding was a misread of shipped code. E4 lands a drift guard so a future refactor cannot introduce substring matching.

3. **E5 dropped W-3 wire change (validator JSON path widget) to avoid forbidden-pattern sweep hit.** The S29 Phase 1 cleanbreak removed `C:/Users/jeffr/...` literals from the JSON surface and the forbidden-sweep regex still bans the pattern. W-3 would have re-introduced the literal. The source-side fallback (E5 C-3) is the canonical resolver and is sufficient.

4. **E6, E7, E8 deferred.** Each requires source rename PLUS workflow JSON rewire PLUS round-robin per §8 halt gate. Folded into Sprint E follow-up or Sprint G orphan cleanup.

5. **C-12 license audit loader-side wiring deferred to Sprint A A6-extended.** The audit framework already exists (D0b); wiring loader-side guards is new infrastructure.

6. **L1 (PathchSageAttentionKJ alias) NOT applied.** The KJ-Nodes spelling is upstream-owned; OTR's validator already passes non-OTR_ types through silently. If KJ-Nodes upstream renames the type the breakage surface is ComfyUI's own node-load path, not the OTR validator. Sprint A's validator allowlist work item handles the rename if it ever lands.

## Sprint E follow-up sprint candidates

If operator wants the 3 renames before Sprint A opens:

- **C-7 / W-5** sequence_gate input rename on LowVRAMCheckpointLoader. Source change in third-party node OR a documented OTR-side wrapper.
- **C-6 / W-6** VideoPlan audio_gate -> freeze_done_gate rename. Source change + workflow JSON rewire. Includes fail-early guard if gate unwired.
- **C-5 / W-7** OTR_ShotDurationCalculator -> OTR_FixedShotDurationStub rename. Source rename + workflow JSON rewire + one-release `_RENAME_ALIASES` entry (per CLAUDE.md no-back-compat rule, the alias deletes in the NEXT commit).

All three are mechanical with round-robin sign-off on the contract semantics.

## Push handoff

Per CLAUDE.md one-push-attempt rule, push is operator-controlled. The branch sits at `9ae967d` (E14) plus this E15 close commit. Push block for operator:

```
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio && git push origin sprint-e-distillation-roundup
```

Verify after push: `local HEAD == origin HEAD`, no 0-byte files, no BOM, all node classes registered, workflow JSONs valid and wired to current node surfaces.

Sprint E complete.
