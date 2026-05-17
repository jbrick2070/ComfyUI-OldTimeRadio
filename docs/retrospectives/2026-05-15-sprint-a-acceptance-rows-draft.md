# Sprint A acceptance rows draft (SA-100 .. SA-103)

**Origin:** triage pass over `docs/retrospectives/2026-05-15-sprint-c-triage-findings.md` §6 Adjudication.
**Captured:** 2026-05-16 on triage branch `triage-sprint-c-retrospective-2026-05-15`.
**For:** Sprint A planning. Paste these four rows verbatim into Sprint A's `SPRINT.md` "Acceptance table" section when Sprint A opens.

Format matches the standard SPRINT.md acceptance table layout:

```
| # | Check | Target |
|--:|---|---|
```

Numbered SA-100+ to avoid colliding with any earlier-numbered Sprint A row that may already be in flight from the Sprint C close handoff (audio C7 baseline capture, empirical visual + audio render quality verification, empirical LTX motion fidelity verification).

---

## Rows (ready to paste)

| # | Check | Target |
|--:|---|---|
| SA-100 | Workflow JSON `widgets_values` schema-positive canonical-shape gate green. Every node's `widgets_values` matches the live `/object_info` canonical preserved-mode shape (linked placeholders + all unlinked defaults, in declared input order). Cross-wiring class regressions hard-fail. **Note:** triage §1 + §6 Adjudication reframed retrospective §6 "Null-State Padding Violation" -- empty strings / `'[]'` / `'{}'` are the BUG-LOCAL-032 canonical fix, not a violation. Rejecting them would re-introduce widget drift. | First Sprint A runtime-verification commit; wires existing `scripts/_schema_sweep.py` into the acceptance gate. |
| SA-101 | Reflection-module `_repair_pass` clamp visibility: one new `log.info("[OTR_StoryBrief] repair pass clamped: base=%.3f bump=%.3f ceiling=%.3f -> repair_temperature=%.3f reasons=%s", ...)` line inserted at the exact site between current lines 490 and 491 of `nodes/_otr_story_brief.py` (immediately after the `min(...)` clamp computation, immediately before the `_build_repair_messages(...)` call). Two pytest tests staged: `test_repair_pass_emits_clamp_log` (monkeypatch logger, force schema-rejection path, assert one `log.info` call with substring "repair pass clamped" and 3-decimal `repair_temperature`); `test_repair_pass_clamp_log_does_not_break_no_change_logs_rule` (AST scan of module, pin existing `log.*` strings to byte-identical snapshot). Purely additive; no existing log string modified. Severity is `log.info` (clamp is designed pre-flight behavior, not unexpected). | First Sprint A runtime-verification commit. |
| SA-102 | `tools/capture_hardware_snapshot.py` lands, co-located with existing `tools/audit_workflow_schema.py` + `tools/validate_workflow_links.py`. CLI: `--out <path>`, `--check <path>` (exit 2 on strict-fail), `--dry-run`. Output schema captures host platform, python info, GPU name + compute capability + driver, torch version + git_version + cuda runtime + cudnn version + backends_flags (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, cuda.matmul.allow_tf32, use_deterministic_algorithms), env_vars_of_interest (`CUBLAS_WORKSPACE_CONFIG`, `PYTHONHASHSEED`, `OTR_REGRESSION_RUNTIME`), package versions on the floating-point-affecting path (transformers, tokenizers, sentencepiece, bitsandbytes, soundfile, numpy, scipy, librosa), ffmpeg first-line out-of-process, seeds_at_capture. First Sprint A runtime-verification commit runs the script once and commits the resulting `tests/fixtures/hardware_snapshot.json` baseline alongside `audio_c7_baseline.wav.b3sum` + `audio_c7_baseline_pre_c5g.wav.b3sum`. Three pytest tests staged (`test_snapshot_capture_runs_clean`, `test_snapshot_required_keys_present`, `test_snapshot_check_mode_passes_against_self`). | First Sprint A runtime-verification commit -- same commit that closes the inherited acceptance rows for audio C7 baseline reset captures. |
| SA-103 | VRAM telemetry in S-A.4 multi-model regression. After each generation cycle, capture `torch.cuda.memory_summary()` output to `logs/sprint_a_vram_<cycle>.txt`. Aggregator extracts peak allocated, peak reserved, allocator-cached-but-unused, fragmentation indicators. **Strict fail** if any cycle exceeds 14.5 GB peak (existing VRAM ceiling per `nodes/_otr_model_catalog.py:DEFAULT_VRAM_CEILING_GB`). **Advisory fail** if allocator-cached-but-unused fragmentation exceeds 20% of peak across the cycle. Closes triage §4 + retrospective §7 Surface Metric Bias gap. | S-A.4 multi-model regression commit. |

---

## Notes for the Sprint A planner

- **Numbering.** SA-100..SA-103 are additions, not replacements. Sprint A's existing acceptance rows from the Sprint C close handoff (audio C7 baseline reset captures, empirical visual + audio render quality verification, empirical LTX motion fidelity verification per `docs/closed-sprints/2026-05-15-sprint-c-story-brief-v2.md` §C-final.5) keep their existing scope and ordering. The high starting number (100) avoids collisions with any earlier-numbered row Sprint A may already have in flight.
- **Why no SA-104 in this draft.** The retrospective §7 perceptual audio hash supplement (Chromaprint via `fpcalc` subprocess as tier-4 fallback) was DEFERRED in §6 Adjudication. SA-102's hardware snapshot already covers env-drift detection; building a four-tier fallback ladder for a solo-developer single-fixed-machine setup is dragon-chasing. SA-104 parked separately for the v2.1+ watch-list. See `docs/retrospectives/2026-05-15-sprint-d-watch-list-addition-sa-104.md`.
- **Why no SA-105 / SA-106.** Operator directive: don't pad; only real actionable findings. SA-100..SA-103 + the inherited rows from Sprint C close cover every actionable item the triage pass surfaced. The rest of the retrospective's recommendations were framing arguments that the closed-sprint plan already accepted or that §6 Adjudication refuted.
- **No leakage from §5 NUL padding.** The §5 finding was a sandbox/mount read artifact, not on-disk corruption. The forensic note at `docs/retrospectives/UNEXPECTED_FINDING_nul_padding.md` is closure; nothing from §5 enters any Sprint A planning surface. Pre-Sprint-C commit-hygiene of `068bf54` and `af4e655` deferred to Sprint G's broad cleanup sweep.

---

## Source references

- `docs/retrospectives/2026-05-15-sprint-c-triage-findings.md` §1, §2, §3, §4, §6 Adjudication
- `docs/closed-sprints/2026-05-15-sprint-c-story-brief-v2.md` §7 (acceptance table format), §C-final.5 (Sprint A inherited rows)
- `nodes/_otr_story_brief.py:487-494` (clamp site)
- `nodes/_otr_model_catalog.py:DEFAULT_VRAM_CEILING_GB`
- `scripts/_schema_sweep.py` (existing canonical-shape sweep, to be wired in for SA-100)
- `tools/audit_workflow_schema.py` + `tools/validate_workflow_links.py` (co-location reference for SA-102's new `tools/capture_hardware_snapshot.py`)
