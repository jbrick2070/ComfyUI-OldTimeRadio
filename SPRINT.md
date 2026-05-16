# Sprint D -- period-LLM CATEGORY -- CLOSED 2026-05-16

## Status

- Phase: closed
- Final commit: D-final
- Branch: sprint-d-period-llm (pushed to origin)
- Closed-sprint archive: docs/closed-sprints/2026-05-16-sprint-d-period-llm.md
- v1.9 tag: pending operator action (per CLAUDE.md "Only Jeffrey
  merges to main and tags releases")

Sprint D shipped the period-LLM CATEGORY: catalog metadata schema
extension (6 new CuratedModel fields), license-audit framework
over 7 per-row markdown files, loader-backend duck-typed protocol
with 3 concrete adapters (transformers_safetensors,
transformers_multimodal_text_only, transformers_gptq_int4),
creative-phase prompt resolver wired into 4 phase sites with
audio C7 byte-identity preserved at default config, and runtime-
gated test surfaces staged for Sprint A's empirical-verification
pass.

13 commits including D-final. ~71 new active pytest tests. ~7
runtime-staged-skip tests. All gates green at every commit
boundary. Audio C7 byte-identical proxy HELD against v1.5
fixture end-to-end. Bug Bible regression 23/1/2 held end-to-end.
Forbidden-pattern sweep 0 runtime hits at every commit boundary.

See `docs/closed-sprints/2026-05-16-sprint-d-period-llm.md` for
the full record including 5 documented deviations from the v3
plan and Sprint G handoff for the `_LegacyTransformersBackendBase`
orphan-candidate.

## Sprint A pending handoffs (inherited from Sprint D)

- D1c runtime tests (3) -- talkie GPTQ int4 loader smoke + VRAM
  cleanup + tokenizer chat_template guard. Require HF_TOKEN +
  GPU + OTR_REGRESSION_RUNTIME=1. Sprint A unblocks by running
  with HF login configured.
- D4 runtime tests (4) -- period-creative VRAM peak + xfail
  determinism + no-news diction guard + modern-news warning.
  Require writer runtime harness for end-to-end fixture-based
  runs. Sprint A wires the harness then unblocks.
- SA-100 schema-positive widgets_values canonical-shape gate
  (from Sprint C triage retrospective, not introduced by Sprint D).
- SA-102 tools/capture_hardware_snapshot.py (from triage).
- SA-103 VRAM telemetry in S-A.4 multi-model regression (from triage).
- SA-101 silent reflection clamp log -- **dropped from Sprint A
  backlog** because D0c (commit `d00450f`) shipped the patch
  forward. Documented at closed-sprint archive §D-final shipped
  state.

## Sprint G pending handoffs (inherited from Sprint D)

- `_LegacyTransformersBackendBase` orphan-candidate review per
  D-final note. Case-by-case judgment call after Sprint A's
  empirical pass validates per-backend split need.

## Next sprint

To be decided by operator. Sprint A (downstream verification +
runtime empirical pass) and Sprint G (broader cleanup sweep) are
both queued.
