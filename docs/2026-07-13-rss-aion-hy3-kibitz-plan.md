# RSS source floor and Aion/Hy3 smoke resilience

## Scope

Review the live canonical ComfyUI evidence from 2026-07-13 for two coupled
operator concerns: (1) strict `scifi_codex` RSS admission intermittently
rejecting a 42-word run even though a 120-word run selected a qualifying RSS
body, and (2) whether the free `tencent/hy3:free` creative lane is a valid
comparison against `aion-labs/aion-3.0-mini` without changing workflow wiring.

## Grounded evidence

- `tmp/scifi_120_p5_recovery_harness3.out.log` is `RESULT SUCCESS`; server
  log records two candidates passing the v4 floor, `RSS_FETCH OK`,
  `obs_publish OK`, and a 00:19:40 prompt.
- `tmp/scifi_42_aion_final_harness4.out.log` is `RESULT FAIL`; its server log
  records zero inline candidates prioritized and a fail-closed v4 source-floor
  error before model dispatch.
- `tmp/scifi_42_hy3_compare_harness.out.log` is empty/incomplete; its server
  log shows the canonical workflow queued Hy3, then no terminal result in the
  captured file. This is evidence of an incomplete comparison, not a model
  quality verdict.
- The canonical workflow remains `workflows/otr_canonical.json`; no wiring
  change is in scope.

## Questions for reviewers

1. Is the 42-word Aion failure a source-selector starvation/ordering defect,
   or evidence of a deeper contract mismatch? Identify the narrow root repair.
2. What minimum telemetry and paired-run protocol distinguishes RSS admission,
   provider failure, and generated-script validation for Aion versus Hy3?
3. Which claims require a fresh canonical 42-word Hy3 run, and what must remain
   unchanged to make the comparison valid?

## Constraints

Preserve the strict source floor and fail-closed behavior; do not hide source
failure with an automatic premise fallback. Do not alter canonical workflow
wiring unless a grounded validator proves drift. Keep the GUI port 8001 alive;
headless tests reset only port 8000. Reviewers propose; the Codex driver judges
against the Windows files and logs.

