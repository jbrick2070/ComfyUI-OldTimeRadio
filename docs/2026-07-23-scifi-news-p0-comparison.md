# Open bug comparison: `PBUG-20260722-02`

Review this one live, open issue only. Compare two independent read-only
diagnoses of the same bug: (A) Antigravity through the Kibitz review path and
(B) the remote RTX 4060 `qwen3-coder-30b-a3b-instruct` worker.

## Issue

`PBUG-20260722-02` is an open `scifi_news` P0 source-span failure. In two live
post-seam-fix qualification runs, P0 accepted a normalized A0 source payload,
but the technical model plus bounded repair returned an `F01` quote that did
not equal the declared `full_text[start:end]` slice. Both attempts failed
closed before ledger or OBS output. This is not a word-count or prose gate.

## Grounding targets

- `docs/PROD_BUG_LOG.md` entry `PBUG-20260722-02` and its cited prompts.
- `_otr_scifi_codex.py`, especially source-span validation and bounded repair.
- The normalized A0 payload/admission contract and tests covering literal
  source spans.
- `scripts/otr_canonical_api_run.py` only for terminal evidence semantics.
- Live artifacts under `tmp/six_bank_sweep_20260722_200609_509/` and
  `tmp/six_bank_sweep_20260722_201449_793/` if present.

## Questions

1. What is the smallest root fix at the accepted-object boundary?
2. Is the defect in prompt/repair behavior, source normalization, validator
   ownership, or telemetry/transport? Cite evidence.
3. What focused regressions and two live requalification checks are required?
4. Identify any attractive fix that would weaken literal-span safety or the
   no-prose-gate law.

Return a concise diagnosis, proposed fix, and test plan. Do not edit the repo.
