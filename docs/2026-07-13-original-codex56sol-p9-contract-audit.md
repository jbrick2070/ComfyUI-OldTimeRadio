# Original Codex56Sol P9 contract-audit recovery

## Scope

Audit the live Aion 3.0 Mini / Mistral-Nemo canonical 42-word failure
(`tmp/scifi_42_aion_final_server.log`, prompt `33b2cd95-cc1d-4f74-a389-ac866d148013`; the operator also referenced `e0a03830-aa18-42c4-8c47-89c6cff51a46`) where P9 raised `final contract audit rejected the script without actionable grounded findings`.

## Grounded evidence

- `nodes/_otr_original_codex56sol.py:324` declares `ContractFinding.exact_span: str`.
- The live P9 raw response logged at `tmp/scifi_42_aion_final_server.log:400-406` emitted `exact_span` as JSON arrays (`[5, 9]`, `[15, 28]`), causing two Pydantic `string_type` failures before the retry.
- `_call` (`nodes/_otr_original_codex56sol.py:699-812`) retries typed schema failures, but the subsequent P9 response was not persisted as an actionable finding; `run_original_codex56sol_episode` (`:1974-1981`) then fail-closes whenever `accepted` is false and `_audit_blocks` is empty.
- `_audit_blocks` (`:1772-1780`) requires a blocking finding whose `item_id` resolves to a script line, whose `exact_span` is a string contained in that line, and whose field/category/correction are non-empty. Manifest paths such as `manifest.lines[4].clue_ids` cannot produce an actionable script retake by design.
- No workflow wiring change is implied; the failure occurs after P1-P8 and before ledger save.

## Questions

1. What is the narrow root repair for P9 schema/transport mismatch while preserving fail-closed behavior?
2. Should `exact_span` remain a string, or should a bounded normalization/schema change accept a character range and convert it deterministically? What tests prove both raw and typed retry paths?
3. How should non-script findings (manifest graph references) be handled so the audit does not repeatedly request an un-actionable retake?

## Constraints

Preserve strict final-contract validation and no blind acceptance. Do not mine failed prose or add workflow wiring. Any recommendation must be verified against the Windows files and this live log.
