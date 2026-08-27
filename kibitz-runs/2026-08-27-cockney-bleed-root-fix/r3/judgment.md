# R3 Judgment — Wiring / Integration / Sequencing

Driver/panelist/judge: Codex
External reviewers: Cursor (Grok) and Antigravity (Gemini)
Reviewer calls this round: 2

## Accepted and integrated

1. **Atomic migration reiterated — CONFIRMED.** Helper signature, both callers, obsolete import removal, and replacement tests are one working-tree change before any focused test runs.
2. **Two-stage production grouping — CONFIRMED.** `run_exchange_prepass` first splits runs on invalid slot/empty/reserved speaker, then calls `group_voiced_beats`. The live audit now reproduces this order and filters against accepted beat IDs only after grouping.
3. **Final ledger join — CONFIRMED by judge artifacts.** Published `beats[]` preserve order but do not provide usable dialogue slot IDs; `lines[]` carry `beat_id`, `dialogue_slot_id`, and speaker. The plan now indexes lines by beat ID and walks beats in canonical order.
4. **Distinct test-fake shapes — CONFIRMED.** Line capture is a list of message lists; exchange capture is a list of dicts containing `messages`; prepass's dynamic fake stores nothing. Exact access paths and wrapper signature are now pinned.
5. **System-only repair equality — CONFIRMED.** Exchange failure reasons intentionally alter the user message. Only the system content must remain identical across attempt and repair.
6. **Retain exchange cast/persona wiring — CONFIRMED.** `cast` remains necessary for the per-active-slot persona block. It is removed only from the policy trigger.
7. **Port semantics — CONFIRMED.** Direct wrapper default `Port=0` chooses an ephemeral port. The plan still performs the project's required port-8000/process/VRAM reset, omits `-Port`, and records the actual chosen port.
8. **Wrapper VRAM log is not a gate — CONFIRMED.** The wrapper logs post-reset VRAM but does not fail on excess. The hard baseline check remains an explicit precondition.
9. **Applied patch receipt — CONFIRMED.** The live leg must show the two exact writer patches before it can qualify.
10. **Stop condition uses policy identity — CONFIRMED.** The gate now checks the canonical policy constant in the captured system message rather than any generic word occurrence.

## Rejected or refined

1. **Add a `None` system-prompt test — REJECTED.** Both production callers resolve a non-null string before invoking the helper. Preserving `system_prompt or ""` is sufficient; testing an unused typed-contract violation adds no defect-killing value.
2. **Treat profile port 8000 as the wrapper listen port — REJECTED.** The local profile describes the usual production endpoint, but this wrapper's actual default is ephemeral. The hard reset still checks 8000 because repository rules require it.
3. **Add persistent group-boundary ledger metadata — REJECTED.** Existing beat order, line identity, audit IDs, and the production grouping helper are sufficient for a temporary qualification audit.
4. **Second live fallback leg — REJECTED.** Per-line scope is deterministically captured in tests; forcing a production fallback would require an unsupported workflow toggle or manufactured failure.
5. **Antigravity “build-ready as-is” — REFINED.** The core wiring is build-ready, but Cursor and the driver independently found the receipt-reconstruction ambiguity. R3 therefore integrates that fix before convergence.

## Verify at build

- Confirm the final ledger join produces valid d### slots without relying on `lines[]` iteration order.
- Confirm the direct PowerShell `-Set @(...) ` binding yields both applied patch receipts.
- Confirm the wrapper logs a chosen ephemeral port and the final asset/ledger correspond to that run.
- Preserve the full user roster/persona behavior while removing cast only from the policy decision.
