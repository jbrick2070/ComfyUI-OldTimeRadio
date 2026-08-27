# R4 Judgment — Convergence / Residual Defects

Driver/panelist/judge: Codex
External reviewers: Cursor (Grok) and Claude Code (Claude)
Reviewer calls this round: 2
Campaign external calls: 8

VERDICT: yes — plan converged after integrating the final test-import and live-audit shape corrections.

## Accepted and integrated

1. **Integration-test constant imports — CONFIRMED.** Neither integration file currently imports `_COCKNEY_ORTHOGRAPHY_RULE`. Both imports are now explicit build steps, preventing NameError.
2. **Audit object shape — CONFIRMED.** Production `VoicedSlot` has no `beat_id`; `group_voiced_beats` preserves arbitrary input objects. The plan now requires `SimpleNamespace`/equivalent objects carrying beat ID, slot ID, and speaker.
3. **Fail-closed ledger join — CONFIRMED against a published ledger.** Opening/closing music rows have blank beat IDs; dialogue rows carry beat/slot identity; final `beats[]` preserve order but not slot IDs. The plan now ignores normal blank music identities, rejects duplicate nonblank IDs, and treats zero/invalid joins as run breaks.
4. **Full policy literal — CONFIRMED.** The plan now shows the actual Python constant including the leading `"\n\n"`, eliminating an implementer interpretation gap.
5. **Real fake signature — CONFIRMED.** `_fake_gen_valid` has defaulted keyword-only arguments. The recorder contract now matches `(messages, *, temperature=0.0, max_new_tokens=0)`.
6. **Helper docstring boundary — CONFIRMED.** The plan now names `LineRequest.speaker` and `VoicedSlot.speaker` as the only intended active-output sources.
7. **Direct-versus-repair test split — CONFIRMED.** Direct builder calls cover presence/absence; `compose_exchange` plus stateful Tier-A covers repair identity.
8. **Red/green mutation receipt — CONFIRMED.** The two core tests must fail before the code change and pass after it, preventing a false-green regression addition.
9. **GO_FORWARD contradiction removal — CONFIRMED.** The records step now replaces, rather than merely annotates, the stale global-orthography instruction.
10. **Bounded live retry and bank integrity — CONFIRMED.** One rerun maximum; media-archive failure cannot be silently relabeled as another bank.
11. **Expected diff budget — CONFIRMED.** Six implementation/test files are named; aliases, persistent one-off scripts, and unrelated production modules are excluded.
12. **Campaign receipt — CONFIRMED.** Eight external calls, exactly two per round, Cursor present in every round, and Codex anchor/judgment in every round.

## Rejected or refined

1. **Claude verdict “no must-fix” — REFINED.** Claude found no architectural blocker, but the driver and Cursor independently verified the missing integration imports and audit object/identity gaps. Those were integrated before declaring convergence.
2. **Fail on every blank line beat ID — REFINED.** Blank music-opening/closing line identities are normal and are ignored. Duplicate nonblank IDs fail the receipt; missing per-beat joins break runs.
3. **Checked-in qualification script — REJECTED.** A deleted temporary probe using the production grouping helper is sufficient.
4. **`append_dialogue_policy(None, ...)` test — REJECTED.** Both real callers pass resolved strings; this is outside the runtime contract.
5. **`roster_has_lemmy` compatibility alias — REJECTED.** It would preserve the wrong category and weaken the root fix.
6. **Native `python -c` verification snippets proposed by Claude — REJECTED for execution form.** Repository PowerShell rules forbid nested-quote `python -c`; the same invariants are covered by pytest, AST, JSON, and workflow gates.

## VERIFY-AT-BUILD checklist

- New ALICE/full-cast and mixed-exchange tests fail on the old implementation, then pass after the atomic patch.
- `rg -n "roster_has_lemmy" nodes tests` returns no matches.
- The updated constant starts with `"\n\n"` and is appended once.
- Both real callers pass tuples of current output speaker strings.
- First and repair calls carry identical system content.
- `git diff -- workflows/otr_canonical.json` is empty and widget slot 13 remains true.
- Focused, full, Bug Bible, and workflow gates pass.
- The canonical live run records both explicit writer patches, a chosen ephemeral port, canonical asset receipts, and a reconstructable accepted LEMMY+other exchange group.
- The GO_FORWARD global-orthography wording is replaced and cross-repository commit/push receipts are complete.

No residual must-fix remains in the plan itself.
