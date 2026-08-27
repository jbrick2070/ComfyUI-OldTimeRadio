# R1 Judgment — High-Level Arc / Creative Coherence

Driver/panelist/judge: Codex
External reviewers: Cursor (Grok) and Antigravity (Gemini)
Reviewer calls this round: 2

## Accepted and integrated

1. **Scalar-string category trap — CONFIRMED.** Both reviewers independently identified that `str` satisfies `Iterable[str]`/`Sequence[str]`. The plan now rejects `str` and `bytes` as containers, validates each element, and adds an executable failure case.
2. **Protect unrelated qualification tests — CONFIRMED.** `tests/test_otr_dialogue_policy.py` contains BUG-12.86 route-receipt tests below the two roster tests. The plan now names only `test_roster_has_lemmy` and `test_append_dialogue_policy` for replacement.
3. **One canonical policy constant — CONFIRMED.** Keeping `_COCKNEY_ORTHOGRAPHY_RULE` avoids duplicate text sources. One static scoped block is intentionally used for Lemmy-only and mixed calls.
4. **GO_FORWARD ambiguity — CONFIRMED.** `docs/GO_FORWARD_PLAN.md:167-169` says to leave orthography global. That can be misread as injecting policy bytes into every non-Lemmy call. The plan now explicitly supersedes that sketch: no active Lemmy means no appended policy bytes; “global” applies only within a Lemmy-containing mixed response.
5. **Published evidence must be named — CONFIRMED.** Four existing ledgers were opened and their bank/model/cast/exchange/text evidence verified. They are now named in P1.5.
6. **Existing exchange seam is not an absence lock — CONFIRMED.** `tests/test_exchange_seam_lane2.py:68-80` asserts prefix/grounding behavior, not full-system equality or Cockney absence. The plan now assigns explicit absence proof to new exchange tests.
7. **Live mixed-group reachability — CONFIRMED with refinement.** `exchange_prepass_audit` records accepted beat ids, not group speaker sets. A live qualification must reconstruct groups with the real `group_voiced_beats` rule and ledger order; a new metadata schema is unnecessary.
8. **Deterministic versus listening proof — CONFIRMED.** Captured-prompt tests prove scope; a canonical episode proves production reachability and gives the human listening gate. Lexical examples remain evidence, not a vocabulary blacklist.
9. **Separate repository receipts — CONFIRMED.** The project and Bug Bible are different Git repositories and require separate tests, commits, pushes, and HEAD/origin checks.
10. **Full-cast voice cards as residual diagnostic — CONFIRMED.** Non-Lemmy line prompts still contain correctly labeled Lemmy voice cards. This is not the subjectless-system defect; inspect it only if bleed persists after the root fix.
11. **No live object-info gate for an unchanged contract — CONFIRMED.** Static workflow gates remain mandatory. Live `/object_info` becomes necessary only if a node/workflow contract unexpectedly changes.

## Rejected

1. **Coerce active speakers with `str(...)` at call sites — REJECTED.** Production exchange construction already normalizes beat speakers to strings at `run_exchange_prepass._speaker`; `LineRequest.speaker` is explicitly typed `str`. Coercing an arbitrary object would convert a category defect into a plausible repr and defeat the fail-loud boundary.
2. **Require the isolation sentence to be absent on a Lemmy-only line — REJECTED.** A single canonical block is deterministic and harmless on a one-speaker turn. A second branch adds no acceptance value.
3. **Add an ANNOUNCER-specific prompt test — REJECTED as redundant.** The active-speaker line test exercises the same helper and full-cast mechanism. The plan remains lean.
4. **Add new `scifi_news_pro` tests/live leg unconditionally — REJECTED.** The planned diff does not reach that runner; existing tests plus full suite are the correct unchanged-code control.
5. **Treat port-8000 competition as established for this campaign — REJECTED/MISREAD.** The sanctioned wrapper chooses a free dynamic port when none is supplied. Selective reset remains mandatory, but no unverified competing-process claim enters the plan.

## Verify at build

- Confirm the forced-Lemmy production ledger contains an accepted mixed group after deterministic reconstruction.
- Confirm the resolved live creative model; Gemma 4 E2B is preferred but must not be falsely claimed if unavailable.
- If any non-Lemmy bleed remains after captured system prompts are clean, inspect labeled cast cards and rolling dialogue context before expanding scope.
