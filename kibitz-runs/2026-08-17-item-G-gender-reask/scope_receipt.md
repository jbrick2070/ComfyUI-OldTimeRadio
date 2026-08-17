# Scope receipt -- item G, EXPLICITLY SCOPED SINGLE ROUND, AND NOT A KIBITZ FAN-OUT

**This is NOT a four-round arc, and it is NOT a kibitz panel.** Both external
kibitz lanes were unavailable, so the round ran on Cowork subagents instead. Do not
describe it as either.

| field | value |
|---|---|
| requested range | **r1 only** |
| rounds NOT run | **r2, r3, r4** |
| input doc | `docs/2026-08-17-item-G-gender-reask-PROBLEM-STATEMENT.md` |
| repo HEAD | `76168ebb` (== origin/v2.0-alpha) |
| driver | **claude** (Cowork) -- panelist AND sole judge |
| kibitz external calls | **1 ATTEMPTED, 0 SUCCEEDED** |
| actual panel | **Fable** (fork 2, the detection heuristic) + **Sonnet** (fork 1, the mechanism) as Cowork subagents, plus `r1/driver_anchor.md` |

## BOTH KIBITZ LANES ARE QUOTA-HELD

* **Codex** -- held until **2026-08-19 20:31**, carried from the item B window.
* **Antigravity** -- the r1 fan-out FAILED (exit 1) on **confirmed provider markers**,
  not a timeout: `"code": 429`, `"status": "RESOURCE_EXHAUSTED"` in
  `~/.gemini/antigravity-cli/log/cli-20260817_141301.log`. Kibitz wrote
  `r1/antigravity_quota_hold.md` and suggested retry after **~2026-08-17T15:36-07:00**
  (1h). Override the window with `KIBITZ_QUOTA_RETRY_AFTER` if needed.

**THE DRIVER CAUSED THE AGY EXHAUSTION AND SHOULD OWN IT.** The H-receipt round
spent TWO agy calls on one question: the kibitz default `Gemini 3.5 Flash (High)`,
then `Gemini 3.1 Pro (High)` after noticing the default had not honoured the
operator's explicit request for Pro 3.1. **Set `KIBITZ_AGY_MODEL` BEFORE the first
call** -- the default is Flash and COMPAT.md says so; reading it first would have
cost one call instead of two.

**A second `claude -p` CLI lane was NOT launched.** CLAUDE.md forbids it when Claude
drives from Cowork ("you do NOT launch a second `claude -p` lane against your own
family"), and the kibitz driver-exclusion rule agrees.

## Why subagents rather than waiting the hour

The 2026-08-17 D-BIS finding 1 decision set the precedent: a scoped single-question
panel of **Fable + Sonnet + Antigravity + the driver anchor**, with Codex excluded
and recorded. This is that roster minus the second dead lane. The operator asked for
work to continue while he was away, and item G's design fork is the gate before
code, so the fork gets broken now and the record says exactly who broke it.

## What the driver still owes

* `judgment.md` -- accepted / rejected-with-reason / verify-at-build.
* `final.md` -- the synthesized answer.
* Every claim grounded against the real Windows files before folding.
* **G may not be reported as FIXED on green units.** Its acceptance is a live leg
  plus a re-run of `scripts/audit_voice_gender_consistency.py` against the 34 BEFORE
  number, and that rides the operator's declared GPU batch.
