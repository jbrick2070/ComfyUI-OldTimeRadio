# Scope receipt -- item H, EXPLICITLY SCOPED SINGLE-ROUND PANEL

**This is NOT a four-round arc and may never be reported as one.**

| field | value |
|---|---|
| requested range | **r1 only** |
| rounds NOT run | **r2, r3, r4** |
| input doc | `docs/2026-08-17-item-H-lumina-hygiene-receipt-PROBLEM-STATEMENT.md` |
| input SHA-256 | `30078ED82CF30F85683A4C202C15AD01594AFCB9E7FA3B92B606B7BA072BB297` |
| input bytes | 6560 |
| repo HEAD | `9804f7a2` (== origin/v2.0-alpha) |
| driver | **claude** (Cowork UI) -- panelist AND sole judge |
| reviewer lanes | **antigravity only** |
| lanes excluded | **codex** -- quota-held until 2026-08-19 20:31 |
| lanes excluded | **claude** -- the active driver, never launched as its own reviewer |
| profiles | shipped `comfyui` + repo-local `.kibitz/comfyui.local.md` (auto-detected, refreshed 2026-08-16) |
| external calls expected | **2** (see the amendment below) |

## AMENDMENT -- the model auto-selection did not honour the operator's ask

The operator asked for **Antigravity Pro 3.1**. Kibitz's Antigravity default is
`Gemini 3.5 Flash (High)` (COMPAT.md's model policy), and the first run recorded
exactly that in `r1/agy_model_selected.txt` -- so the run in
`kibitz-runs/2026-08-17-item-H-receipt/` is FLASH, not Pro, and must not be
described as the Pro read.

A second run was launched with `KIBITZ_AGY_MODEL="Gemini 3.1 Pro (High)"` into the
separate topic `2026-08-17-item-H-receipt-pro/` so neither artifact clobbers the
other. `agy models` confirms `gemini-3.1-pro-high` is available; agy takes the
display-name slug with spaces and parentheses, not the API-style name.

**So the real provenance is: TWO Antigravity calls (Flash High + Pro 3.1), one
Fable judgment pass on the queue-order question, and the driver anchor. Codex
excluded on quota. Still r1 only; r2/r3/r4 NOT run.** The Fable pass was spawned
separately from kibitz and is not one of the two kibitz calls.

## Why scoped rather than a full arc

Item H is ONE question with two named options already written down in
`GO_FORWARD_PLAN.md`. There is no multi-stage plan to pressure-test across arc /
coding / wiring / convergence, so three of the four rounds would have nothing to
bite on. The precedent is the 2026-08-17 D-BIS finding 1 decision, which was
recorded explicitly as *"a single-round scoped panel on ONE question, not an
r1-r4 arc"* and was reported that way.

The operator asked for Antigravity Pro 3.1 as a second opinion on this specific
call, which is exactly the shape above.

## What the driver owes regardless

* `driver_anchor.md` written BEFORE fan-out (done).
* Every panel claim grounded against the real Windows files before folding.
* `judgment.md` recording accepted / rejected-with-reason / verify-at-build.
* `final.md` as the synthesized answer.
* One external call reported as one, never inflated.
