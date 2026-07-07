# Pass 04 Codex Anchor - Convergence

## Proposed Final Decision

Ship a Seedance-only prompt conditioner plus tests. Do not edit the workflow JSON
or style JSON in this first pass.

## Why No Workflow JSON

The change is internal adapter behavior. It uses the existing `text_prompt`
input and keeps the existing Partner Node request shape. No node/widget/link
surface changes.

The operator is fine with JSON edits if they improve output. That means we
should not avoid JSON on principle; it does not mean we should add unwired or
unneeded workflow churn.

## Why No Style JSON Yet

`sci_fi_radio.json` does contain aggressive motion verbs, but those verbs are
part of a deliberately restored LTX opener register. A global pack edit changes
non-Seedance motion policy too. Seedance-specific stabilization is narrower and
better grounded.

## Must-Have Details

- Detect the smooth marker before applying softeners, or idempotence breaks.
- Use regex replacements in specific-first order.
- Keep `_text_prompt_input()` as the loud missing-prompt gate.
- Do not add unsupported Seedance fields.
- Keep under-minimum duration clamp and trim policy.
- Add tests proving request shape is unchanged.

## Remaining Question

Any final blocker before implementation?
