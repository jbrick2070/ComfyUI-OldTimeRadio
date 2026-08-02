# Episode rating

**Operator, 2026-08-02.** Backlog -- behind local 6/6, cloud grounding, the
all-cloud campaign.

## The rule

Ask four yes/no questions about the episode:

    violence?  smoking?  profanity?  sexual content?

Any yes -> the credits say **FOR MATURE AUDIENCES**. None -> the default label.

That is the whole methodology.

## Why yes/no and not a 0-3 scale

A local 12B model judges this. Scoring intensity means calibrating -- what
separates peril 2 from peril 3? -- and nothing in a rubric makes that stable.
This repo has watched this exact model drift under load: it restated speakers
with their cast role and broke the markup parser for a whole leg, and it wrote
five lines of "(static crackles)" where dialogue was asked for.

Presence is an observation, not a judgement, and observation is what a small
model is reliable at. It also makes the formula trivial and auditable, which is
the point of an open methodology: any(four) -> mature. Nothing to weight,
nothing to defend.

(Tobacco is a real descriptor -- BBFC and ESRB both surface it -- not an
idiosyncratic addition.)

## Not copied

The letter marks (G/PG/R) are administered trademarks; stamping them implies a
certification nobody granted. The QUESTIONNAIRE SHAPE is the reusable part, and
IARC and Kijkwijzer are the precedent for it. Our vocabulary, their structure.

## Wiring, when it is built

* The ledger owns the field, one writer, stamped before the opening credits and
  frozen -- both credit rolls must read the same value.
* Stamp WHICH descriptor fired, even though the label is binary. The model
  answered all four anyway, and it turns "the rating is wrong" into "which
  question did it get wrong".
* It is a LABEL, never a publish gate. If that ever changes it is an explicit
  decision -- the "advisory that quietly became fatal" shape that bit the codex
  lane's cast-coverage field.
* Same frozen ledger must rate identically on a re-run.

## Open question for the operator

Do the CREDITS name the descriptor ("violence") or only the label? The ledger
should carry both either way.
