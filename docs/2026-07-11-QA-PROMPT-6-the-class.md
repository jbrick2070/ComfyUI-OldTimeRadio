# OTR QA #6 -- THE CLASS. Stop fixing instances. (paste into agy AND into codex)

REVIEWER ONLY. Do NOT edit source, do NOT git add/commit/push. Write to
`qa6_<yourname>.md`. Pull first. CONFIRMED or [ASSUMPTION] on every claim.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha

## The count is now SEVEN, and I am done playing whack-a-mole

Every single kill on this build, once diagnosed, has been the same defect wearing a
different coat: **a gate that blocks production on something the thing being judged cannot
satisfy, or that is not a defect at all.**

1. A word-count quota the writer could never hit (5 words a beat, exact equality demanded).
2. An episode-level fact rule enforced per scene -- scene 2 failed for not containing a
   fact that belongs to scene 1.
3. A seam demanding a field its own strict schema forbids (`fact_ids` vs `fact_uses`) --
   so the model obeyed the seam and the repair then DELETED its fact attribution to make
   it validate.
4. A rewrite told to "retain the outline" and therefore preserving the exact wording it
   was summoned to replace.
5. An auditor failing an episode because a line "adds no new information" -- in a 30-word
   script with a two-fact dossier, where saying something new is not possible.
6. `cites` with `min_length=1`, so a ceremonial line that states NO fact could not
   validate -- and the lane satisfied the schema by citing a sentinel `fact_0` that the
   one-based dossier can never produce. **The schema forced the lane to lie.**
7. (today, live) The Sonnet auditor failed the episode with *"line 3 contradicts line 1's
   claim about laboratory navigation"*. The accused line, printed in full:

       [3] THESSALY: 'If these tests prove successful, could this technology reshape
                      global communication?'   cites=['fact_2']   invented_facts=[]

   THESSALY is the SPECULATOR. Her entire contract is "dramatic license on what this might
   mean or lead to." **The auditor failed her for doing her job.** A conditional question
   asserts nothing and can contradict nothing.

The law, stated once more, and it is the only thing I want you looking for now:

> **A gate that blocks production may only block on something that is (a) objectively
> checkable, (b) actually fixable by the party being asked to fix it, and (c) genuinely a
> defect rather than a note, a preference, or the thing's own contract being honoured.**

## JOB 1 -- find every remaining instance of the class. This is the whole job.

Sweep ALL FOUR lanes (codex, gemini, sonnet, fable2), the shared writer tail, the freeze
cascade, cast lock, shot lock, the render/media path, captions, credits. For EVERY place
that BLOCKS -- raises, fails a run, exhausts a bounded loop, returns a fatal verdict:

| where (file:line) | blocks on | objectively checkable? | fixable by whom? | is it actually a defect? | verdict |

Verdict is one of: **KEEP** (a real, checkable, fixable defect), **DOWNGRADE** (it is a
note -- record it, do not block), **REPAIR** (it is mechanical -- derive it, do not block),
**IMPOSSIBLE** (the party being judged cannot satisfy it -- the contract itself is broken).

Confirmed starting points you have both already found:
- `_SonnetTailFinalizer` / `_GeminiTailFinalizer` / `_CodexTailFinalizer` treat any ledger
  **warning** as fatal, and demand `freeze_verdict == "frozen_clean"`. agy enumerated ten
  warnings from `_otr_ledger_freeze.py` -- classify EACH one: real defect, or note?
- `_otr_scifi_fable2.py` blocks on a word-count variance of +/-20% -- that is the quota
  disease again, in the fourth lane.
- Any gate whose failure message is a bare string that names neither the offending item nor
  the reason (every one of those cost us a roll of pure guessing).

## JOB 2 -- the roles question (Sonnet), and its generalisation

Sonnet's audit judged ORUM (Literalist -- may state only what the wording supports) and
THESSALY (Speculator -- licensed to extrapolate) by the SAME rule. That is why it failed a
question for contradicting a fact.

- Is there anywhere else in this system where a validator applies ONE standard to parties
  who have DIFFERENT contracts? (Announcer vs character. Literalist vs speculator.
  Ceremonial vs cited. Fact-bearing lines vs non-fact lines.)
- Sonnet's `_spoken_error` hygiene runs identically over ceremonial and dialogue lines --
  is that right?
- Generalise: list every validator that should be role-aware and is not.

## JOB 3 -- would a fresh lane inherit the lessons?

The operator's question, and the honest answer so far is NO: three lanes were written
independently and each re-made the same mistakes. Two cross-lane guards now exist:
- an AST guard rejecting a literal assigned to a spoken field (`text=`, `premise=`, ...)
- a guard that every seam's worked example must VALIDATE against the schema it feeds

Design the rest of the guard suite. What executable, cross-lane test would have caught
each of the seven kills above at commit time rather than 15 minutes into a render? Be
concrete: the test, what it walks, what it asserts. I will write the ones that are worth
writing. Rank them by kills-prevented.

## Output (`qa6_<yourname>.md`)

JOB 1 THE BLOCKING TABLE (every gate, with a verdict) -- this is the deliverable
JOB 2 ROLE-BLIND VALIDATORS
JOB 3 THE GUARD SUITE THAT WOULD HAVE CAUGHT ALL SEVEN
