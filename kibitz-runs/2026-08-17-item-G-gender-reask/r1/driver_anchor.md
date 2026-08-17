# Driver anchor -- item G (Claude, Cowork). Written BEFORE fan-out.

Claims labelled against files read at HEAD `76168ebb`.

## PANEL REALITY FOR THIS ROUND -- both kibitz lanes are down

* **Codex: quota-held** until 2026-08-19 20:31 (carried from the item B window).
* **Antigravity: quota-held.** The r1 fan-out FAILED with hard provider markers
  (`"code": 429`, `RESOURCE_EXHAUSTED`) in `~/.gemini/antigravity-cli/log/`; kibitz
  wrote `antigravity_quota_hold.md` suggesting retry after ~2026-08-17T15:36-07:00.
  **I caused it:** the H-receipt round spent TWO agy calls (the kibitz default
  `Gemini 3.5 Flash (High)`, then `Gemini 3.1 Pro (High)` when the default did not
  honour the operator's request). Two calls where one would have done.
* **A second `claude -p` CLI lane is forbidden** by CLAUDE.md when Claude drives.
* **So this round's panel is Fable + Sonnet as Cowork subagents plus this anchor** --
  the 2026-08-17 D-BIS finding 1 roster minus the dead lane. It is NOT a kibitz
  fan-out and must not be described as one, and it is NOT a four-round arc.

## VERDICT

**Build it, and take FORK 1(a) -- the `post_validator` on the existing ladder --
but only with an explicit degrade path. FORK 2 is the one I do not want to call
alone.**

## MUST-FIX

**MF1 -- `structured_call` already has the hook, so a hand-rolled loop is the wrong
default. CONFIRMED.** `post_validator` is a live parameter in production use
(`nodes/_otr_content_safety.py:262`) with ladder-advance coverage
(`tests/test_a4_capacity_phase_advances_the_ladder.py:253-258`), and
`llm_write_description`'s own docstring records that a voice-pool post_validator
used to be wired here. Bounded retries and the below-base structural-retry
temperature come free; re-implementing them beside the ladder would be a second
mechanism to keep in sync.

**MF2 -- THE EXHAUSTION PATH IS THE REAL PROBLEM AND IT IS NOT COSMETIC.
CONFIRMED.** On exhaustion `structured_call` raises `StructuredCallFailedError`,
which `llm_write_description` converts to `CastingFailedError` (`:856-869`), and the
comment there states `lock_cast`'s `CastValidationLLMError` promotion keys on
`len(attempts) == max_attempts`. The ruling says a render must DEGRADE, not die. So
1(a) is only correct with a way to distinguish "gender never agreed" (keep the last
answer, carry on) from a structural failure (raise, exactly as today). Get that
wrong in one direction and a cosmetic mismatch kills renders; wrong in the other and
real structural failures get swallowed. **This is the single highest-risk line of
the item.**

**MF3 -- do not touch the prompt text.** `character_description` feeds the line
composer -> dialogue -> AUDIO, which is why the hybrid voice-fit was kept as a
separate call rather than folded into this one. A re-ask must be a NEW bounded
attempt, not a reworded base prompt, or dialogue audio re-baselines as collateral.

## SHOULD-FIX

**SF1 -- FORK 2 is where I am weakest, and the repo has a directly relevant
ruling.** The gender ladder's margin (floor 4, ratio 3x) must not be loosened
because DOROTHY of Oz measures 8/3 male when her scene is crowded, and *a confident
WRONG pin must stay impossible*. Prose is a different domain from a scene's name
census, so I do NOT know whether reusing that estimator at that margin is right
here. My instinct is to trigger ONLY on explicit contradiction of the handed value
(an opposite-gender pronoun or honorific referring to the character), never on
ambient gendered vocabulary -- but that is instinct, not evidence.

**SF2 -- "keep the last answer" may be the wrong tie-break and the ruling names it
anyway.** If attempt 2 also mismatches but is worse prose, "last" is strictly worse
than "first". The defensible rule is probably "the first answer that agreed, else
the first answer" -- but the operator's ruling says keep the LAST, so changing it is
his call, not mine. Flagging rather than quietly implementing my preference.

**SF3 -- the acceptance is GPU-gated, and that is fine.** A live leg plus a re-run
of `scripts/audit_voice_gender_consistency.py` against the 34 BEFORE number. The
operator asked to batch fixes and run one declared GPU session, so building this
unproven and queueing its leg is the requested shape -- but the chunk may NOT be
reported as fixed until that leg runs. Green units here prove wiring, not the 34.

## Claims I did NOT verify

* Whether `structured_call`'s `post_validator` failures are distinguishable from
  schema failures in its exhaustion error -- I have not read `structured_call`
  itself, only its call sites and tests. MF2's fix depends on this and it is the
  first thing to check before writing code.
* The current shape of `scripts/audit_voice_gender_consistency.py` and whether 34
  is reproducible from committed ledgers without a fresh render.
