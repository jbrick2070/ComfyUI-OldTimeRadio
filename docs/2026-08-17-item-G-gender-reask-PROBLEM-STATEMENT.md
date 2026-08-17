# Item G / PBUG-20260815-11 -- the gender re-ask, and its one real fork

**One question, scoped.** HEAD `76168ebb`, branch `v2.0-alpha`. Every code claim
below was read at the real files on 2026-08-17.

## The defect and the operator's ruling

34 characters sound one gender and look the other, measured over 1,686 real
ledgers. The ruling (2026-08-15) is NARROWER than the option put to the operator,
and it stands as written:

* Nothing downstream of casting may reject, rewrite or block a prompt.
  **Node 89 gets NO classifier.** Do not edit the ruling comment away.
* What IS permitted is upstream: in `_otr_casting.py`, where the comment at
  `:365-369` states that gender is a Python-decided fact the LLM writes into,
  **the description producer may check its own returned prose against the gender it
  was handed and RE-ASK.**
* **Bounded retries, then keep the last answer. A render must not die: it degrades
  rather than raising.**
* No new field (nothing for `Ledger.set_cast` to drop).
* **Acceptance:** a live leg plus a re-run of
  `scripts/audit_voice_gender_consistency.py`. **34 portrait conflicts is the
  BEFORE number.**

## The surface, read

`llm_write_description` (`nodes/_otr_casting.py:792-878`) is the lone LLM call in
casting. It builds the prompt via `_build_user_prompt` (which emits the
Python-decided `Gender:` line at `:368-369`), then calls `structured_call` with the
existing retry ladder -- base -> structural retry at half temperature -> typed
repair -- `max_attempts=3` by default, schema `DescriptionResponse` (one field).

**`structured_call` ALREADY TAKES A `post_validator`.** This is the pivot. Its
docstring here even notes the old "voice-pool post_validator is gone", and the hook
is alive and in production use elsewhere (`nodes/_otr_content_safety.py:262`, with
ladder-advance coverage in
`tests/test_a4_capacity_phase_advances_the_ladder.py:253-258`). So a gender check
can ride the EXISTING bounded ladder instead of a hand-rolled loop.

**But the ladder RAISES on exhaustion, and the ruling forbids raising.** When the
ladder is exhausted `structured_call` raises `StructuredCallFailedError`, and
`llm_write_description`'s existing handler converts that to `CastingFailedError`
(`:856-869`) -- which propagates and, per the comment there, is what
`lock_cast`'s `CastValidationLLMError` promotion keys on via
`len(attempts) == max_attempts`. A gender mismatch that survives the ladder must
NOT take that path, or a cosmetic mismatch kills the render.

## FORK 1 -- where the re-ask lives

* **(a) `post_validator` on the existing ladder.** Bounded retries for free, one
  code path, consistent with how every other pass in this repo validates. Cost: the
  exhaustion path currently raises, so it needs a way to distinguish "gender never
  agreed" (degrade, keep last answer) from a genuine structural failure (raise, as
  today). Getting that distinction wrong either kills renders or silently swallows
  real failures.
* **(b) A separate bounded re-ask AFTER `structured_call` returns.** Full control
  over degrade-not-raise, and it cannot disturb the existing exhaustion semantics
  or the `len(attempts)` promotion contract. Cost: a second retry mechanism beside
  the ladder, and the temperature discipline (structural retry strictly below base)
  would have to be re-implemented or deliberately skipped.

## FORK 2 -- how a mismatch is DETECTED in prose, and this is the sharp one

The prose is audience-facing character description. Detecting "this reads male when
Python said female" is a heuristic, and this repo has already ruled on the
neighbouring heuristic: **the gender ladder's decision margin must not be
loosened**, because DOROTHY of Oz measures 8/3 male under a looser estimator when
her scene is crowded, and *a confident WRONG pin must stay impossible;
decline-and-roll is the accepted behaviour*.

**CORRECTED 2026-08-17: the shipped floor is `SCORE_FLOOR = 8`, not 4.** An earlier
draft of this section said "floor 4, ratio 3x", copied from `GO_FORWARD_PLAN.md`,
which had it wrong for as long as the bullet existed. `_otr_gender_pronoun_scan.py`
states it plainly: *"THE FLOOR WAS 4 AND THAT WAS TOO LOW -- it shipped a
confidently WRONG pin."* `DOMINANCE_RATIO = 3.0` is right. **This decides fork 2:**
at floor 8 the census can never fire on a 40-word description, so "reuse the census
at its own margin" is not a live option, and lowering the floor to make it fire is
the forbidden loosening.

So the question is not "can we detect gender" but **what evidence is strong enough
to spend a re-ask on, and what must never trigger one.** Candidates, none chosen:
pronoun counting; gendered nouns (his/her, man/woman, sir/madam, actress);
honorifics; the existing estimator reused at the ladder's own margin; or only
explicit contradictions of the handed value. A false positive costs an LLM call and
risks talking a correct description into a worse one; a false negative leaves the
PBUG partly unfixed.

## What the panel is asked to break

1. Pick FORK 1 and say why. If (a), specify exactly how "gender never agreed" is
   distinguished from a structural failure without weakening the
   `len(attempts) == max_attempts` promotion contract that `lock_cast` depends on.
2. Pick FORK 2's evidence bar. Is reusing the existing estimator at its existing
   margin correct here, or is prose a different enough domain to need its own rule?
   Name the specific words or patterns you would and would NOT trigger on.
3. What is the FAILURE MODE of re-asking? If the second answer is also mismatched
   but WORSE prose, we kept the last answer -- is "last" right, or should it be
   "first", or "the one that agreed if any did"? The ruling says keep the last
   answer; say if that is wrong and why.
4. Is there any way this reaches back from node 89, adds a ledger field, or edits
   the ruling comment -- the three things explicitly forbidden?
5. The cheapest LIVE observation that would move the 34 BEFORE number, given
   acceptance needs a real leg and the operator batches GPU work.

## Invariants a fix may not break

* No classifier at node 89; nothing downstream of casting rejects or rewrites.
* A render must not die -- degrade, never raise, on a cosmetic mismatch.
* No new ledger field.
* Do not loosen the gender ladder's decision margin (`SCORE_FLOOR = 8`,
  `DOMINANCE_RATIO = 3.0` -- read the constants, not any doc that names numbers).
* `character_description` feeds the line composer -> dialogue -> AUDIO, so
  perturbing this prompt re-baselines dialogue audio as collateral. That is why the
  hybrid voice-fit was kept as a SEPARATE call rather than folded in here.
* Story quality is DONE and is not to be chased. This is a CORRECTNESS fix (a
  character's gender contradicting the source), which the directive explicitly
  leaves open -- not a prose-improvement pass.
