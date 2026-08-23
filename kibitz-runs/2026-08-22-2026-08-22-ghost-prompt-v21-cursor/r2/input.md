# Driver anchor -- Ghost Prompt v2.1, one scoped round

**Driver:** Claude (Cowork), sole judge. **Round:** ONE, at the operator's
instruction -- this is NOT a four-round arc and must not be reported as one.
**Repo:** `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`,
branch `v2.0-alpha`, HEAD `d1ab7037`.

Everything below is checkable against the real Windows files. Check it. I want
the claims that are wrong named, not agreed with.

## What happened, in order

1. **Ghost Prompt v2 shipped** (`a8fad82c`). The Ghost Signal video lane
   (AnimateDiff SD1.5) had a real text defect: `resolve_action()` copied the
   first six regex words of an unmapped free-text `beat_intent` after the
   literal `moves with`, which put CAST NAMES in the picture and ended
   mid-clause. Proven against a published render and hash-matched 8/8 from the
   pre-change composer.
2. v2 replaced it with "controlled abstraction": a Python-owned recurrence
   motif plus ONE short LLM-authored drawable leaf per beat, three
   representations (`figure`/`object`/`signal`), the model never seeing
   dialogue, title, M4 wall, cast prose or names.
3. **Every gate passed and the pictures got worse.** Suite 12225 green, token
   window measured, control vector exact, three episodes published. The
   operator watched them and said the AnimateDiff output had tanked.
4. **He was right.** Counting recognisable subjects at frame 40 of every beat:
   v1 archival 4/4 sampled; v2 archival with LLM leaves 2/8; v2 deterministic
   0/4; v2 anime 0/8. The two v2 survivors are the only beats whose leaf named
   a real object (a lantern, a radio dial).
5. **v2.1** (`f6075592`, `633e68a7`) put concrete nouns back. Re-rendered with
   real LLM leaves: **6/8 recognisable, 2 human figures**.

## The claim I want attacked

**Legibility tracks concrete nouns, and nothing else.** v1 handed SD1.5
`"a man, a broad steady figure, a charcoal coat, holding a folded chart"`.
v2 handed it `"charcoal ledger emblem"` plus
`"an abstract signal field filling the composition"`. A 512x288 SD1.5 obliges by
painting texture.

v2.1 therefore: mode laws name a physical thing; the motif is the prop itself
(`a charcoal lantern`, not `charcoal lantern emblem`); `figure` says FIGURE and
wears a garment; bookends name real radio hardware; `GHOST_CHARACTER_CYCLE` is
`figure/object/figure/signal` so half the character beats show a person; a leaf
naming a texture (static, waveform, gradient, noise, grain) is rejected.

**Is that diagnosis right, or did something else cause the regression?**
Alternatives I have NOT ruled out and want tested: the prompts also got SHORTER
(208-317 chars down to 164-203); the negative prompt is unchanged but its
relative weight against a shorter positive is not; the mode laws add three
trailing clauses that may dilute the subject; `video_art` and `anime` packs may
simply be abstract by design and unfairly counted.

## Read these

* `nodes/_otr_video_engines/ghost_signal_author.py` -- `GHOST_CHARACTER_CYCLE`,
  `motif_for_character`, `_ABSTRACT_SUBJECT_WORDS`, `_HUMAN_WORDS`,
  `validate_drawable_beat`, `GHOST_FALLBACK_CLAUSES`, `GHOST_BATCH_RULES`.
* `nodes/_otr_video_engines/ghost_signal_prompt.py` -- `GHOST_MODE_LAWS_V2`,
  `compose_ghost_prompt_v2`.
* `docs/2026-08-22-ghost-prompt-v2-publish-receipt.md` section 4A.

## Specific questions

1. **`_HUMAN_WORDS` was cut on live evidence.** It carried
   hand/arm/shoulder and rejected `"the silver ledger sits on a desk as a clock
   hand ticks"` -- a CLOCK hand -- killing all eight beats twice because one bad
   leaf rejects the whole batch. I removed body parts and kept
   person/people/man/woman/boy/girl/child/crowd/figure/silhouette/face/portrait/
   someone. **Is that set now too permissive for `object`/`signal`?** "a
   shoulder passes through the light" now passes in object mode. Is that right?
2. **Whole-batch rejection.** One bad leaf costs all eight beats and two
   generations. It is deliberate (no partial salvage across attempts, so an
   episode's prompts are not a function of how many tries each row took). Given
   it has now twice dropped an episode to fallback over ONE leaf, is the
   trade still correct, or should a single row be re-asked?
3. **`GHOST_CHARACTER_CYCLE` is period 4 with `figure` twice.** Check the
   collision-correction interaction in `schedule_ghost_modes`: bookends
   alternate `object`/`signal` from their own offset and only a BOOKEND is ever
   flipped. Can any seed produce a run of three, or starve `figure` below half?
4. **The abstract-word reject list.** `static`, `waveform`, `gradient`,
   `texture`, `noise`, `grain`, `geometry`, `scanlines`, `interference`. Does
   any of these have a legitimate concrete use a beat would want -- "grain" in
   an archival pack, say -- and would rejecting it cost a good leaf?
5. **Anything in `ghost_signal_author.py` that is simply a bug.** Hash key set,
   parser, tokenizer measurement, fallback collision probing.

Be specific, cite file:line, and prefer "I could not verify X" to a confident
guess. Do not restate the design back to me.
