# Is a bespoke in-decode guard the right shape? -- r1 question

**Decide the SHAPE, not the code.** An implementation already exists and passes
its tests; the question is whether it is the right thing to ship in a project
that is about to go open source. If a stock mechanism does this job, the
bespoke one should go.

## The defect, stated once

A local schema-constrained decode (Mistral-Nemo, lm-format-enforcer,
transformers) fell into a verbatim repetition loop inside ONE JSON string and
generated **13,912 tokens over 22 minutes**, then discarded all of it at the
context ceiling. Measured, twice, on live legs. lmfe masks EOS until the JSON
document is complete, so the only exits from a string are the closing quote and
the ceiling; once the loop locks, `top_p=0.95` + `min_p=0.05` prune the quote to
probability exactly zero, and `repetition_penalty` cannot help because HF's
penalty is not frequency-aware.

## The operator's constraint, which rules out the obvious answers

**A cap on length is not available as the primary fix.** Episodes must be
allowed to get large; a legitimately long field and a looping field are
indistinguishable by total length. THE LAW of this project: an audit may improve
a story, never fail one for length, language, style or quality. `target_words`
is a REQUEST, never a limit.

**And the adoption constraint (operator, 2026-08-13):** this ships as an open
source ComfyUI custom-node pack. Anything that looks like an extra install
burden, a new dependency, or a bespoke subsystem a user must understand is a
cost. Prefer a stock mechanism if one genuinely does the job.

## What is already shipped and is NOT in question

Structural `max_length` ceilings on authored strings, enforced by lmfe DURING
decoding, with a reroll when a string reaches the threshold. That is landed and
live-proven (it fired on `premise` on 2026-08-13). It bounds ONE string. It does
NOT bound aggregate cost across ~51 authored strings, which is the remaining
gap.

## The candidate shapes -- rank them, and say which you would ship

**A. The bespoke latched StoppingCriteria (BUILT).** One file in `nodes/`, no
new dependency, ~200 lines. Tracks whether the decode is inside a JSON string
and how many tokens it has been open; latches at 2,048 open-string tokens;
`generate()` returns normally; the transport classifies and raises a rerollable
`decode_degeneracy` phase. Repetition is telemetry only.
*Cost:* a hand-rolled lexer to maintain, and a reviewer has to trust it.

**B. `no_repeat_ngram_size` (stock, one kwarg).** A CPU probe shows it restores
escape probability from 0 to 1 in a locked loop. **Known risk:** installed
transformers includes the PROMPT in decoder-only n-gram history, and JSON keys
repeat by design, so the ban could intersect with lmfe's mask to an EMPTY
allowed set -- which fails the decode outright rather than slowly. Unproven
against the real grammar.

**C. `transformers` `MaxTimeCriteria` or a token ceiling per call (stock).**
Trivial, no new code. But it is a LENGTH/TIME cap by another name, and hits the
operator constraint above: it cannot distinguish a long episode from a loop.

**D. `stop_strings` / the existing substring stop (stock).** The prompt already
asks the model to end with `END.`. A locked loop never emits it, so this cannot
fire on the failure mode. Include only to be dismissed with a reason.

**E. Post-hoc detection only** -- let the decode run, detect degeneracy after
`generate()` returns, reroll. No in-decode machinery at all.
*Cost:* still pays the full 22 minutes. Cheapest to maintain, most expensive to
run.

**F. Something none of the above.** Sampler-level (a custom LogitsProcessor that
un-prunes the exit token), a grammar-level fix in lmfe, a smaller per-field
`max_new_tokens` derived from structure rather than word count, or another
mechanism. If a better shape exists, name it.

**G. PROMPT THE MODEL SO IT DOES NOT TIP INTO THE LOOP (operator, 2026-08-13:
"the key thing is we should prompt Mistral and the other LLMs so they don't go
crazy").** No code, no maintenance, and -- uniquely among these candidates --
it reaches EVERY lane and EVERY provider, including the fable2 markup lane and
cloud models no local StoppingCriteria can touch. Evaluate it as PREVENTION
rather than as a guarantee, and rank it against the others on that basis.

Two facts the panel must weigh rather than assume:

* **It has already been tried once, this same day, and did not hold.** Commit
  `6ca20028` (03:07) replaced "use the available provider capacity" with
  "capacity is a ceiling, never a target", explicitly to remove runaway fuel.
  That wording was LIVE in the server that ran away at 10:44.
* **A locked loop cannot read.** Once the repetition attractor forms, the
  closing quote sits at probability exactly zero after top_p/min_p; an
  instruction cannot be heard by a sampler with no path to the exit. This is
  the same reason the prompt's literal `END.` never fires.

So the real question for G is not "does it work" but **"how much does it reduce
the RATE, and what wording actually does that"**, given the specimen: the
runaway text was enumerative boilerplate ("the lab's visual prompts include: a
large screen ..., a row of servers ..., and a whiteboard ...") -- a list with no
natural end -- and it was SHOT-level visual-prompt content emitted into a
SCENE-level prose field. The current surface instruction says "There is ample
room, so write each field at its natural length."

Judge whether a field-shape instruction (what the field is FOR, an explicit
shape such as one or two sentences, and a ban on open-ended enumeration in
description fields) is the higher-value change than any code guard -- and say
whether it REPLACES a guard or PAIRS with one. Note that field-shape guidance is
legal here: THE LAW forbids an AUDIT rejecting a story for length; prompt
guidance that nothing enforces is craft direction, the same category as
`target_words` being a request.

## Questions I want answered, not opinions

1. **Is B safe?** Can `no_repeat_ngram_size` + an lmfe grammar ever produce an
   empty allowed set on a real schema? If it cannot, B is one kwarg and A should
   be deleted. This is the highest-value question here.
2. **Does anything stock detect NON-TERMINATION rather than LENGTH?** That is
   the actual requirement. If transformers or lmfe already offers it, use it.
3. **Is a ~200-line lexer a real adoption cost** for a pack that already ships
   ~30 `_otr_*` modules and depends on transformers, torch and lmfe? Be concrete
   rather than sympathetic -- what would a user actually have to do?
4. **If A survives, what is the smallest version of it?** Could it drop the
   lexer and count consecutive tokens since the last structural character, or
   watch the lmfe parser's own state instead?
5. **What breaks A in production?** Name the tokenizer or schema shape where the
   open-string lexer miscounts.

Answer with a ranked recommendation and the reasoning, citing real files where
it matters. The measured facts above are settled -- do not re-derive them.
