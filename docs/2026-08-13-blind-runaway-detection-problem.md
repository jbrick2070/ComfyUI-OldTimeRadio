# A local LLM writes forever in three different ways. What detects all three?

**This is a blind problem statement.** It deliberately contains no proposed
solution, no mechanism, and no description of anything already built, because
the driver has twice been wrong about the SHAPE of the answer and wants
independent reasoning rather than validation. Please do not ask what has been
tried architecturally; everything ruled out by MEASUREMENT is listed below.

## The system

A fully local, offline pipeline writes old-time-radio drama scripts on one
16 GB consumer GPU, then renders them to audio and video. No cloud inference,
no API keys.

* Writers: `mistralai/Mistral-Nemo-Instruct-2407` (12B, context cap 16,384) and
  `google/gemma-4-12b-it` (cap 8,192), via HuggingFace transformers.
* Sampling: `temperature=0.72, top_p=0.95, min_p=0.05, repetition_penalty=1.03`.
* Two output paths: **schema-constrained JSON** through lm-format-enforcer
  (`prefix_allowed_tokens_fn`), and **free-text markup** parsed after the fact.
* The writer runs several passes per episode (evidence extraction, dramatic
  question, cast, score draft, spoken script). Each pass is one `generate()`
  call inside one node execution -- nothing downstream can interrupt it.
* A pass that fails is retried by a ladder: base call, then a lower-temperature
  structural retry, then a typed repair. Three candidate cycles maximum, then
  the episode dies.

## The failure, in three captured specimens

All three are real production runs, captured with full text. All three consumed
essentially the entire remaining context window inside a SINGLE pass and were
then discarded, costing 20+ minutes of GPU each.

**Specimen A -- anaphoric peroration loop. 13,828 tokens.**
The model locked into a rhetorical refrain and cycled it: "Let the echoes
inspire us / echo / resound", over and over, verbatim, until the budget ran out.

**Specimen B -- escalating repetition. 14,521 tokens.**
Same shape, different flavour: "Before it's too late..." repeated with small
escalations, again until the ceiling.

**Specimen C -- a pure ELABORATION SPIRAL. 15,355 tokens, 83,000 characters,
and NO REPETITION ANYWHERE.**
This is the hard one. The model never repeated itself. It simply kept
elaborating -- each sentence new, each sentence plausible, none of it ever
arriving at an end. It was writing meta-commentary about its own output
("the draft is designed to be compatible with the provided contract, ensuring
that it meets all required criteria...") and kept finding more to say.

A fourth, related observation: under grammar-constrained JSON, the enforcer
masks the end-of-sequence token until the JSON document is complete, so the
only exit from a string field is its closing quote. Once a decode has locked,
`top_p`/`min_p` prune that quote to effectively zero probability.

## The hard constraint that makes this difficult

**We must not cap length, and we cannot use length as the signal.**

The project's governing rule is that an audit may improve a story but may NEVER
fail one for length, language, style or quality. Episodes must be allowed to get
much longer than they are today -- a 12-minute episode is a near-term target and
needs roughly three times the current output. A legitimately long field and a
runaway field are indistinguishable by total length.

Requested word counts are also unusable as a control: this exact Mistral
checkpoint complies with a word target only 14.5-34% of the time (measured,
published), and asking a model to count its own output is not reliable.

## Ruled out BY MEASUREMENT -- do not re-propose without new evidence

* **`repetition_penalty`** -- inert here, up to and including its maximum. HF's
  penalty is not frequency-aware, so a token emitted 60 times is penalised
  exactly as much as one emitted once.
* **`no_repeat_ngram_size`** -- unusable with the grammar. The n-gram ban and
  the grammar mask both write `-inf`, prefix validation only checks that the
  grammar returned a non-empty list rather than that a token survived the ban,
  so the intersection can be EMPTY and sampling crashes.
* **Lower temperature** -- helped one run, failed another. Not reliable.
* **A stop string / "end with END."** -- a locked decode never emits it.
* **Prompt wording alone** -- "capacity is a ceiling, never a target" was
  shipped and a runaway recurred hours later with it live.
* **A hard token ceiling** -- forbidden by the constraint above.

## The question

**What mechanism detects all three specimens, and only degenerate output?**

Specifically:

1. Is there a single signal that catches A, B and C, or does this genuinely
   require more than one? If more than one, what is the minimum set, and what
   does each one see that the others cannot?
2. Specimen C is the crux. It never repeats, it is never off-topic, and every
   sentence is individually well-formed. What is OBSERVABLE about it, early,
   that distinguishes it from a model legitimately writing at length?
3. Whatever you propose: what would make it fire on a healthy long episode --
   a refrain, a callback, a repeated station ident, parallel rhetoric, a list of
   similar dialogue turns? A guard that damages real writing is worse than the
   bug.
4. When it fires, what should happen? The pass can be retried, but retries are
   finite and each one costs a candidate cycle.
5. What could be measured cheaply, offline, from 1,600 already-shipped episode
   transcripts, to calibrate any threshold you propose rather than guessing it?

Ground your reasoning in what is actually observable during token-by-token
generation on a local transformers model. Say plainly where you are reasoning
from evidence and where you are speculating.
