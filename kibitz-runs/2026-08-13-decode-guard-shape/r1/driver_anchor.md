# Driver anchor -- r1, decode-guard shape (Claude, Cowork)

Written BEFORE the fan-out. The operator's question is "is this best practice,
and will people go for it" -- an ADOPTION question as much as a technical one,
because this pack is about to go open source.

## VERDICT (mine, held loosely pending the panel)

**The SHAPE is stock; the SIGNAL is domain-specific and I do not think that is
avoidable.** I would ship A, and I would take B only if the panel can prove it
safe. But question 4 in the input doc is the one I most want broken -- if A can
lose its lexer, it should.

## CONFIRMED, by reading the real files and the installed library

* **`StoppingCriteria` is the documented transformers extension point for
  exactly this.** Subclassing it is not exotic; it is the mechanism the library
  publishes for "stop generation on a condition you define". The repo ALREADY
  does this: `_get_substring_stop_class()` in `OTR_LedgerScriptWriter.py` is a
  hand-written `StoppingCriteria` subclass that has shipped for months. So the
  guard adds no new KIND of thing to this codebase.
* **No new dependency.** `_otr_decode_guard.py` imports `StoppingCriteria` from
  transformers (already a hard requirement) and nothing else. No ComfyUI import,
  so it unit-tests without a runtime. Nothing installs, nothing runs at boot,
  nothing is user-visible.
* **The stock alternatives each fail for a stated reason, not a stylistic one:**
  `MaxTimeCriteria` and a smaller `max_new_tokens` are LENGTH/TIME caps and hit
  the operator's large-episode constraint head on; `stop_strings` cannot fire on
  a loop that never emits the stop; `repetition_penalty` is measured inert
  (HF's penalty is not frequency-aware, and even its 1.2 maximum leaves escape
  probability at exactly 0).
* **B (`no_repeat_ngram_size`) is one kwarg and would delete ~200 lines**, which
  is why it deserves the hardest look. The blocker is real but unproven: the
  installed `NoRepeatNGramLogitsProcessor` includes prompt history on
  decoder-only models, and JSON keys repeat by design, so the ban can intersect
  lmfe's mask. An empty allowed set is a HARDER failure than the slow decode it
  replaces.

## What I am NOT confident about, and want the panel on

1. **Whether B's empty-intersection risk is real in practice.** If a reviewer
   can show it cannot happen against `RadioScoreDraftV4`, ship B and delete A.
   I have not proved it either way and I will not assert it.
2. **Whether the lexer is the smallest sufficient signal.** Alternatives I can
   see: count consecutive tokens since the last structural JSON character; ask
   lmfe's parser for its own state (rejected in the settled design as coupling
   to third-party internals, but worth re-testing); or watch the streamer's
   decoded tail rather than tracking state.
3. **Tokenizer edge cases.** The lexer decodes token-by-token with
   `clean_up_tokenization_spaces=False` and caches, because isolated-token
   decoding is not formally compositional for every tokenizer. I have tests for
   escaped quotes, even/odd backslash runs and quotes inside multi-character
   tokens -- but a tokenizer that emits a quote as part of a byte-fallback piece
   could still miscount, and I have not proved it cannot.

## The adoption answer I would give, and want checked

A user installing this pack does exactly what they did before. The guard is one
more `_otr_*.py` among ~30, invisible unless a decode runs away, and its only
user-visible behaviour is a clearer log line and a reroll instead of a 22-minute
stall. **The thing that would genuinely be "not best practice" is the opposite
choice** -- shipping a writer that can silently burn a user's GPU for 22 minutes
and discard the result, on a pack whose whole promise is local, offline, on
consumer hardware.

## What the panel should NOT spend the round on

Re-deriving the defect or the measurements. They are settled and cited. Spend it
on question 1 (is B safe), question 4 (can A be smaller), and a concrete answer
on adoption cost.
