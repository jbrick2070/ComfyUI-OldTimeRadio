"""In-decode liveness guard -- detect a decode that is CYCLING, not one that is long.

WHY THIS EXISTS. On 2026-08-13 a live leg spent 13,912 tokens over 22 minutes
rewriting the same ~384-token paragraph about 36 times, then discarded all of it
at the context ceiling. Nothing watched the decode: the structured path called
``model.generate()`` with no stopping criteria at all.

WHY IT MEASURES REPETITION AND NOT LENGTH. A first version of this module
counted how long one JSON string had stayed open, and an r1 panel correctly
called that a per-string LENGTH ceiling wearing a liveness costume: it cannot
tell a legitimately long field from a loop, which is the exact distinction the
operator requires ("we can't just arbitrarily cap, we need to allow for large
episodes"). It was also JSON-specific while being installed on every call, so an
ordinary quotation mark in spoken dialogue would open its state.

So the trigger is now VERBATIM CYCLING: the same run of tokens repeating back to
back. That is what a runaway actually is, and it is:

* **length-independent** -- a 20,000-token field that never repeats is left
  alone, and a loop is caught at the third lap regardless of how long the field
  is allowed to be;
* **format-independent** -- it reads token IDs, so it works identically for
  constrained JSON, raw markup and free prose, which is what the unconstrained
  lanes need;
* **tokenizer-independent** -- nothing is decoded, so isolated-token decoding
  compositionality, byte-fallback pieces and escape handling stop being risks.

WHAT THIS IS NOT:
* NOT a word-count gate. ``target_words`` is never read here and must never be.
  A word target is a REQUEST, never a limit.
* NOT a quality judgement. It cannot see style, only exact repetition.
* NOT terminal by itself -- it raises a REROLLABLE phase. But note that
  ``MAX_CANDIDATE_CYCLES`` is 3, so three FALSE positives could still end an
  episode. That is why the thresholds below are deliberately conservative:
  this must fire on a runaway and essentially never on writing.

DESIGN NOTES that are load-bearing (settled review, 2026-08-13):
* The criterion LATCHES rather than raising. Raising from inside a criterion
  skips the writer's decode/evidence construction and Transformers'
  ``streamer.end()``, and arrives at the wrapper unclassified. So: detect ->
  latch -> return True -> let ``generate()`` return -> classify and raise
  outside.
* Construction failure is the CALLER's problem to surface loudly. A liveness
  guard that silently fails to install is worse than none, because the log then
  claims protection that is not there.

UTF-8, no BOM. ASCII only. No ComfyUI imports -- a leaf module so every local
transport can install it.
"""
from __future__ import annotations

from typing import Any

#: The shortest cycle worth calling degenerate, in tokens.
#:
#: Short repeats are NORMAL and must never trip this: JSON structural keys
#: recur in every array element, dialogue has refrains, and lists repeat their
#: separators. The measured runaway cycled ~384 tokens. 48 sits well above
#: routine structural repetition and well below the real defect.
MIN_CYCLE_TOKENS = 48

#: How many BACK-TO-BACK verbatim repeats must be seen before halting.
#:
#: Two could be deliberate parallel construction -- a writer echoing a phrase
#: for effect, a schema emitting two similar objects. Three consecutive exact
#: repeats of a 48+ token run is not a style choice; the measured runaway did
#: it 36 times. At ~11 tok/s this halts a 384-token cycle near 105 seconds
#: instead of 22 minutes.
REQUIRED_REPEATS = 3

#: Longest cycle we will look for. Bounds the per-check cost.
MAX_CYCLE_TOKENS = 1024

#: Check every N tokens rather than every token, so the scan cost is amortised.
_CHECK_EVERY = 32


def find_repeating_cycle(
    tokens: list,
    *,
    min_cycle: int = MIN_CYCLE_TOKENS,
    max_cycle: int = MAX_CYCLE_TOKENS,
    required_repeats: int = REQUIRED_REPEATS,
) -> int | None:
    """Return the cycle length if the tail is repeating verbatim, else None.

    Looks only at the TAIL, because a runaway is cycling *now*; earlier healthy
    text is irrelevant and scanning it would only add cost and false positives.

    Pure and side-effect free so it can be tested directly.
    """
    total = len(tokens)
    longest = min(max_cycle, total // required_repeats)
    for cycle in range(min_cycle, longest + 1):
        span = cycle * required_repeats
        tail = tokens[total - span:]
        reference = tail[:cycle]
        if all(
            tail[index * cycle:(index + 1) * cycle] == reference
            for index in range(1, required_repeats)
        ):
            return cycle
    return None


def make_degeneracy_criterion(
    prompt_len: int,
    *,
    min_cycle_tokens: int = MIN_CYCLE_TOKENS,
    required_repeats: int = REQUIRED_REPEATS,
):
    """Build the latched StoppingCriteria. Raises if transformers is absent.

    Takes no tokenizer: the detector reads token IDs and never decodes, which
    is what makes it safe across every tokenizer and every output format.
    """
    from transformers import StoppingCriteria  # raises if unavailable

    class _DegeneracyHalt(StoppingCriteria):
        def __init__(self) -> None:
            self.hit = False
            self.reason: str | None = None
            self.cycle_tokens: int | None = None
            self.generated_seen = 0
            self._tail: list[int] = []
            self._since_check = 0

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: D401
            if self.hit:
                return True
            try:
                row = input_ids[0]
                total = int(row.shape[-1])
                for index in range(prompt_len + self.generated_seen, total):
                    self._tail.append(int(row[index]))
                    self.generated_seen += 1
                    self._since_check += 1
                # Keep only what a maximal cycle check can need.
                window = MAX_CYCLE_TOKENS * required_repeats
                if len(self._tail) > window:
                    del self._tail[: len(self._tail) - window]
                if self._since_check < _CHECK_EVERY:
                    return False
                self._since_check = 0
                cycle = find_repeating_cycle(
                    self._tail,
                    min_cycle=min_cycle_tokens,
                    required_repeats=required_repeats,
                )
                if cycle is not None:
                    self.hit = True
                    self.reason = "verbatim_cycle"
                    self.cycle_tokens = cycle
                    return True
            except Exception:  # pragma: no cover - never break a live decode
                # A guard that throws mid-decode would take down the render it
                # exists to protect. Failing OPEN is correct here: the context
                # ceiling is still behind us.
                return False
            return False

        def telemetry(self) -> dict:
            return {
                "halt_reason": self.reason,
                "cycle_tokens": self.cycle_tokens,
                "required_repeats": required_repeats,
                "min_cycle_tokens": min_cycle_tokens,
                "generated_tokens_seen": self.generated_seen,
            }

    return _DegeneracyHalt()
