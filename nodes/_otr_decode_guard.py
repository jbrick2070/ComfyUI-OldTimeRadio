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

WHAT THIS IS, EXACTLY -- framing corrected by a blind frontier panel 2026-08-13
(`docs/2026-08-13-blind-runaway-detection/`), which was given the three
specimens and NO description of this module, and independently landed on a
StoppingCriteria with a ~2,000-token per-string bound. Two of its corrections
change what this file may claim about itself:

* **This is a COST-BOUNDING mechanism, not a correctness oracle.** You cannot
  prove non-termination from a prefix: any finite prefix of a spiral is also a
  possible prefix of a legitimate output that closes later. So the honest
  targets are a low false-abort rate, short detection latency and GPU minutes
  saved -- never "catches every runaway and nothing else". Do not write that
  claim back into this file.
* **LOCAL length is the bound, GLOBAL length is not.** A 15,000-token EPISODE
  is valid; a 15,000-token single field is a runaway. Every bound here is
  scoped to ONE structural unit for that reason, which is what lets episodes
  grow without weakening the guard.
* The elaboration spiral is properly a per-pass DISCOURSE-CONTRACT failure --
  the model writing commentary about its own output instead of the story. This
  module bounds its COST; the cure is the pass's own prompt contract, and the
  two should not be confused for each other.

WHAT THIS IS NOT:
* NOT a word-count gate. ``target_words`` is never read here and must never be.
  A word target is a REQUEST, never a limit.
* NOT a quality judgement. It cannot see style, only exact repetition.
* NOT terminal by itself -- it raises a REROLLABLE phase, so a false positive
  costs a retry rather than the episode. The thresholds below stay deliberately
  conservative anyway: repeated false fires would burn a run's retry budget and
  then end it. (This paragraph used to justify itself with
  ``MAX_CANDIDATE_CYCLES`` from ``_otr_scifi_codex.py`` -- a retry budget in a
  module that no longer exists. Tuning against a deleted mechanism is how a
  threshold ends up defended by nothing.)

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


#: Word-level floor for the CLOUD guard. The local floor is 48 TOKENS; at the
#: ~1.5 tokens/word this project's prose measures, 32 words is the same size of
#: paragraph fragment expressed in the only unit a remote reply gives us.
MIN_CYCLE_WORDS = 32

#: Tokens ONE JSON string may stay open before the decode is called degenerate.
#:
#: THE SECOND SIGNAL, AND IT IS NOT REDUNDANT WITH THE FIRST. Three runaway
#: specimens were captured on 2026-08-13 (docs/HANDOFF_LOG.md):
#:   P3 -- an anaphoric peroration loop        13,828 tok  (repeats)
#:   P5 -- escalating repetition               14,521 tok  (repeats)
#:   P2 -- a pure ELABORATION SPIRAL           15,355 tok  (NEVER repeats)
#: The cycle detector below catches the first two and is structurally blind to
#: the third: a spiral has no cycle to find. Its only observable property is
#: that ONE string has stayed open absurdly long, so an open-string counter is
#: the only thing that sees it. The handoff log says exactly this -- "only an
#: unclosed-string token counter catches all three" -- and the settled design
#: (GO_FORWARD item 2) names it the PRIMARY signal for that reason.
#:
#: WHY IT IS NOT A LENGTH GATE, which is the objection that got the first
#: version deleted: it does not measure the OUTPUT, it measures how long one
#: string has gone without closing. A 20,000-token draft of many complete
#: fields never approaches it; a single field that has run 2,048 tokens without
#: a closing quote is not long writing, it is a decode that has lost its exit.
#:
#: 2,048 tokens is ~8,200 chars -- about 1.8x the longest authored string this
#: project has ever shipped (4,549 chars) and well under the 12,000-char schema
#: ceiling, so the ceiling stays the structural backstop behind it.
MAX_OPEN_STRING_TOKENS = 2048


def assert_no_verbatim_cycle(
    text: str,
    *,
    label: str,
    min_cycle_words: int = MIN_CYCLE_WORDS,
    required_repeats: int = REQUIRED_REPEATS,
) -> None:
    """Post-hoc cycle check for a REMOTE reply. Raises on degeneracy.

    WHY THIS IS SEPARATE FROM THE CRITERION. A `StoppingCriteria` attaches to a
    token loop, and a cloud call has no token loop to attach to -- the request
    goes out, the whole reply comes back. So this cannot SAVE the spend the way
    the local guard saves GPU minutes; by the time we can see the text, the
    money and the latency are gone.

    WHAT IT STILL BUYS, and why it is worth having anyway:

    * A degenerate reply never reaches the ledger. Without this, a cloud lane
      writes a paragraph repeated forty times into an episode and the first
      thing that notices is a listener.
    * It raises the SAME rerollable phase as the local guard, so a cloud
      runaway and a local runaway dispose identically -- the retry ladder does
      not need to know which transport produced it.
    * It removes the asymmetry where local lanes are protected and cloud lanes
      silently are not, which is the sort of gap that is discovered a year
      later by someone reading an exemption list.

    Measured in WORDS rather than tokens because a remote provider does not
    return its token ids. That makes the check tokenizer-independent by
    construction, and it is the same detector -- `find_repeating_cycle` is
    generic over any list of hashables, so words work unchanged.

    A streaming provider could do better (abort the stream mid-flight and save
    the remainder of the spend). That is a per-provider capability and belongs
    in the provider's own client, not here; this is the floor every provider
    can meet.
    """
    words = (text or "").split()
    cycle = find_repeating_cycle(
        words,
        min_cycle=min_cycle_words,
        max_cycle=MAX_CYCLE_TOKENS,
        required_repeats=required_repeats,
    )
    if cycle is None:
        return
    try:
        from ._otr_generation_budget import GenerationDegeneracyError
    except ImportError:  # pragma: no cover - flat/standalone import path
        from _otr_generation_budget import (  # type: ignore
            GenerationDegeneracyError,
        )
    raise GenerationDegeneracyError(
        f"{label}: the remote reply repeated a {cycle}-word run verbatim "
        f"{required_repeats} times, which is a degenerate generation rather "
        f"than a long one",
        halt_reason="verbatim_cycle_remote",
        repetition={
            "cycle_words": cycle,
            "required_repeats": required_repeats,
            "min_cycle_words": min_cycle_words,
            "total_words": len(words),
        },
        raw_completion=text,
    )


class OpenStringTracker:
    """Track whether a constrained decode is inside a JSON string, and how long.

    SCOPED ON PURPOSE. This is only installed where a SCHEMA IS BOUND, because
    it reads quotes as structure. On a free-prose or markup pass an ordinary
    quotation mark in spoken dialogue would open its state and it would count
    the rest of the scene -- which is precisely why the first version of this
    module was wrong to install a quote lexer on every route.

    A tiny hand-rolled lexer rather than a reach into lm-format-enforcer's
    parser state: coupling a liveness contract to a third party's internals is
    how a guard silently stops guarding after an upgrade.
    """

    __slots__ = ("in_string", "escaped", "open_tokens", "max_open_tokens")

    def __init__(self) -> None:
        self.in_string = False
        self.escaped = False
        self.open_tokens = 0
        self.max_open_tokens = 0

    def feed(self, text: str) -> None:
        """Consume the text contributed by ONE generated token."""
        if self.in_string:
            self.open_tokens += 1
            if self.open_tokens > self.max_open_tokens:
                self.max_open_tokens = self.open_tokens
        for ch in text:
            if self.escaped:
                # Whatever follows a backslash is literal, including a quote
                # and a second backslash. This is what makes an odd/even
                # backslash run behave correctly across token boundaries.
                self.escaped = False
                continue
            if ch == "\\":
                if self.in_string:
                    self.escaped = True
                continue
            if ch == '"':
                self.in_string = not self.in_string
                self.open_tokens = 0


def make_degeneracy_criterion(
    prompt_len: int,
    *,
    min_cycle_tokens: int = MIN_CYCLE_TOKENS,
    required_repeats: int = REQUIRED_REPEATS,
    tokenizer: "Any | None" = None,
    max_open_string_tokens: int = MAX_OPEN_STRING_TOKENS,
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
            self.open_string_tokens: int | None = None
            self._tail: list[int] = []
            self._since_check = 0
            # The open-string half only runs when a tokenizer was supplied,
            # which the caller does ONLY on a schema-bound route.
            self._tracker = OpenStringTracker() if tokenizer is not None else None
            self._decoded: dict[int, str] = {}

        def _text(self, token_id: int) -> str:
            cached = self._decoded.get(token_id)
            if cached is None:
                cached = tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False,
                )
                self._decoded[token_id] = cached
            return cached

        def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: D401
            if self.hit:
                return True
            try:
                row = input_ids[0]
                total = int(row.shape[-1])
                for index in range(prompt_len + self.generated_seen, total):
                    token_id = int(row[index])
                    self._tail.append(token_id)
                    self.generated_seen += 1
                    self._since_check += 1
                    # SIGNAL 2: the elaboration spiral. Checked per token
                    # because a spiral has no periodic structure to wait for --
                    # the only thing that changes is that the string is still
                    # open. This is what catches the P2 specimen (15,355 tokens,
                    # 83k chars, no repetition anywhere) that the cycle
                    # detector below is structurally blind to.
                    if self._tracker is not None:
                        self._tracker.feed(self._text(token_id))
                        if (self._tracker.in_string
                                and self._tracker.open_tokens
                                >= max_open_string_tokens):
                            self.hit = True
                            self.reason = "open_string"
                            self.open_string_tokens = self._tracker.open_tokens
                            return True
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
                "open_string_tokens": self.open_string_tokens,
                "max_open_string_tokens": (
                    self._tracker.max_open_tokens if self._tracker else None
                ),
                "open_string_bound": (
                    max_open_string_tokens if self._tracker else None
                ),
                "required_repeats": required_repeats,
                "min_cycle_tokens": min_cycle_tokens,
                "generated_tokens_seen": self.generated_seen,
            }

    return _DegeneracyHalt()
