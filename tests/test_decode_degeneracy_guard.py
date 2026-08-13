"""The in-decode liveness guard: catch a CYCLING decode, never a long one.

The guard exists because a live leg spent 13,912 tokens over 22 minutes
rewriting the same ~384-token paragraph about 36 times. It must not exist at
the cost of refusing long writing -- THE LAW says an audit may improve a story
but never fail one for length -- so these tests are as much about what it leaves
alone as what it stops.

THE FALSE-POSITIVE TESTS ARE THE POINT. An r1 panel killed the first version of
this module for being a length ceiling in disguise, and caught that its
"long string never halts" test used 50 tokens against a bound of 64 -- it never
approached the threshold, so it proved nothing. Every no-halt test below now
generates FAR more tokens than any threshold, and differs only in whether the
content repeats.
"""
from __future__ import annotations

from nodes import _otr_decode_guard as guard
from nodes._otr_generation_budget import (
    CAPACITY_PHASE_DECODE_DEGENERACY,
    CAPACITY_PHASE_OUTPUT_LIMIT,
    CAPACITY_PHASE_PROMPT_NO_ROOM,
    GenerationDegeneracyError,
    PromptContextOverflowError,
    is_rerollable_generation_error,
)


# --------------------------------------------------------------------------
# The detector, tested directly: it is pure, so it needs no model or tokenizer.
# --------------------------------------------------------------------------

def test_a_verbatim_cycle_is_found():
    cycle = list(range(100, 100 + 60))          # a 60-token run
    tokens = cycle * 3
    assert guard.find_repeating_cycle(tokens) == 60


def test_the_measured_runaway_shape_is_found():
    """The real specimen: a ~384-token paragraph repeating back to back."""
    paragraph = list(range(1000, 1000 + 384))
    tokens = list(range(500)) + paragraph * 4    # healthy text, then the loop
    assert guard.find_repeating_cycle(tokens) == 384


def test_TWO_repeats_are_not_enough():
    """Two could be deliberate parallel construction. Three is not a style."""
    cycle = list(range(100, 100 + 60))
    assert guard.find_repeating_cycle(cycle * 2) is None


def test_REPEATED_JSON_KEYS_WITH_VARYING_VALUES_never_trip_the_detector():
    """The realistic structural-repetition case, and the one that matters.

    A JSON array emits the SAME key tokens in every element while the VALUES
    differ -- so the token stream is not periodic, and the detector must leave
    it alone no matter how many elements there are. This is what
    `no_repeat_ngram_size` gets wrong and what the detector must not.
    """
    keys = list(range(10))                       # the repeated key tokens
    tokens = []
    for element in range(60):                    # 60 array elements
        tokens += keys + [90000 + element]       # ... each with its own value
    assert guard.find_repeating_cycle(tokens) is None


def test_a_PERFECTLY_PERIODIC_stream_is_degenerate_even_if_the_unit_is_short():
    """A short unit repeating with NOTHING varying is a runaway, not structure.

    Deliberately asserted so the boundary is documented rather than discovered:
    the floor is on the CYCLE length, and a 10-token unit repeated 20 times
    contains a 50-token cycle, which is above the floor and is caught. Real
    structural repetition never looks like this, because values differ -- see
    the test above.
    """
    assert guard.find_repeating_cycle(list(range(10)) * 20) is not None


def test_LONG_VARIED_output_never_trips_the_detector():
    """A big episode: thousands of tokens, no verbatim cycle. Must be ignored."""
    tokens = [(i * 7919) % 30000 for i in range(20000)]
    assert guard.find_repeating_cycle(tokens) is None


def test_NEAR_repetition_is_not_a_cycle():
    """Similar is not identical. Only verbatim repetition counts."""
    cycle = list(range(100, 100 + 60))
    tokens = cycle + [x + 1 for x in cycle] + [x + 2 for x in cycle]
    assert guard.find_repeating_cycle(tokens) is None


# --------------------------------------------------------------------------
# The criterion. No tokenizer at all now -- it reads token ids.
# --------------------------------------------------------------------------

class _FakeIds:
    def __init__(self, row):
        self._row = list(row)

    def __getitem__(self, index):
        return _FakeRow(self._row)


class _FakeRow:
    def __init__(self, row):
        self._row = row
        self.shape = (len(row),)

    def __getitem__(self, index):
        return self._row[index]


def _feed_all(token_ids, prompt_len=0, **kwargs):
    """Feed the whole sequence in one call, as a late generate() step would."""
    criterion = guard.make_degeneracy_criterion(prompt_len, **kwargs)
    verdict = criterion(_FakeIds(token_ids), None)
    return criterion, verdict


def test_the_criterion_halts_on_a_cycle_and_latches():
    cycle = list(range(200, 200 + 60))
    criterion, verdict = _feed_all(cycle * 4)
    assert verdict is True
    assert criterion.hit is True
    assert criterion.reason == "verbatim_cycle"
    assert criterion.cycle_tokens == 60
    # latched
    assert criterion(_FakeIds(cycle * 4), None) is True


def test_the_criterion_leaves_a_LONG_NON_REPEATING_decode_alone():
    """20,000 tokens of varied output. A length-based guard would have fired."""
    tokens = [(i * 7919) % 30000 for i in range(20000)]
    criterion, verdict = _feed_all(tokens)
    assert verdict is False
    assert criterion.hit is False


def test_the_criterion_ignores_the_prompt():
    """A repetitive PROMPT is not the model's fault and must not halt it."""
    cycle = list(range(300, 300 + 60))
    prompt = cycle * 4
    criterion, verdict = _feed_all(prompt, prompt_len=len(prompt))
    assert verdict is False
    assert criterion.hit is False


def test_the_criterion_is_format_agnostic():
    """No decoding, so JSON, markup and free prose behave identically.

    This is what the first version got wrong: its JSON-string lexer was blind
    to raw-markup runaways and would open its state on an ordinary quotation
    mark in spoken dialogue.
    """
    criterion = guard.make_degeneracy_criterion(0)
    assert not hasattr(criterion, "tracker")
    import inspect
    source = inspect.getsource(guard)
    assert "tokenizer" not in source.split("UTF-8")[0].replace(
        "tokenizer-independent", ""
    )


def test_telemetry_names_what_fired():
    cycle = list(range(400, 400 + 50))
    criterion, _ = _feed_all(cycle * 4, min_cycle_tokens=48)
    data = criterion.telemetry()
    assert data["halt_reason"] == "verbatim_cycle"
    assert data["cycle_tokens"] == 50
    assert data["required_repeats"] == guard.REQUIRED_REPEATS


# --------------------------------------------------------------------------
# Disposition. A halt must REROLL, or it is a writer veto.
# --------------------------------------------------------------------------

def test_a_degeneracy_halt_is_rerollable():
    error = GenerationDegeneracyError("halted", halt_reason="verbatim_cycle")
    assert error.phase == CAPACITY_PHASE_DECODE_DEGENERACY
    assert is_rerollable_generation_error(error) is True


def test_an_output_limit_is_still_rerollable():
    error = PromptContextOverflowError(
        "ran out", phase=CAPACITY_PHASE_OUTPUT_LIMIT,
    )
    assert is_rerollable_generation_error(error) is True


def test_a_prompt_no_room_refusal_is_still_never_rerollable():
    error = PromptContextOverflowError(
        "no room", phase=CAPACITY_PHASE_PROMPT_NO_ROOM,
    )
    assert is_rerollable_generation_error(error) is False


def test_degeneracy_is_a_DISTINCT_phase_from_capacity():
    """Reusing output_limit would make every diagnostic lie.

    A halted decode stops with most of its allowance unspent, so reporting it
    as "ended at the provider capacity limit" would send the next reader
    hunting a budget defect that does not exist.
    """
    assert CAPACITY_PHASE_DECODE_DEGENERACY != CAPACITY_PHASE_OUTPUT_LIMIT
    halt = GenerationDegeneracyError("halted")
    assert halt.phase != CAPACITY_PHASE_OUTPUT_LIMIT
    assert hasattr(halt, "raw_completion")
    assert hasattr(halt, "generated_tokens")


def test_the_guard_never_reads_target_words():
    """A word target is a REQUEST, never a limit. Checked structurally.

    Against IDENTIFIERS via the AST, not raw text: the module's own docstring
    says it never reads `target_words`, and a text search would flag that
    sentence -- which is documentation working as intended.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(guard.__file__).read_text(encoding="utf-8"))
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            identifiers.add(node.arg)

    offenders = {name for name in identifiers if "target_word" in name}
    assert not offenders, (
        f"the liveness guard must never read a word target; found {offenders}"
    )
