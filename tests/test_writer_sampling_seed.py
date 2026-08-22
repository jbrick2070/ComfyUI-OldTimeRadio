"""OTR_WRITER_SEED -- reproducible sampling, so a bake-off can hold the script.

WHY THIS EXISTS, in the operator's own words: *"its a different story so not a
good comparison"*.

On 2026-08-22 four separate attempts to judge a video-lane change all collapsed
for the same reason. The writer's ``gen_kwargs`` carries ``do_sample=True`` with
no seed and no generator, so every leg writes a DIFFERENT episode -- and every
impression the operator formed while comparing two legs turned out to be about
the script rather than the thing under test: more colour at the start (different
opening scene), a better story (the writer), better audio match (same cadence in
both, so it was mood), more character beats (structurally impossible for a video
engine to affect -- the ledger is frozen before any engine exists).

Unseeded sampling is CORRECT for production; every episode should be its own.
This is an opt-in bake-off tool and it must stay opt-in, which is what most of
these tests are about.

KEYED ON THE PROMPT, NOT A CALL COUNTER. An episode makes many generate calls,
and a counter would make each seed depend on call ORDER -- so one conditional
pass firing in run A but not run B would shift every later seed and the
reproduction would drift silently. Hashing the input tokens makes a call's seed
a function of what that call is being asked.
"""
from __future__ import annotations

import pytest

from nodes import OTR_LedgerScriptWriter as writer


class _Ids:
    """Stands in for a tokenizer's input_ids tensor."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


def _inputs(values=((1, 2, 3),)):
    return {"input_ids": _Ids([list(v) for v in values])}


# --------------------------------------------------------------------------- #
# OFF BY DEFAULT. Production must be untouched.
# --------------------------------------------------------------------------- #

def test_unset_is_a_no_op(monkeypatch):
    monkeypatch.delenv(writer.WRITER_SEED_ENV, raising=False)
    assert writer._seed_writer_sampling(_inputs()) is None


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_is_a_no_op(blank, monkeypatch):
    monkeypatch.setenv(writer.WRITER_SEED_ENV, blank)
    assert writer._seed_writer_sampling(_inputs()) is None


@pytest.mark.parametrize("junk", ["abc", "1.5", "None", "seed"])
def test_an_unusable_value_leaves_sampling_unseeded_rather_than_raising(
        junk, monkeypatch):
    """A bake-off convenience may never break a render."""
    monkeypatch.setenv(writer.WRITER_SEED_ENV, junk)
    assert writer._seed_writer_sampling(_inputs()) is None


def test_a_malformed_inputs_dict_does_not_raise(monkeypatch):
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "7")
    assert writer._seed_writer_sampling({}) is None
    assert writer._seed_writer_sampling({"input_ids": object()}) is None


# --------------------------------------------------------------------------- #
# WHAT IT BUYS: the same prompt seeds the same way, twice.
# --------------------------------------------------------------------------- #

def test_the_same_prompt_seeds_identically(monkeypatch):
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "42")
    first = writer._seed_writer_sampling(_inputs())
    second = writer._seed_writer_sampling(_inputs())
    assert first is not None
    assert first == second


def test_a_different_prompt_seeds_differently(monkeypatch):
    """Two passes in one episode ask different questions and must not collapse
    onto one seed -- that would make every pass sample the same way."""
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "42")
    a = writer._seed_writer_sampling(_inputs(((1, 2, 3),)))
    b = writer._seed_writer_sampling(_inputs(((9, 9, 9),)))
    assert a != b


def test_a_different_base_seeds_differently(monkeypatch):
    """The base is the knob: two bake-off arms that want DIFFERENT scripts set
    different bases, and two that want the same script share one."""
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "1")
    one = writer._seed_writer_sampling(_inputs())
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "2")
    two = writer._seed_writer_sampling(_inputs())
    assert one != two


def test_the_seed_is_order_independent(monkeypatch):
    """THE REASON IT IS NOT A COUNTER. Asking the same three prompts in a
    different order must produce the same three seeds, so a conditional pass
    cannot shift everything after it."""
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "42")
    prompts = [_inputs(((1,),)), _inputs(((2,),)), _inputs(((3,),))]
    forward = [writer._seed_writer_sampling(p) for p in prompts]
    backward = [writer._seed_writer_sampling(p) for p in reversed(prompts)]
    assert forward == list(reversed(backward))


def test_the_seed_is_in_torchs_legal_range(monkeypatch):
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "999999999")
    seed = writer._seed_writer_sampling(_inputs())
    assert 0 <= seed <= 0x7FFF_FFFF


def test_it_actually_seeds_torch(monkeypatch):
    """Not just arithmetic -- the point is that the RNG moves."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "42")
    writer._seed_writer_sampling(_inputs())
    first = torch.rand(4).tolist()
    writer._seed_writer_sampling(_inputs())
    second = torch.rand(4).tolist()
    assert first == second, (
        "two identical seedings produced different draws; sampling is not "
        "actually reproducible")


def test_two_different_seeds_really_do_diverge(monkeypatch):
    """The negative control. Without it, a helper that seeded nothing at all
    would still pass the test above."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "1")
    writer._seed_writer_sampling(_inputs())
    one = torch.rand(4).tolist()
    monkeypatch.setenv(writer.WRITER_SEED_ENV, "2")
    writer._seed_writer_sampling(_inputs())
    two = torch.rand(4).tolist()
    assert one != two


# --------------------------------------------------------------------------- #
# WIRING: both generate call sites are covered.
# --------------------------------------------------------------------------- #

def test_both_generate_paths_are_seeded():
    """The min_p retry re-generates after a failed attempt has already consumed
    RNG state. Without a re-seed there, a run that hit the TypeError would
    diverge from one that did not -- reproducible only by luck."""
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(writer)
    tree = ast.parse(source)
    generate_lines, seed_lines = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "generate":
            generate_lines.append(node.lineno)
        if isinstance(func, ast.Name) and func.id == "_seed_writer_sampling":
            seed_lines.append(node.lineno)
    assert len(generate_lines) >= 2, generate_lines
    for line in generate_lines:
        assert any(0 < line - seed < 12 for seed in seed_lines), (
            "model.generate at line %d has no _seed_writer_sampling call "
            "shortly before it; that path is not reproducible" % line)
    del textwrap


def test_production_sampling_is_still_unseeded_by_default():
    """The writer must keep rolling a fresh story per episode unless asked."""
    import inspect

    source = inspect.getsource(writer)
    assert '"do_sample": True' in source
    assert "WRITER_SEED_ENV" in source
    assert writer.WRITER_SEED_ENV == "OTR_WRITER_SEED"
