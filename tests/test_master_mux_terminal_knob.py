"""The mux may not lose a finished episode to a typo, and may not claim a
proof it does not have.

Both found by a Fable consult (2026-07-27) after two mechanical panels missed
them, on a file neither panel had been pointed at.

WHY IT MATTERS THAT THIS IS THE MUX. `OTR_MasterAudioMux` is the LAST node of
the graph. Everything it can raise on, it raises on AFTER the whole episode has
rendered -- so a defect here costs the entire run, not a retry. That is the
opposite end of the pipeline from where this build usually pays for a knob.

CPU-only: pure helpers over plain data. No ffmpeg, no GPU, no model load.
"""

from __future__ import annotations

import logging

import pytest

from nodes import otr_master_audio_mux as m


# --------------------------------------------------------------------------- #
# the knob: NAMED and ignored when malformed, never fatal
# --------------------------------------------------------------------------- #
def test_an_unset_knob_gets_the_documented_default(monkeypatch):
    monkeypatch.delenv("OTR_MAX_CREDITS_TAIL_S", raising=False)
    assert m._credits_tail_ceiling() == m._MAX_CREDITS_TAIL_S_DEFAULT


def test_a_valid_knob_binds(monkeypatch):
    # OPPOSES the default, so this can tell "the env won" from "the default won".
    assert m._MAX_CREDITS_TAIL_S_DEFAULT != 12.5
    monkeypatch.setenv("OTR_MAX_CREDITS_TAIL_S", "12.5")
    assert m._credits_tail_ceiling() == 12.5


@pytest.mark.parametrize("garbage", ["45s", "forty-five", "", "  ", "45,0"])
def test_a_MALFORMED_knob_can_NOT_kill_a_finished_episode(monkeypatch, garbage):
    """The defect, stated directly.

    It was `float(os.environ.get("OTR_MAX_CREDITS_TAIL_S", "45"))` -- so
    `OTR_MAX_CREDITS_TAIL_S=45s` in a server's launch environment raised an
    uncaught ValueError at the mux and lost the episode at the finish line,
    over a value that only widens a sanity ceiling. `45s` is the realistic
    typo: the knob's name ends in `_S`."""
    monkeypatch.setenv("OTR_MAX_CREDITS_TAIL_S", garbage)
    assert m._credits_tail_ceiling() == m._MAX_CREDITS_TAIL_S_DEFAULT


def test_a_malformed_knob_is_NAMED_not_silently_swallowed(monkeypatch, caplog):
    """Ignored is not the same as unmentioned. A ceiling that silently reverts
    to the default is a ceiling the operator believes they moved -- which is
    the complaint that produced the demotion-notice rule in the first place."""
    monkeypatch.setenv("OTR_MAX_CREDITS_TAIL_S", "45s")
    with caplog.at_level(logging.WARNING, logger=m.log.name):
        m._credits_tail_ceiling()
    text = " ".join(r.getMessage() for r in caplog.records
                    if r.name == m.log.name)
    assert "OTR_MAX_CREDITS_TAIL_S" in text
    assert "45s" in text                       # the offending value, verbatim
    assert "IGNORING" in text


def test_a_VALID_knob_says_nothing(monkeypatch, caplog):
    # Not vacuous: this is what fails if the warning is moved outside the
    # except branch and starts firing on every well-configured episode.
    monkeypatch.setenv("OTR_MAX_CREDITS_TAIL_S", "30")
    with caplog.at_level(logging.WARNING, logger=m.log.name):
        assert m._credits_tail_ceiling() == 30.0
    assert [r for r in caplog.records if r.name == m.log.name] == []


def test_the_mux_has_no_OTHER_unguarded_env_read():
    """The sibling knob in this file (`_sfx_gain` via `_clamp01`) was already
    guarded; this one was not. Pin that no third one appears -- a bare numeric
    cast wrapped straight around an env read, at the terminal node, is the whole
    bug class.

    WALKS THE AST, NOT THE TEXT. The first draft grepped the source for
    `float(os.environ.get(` and went red on this module's own DOCSTRING, which
    quotes the defect it describes. A source grep cannot tell a call from prose
    about a call -- and the version that can be fooled by a comment is the
    version that gets deleted the first time it cries wolf."""
    import ast
    import inspect

    def _is_env_read(node):
        if not isinstance(node, ast.Call):
            return False
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in ("get", "getenv"):
            return "environ" in ast.dump(fn) or fn.attr == "getenv"
        return False

    tree = ast.parse(inspect.getsource(m))
    offenders = [
        ast.dump(n) for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id in ("float", "int")
        and n.args and _is_env_read(n.args[0])
    ]
    assert offenders == [], (
        "a numeric cast wraps an env read directly -- guard it like "
        "_credits_tail_ceiling: %s" % offenders)
