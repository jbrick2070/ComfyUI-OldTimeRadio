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

import ast
import logging

import pytest

from nodes import otr_master_audio_mux as m


# --------------------------------------------------------------------------- #
# the predicate the last test walks with, at module scope so it can be tested
# itself -- a guard nobody points at goes blind quietly
# --------------------------------------------------------------------------- #
def _env_owner_names(tree):
    """Whatever name this module binds ``nodes/_otr_shared/env.py`` to.

    Resolved from the module's OWN imports, honouring ``asname``, at both the
    packaged depth (``from ._otr_shared import env as otr_env``) and the flat
    one (``from _otr_shared import env as otr_env``) -- so a re-alias cannot
    dodge the rule and no spelling has to be guessed."""
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom)
                and (node.module or "").endswith("_otr_shared")):
            names.update(a.asname or a.name for a in node.names
                         if a.name == "env")
    return names


def _is_env_read(node, owners):
    """Does this call READ the environment -- by either spelling."""
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute) and fn.attr in ("get", "getenv"):
        if "environ" in ast.dump(fn) or fn.attr == "getenv":
            return True
        return isinstance(fn.value, ast.Name) and fn.value.id in owners
    return False


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
    """Pin that no OTHER unguarded env read appears in the terminal mux -- a
    bare numeric cast wrapped straight around an env read, at the terminal
    node, is the whole bug class. (The once-guarded sibling knob, the SFX bed
    gain, died with the bed in the 2026-08-06 rip; the credits-tail ceiling is
    now the mux's one env read.)

    WALKS THE AST, NOT THE TEXT. The first draft grepped the source for
    `float(os.environ.get(` and went red on this module's own DOCSTRING, which
    quotes the defect it describes. A source grep cannot tell a call from prose
    about a call -- and the version that can be fooled by a comment is the
    version that gets deleted the first time it cries wolf.

    IT ALSO KNOWS THE OWNER. Once the mux asks `nodes/_otr_shared/env.py` for
    its knob instead of spelling `os.environ` itself, the read is
    `otr_env.get("OTR_MAX_CREDITS_TAIL_S", ...)` -- which the `environ` branch
    below cannot see. So the owner's bound name is resolved from the MUX'S OWN
    imports, honouring `asname`: a re-alias cannot dodge this, and the old
    `os.environ` / `getenv` branch is KEPT, so a regression to the old spelling
    is caught too."""
    import inspect

    tree = ast.parse(inspect.getsource(m))
    offenders = [
        ast.dump(n) for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id in ("float", "int")
        and n.args and _is_env_read(n.args[0], _env_owner_names(tree))
    ]
    assert offenders == [], (
        "a numeric cast wraps an env read directly -- guard it like "
        "_credits_tail_ceiling: %s" % offenders)


def test_the_predicate_sees_the_owner_spelling_and_the_old_one():
    """The rewrite that carries this test through the env migration, pinned.

    Without the owner branch, `float(otr_env.get("OTR_MAX_CREDITS_TAIL_S"))`
    would read as clean and this whole gate would pass vacuously the moment the
    mux stopped saying `os.environ` -- which is exactly what the migration does
    to it."""
    def _offends(src):
        tree = ast.parse(src)
        owners = _env_owner_names(tree)
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id in ("float", "int")
                and n.args and _is_env_read(n.args[0], owners)]

    # the OLD spelling, which is the defect this file was written for
    assert _offends("import os\nfloat(os.environ.get('A', '45'))\n")
    assert _offends("import os\nint(os.getenv('A', '45'))\n")
    # the MIGRATED spelling, at both depths and under a re-alias
    assert _offends("from ._otr_shared import env as otr_env\n"
                    "float(otr_env.get('A', '45'))\n")
    assert _offends("from _otr_shared import env as otr_env\n"
                    "float(otr_env.get('A', '45'))\n")
    assert _offends("from _otr_shared import env as _e\n"
                    "float(_e.get('A', '45'))\n")
    # and it still does not cry wolf
    assert not _offends("d = {}\nfloat(d.get('A', '45'))\n")
    assert not _offends("s = \"float(os.environ.get('A'))\"\n")
    assert not _offends("from ._otr_shared import env as otr_env\n"
                        "float(_guarded(otr_env.get('A', '45')))\n")
