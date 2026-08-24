"""The commercial-clean JOIN on the cast lock -- two licences, one verdict.

THE DEFECT THIS FILE EXISTS FOR. A voice-bank row's ``commercial_clean``
describes the reference CLIP: the 40 indextts2 rows are public-domain
recordings, so they say ``true``, and that is CORRECT. The MODEL that speaks
with that clip carries a licence of its own, and indextts2's is the bilibili
Model Use License -- non-commercial. Reading only the clip made the cast report
print ``clean=True`` for audio nobody could ship, and stamped that same claim
onto the ledger.

The engine-profile layer already knew: ``char_indextts2_v1`` carries
``commercial_clean: false  # Bilibili license -- non-commercial use gated``.
Nothing joined the two facts, so the cast layer contradicted the profile layer
about the same delivered audio.

WHY THE RATCHET AT THE BOTTOM IS THE POINT. The failure mode is not "someone
deletes the join" -- it is someone adding a SIXTH counter, or a new report
line, that reads the clip flag directly again. Then the report prints
``clean=True`` beside a ledger row saying ``False``, which is worse than the
original bug because it looks audited. The AST guard makes that unwritable, and
the negative control below proves the guard actually bites.

ENFORCEMENT STAYS OFF. Nothing here blocks a render; ``gated`` feeds one
non-blocking warning (I-8). The release gate is a separate, unarmed mechanism
fed by the profile layer, which was already truthful.
"""
from __future__ import annotations

import ast
import io

import pytest

from nodes import cast_lock as CL


CHARACTER_ROW = {"char_id": "SHERIFF", "speaker_role": "character"}
ANNOUNCER_ROW = {"char_id": "announcer", "speaker_role": "announcer"}


class _Ref:
    """The shape the cast lock stamps from -- a voice-bank row."""

    def __init__(self, engine, clip_clean):
        self.engine = engine
        self.commercial_clean = clip_clean
        self.voice_ref_id = "vr_test"


# --------------------------------------------------------------------------- #
# THE JOIN
# --------------------------------------------------------------------------- #

def test_a_public_domain_clip_on_a_noncommercial_model_is_not_clean():
    """THE BUG, stated as a test. Both halves were true-looking; the delivered
    audio is still not commercially clean."""
    assert CL._delivered_commercial_clean(
        CHARACTER_ROW, _Ref("indextts2", True)) is False


def test_a_clean_clip_on_a_clean_model_stays_clean():
    """The join must not over-gate. If it gated everything it would be a
    constant, not a fix."""
    assert CL._delivered_commercial_clean(
        CHARACTER_ROW, _Ref("kokoro", True)) is True


def test_a_gated_clip_stays_gated_whatever_the_model():
    assert CL._delivered_commercial_clean(
        CHARACTER_ROW, _Ref("kokoro", False)) is False


def test_an_unknown_engine_leaves_the_clip_flag_standing():
    """Absence of a profile is not a licence. A partial install, or an engine
    with no curated row, behaves exactly as it did before the join existed --
    the join DOWNGRADES only on a known-gated model."""
    assert CL._delivered_commercial_clean(
        CHARACTER_ROW, _Ref("no_such_engine_at_all", True)) is True
    assert CL._delivered_commercial_clean(
        CHARACTER_ROW, _Ref("no_such_engine_at_all", False)) is False


# --------------------------------------------------------------------------- #
# ONE PROFILE, RESOLVED BY (role, engine) -- never by engine name alone
# --------------------------------------------------------------------------- #

def test_the_row_decides_which_role_is_resolved():
    """The same engine can be curated differently for the announcer than for a
    character, so the lookup is role-scoped."""
    assert CL._profile_role_for_entry(CHARACTER_ROW) == "char_voice"
    assert CL._profile_role_for_entry(ANNOUNCER_ROW) == "announcer_voice"


def test_the_profile_layer_is_the_authority_on_the_model():
    """Pinned against the real YAML: this is the fact the cast layer used to
    contradict."""
    assert CL._model_license_clean("char_voice", "indextts2") is False
    assert CL._model_license_clean("char_voice", "kokoro") is True
    assert CL._model_license_clean("char_voice", "no_such_engine_at_all") is None


# --------------------------------------------------------------------------- #
# THE STAMP IS THE FUNNEL
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("engine,expected", [("indextts2", False),
                                             ("kokoro", True)])
def test_the_stamp_writes_the_join_not_the_clip_flag(engine, expected):
    """``_stamp`` is the one place every stamped row passes through -- the
    ordinary draw, the announcer, both route tiers and the gender fallback. The
    join belongs here so no call site can miss it."""
    entry = dict(CHARACTER_ROW)
    CL.CastLock._stamp(entry, _Ref(engine, True))
    assert entry["commercial_clean"] is expected


# --------------------------------------------------------------------------- #
# THE RATCHET -- no counter or report may read the clip flag directly
# --------------------------------------------------------------------------- #

SOURCE_PATH = CL.__file__
JOIN_FUNCTION = "_delivered_commercial_clean"


def _clip_flag_reads_outside_the_join(source: str) -> list:
    """Line numbers of every ``<x>.commercial_clean`` attribute read that is
    NOT inside the join helper."""
    tree = ast.parse(source)
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == JOIN_FUNCTION:
            allowed.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "commercial_clean"
        and node.lineno not in allowed
    )


def test_nothing_outside_the_join_reads_the_clip_flag():
    """The atomicity guard. A new counter or report line that reads the clip
    flag directly would make the report disagree with the ledger it was printed
    beside -- the exact contradiction this campaign removed."""
    source = io.open(SOURCE_PATH, encoding="utf-8").read()
    stray = _clip_flag_reads_outside_the_join(source)
    assert not stray, (
        "cast_lock.py reads a voice-bank row's clip licence directly at line(s) "
        "%s -- route it through %s() so the clip AND the model are joined"
        % (stray, JOIN_FUNCTION))


def test_the_guard_would_have_caught_the_original_defect():
    """NEGATIVE CONTROL. A guard nobody has seen fail is a guard nobody can
    trust: this is the code as it was before the join, and it must be flagged."""
    evasive = (
        "def _delivered_commercial_clean(entry, ref):\n"
        "    return bool(getattr(ref, 'commercial_clean', False))\n"
        "\n"
        "def cast(entry, ref, report):\n"
        "    gated = 0 if ref.commercial_clean else 1\n"
        "    report.append(f'clean={ref.commercial_clean}')\n"
        "    return gated\n"
    )
    assert _clip_flag_reads_outside_the_join(evasive) == [5, 6]
