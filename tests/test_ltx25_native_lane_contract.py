# -*- coding: utf-8 -*-
"""What the three native (non-GGUF) LTX 2.5 lanes owe their operator.

Written 2026-09-22 after the codex QA lane read the lanes' first commits and
found three defects the author could not see. Each one is pinned here, because
each is invisible to the checks that were run at the time: the suite was green,
``ast.parse`` was happy, and a live leg rendered successfully with all three
present.

  * the GGUF check was skipped by MUTATING THIS MODULE with ``unittest.mock``
    inside ``assert_usable``. Overlapping calls unwind out of order and the
    bypass can survive both -- silently disabling a real safety check on the
    GGUF lanes, which is a different lane entirely.
  * operator-facing text named ComfyUI-GGUF at two sites where the failure has
    nothing to do with that pack, sending the reader to the wrong fix.
  * the lanes had no publication shortcode, so an episode they dominate
    publishes its video identity as ``unk``.
"""
import ast
import io

import pytest

from nodes._otr_video_engines import eng_ltx25
from nodes._otr_shared import shortcodes

NATIVE = (
    eng_ltx25.Ltx25NativeFoley16gbEngine,
    eng_ltx25.Ltx25NativeFoleyWideEngine,
    eng_ltx25.Ltx25NativeFoleyBlackwellEngine,
)


def test_the_gguf_check_is_declined_by_an_override_not_by_patching_the_module():
    """The seam has to be an instance method, or concurrency can leak it.

    ``mock.patch.object`` on the module is not re-entrant: with two overlapping
    checks the restores run A-enter, B-enter, A-exit, B-exit and the last exit
    writes back the PATCHED value, leaving the inspector bypassed for the rest
    of the process. This repo's render routes run in unguarded daemon threads,
    so that shape is reachable, and the lane it would silently un-gate is the
    GGUF one.
    """
    # By AST, not by grep: the docstrings below deliberately SAY
    # "unittest.mock" to explain why it is gone, and a text search cannot tell
    # an explanation apart from the thing it explains.
    tree = ast.parse(io.open(eng_ltx25.__file__, encoding="utf-8").read())
    imported = [
        n.names[0].name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom)) and n.names
    ] + [
        (n.module or "")
        for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    ]
    assert not [m for m in imported if "mock" in m], imported
    # The real answer: a hook the parent defines and the native base overrides.
    assert eng_ltx25.Ltx25NativeFoleyBase._inspect_te_loader is not \
        eng_ltx25.Ltx25FoleyPlusEngine._inspect_te_loader


@pytest.mark.parametrize("cls", NATIVE)
def test_a_native_lane_declines_the_gguf_inspection_without_touching_anything(cls):
    before = eng_ltx25._inspect_ltx25_gguf_patch
    assert cls()._inspect_te_loader(object()) == ("", [])
    # The module-level function is exactly the object it was.
    assert eng_ltx25._inspect_ltx25_gguf_patch is before


def test_the_gguf_family_still_runs_the_inspection():
    """The override must not have neutered the check for the lanes that need it."""
    calls = []

    class Probe(eng_ltx25.Ltx25FoleyPlusEngine):
        pass

    real = eng_ltx25._inspect_ltx25_gguf_patch
    try:
        eng_ltx25._inspect_ltx25_gguf_patch = lambda cls: (calls.append(cls), ("p", []))[1]
        assert Probe()._inspect_te_loader("SENTINEL") == ("p", [])
    finally:
        eng_ltx25._inspect_ltx25_gguf_patch = real
    assert calls == ["SENTINEL"]


@pytest.mark.parametrize("cls", NATIVE)
def test_no_operator_facing_string_on_a_native_lane_names_the_gguf_pack_as_the_fix(cls):
    eng = cls()
    assert "GGUF" not in eng._weight_family()
    remedy = eng._missing_node_remedy()
    assert "update ComfyUI itself" in remedy
    # It may MENTION the pack to say it is not used; it must not ask for it.
    assert "install" not in remedy.lower()


def test_the_gguf_lane_still_asks_for_the_gguf_pack():
    eng = eng_ltx25.Ltx25FoleyPlusEngine()
    assert eng._weight_family() == "GGUF"
    assert "ComfyUI-GGUF" in eng._missing_node_remedy()


@pytest.mark.parametrize("cls", NATIVE)
def test_a_missing_native_weight_is_labelled_by_its_real_format(cls, monkeypatch):
    labels = [label for label, _path, _floor in cls()._weight_paths()]
    assert labels, "a lane with no weights to check would gate on nothing"
    assert not any("GGUF" in label for label in labels), labels
    assert any("native safetensors" in label for label in labels), labels


@pytest.mark.parametrize("cls", NATIVE)
def test_every_native_lane_has_a_publication_shortcode(cls):
    code = shortcodes.VIDEO_LANE.get(cls.name)
    assert code, "%s would publish as 'unk'" % cls.name
    assert len(code) == 4 and code.isalnum()


def test_the_native_shortcodes_collide_with_nothing():
    codes = list(shortcodes.VIDEO_LANE.values())
    assert len(codes) == len(set(codes))
