"""Every declared widget/socket must be a parameter the execute function accepts.

THE DEFECT THIS PINS, and it shipped: `OTR_SceneSequencer` declared
`music_cue_audio` and `music_cue_manifest_json` in `INPUT_TYPES` long after
commit 59286499 removed them from `sequence()`. ComfyUI passes declared inputs
as keyword arguments, so wiring either socket in the UI raised
``TypeError: sequence() got an unexpected keyword argument`` -- a crash trap
sitting behind a tooltip that described what the input would do.

It survived because nothing could see it: the canonical graph never wired
those sockets, so no test, no validator and no render ever passed them. Only
an operator clicking the socket would have found it.

THE TRAP THIS TEST ALSO GUARDS: the identically-named sockets on
`OTR_EpisodeAssembler` are LIVE -- canonical node 7, fed by node 83 over links
282/283 -- and they carry every episode's opening, closing and interstitial
music. A cleanup that greps the pack for `music_cue_audio` and deletes what it
finds would cut the music bus. This test passes only when the Assembler still
declares AND accepts them.

Hidden inputs (`unique_id`, `prompt`, `extra_pnginfo`, ...) are excluded --
ComfyUI supplies those by its own contract, not from the graph.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _mappings() -> dict:
    """The pack's registered nodes, imported the way ComfyUI imports them."""
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        pkg = importlib.import_module(_REPO_ROOT.name)
    finally:
        sys.path.pop(0)
    return dict(getattr(pkg, "NODE_CLASS_MAPPINGS", {}) or {})


NODE_CLASS_MAPPINGS = _mappings()


def _declared_input_names(cls) -> set:
    spec = cls.INPUT_TYPES()
    names = set()
    for section in ("required", "optional"):
        block = spec.get(section) or {}
        if isinstance(block, dict):
            names |= set(block.keys())
    return names


def _accepted_parameter_names(fn):
    sig = inspect.signature(fn)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    return set(sig.parameters) - {"self"}, accepts_kwargs


@pytest.mark.parametrize("node_name", sorted(NODE_CLASS_MAPPINGS))
def test_every_declared_input_is_a_parameter_of_the_execute_function(node_name):
    cls = NODE_CLASS_MAPPINGS[node_name]
    if not hasattr(cls, "INPUT_TYPES") or not getattr(cls, "FUNCTION", None):
        pytest.skip("%s declares no INPUT_TYPES/FUNCTION" % node_name)
    fn = getattr(cls, cls.FUNCTION, None)
    if fn is None:
        pytest.fail("%s.FUNCTION=%r names no method" % (node_name, cls.FUNCTION))

    accepted, accepts_kwargs = _accepted_parameter_names(fn)
    if accepts_kwargs:
        return  # **kwargs absorbs anything ComfyUI passes

    orphans = sorted(_declared_input_names(cls) - accepted)
    assert not orphans, (
        "%s declares %s in INPUT_TYPES but %s() cannot accept %s -- ComfyUI "
        "passes declared inputs as keyword arguments, so wiring one raises "
        "TypeError. Either remove the declaration or add the parameter."
        % (node_name, orphans, cls.FUNCTION,
           "it" if len(orphans) == 1 else "them"))


def test_the_scene_sequencer_no_longer_advertises_the_music_bus():
    """The specific removal, asserted by name so it cannot quietly return."""
    scene = NODE_CLASS_MAPPINGS["OTR_SceneSequencer"]
    declared = _declared_input_names(scene)
    assert "music_cue_audio" not in declared
    assert "music_cue_manifest_json" not in declared


def test_the_episode_assembler_STILL_owns_the_music_bus():
    """The live half. If this ever fails, the music bus was cut -- opening,
    closing and interstitial cues all arrive through these two sockets."""
    assembler = NODE_CLASS_MAPPINGS["OTR_EpisodeAssembler"]
    declared = _declared_input_names(assembler)
    assert "music_cue_audio" in declared
    assert "music_cue_manifest_json" in declared

    accepted, accepts_kwargs = _accepted_parameter_names(
        getattr(assembler, assembler.FUNCTION))
    assert accepts_kwargs or {
        "music_cue_audio", "music_cue_manifest_json"} <= accepted
