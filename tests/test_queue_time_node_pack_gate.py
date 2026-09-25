"""The queue-time node-pack gate (PBUG-20260925-02).

A video engine whose ComfyUI node classes are not registered is refused when
the graph is QUEUED -- before any weight download, writer pass or render --
and the refusal says what to do first. Grounded on the 4060's fresh-install
walk: 18 minutes 22 seconds and 3.9 GB before the old failure.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import _otr_visual_assets as VA  # noqa: E402
from nodes._otr_video_engines import eng_ghost_signal as GS  # noqa: E402
from nodes._otr_video_engines import registry as VREG  # noqa: E402
from nodes._otr_video_engines import wrapper_bridge as WB  # noqa: E402

GHOST = "animatediff15_v3_haunted_video"


def _install(monkeypatch, *names):
    """Stand in for ComfyUI's registry with exactly these classes registered.

    ``resolve_node_class`` calls ``node_class_mappings(mapping)`` with the map
    it was handed, so the stand-in must accept that argument and hand it back.
    """
    registered = {n: object for n in names}
    monkeypatch.setattr(
        WB, "node_class_mappings",
        lambda mapping=None: mapping if mapping is not None else registered)


@pytest.fixture
def every_ghost_class():
    """The engine's OWN table -- it can hold more than GHOST_NODE_CANDIDATES
    (the lora seam adds a loader on lanes that declare one)."""
    return sorted({n for names in VREG.get_engine(GHOST)._node_candidates().values()
                   for n in names})


def test_a_missing_node_pack_is_refused_at_queue_time_with_the_fix_first(monkeypatch, every_ghost_class):
    _install(monkeypatch, *[n for n in every_ghost_class if not n.startswith("ADE_")])
    with pytest.raises(VA.VisualAssetError) as err:
        VA._refuse_missing_node_packs({GHOST})
    text = str(err.value)
    assert text.startswith("Install ComfyUI-AnimateDiff-Evolved"), text
    assert "ADE_AnimateDiffLoaderGen1" in text and "ADE_StandardStaticContextOptions" in text
    assert "CheckpointLoaderSimple" not in text          # only the MISSING classes are named
    assert GHOST in text and "Nothing was downloaded" in text
    assert "FailureKind" not in text


def test_every_class_present_passes(monkeypatch, every_ghost_class):
    _install(monkeypatch, *every_ghost_class)
    VA._refuse_missing_node_packs({GHOST})


def test_an_engine_without_a_node_table_and_an_unknown_id_pass_through(monkeypatch):
    _install(monkeypatch)
    VA._refuse_missing_node_packs({"viz_mxc_cpu", "no_such_engine_xyz"})


def test_an_empty_node_registry_is_nothing_to_check(monkeypatch):
    """`node_class_mappings` returns {} with no ComfyUI registry rather than
    raising (agy QA): that must skip, not refuse every core class."""
    monkeypatch.setattr(WB, "node_class_mappings", lambda mapping=None: mapping or {})
    VA._refuse_missing_node_packs({GHOST})


def test_no_registry_at_all_is_nothing_to_check(monkeypatch):
    """The runtime-bridge tests fake the package tree; a gate that cannot reach
    the engine registry skips, it does not raise."""
    import builtins
    real = builtins.__import__

    def no_registry(name, *a, **k):
        if "_otr_video_engines" in name:
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_registry)
    VA._refuse_missing_node_packs({GHOST})


def test_the_gate_runs_before_any_download():
    src = inspect.getsource(VA.ensure_prompt_visual_assets)
    call = src.index('_refuse_missing_node_packs(plan["engines"])')
    assert call < src.index("native_requests(")
    assert call < src.index("_COVERED")


def test_ghost_signals_own_refusal_leads_with_the_fix():
    assert GS.GHOST_NODE_PACK_HINT.startswith("Install ComfyUI-AnimateDiff-Evolved")
    body = inspect.getsource(type(VREG.get_engine(GHOST)).assert_usable)
    assert "GHOST_NODE_PACK_HINT" in body
