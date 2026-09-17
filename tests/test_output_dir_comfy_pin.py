"""The boot pin must honor ComfyUI's live output directory.

Desktop can load this pack through a custom_nodes junction under
ComfyUI-Installs while --output-directory points at the operator's
Documents tree. Walking up from __file__ then publishes to the install
output and misses <output>/otr/obs. The pin must call
folder_paths.get_output_directory() and must not reconstruct output/
from this pack's load path.

Every shipping graph, including canonical, publishes through
OTR_MasterAudioMux with an empty output_path widget so the mux uses
that same <output>/otr/obs root. A hardcoded Documents or Installs
path in any JSON would re-split the trees for that graph only.

Pure CPU. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_paths as P
from nodes._otr_shared import env as otr_env

REPO = Path(__file__).resolve().parent.parent
INIT = REPO / "__init__.py"
CANONICAL = REPO / "workflows" / "otr_canonical.json"
VARIANTS = REPO / "workflows" / "variants"
_FORBIDDEN_OBS = (
    "ComfyUI-Installs",
    r"Documents\ComfyUI\output",
    r"Documents/ComfyUI/output",
)


def _shipping_graphs():
    graphs = [CANONICAL]
    graphs.extend(sorted(VARIANTS.glob("otr_*.json")))
    return [p for p in graphs if p.is_file() and ".env." not in p.name]


def _mux_nodes(graph):
    return [n for n in graph.get("nodes") or [] if n.get("type") == "OTR_MasterAudioMux"]


def _widget_value(node, name):
    names = []
    for inp in node.get("inputs") or []:
        widget = inp.get("widget")
        if isinstance(widget, dict) and widget.get("name"):
            names.append(str(widget["name"]))
    values = node.get("widgets_values") or []
    if name not in names:
        return None
    idx = names.index(name)
    if idx >= len(values):
        return None
    return values[idx]


@pytest.fixture()
def clean_output_pin(monkeypatch):
    monkeypatch.delenv("OTR_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("OTR_OBS_DIR", raising=False)
    yield
    otr_env.unpin("OTR_OUTPUT_DIR")


def test_pin_uses_folder_paths_not_pack_walkup(clean_output_pin, tmp_path, monkeypatch):
    remapped = tmp_path / "desktop-output"
    remapped.mkdir()
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(remapped))
    pinned = P.pin_output_dir_from_comfy()
    assert pinned is not None
    assert Path(pinned).resolve() == remapped.resolve()
    assert Path(otr_env.get("OTR_OUTPUT_DIR")).resolve() == remapped.resolve()
    assert P.comfy_output_dir().resolve() == remapped.resolve()
    assert P.otr_obs_dir().resolve() == (remapped / "otr" / "obs").resolve()


def test_pin_skips_when_operator_already_set(clean_output_pin, tmp_path, monkeypatch):
    already = tmp_path / "launcher-output"
    already.mkdir()
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(already))
    import folder_paths
    monkeypatch.setattr(
        folder_paths, "get_output_directory", lambda: str(tmp_path / "should-not-win")
    )
    assert P.pin_output_dir_from_comfy() is None
    assert Path(otr_env.get("OTR_OUTPUT_DIR")).resolve() == already.resolve()


def test_pin_skips_empty_folder_paths(clean_output_pin, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: "")
    assert P.pin_output_dir_from_comfy() is None
    assert not (otr_env.get("OTR_OUTPUT_DIR") or "").strip()


def test_init_wires_the_comfy_pin_and_does_not_walk_up_from_file():
    src = INIT.read_text(encoding="utf-8")
    start = src.index("OTR output base pin")
    end = src.index("ISOLATED PER-NODE LOADING")
    block = src[start:end]
    assert "pin_output_dir_from_comfy" in block
    assert "OTR_OUTPUT_DIR pinned (ComfyUI output)" in src
    assert "abspath(__file__)" not in block
    assert "node-relative" not in block


def test_every_shipping_graph_muxes_to_empty_output_path():
    graphs = _shipping_graphs()
    assert len(graphs) >= 18, graphs
    missing = []
    hardcoded = []
    for path in graphs:
        graph = json.loads(path.read_text(encoding="utf-8"))
        muxes = _mux_nodes(graph)
        rel = str(path.relative_to(REPO)).replace("\\", "/")
        if not muxes:
            missing.append(rel)
            continue
        for node in muxes:
            out = _widget_value(node, "output_path")
            if out not in ("", None):
                hardcoded.append("%s output_path=%r" % (rel, out))
            blob = json.dumps(node.get("widgets_values") or [])
            if any(token in blob for token in _FORBIDDEN_OBS):
                hardcoded.append("%s widgets=%s" % (rel, blob))
    assert missing == [], "no OTR_MasterAudioMux in: " + ", ".join(missing)
    assert hardcoded == [], "hardcoded mux dest: " + "; ".join(hardcoded)


def test_canonical_is_in_the_shipping_graph_walk():
    assert CANONICAL in _shipping_graphs()
    assert CANONICAL.is_file()
