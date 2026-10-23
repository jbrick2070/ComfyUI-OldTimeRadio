"""js/docs/<node>.md -- the help page ComfyUI shows for a node's ? button -- is
exactly what scripts/build_node_docs.py writes from config/node_docs.json and
the node's own definition, for every registered node and for nothing else.
Edit the config or the node, run the script, commit both; this fails otherwise.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "js" / "docs"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "otr_build_node_docs_parity", REPO / "scripts" / "build_node_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pages():
    return _generator().render_all()


def test_every_registered_node_has_its_committed_page_and_no_page_is_stale(pages):
    committed = {p.stem for p in DOCS.glob("*.md")}
    assert sorted(set(pages) - committed) == [], "pages the generator writes that are not committed"
    assert sorted(committed - set(pages)) == [], "committed pages for nodes that do not exist"


@pytest.mark.parametrize("key", sorted(json.loads(
    (REPO / "config" / "node_docs.json").read_text(encoding="utf-8")).keys() - {"_comment"}))
def test_the_committed_page_is_the_generators_output(pages, key):
    committed = (DOCS / f"{key}.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == pages[key], (
        f"js/docs/{key}.md differs from what scripts/build_node_docs.py writes. Edit "
        "config/node_docs.json or the node, run `python scripts/build_node_docs.py`, "
        "and commit both.")


def _shipped_guides() -> set:
    """The apple/ pages the registry zip carries: on disk and not named in .comfyignore."""
    ignored = {line.strip() for line in (REPO / ".comfyignore").read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")}
    return {p.name for p in (REPO / "apple").glob("*.md") if f"apple/{p.name}" not in ignored}


def test_every_guide_a_page_points_at_ships():
    config = json.loads((REPO / "config" / "node_docs.json").read_text(encoding="utf-8"))
    shipped = _shipped_guides()
    for key, entry in config.items():
        if key == "_comment":
            continue
        assert entry["guides"], key
        for guide in entry["guides"]:
            assert guide in shipped, f"{key} points at apple/{guide}, which does not ship"
        assert entry["intro"].strip() and entry.get("touch") in ("yes", "no"), key


def test_the_pages_carry_the_nodes_own_words(pages):
    """A page is not decoration: it names the node's inputs and its tooltips."""
    writer = pages["OTR_LedgerScriptWriter"]
    assert "**Story bank** (`source_bank`)" in writer, "the form's label, then the raw name"
    assert "## Inputs" in writer and "## Outputs" in writer and "## Read more" in writer
    mux = pages["OTR_MasterAudioMux"]
    assert "This is an output node" in mux
