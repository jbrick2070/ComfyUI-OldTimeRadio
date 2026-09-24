# -*- coding: utf-8 -*-
"""Every wired socket in every shipped graph resolves against the live classes.

WHY THIS EXISTS. CLAUDE.md section 0 names THREE checks after any graph change:
widget-count vs live INPUT_TYPES, link referential integrity, and **every wired
input-name present in INPUT_TYPES**. The first two have had tests for months
(`test_canonical_widget_input_parity.py`, `test_workflow_link_target_indexes.py`).
The third had never been run at all until it was done by hand on 2026-09-24,
during the voice-route deletion, and only because someone thought to ask.

A check that runs when a person remembers to ask is not a check. This is that
third one, plus the output side nobody had covered either, as something the
suite runs on every change.

WHAT IT CATCHES THAT THE SIBLINGS DO NOT. A link can carry a valid `dst_slot`
into a valid `inputs` array and still name a socket the class no longer
declares -- rename an input in `INPUT_TYPES` and every saved graph keeps the old
name. LiteGraph loads it happily, the widget count still matches, `dst_slot`
still resolves, and the node fails at EXECUTE time with an error that names the
value rather than the rename. The same goes for an output slot pointing past
`RETURN_TYPES` after a return value is removed.

IT READS EVERY SHIPPED GRAPH, not just the canonical one. The variants are
generated, so a defect in the generator corrupts all 24 identically and a
canonical-only check would see nothing wrong.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / "workflows"


def _load_mappings():
    """NODE_CLASS_MAPPINGS from the pack's own __init__.

    Loaded by FILE LOCATION: the package directory name carries a hyphen, so a
    plain import cannot reach it.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "_otr_socket_audit_init", REPO_ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_otr_socket_audit_init"] = mod
    spec.loader.exec_module(mod)
    return mod.NODE_CLASS_MAPPINGS


@pytest.fixture(scope="module")
def mappings():
    return _load_mappings()


def _workflows():
    """Every shipped graph, INCLUDING the variants in their subdirectory.

    `glob` rather than `rglob` was the first version of this and it found
    exactly ONE file -- the canonical -- because the 24 variants live under
    `workflows/variants/`. The audit reported clean having examined 4% of the
    graphs, and only the non-vacuity test at the bottom of this file caught it.
    """
    files = sorted(WORKFLOW_DIR.rglob("*.json"))
    assert files, "no workflows found; this test would pass by checking nothing"
    return files


def _declared_inputs(cls):
    """{name: declared_type} across required and optional."""
    spec = cls.INPUT_TYPES()
    out = {}
    for section in ("required", "optional"):
        for name, value in (spec.get(section) or {}).items():
            out[name] = value[0] if isinstance(value, tuple) and value else None
    return out


def _graphs():
    for path in _workflows():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 -- a broken graph is a finding
            pytest.fail("%s is not readable JSON: %s" % (path.name, exc))
        yield path, doc


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_wired_input_name_is_declared_by_its_class(path, mappings):
    """THE SECTION-0 CHECK THAT HAD NEVER RUN.

    An input descriptor naming a socket the class does not declare survives
    every structural check and dies at execute time.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for node in doc.get("nodes") or ():
        node_type = node.get("type")
        cls = mappings.get(node_type)
        if cls is None:
            continue                     # covered by its own test below
        declared = _declared_inputs(cls)
        for slot in (node.get("inputs") or ()):
            name = slot.get("name")
            if name not in declared:
                bad.append("node %s (%s): input %r is not in INPUT_TYPES"
                           % (node.get("id"), node_type, name))
    assert not bad, "%s:\n  %s" % (path.name, "\n  ".join(bad))


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_link_type_matches_the_socket_it_feeds(path, mappings):
    """A link's declared type agrees with the input it lands on.

    `*` is wild on either side and is accepted; anything else disagreeing means
    the graph would feed an engine the wrong kind of value.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    links = {l[0]: l for l in (doc.get("links") or ()) if l}
    bad = []
    for node in doc.get("nodes") or ():
        cls = mappings.get(node.get("type"))
        if cls is None:
            continue
        declared = _declared_inputs(cls)
        for slot in (node.get("inputs") or ()):
            link_id = slot.get("link")
            if link_id is None or link_id not in links:
                continue
            link = links[link_id]
            link_type = link[5] if len(link) > 5 else None
            want = declared.get(slot.get("name"))
            if (isinstance(want, str) and isinstance(link_type, str)
                    and want != link_type and "*" not in (want, link_type)):
                bad.append(
                    "node %s (%s): input %r declares %s but link %s carries %s"
                    % (node.get("id"), node.get("type"), slot.get("name"),
                       want, link_id, link_type))
    assert not bad, "%s:\n  %s" % (path.name, "\n  ".join(bad))


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_no_node_claims_more_outputs_than_its_class_returns(path, mappings):
    doc = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for node in doc.get("nodes") or ():
        cls = mappings.get(node.get("type"))
        if cls is None:
            continue
        returns = tuple(getattr(cls, "RETURN_TYPES", ()) or ())
        outputs = node.get("outputs") or ()
        if len(outputs) > len(returns):
            bad.append("node %s (%s): %d outputs but RETURN_TYPES has %d"
                       % (node.get("id"), node.get("type"),
                          len(outputs), len(returns)))
    assert not bad, "%s:\n  %s" % (path.name, "\n  ".join(bad))


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_link_leaves_a_slot_its_source_actually_has(path, mappings):
    """The mirror of the dst_slot corruption that broke 63 workflows in
    2026-08: an out-of-range `src_slot` reads a return value that is not there."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n.get("id"): n for n in (doc.get("nodes") or ())}
    bad = []
    for link in (doc.get("links") or ()):
        if not link or len(link) < 5:
            continue
        link_id, src_id, src_slot = link[0], link[1], link[2]
        src = nodes.get(src_id)
        if src is None:
            bad.append("link %s leaves node %s, which is not in the graph"
                       % (link_id, src_id))
            continue
        cls = mappings.get(src.get("type"))
        if cls is None:
            continue
        returns = tuple(getattr(cls, "RETURN_TYPES", ()) or ())
        if src_slot is None or src_slot < 0 or src_slot >= len(returns):
            bad.append("link %s takes slot %s of node %s (%s), which returns %d"
                       % (link_id, src_slot, src_id, src.get("type"),
                          len(returns)))
    assert not bad, "%s:\n  %s" % (path.name, "\n  ".join(bad))


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_no_link_dangles_and_no_endpoint_is_missing(path):
    """Orphans in both directions: an input naming a link that is not in the
    table, and a link naming a node that is not in the graph."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    links = {l[0]: l for l in (doc.get("links") or ()) if l}
    nodes = {n.get("id"): n for n in (doc.get("nodes") or ())}
    bad = []
    for node in doc.get("nodes") or ():
        for slot in (node.get("inputs") or ()):
            link_id = slot.get("link")
            if link_id is not None and link_id not in links:
                bad.append("node %s: input %r names link %s, which is not in "
                           "the links table"
                           % (node.get("id"), slot.get("name"), link_id))
    for link_id, link in links.items():
        if len(link) < 5:
            continue
        for role, node_id in (("source", link[1]), ("target", link[3])):
            if node_id not in nodes:
                bad.append("link %s names a %s node %s that is not in the graph"
                           % (link_id, role, node_id))
    assert not bad, "%s:\n  %s" % (path.name, "\n  ".join(bad))


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_OTR_node_type_in_the_graph_is_still_registered(path, mappings):
    """A graph naming a class the pack no longer registers is a node that fails
    to load for the user, with a message about a missing type rather than about
    the change that removed it.

    Scoped to OTR types: a graph may legitimately carry core ComfyUI nodes that
    this pack does not register.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted({
        str(n.get("type")) for n in (doc.get("nodes") or ())
        if str(n.get("type")).startswith("OTR_") and n.get("type") not in mappings
    })
    assert not missing, "%s names unregistered OTR nodes: %s" % (
        path.name, ", ".join(missing))


def test_the_audit_actually_reaches_the_shipped_graphs(mappings):
    """Guard against the whole file passing by checking nothing.

    Every assertion above is "no violations found", which is also what an empty
    scan reports. This pins that the scan has real material: the canonical
    exists, it carries OTR nodes, and those nodes resolve to live classes.
    """
    files = _workflows()
    assert len(files) >= 20, "expected the canonical plus its variants, got %d" % len(files)

    canonical = WORKFLOW_DIR / "otr_canonical.json"
    assert canonical.exists(), "the canonical workflow is missing"
    doc = json.loads(canonical.read_text(encoding="utf-8"))

    otr_nodes = [n for n in (doc.get("nodes") or ())
                 if str(n.get("type")).startswith("OTR_")]
    assert len(otr_nodes) >= 15, (
        "only %d OTR nodes in the canonical; the audit would be nearly empty"
        % len(otr_nodes))
    assert all(n.get("type") in mappings for n in otr_nodes)
    assert doc.get("links"), "the canonical has no links to audit"
