"""Every canonical widget VALUE must have a matching input DESCRIPTOR (7b C1).

THE DEFECT THIS GUARDS (2026-07-27). ``OTR_VideoDirector`` gained an optional
``max_render_frames`` widget (``otr_video_director.py:302-318``), and node 87 in
``workflows/otr_canonical.json`` duly carried its default as the 15th entry of
``widgets_values`` -- but no corresponding entry was ever added to that node's
``inputs`` array, which stopped at ``dtype_policy``. The value sat there
unbound.

Why that mattered rather than being cosmetic: the WAN 8GB launch contract
routes its render-length ceiling profile -> director widget -> ledger
``video.max_render_frames`` -> ``build_episode_render_policy`` -> ``prepare`` ->
``motion_common.profile_max_render_frames()``. With no input descriptor the
first hop is dead, so the entire profile-carried ceiling channel was
unreachable from the real workflow -- the "unwired code is dead code" case
CLAUDE.md section 0 exists to prevent, and it had already shipped.

The guard is deliberately REPO-WIDE rather than node-87-specific: the defect
class is "a widget was added to a node class and the canonical was not
re-wired", which can recur on any node. Verified at the time of writing that
all 23 canonical nodes carrying ``widgets_values`` satisfy it, so this asserts
a property that currently holds everywhere, not a special case.

Note this is parity of COUNT plus name-reachability, not positional value
checking. ``widgets_values`` is POSITIONAL (BUG-LOCAL-097) and its VALUES are
the operator's to set; what must never drift is the descriptor list.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "workflows" / "otr_canonical.json"


def _canonical():
    with open(CANONICAL, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _widget_inputs(node):
    return [i for i in (node.get("inputs") or []) if i.get("widget")]


def _nodes_with_widget_values():
    out = []
    for node in _canonical().get("nodes") or []:
        wv = node.get("widgets_values")
        if isinstance(wv, list):
            out.append(node)
    return out


def test_the_canonical_actually_parses_and_has_nodes():
    data = _canonical()
    assert data.get("nodes"), "canonical has no nodes"
    assert len(_nodes_with_widget_values()) >= 20


@pytest.mark.parametrize(
    "node_id",
    [n.get("id") for n in _nodes_with_widget_values()],
)
def test_every_widget_value_has_an_input_descriptor(node_id):
    """THE MUTATION TARGET. Delete node 87's ``max_render_frames`` input
    descriptor and this must fail for node 87 -- naming the node and both
    counts, not just 'the workflow changed'."""
    node = next(n for n in _canonical()["nodes"] if n.get("id") == node_id)
    winputs = _widget_inputs(node)
    values = node.get("widgets_values") or []
    assert len(winputs) == len(values), (
        "node %s (%s) declares %d widget input descriptor(s) but carries %d "
        "widget value(s) -- a value with no descriptor is an UNWIRED widget: "
        "the node class accepts it, the saved graph stores it, and nothing "
        "connects the two. Descriptor names: %r"
        % (node_id, node.get("type"), len(winputs), len(values),
           [i.get("name") for i in winputs]))


def test_node_87_carries_max_render_frames_specifically():
    """The concrete case, pinned by name.

    The parametrized test above would also pass if someone 'fixed' the count by
    DELETING the trailing widget value instead of adding the descriptor. That
    would silently remove the 8GB tier's ceiling control, so the name is pinned
    here separately from the count.
    """
    node = next(n for n in _canonical()["nodes"] if n.get("id") == 87)
    assert node.get("type") == "OTR_VideoDirector"
    names = [i.get("name") for i in _widget_inputs(node)]
    assert "max_render_frames" in names, (
        "node 87 lost its max_render_frames input descriptor -- the "
        "profile-carried render-length ceiling (the WAN 8GB launch contract) "
        "has no path from the director widget to the ledger. Descriptors: %r"
        % (names,))
    assert names[-1] == "max_render_frames", (
        "max_render_frames must remain LAST so widgets_values stays "
        "positionally stable (BUG-LOCAL-097); got %r" % (names,))


def test_the_descriptor_matches_what_the_node_class_declares():
    """The descriptor must describe the REAL widget, not a plausible one.

    A descriptor whose type or optionality disagrees with ``INPUT_TYPES`` is
    worse than none: the graph looks wired and behaves unpredictably.
    """
    from nodes.otr_video_director import OTRVideoDirector

    spec = OTRVideoDirector.INPUT_TYPES()
    optional = spec.get("optional") or {}
    assert "max_render_frames" in optional, (
        "the node class no longer declares max_render_frames as optional; the "
        "canonical descriptor is now describing something that does not exist")
    declared_type = optional["max_render_frames"][0]

    node = next(n for n in _canonical()["nodes"] if n.get("id") == 87)
    desc = next(i for i in _widget_inputs(node)
                if i.get("name") == "max_render_frames")
    assert desc.get("type") == declared_type, (
        "canonical says type=%r, the node class says %r"
        % (desc.get("type"), declared_type))
    assert desc.get("shape") == 7, (
        "an OPTIONAL input carries shape 7 in this schema (matching "
        "device_policy / dtype_policy); got %r" % (desc.get("shape"),))
    assert desc.get("widget", {}).get("name") == "max_render_frames"
    assert desc.get("link") is None, "the widget must not arrive pre-linked"
