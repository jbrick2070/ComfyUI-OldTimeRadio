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


# ---------------------------------------------------------------------------
# C1b -- the SHIPPED VARIANTS had the identical defect (2026-07-27)
# ---------------------------------------------------------------------------
#
# C1 wired the canonical and stopped there. A verify-at-build item from the r4
# panel asked whether the variants carried the same node -- they did, all
# eleven of them, including the two 8GB tiers this ceiling exists FOR.
#
# The worst instance was ``variants/otr_8gb_wan.json``: its orphan trailing
# widget value was not the harmless 0 default but a REAL **17**, matching
# ``config/profiles/otr_8gb_wan.json`` -- the only shipped profile that pins
# ``max_render_frames`` at all. So the WAN 8GB launch contract's ceiling had
# been deliberately configured and silently ignored, because the value had no
# descriptor to arrive through. That is the precise failure
# ``test_floor_max_override_is_an_absolute_hard_cap`` was written after: an 8GB
# leg inheriting the 177-frame engine max and dying in the cost model.
#
# Guarding the canonical alone would have left that live.

VARIANTS_DIR = REPO_ROOT / "workflows" / "variants"


def _graph_workflows():
    """Every tracked workflow that is a real litegraph (has ``nodes``).

    The ``*.env.json`` siblings are API/env dumps with no ``nodes`` array and
    are correctly excluded rather than special-cased by filename.
    """
    out = []
    for path in sorted(list(VARIANTS_DIR.glob("*.json"))
                       + [CANONICAL]):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data.get("nodes"), list):
            out.append(path)
    return out


@pytest.mark.parametrize(
    "wf_path", _graph_workflows(), ids=lambda p: p.name)
def test_every_workflow_has_widget_input_parity(wf_path):
    """No shipped graph may carry a widget value with no descriptor."""
    with open(wf_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    offenders = []
    for node in data.get("nodes") or []:
        values = node.get("widgets_values")
        if not isinstance(values, list):
            continue
        winputs = [i for i in (node.get("inputs") or []) if i.get("widget")]
        if len(winputs) != len(values):
            offenders.append((node.get("id"), node.get("type"),
                              len(winputs), len(values)))
    assert not offenders, (
        "%s carries unwired widget value(s): %r -- each is a value the node "
        "class accepts and the graph stores with nothing connecting the two"
        % (wf_path.name, offenders))


@pytest.mark.parametrize(
    "wf_path",
    [p for p in _graph_workflows() if p.parent.name == "variants"],
    ids=lambda p: p.name)
def test_every_variant_director_carries_max_render_frames(wf_path):
    """THE MUTATION TARGET for C1b. Strip the descriptor from any variant and
    that variant fails BY NAME."""
    with open(wf_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    directors = [n for n in (data.get("nodes") or [])
                 if str(n.get("type") or "") == "OTR_VideoDirector"]
    if not directors:
        pytest.skip("%s has no OTR_VideoDirector" % wf_path.name)
    for node in directors:
        names = [i.get("name") for i in _widget_inputs(node)]
        assert "max_render_frames" in names, (
            "%s node %s has no max_render_frames descriptor -- this tier's "
            "render-length ceiling cannot reach the ledger. Descriptors: %r"
            % (wf_path.name, node.get("id"), names))
        assert names[-1] == "max_render_frames", (
            "%s: max_render_frames must stay LAST so widgets_values remains "
            "positionally stable (BUG-LOCAL-097); got %r"
            % (wf_path.name, names))


def test_the_wan_8gb_variant_still_carries_its_real_17_frame_ceiling():
    """The value, not just the descriptor.

    ``config/profiles/otr_8gb_wan.json`` is the ONLY shipped profile pinning
    ``max_render_frames``, and it pins 17. If this ever reads 0 again, someone
    'fixed' a parity failure by deleting the value instead of wiring it, and
    the 8GB WAN tier has silently lost its cap for the second time.
    """
    wf = VARIANTS_DIR / "otr_8gb_wan.json"
    with open(wf, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    node = next(n for n in data["nodes"]
                if str(n.get("type") or "") == "OTR_VideoDirector")
    names = [i.get("name") for i in _widget_inputs(node)]
    idx = names.index("max_render_frames")
    assert node["widgets_values"][idx] == 17, (
        "otr_8gb_wan.json should pin max_render_frames=17 to match "
        "config/profiles/otr_8gb_wan.json; got %r"
        % (node["widgets_values"][idx],))
