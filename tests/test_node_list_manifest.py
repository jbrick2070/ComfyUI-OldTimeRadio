"""node_list.json -- the ONE static, machine-readable list of this pack's nodes.

WHY THIS FILE EXISTS, and it is not the reason the request came in. The operator
asked for a node list so the Comfy Registry would "recognize" the nodes. It will
not: `GET /nodes/<id>/comfy-nodes` returns 404 for EVERY pack sampled
(comfyui-old-time-radio, comfyui-kjnodes, rgthree-comfy, comfyui-dramabox), so an
empty node panel is universal and is not an OTR defect. What IS an OTR defect is
that this pack's node ids were not statically readable by anything: the loader
builds `NODE_CLASS_MAPPINGS` at runtime from `_NODE_MODULES` plus a merged table,
so an AST scanner keyed on `NODE_CLASS_MAPPINGS` extracts ZERO ids. Any external
extractor -- the Registry's, ComfyUI-Manager's, a future one -- sees nothing.

THE COUNT HAS BEEN WRONG IN THIS REPO THREE DIFFERENT WAYS, which is exactly why
the manifest is generated and then pinned by this test rather than typed:
  * a comment in `__init__.py` still said 34 (pre-lean-mean, retired nodes);
  * `/object_info` reports 29 OTR_* ids -- but 4 of those belong to a DIFFERENT
    pack, `ComfyUI-OTR-UpstreamStoryLab`, and are not ours to declare;
  * this pack declares and loads exactly 25.
A hand-typed list drifts silently. This test makes drift a red build.

THE VACUITY GUARD IS DELIBERATE. Comparing two sets that are both empty passes,
and that is precisely how the scene-coherence gate survived 55 episodes without a
producer. A floor is asserted so a parse failure cannot look like agreement.
"""
from __future__ import annotations

import ast
import io
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO, "node_list.json")
INIT = os.path.join(REPO, "__init__.py")

#: This pack declared 25 nodes when the manifest was pinned. The floor exists so
#: an empty-vs-empty comparison cannot pass; it is not an upper bound.
DECLARED_FLOOR = 20


def _literal_node_modules_keys() -> set:
    """The ids literal in ``__init__._NODE_MODULES`` -- read, not executed."""
    tree = ast.parse(io.open(INIT, encoding="utf-8").read())
    keys: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_NODE_MODULES" \
                    and isinstance(node.value, ast.Dict):
                keys.update(
                    k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str))
    return keys


def _merged_table_keys() -> set:
    """The ids merged in from the class registry, gated on the module existing
    on disk exactly as the loader gates them."""
    from nodes._otr_class_registry import new_node_modules_table

    merged = set()
    for key, value in new_node_modules_table().items():
        rel = value[0].lstrip(".").replace(".", os.sep) + ".py"
        if os.path.exists(os.path.join(REPO, rel)):
            merged.add(key)
    return merged


def _expected_ids() -> set:
    return _literal_node_modules_keys() | _merged_table_keys()


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert os.path.exists(MANIFEST), (
        "node_list.json is missing -- it is the only static list of this pack's "
        "node ids that an external extractor can read")
    return json.load(io.open(MANIFEST, encoding="utf-8"))


def test_the_manifest_is_a_flat_id_to_description_map(manifest):
    assert isinstance(manifest, dict)
    for key, value in manifest.items():
        assert isinstance(key, str) and isinstance(value, str), key
        assert value.strip(), f"{key} has an empty description"


def test_every_id_carries_the_public_prefix(manifest):
    stray = sorted(k for k in manifest if not k.startswith("OTR_"))
    assert not stray, f"non-OTR ids in the manifest: {stray}"


def test_the_manifest_is_not_vacuous(manifest):
    """A guard that passes on an empty file is not a guard."""
    assert len(manifest) >= DECLARED_FLOOR, (
        "node_list.json has only %d entries; this pack declares far more, so "
        "either the manifest was truncated or generation failed silently"
        % len(manifest))
    assert len(_expected_ids()) >= DECLARED_FLOOR, (
        "the DECLARATION side resolved to almost nothing -- the AST scan or the "
        "registry table import is broken, so any agreement below is meaningless")


def test_the_manifest_matches_what_the_pack_actually_declares(manifest):
    """THE RATCHET. Adding or retiring a node without regenerating the manifest
    is a red build, not a silent drift."""
    expected = _expected_ids()
    got = set(manifest)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    assert not missing and not extra, (
        "node_list.json is out of sync with the loader's declaration.\n"
        "  declared but absent from the manifest: %s\n"
        "  in the manifest but not declared:      %s\n"
        "Regenerate it from _NODE_MODULES rather than editing it by hand."
        % (missing or "none", extra or "none"))


def test_the_stale_node_count_is_gone_from_the_init_comment():
    """A comment claiming 34 nodes is how the wrong number kept propagating --
    it was quoted back as fact in a publishing plan."""
    text = io.open(INIT, encoding="utf-8").read()
    assert "34 nodes" not in text, (
        "__init__.py still claims 34 nodes; the pack declares %d"
        % len(_expected_ids()))
