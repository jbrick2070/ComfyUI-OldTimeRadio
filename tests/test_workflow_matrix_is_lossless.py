# -*- coding: utf-8 -*-
"""The workflow matrix renders exactly what the 24 configs render.

WHAT THIS PROVES, AND WHAT IT DOES NOT. `build_variants.py --check`
regenerates the variants from their source and diffs against the committed
graphs. That proves the generator is deterministic; it cannot prove the source is
right, because "regenerate and commit" makes disk equal regeneration BY
CONSTRUCTION. Put a wrong-but-registered engine id in the source, regenerate,
commit both, and `--check` passes forever.

This compares the matrix against the 24 `config/profiles/*.json` files it
replaces and asserts they render the same graph. BE PRECISE ABOUT WHAT THAT BUYS:
the matrix was GENERATED from those files, so this is not two people
independently reaching the same answer. What it proves is that the CONSOLIDATION
was lossless -- that stripping 614 restated values changed no rendered output --
which is the thing that could plausibly have gone wrong and the thing worth a
test. It does NOT prove any value is CORRECT; a wrong engine id in a profile file
is faithfully carried into the matrix and passes here.

THAT PROPERTY HAS A SHELF LIFE, AND IT IS WORTH STATING PLAINLY. The matrix was
GENERATED from the configs once, during the migration. While both exist, this
test is a genuine cross-check of the consolidation. Once `config/profiles/` is
deleted it would be comparing the matrix to itself, and it must be removed in
that same change rather than left behind looking like it still proves something.

WHAT REPLACES IT IS ALREADY ON DISK. `workflows/variants/` holds the 24 graphs
GENERATED FROM THE CONFIGS. So after the source is switched to the matrix,
`--check` passing against those committed graphs is itself a non-circular proof
-- the oracle was produced by the source being replaced. That only holds if the
variants are NOT regenerated during the switchover: running `--all` before
`--check` overwrites the oracle with the very output under test and silently
converts the proof into a tautology.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO / "config" / "workflow_matrix.json"

for _p in (REPO, REPO / "scripts", REPO / "nodes"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture(scope="module")
def matrix():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def modules():
    import build_variants as bv
    from _otr_shared import capability_profiles as cp
    from nodes import _otr_workflow_apply as wa
    return bv, cp, wa


@pytest.fixture(scope="module")
def canonical():
    return json.loads(
        (REPO / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))


def _unflatten(pairs):
    """{'llm.device': 'cuda'} -> {'llm': {'device': 'cuda'}}"""
    out = {}
    for dotted, value in pairs.items():
        parts = dotted.split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _row_document(row):
    """The config a matrix row stands for: identity, metadata, deltas."""
    doc = {"id": row["id"]}
    for key in ("display_name", "status", "platform", "device_backend",
                "gpu_vendor", "allow_sidecars", "toolchains"):
        if key in row:
            doc[key] = row[key]
    doc.update(_unflatten(row.get("deltas") or {}))
    return doc


def _rows(matrix):
    return {r["id"]: r for r in matrix["rows"]}


def test_every_shipping_id_has_exactly_one_row(matrix, modules):
    """No workflow is lost in the consolidation, and none is invented."""
    bv, _, _ = modules
    shipping = set(bv.SHIPPING_SET)
    ships = {r["id"] for r in matrix["rows"] if r.get("ships")}
    assert ships == shipping, (
        "matrix-only: %s\nSHIPPING_SET-only: %s"
        % (sorted(ships - shipping), sorted(shipping - ships)))


@pytest.mark.parametrize("pid", [r["id"] for r in json.loads(
    MATRIX_PATH.read_text(encoding="utf-8"))["rows"]])
def test_the_row_renders_what_the_config_renders(pid, matrix, modules, canonical):
    """THE LOSSLESS PROOF, one row at a time so a failure names the workflow.

    A row carries deltas, so it will never equal its 39-key config as a dict and
    that comparison would mean nothing anyway. What has to survive consolidation
    is what RENDERS.
    """
    _, cp, wa = modules
    logging.disable(logging.CRITICAL)
    try:
        from_config = wa.apply_profile(canonical, cp.load_profile(pid))
        from_row = wa.apply_profile(canonical, _row_document(_rows(matrix)[pid]))
    finally:
        logging.disable(logging.NOTSET)
    assert from_row == from_config, (
        "row %s renders a different graph than config/profiles/%s.json -- the "
        "consolidation dropped or changed something that reaches a widget" % (pid, pid))


@pytest.mark.parametrize("pid", [r["id"] for r in json.loads(
    MATRIX_PATH.read_text(encoding="utf-8"))["rows"]])
def test_no_delta_merely_restates_the_canonical(pid, matrix, modules, canonical):
    """THE ANTI-DRIFT PROPERTY, MECHANIZED.

    A stated value equal to the canonical's is a fork point: it keeps its value
    when the canonical moves. That is how 82 configs went on pinning a voice
    engine the canonical had already left. A key the matrix does not mention
    follows the canonical forever, so the rule is that a delta must actually
    change something.

    Checked per row, by removing one delta and rendering: if the graph is
    unchanged, that delta was not doing anything and belongs deleted.
    """
    _, _, wa = modules
    row = _rows(matrix)[pid]
    deltas = row.get("deltas") or {}
    logging.disable(logging.CRITICAL)
    try:
        full = wa.apply_profile(canonical, _row_document(row))
        inert = []
        for dotted in sorted(deltas):
            trimmed = dict(row)
            trimmed["deltas"] = {k: v for k, v in deltas.items() if k != dotted}
            if wa.apply_profile(canonical, _row_document(trimmed)) == full:
                inert.append(dotted)
    finally:
        logging.disable(logging.NOTSET)
    assert not inert, (
        "row %s states %d value(s) that change nothing: %s\nDelete them -- an "
        "inert pin is a fork point that silently keeps its value when the "
        "canonical moves." % (pid, len(inert), ", ".join(inert)))


def test_the_matrix_actually_has_material(matrix):
    """Guard against the file passing by being nearly empty.

    Every assertion above is per-row and parametrized, so an empty `rows` list
    collects zero tests and the file reports success having checked nothing --
    the same shape as the socket audit that passed while reading 1 graph of 25.
    """
    rows = matrix["rows"]
    assert len(rows) >= 20, "expected the full shipped set, got %d" % len(rows)
    assert sum(len(r.get("deltas") or {}) for r in rows) >= 200, (
        "the rows carry almost no deltas; the matrix is not describing the "
        "differences it exists to describe")
    assert all(r.get("id") for r in rows), "a row without an id names no workflow"

def _omitted_probe_pairs():
    """(row, key) for every row crossed with the probe keys it does NOT state.

    Two keys of different kinds: `llm.creative_model` is COMBO-label transformed on
    write, `render.fps` is a plain int no row states. Built from the matrix at
    collection time, so nothing skips and every row is covered.
    """
    doc = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    pairs = []
    for row in doc["rows"]:
        for key in ("llm.creative_model", "render.fps"):
            if key not in (row.get("deltas") or {}):
                pairs.append((row["id"], key))
    return pairs


@pytest.mark.parametrize("pid,probe_key", _omitted_probe_pairs(),
                         ids=lambda v: str(v))
def test_a_key_a_row_omits_follows_the_canonical(pid, probe_key, matrix, modules,
                                                 canonical):
    """THE PROPERTY THE WHOLE DESIGN RESTS ON, AND THE ONE THAT WAS UNTESTED.

    Every other test here checks that a STATED delta is not inert. None checked the
    other direction -- that a key a row does NOT state actually tracks the
    canonical. That gap let a real defect through: a `defaults.values` block was
    merged under the deltas and therefore APPLIED, so all 36 omitted keys were
    re-pinned from a stale baseline on every emit. With the canonical's writer moved
    to gemma, `otr_8gb_low` still rendered Qwen.

    THE ENTIRE SUITE PASSED, INCLUDING `--check`, because the regenerated graph and
    the committed graph were stale in the same way -- which is exactly why
    byte-identical regeneration cannot be the only proof.

    Moves the canonical IN MEMORY (never on disk) and asserts the row follows it.
    """
    _, cp, wa = modules
    assert probe_key not in (_rows(matrix)[pid].get("deltas") or {})

    # The node and widget this key drives, from the mapping rather than hardcoded,
    # so the test survives a key being re-pointed at a different node.
    mapping = wa.load_widget_mapping()
    node_type, widget = mapping["managed"][probe_key]["targets"][0]
    schemas = wa.build_offline_schemas()
    slot = wa.serialized_slot_names(node_type, schemas).index(widget)

    # A sentinel of the right TYPE: writing a string into an int widget would be
    # refused by the applier's own validation and the test would fail for the wrong
    # reason.
    original = next(n for n in canonical["nodes"]
                    if n.get("type") == node_type)["widgets_values"][slot]
    sentinel = (original + 1) if isinstance(original, int) and not isinstance(
        original, bool) else "google/gemma-4-12b-it"

    moved = copy.deepcopy(canonical)
    for node in moved.get("nodes") or ():
        if node.get("type") == node_type:
            node["widgets_values"][slot] = sentinel

    logging.disable(logging.CRITICAL)
    try:
        rendered = wa.apply_profile(moved, cp.load_profile(pid))
    finally:
        logging.disable(logging.NOTSET)
    got = next(n for n in rendered["nodes"]
               if n.get("type") == node_type)["widgets_values"][slot]

    assert got == sentinel, (
        "row %s does not state %s, so it must inherit whatever the canonical says. "
        "The canonical was moved to %r and the row rendered %r instead -- something "
        "is re-pinning an omitted key from a stored copy."
        % (pid, probe_key, sentinel, got))
