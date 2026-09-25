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


def test_every_shipping_row_has_a_committed_variant(matrix):
    """No workflow is lost in the consolidation, and none is invented.

    COMPARED AGAINST THE GRAPHS ON DISK, not against `build_variants.SHIPPING_SET`.
    That constant is `shipping_ids()`, which reads this same file -- so the earlier
    version of this test compared the matrix's `ships` set to itself and would have
    passed with any set at all. The variant filenames are an independent artifact:
    they exist because something emitted them.
    """
    ships = {r["id"] for r in matrix["rows"] if r.get("ships")}
    variants = {p.stem for p in (REPO / "workflows" / "variants").glob("otr_*.json")
                if not p.name.endswith(".env.json")}
    assert ships, "the matrix marks no row as shipping"
    assert ships == variants, (
        "rows marked `ships` with no committed graph: %s\n"
        "committed graphs with no shipping row: %s"
        % (sorted(ships - variants), sorted(variants - ships)))


def _key_indicators(matrix):
    """The keys every row must state, from the matrix's own declaration."""
    return tuple(matrix.get("key_indicators") or ())


@pytest.mark.parametrize("pid", [r["id"] for r in json.loads(
    MATRIX_PATH.read_text(encoding="utf-8"))["rows"]])
def test_every_row_states_every_key_indicator(pid, matrix):
    """AN INDICATOR MUST NOT GO QUIETLY MISSING.

    A row that drops one silently starts INHERITING it -- which is the precise
    failure the indicator list exists to prevent, arriving by omission rather than
    by decision. It matters most where a machine requires the value: an 8 GB row
    that stops stating its writer follows the canonical, and a canonical that moves
    to a 12B then puts a model on that card which cannot fit.
    """
    declared = _key_indicators(matrix)
    assert declared, "the matrix declares no key_indicators; nothing is guarded"
    stated = set(_rows(matrix)[pid].get("deltas") or {})
    missing = [k for k in declared if k not in stated]
    assert not missing, (
        "row %s does not state %d key indicator(s): %s\nAn unstated indicator is "
        "inherited from the canonical, which is exactly what stating it prevents."
        % (pid, len(missing), ", ".join(missing)))


@pytest.mark.parametrize("pid", [r["id"] for r in json.loads(
    MATRIX_PATH.read_text(encoding="utf-8"))["rows"]])
def test_no_incidental_value_merely_restates_the_canonical(pid, matrix, modules,
                                                           canonical):
    """THE ANTI-DRIFT PROPERTY, NARROWED TO WHERE IT STILL APPLIES.

    A value stated for no reason and equal to the canonical's is a fork point: it
    keeps its old value when the canonical moves. That is how 82 configs went on
    pinning a voice engine the canonical had already left, and it stays an error.

    KEY INDICATORS ARE EXEMPT, and the exemption is the point rather than a hole.
    Those are stated deliberately -- the writer, the ceiling, the lanes, the device --
    because they define the workflow and because a machine can REQUIRE one. An 8 GB
    card needs the small writer whatever the canonical picks, so that pin must
    survive a canonical that moves upward. What makes an explicit pin safe here is
    that all 25 rows sit in one column of one file; the 82-file drift was invisible
    because it was spread across 82 files, not because it was pinned.

    So the rule is narrower, not weaker: anything NOT declared an indicator must
    actually change something.
    """
    _, _, wa = modules
    row = _rows(matrix)[pid]
    deltas = row.get("deltas") or {}
    indicators = set(_key_indicators(matrix))
    candidates = [k for k in sorted(deltas) if k not in indicators]

    logging.disable(logging.CRITICAL)
    try:
        full = wa.apply_profile(canonical, _row_document(row))
        inert = []
        for dotted in candidates:
            trimmed = dict(row)
            trimmed["deltas"] = {k: v for k, v in deltas.items() if k != dotted}
            if wa.apply_profile(canonical, _row_document(trimmed)) == full:
                inert.append(dotted)
    finally:
        logging.disable(logging.NOTSET)
    assert not inert, (
        "row %s states %d non-indicator value(s) that change nothing: %s\nDelete "
        "them -- an incidental pin is a fork point that silently keeps its value "
        "when the canonical moves. If one of these is actually a decision, declare "
        "it in `key_indicators` instead of leaving it to look accidental."
        % (pid, len(inert), ", ".join(inert)))


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

@pytest.mark.parametrize("pid", [r["id"] for r in json.loads(
    MATRIX_PATH.read_text(encoding="utf-8"))["rows"]])
def test_resolved_values_are_already_in_profile_form(pid, modules, canonical):
    """ONE SPELLING OUT OF `resolved_profile`, WHICHEVER BRANCH ANSWERED.

    It has two: a value the row STATES, and a value inherited from the canonical.
    They must agree on spelling, or a caller gets `ltx25_high_video` or
    `ltx25_video` for the same engine depending on something it cannot see. That was
    a real defect, and every test passed through it because every engine key is a key
    indicator -- so every row states one, and the inherited branch never fired.

    Rather than contriving a row that inherits an engine, this pins a property of the
    ANSWER: the output is already in profile form, so projecting it again is a no-op.
    A branch that skips the projection fails here for any value the projection would
    have changed, with no special fixture needed.
    """
    _, _, wa = modules
    logging.disable(logging.CRITICAL)
    try:
        resolved = wa.resolved_profile(pid, canonical)
        flat = wa._flatten_profile_values(resolved)
    finally:
        logging.disable(logging.NOTSET)

    drifting = {k: (v, wa._to_profile_form(v)) for k, v in flat.items()
                if wa._to_profile_form(v) != v}
    assert not drifting, (
        "row %s resolves %d value(s) that are not in profile form -- applying the "
        "projection again changes them, which means one branch of resolved_profile "
        "skipped it and the two branches disagree on spelling:\n  %s"
        % (pid, len(drifting),
           "\n  ".join("%s: %r -> %r" % (k, a, b) for k, (a, b) in sorted(drifting.items()))))


def test_the_profile_form_projection_is_idempotent_and_keeps_public_spellings():
    """`_to_profile_form` strips a COMBO label and nothing else.

    It deliberately does NOT apply `resolve_engine_id`, which maps public spellings to
    internal ones. Public is what rows state and what a person picks in the dropdown,
    so keeping it is what lets the generated docs read like the menu. The consumer
    that genuinely wants internal ids resolves them itself.
    """
    from nodes import _otr_workflow_apply as wa

    # A stored COMBO label loses its parenthetical; the bare value is untouched.
    assert wa._to_profile_form(
        "Qwen/Qwen3.5-4B (8.7 GB download, mac16-tight nv8-nf4 nv16 nv24)"
    ) == "Qwen/Qwen3.5-4B"
    assert wa._to_profile_form("Qwen/Qwen3.5-4B") == "Qwen/Qwen3.5-4B"

    # THE PUBLIC ALIAS SURVIVES. `resolve_engine_id` would make this ltx25_video.
    assert wa._to_profile_form("ltx25_high_video") == "ltx25_high_video"

    # Idempotent, and non-strings pass straight through.
    for value in ("viz_camera", "kokoro", 25, 6.8, True, None, ["a"]):
        assert wa._to_profile_form(value) == value
        assert wa._to_profile_form(wa._to_profile_form(value)) == wa._to_profile_form(value)
