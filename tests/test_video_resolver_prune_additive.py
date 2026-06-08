"""Additive CPU coverage for the execution-group DAG + prune resolver (AS-2).

New, self-contained tests for nodes/_otr_shared/resolver.py: deterministic
topological order, fail-closed structural errors (cycle / dup id / self-edge /
dangling dep / empty id), and the conditional-skip RESOLVER-PRUNE that drops a
degraded consumer and cascades to the now-orphaned provider (the character_3d ->
humo background case), while never auto-pruning a still-consumed provider. Pure
stdlib + the module under test. UTF-8, no BOM, ASCII, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import resolver as rs


def _g(gid, kind="consumer", depends_on=(), produces_base_for=()):
    return {
        "group_id": gid,
        "kind": kind,
        "engine_id": "e_" + gid,
        "profile_id": "p_" + gid,
        "depends_on": list(depends_on),
        "produces_base_for": list(produces_base_for),
    }


def test_topological_order_linear():
    groups = [_g("c", depends_on=("b",)), _g("b", depends_on=("a",)), _g("a")]
    assert rs.topological_order(groups) == ["a", "b", "c"]


def test_topological_order_is_deterministic_sorted_roots():
    groups = [_g("z"), _g("a"), _g("m")]
    assert rs.topological_order(groups) == ["a", "m", "z"]


def test_cycle_raises():
    groups = [_g("a", depends_on=("b",)), _g("b", depends_on=("a",))]
    with pytest.raises(rs.ResolverError):
        rs.topological_order(groups)


def test_duplicate_group_id_raises():
    with pytest.raises(rs.ResolverError):
        rs.topological_order([_g("a"), _g("a")])


def test_self_edge_raises():
    with pytest.raises(rs.ResolverError):
        rs.topological_order([_g("a", depends_on=("a",))])


def test_dangling_dependency_raises():
    with pytest.raises(rs.ResolverError):
        rs.topological_order([_g("a", depends_on=("ghost",))])


def test_empty_group_id_raises():
    with pytest.raises(rs.ResolverError):
        rs.topological_order([_g("")])


def test_validate_passes_clean_provider_consumer():
    groups = [
        _g("bg", kind="provider", produces_base_for=("fg",)),
        _g("fg", kind="consumer", depends_on=("bg",)),
    ]
    out = rs.validate_execution_groups(groups)
    assert [g["group_id"] for g in out] == ["bg", "fg"]


def test_validate_rejects_produces_base_for_unknown():
    groups = [_g("bg", kind="provider", produces_base_for=("ghost",))]
    with pytest.raises(rs.ResolverError):
        rs.validate_execution_groups(groups)


def test_prune_consumer_cascades_to_orphaned_provider():
    # character_3d (consumer) over an ltx background (provider): prune the
    # consumer -> the background it fed is orphaned -> dropped too.
    groups = [
        _g("bg", kind="provider", produces_base_for=("char",)),
        _g("char", kind="consumer", depends_on=("bg",)),
    ]
    assert rs.prune_orphaned_groups(groups, ["char"]) == []


def test_prune_keeps_provider_still_consumed_by_another():
    groups = [
        _g("bg", kind="provider"),
        _g("c1", kind="consumer", depends_on=("bg",)),
        _g("c2", kind="consumer", depends_on=("bg",)),
    ]
    ids = [g["group_id"] for g in rs.prune_orphaned_groups(groups, ["c1"])]
    assert "bg" in ids and "c2" in ids and "c1" not in ids


def test_prune_never_auto_removes_a_consumer():
    groups = [_g("solo", kind="consumer")]
    assert [g["group_id"] for g in rs.prune_orphaned_groups(groups, [])] == [
        "solo"
    ]


def test_prune_is_order_preserving_and_pure():
    groups = [
        _g("a", kind="provider", produces_base_for=("b",)),
        _g("b", kind="consumer", depends_on=("a",)),
        _g("c", kind="consumer"),
    ]
    snapshot = [dict(g) for g in groups]
    out = rs.prune_orphaned_groups(groups, ["b"])
    # 'a' orphaned -> gone; 'c' independent consumer stays; order preserved.
    assert [g["group_id"] for g in out] == ["c"]
    assert groups == snapshot  # input list not mutated


def test_validate_with_prune_orphans_flag():
    groups = [
        _g("bg", kind="provider", produces_base_for=("char",)),
        _g("char", kind="consumer", depends_on=("bg",)),
        _g("keep", kind="consumer"),
    ]
    out = rs.validate_execution_groups(
        groups, prune_orphans=True, pruned_group_ids=["char"]
    )
    assert [g["group_id"] for g in out] == ["keep"]
