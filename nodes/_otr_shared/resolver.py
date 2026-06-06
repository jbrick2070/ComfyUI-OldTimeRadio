"""Execution-group DAG validation + conditional-skip RESOLVER-PRUNE (AS-2).

``OTR_ShotLock`` stamps ordered ``execution_groups`` into ``ledger['video']``.
A runtime fallback can DEGRADE a consumer to a different family -- e.g. a
``character_3d`` (alpha character over a separately-rendered background) shot
degrades to ``humo`` (an opaque, full-frame face). When that happens the
separate background-provider group (e.g. an ``ltx_video`` group rendered ONLY to
sit behind the 3D alpha) is ORPHANED: nothing consumes it any more.

AS-2's decision: handle that conditional-skip as a RESOLVER-PRUNE -- physically
remove the orphaned group from the plan and re-stamp -- NOT as a "skip edge"
left dangling in the DAG. A skip-edge keeps a dead provider in the topology
(wasted render + a confusing ledger that lists a group whose output is never
used); pruning keeps the plan honest: the ledger only ever lists groups that
actually run.

This module is dependency-free (stdlib only) and pure -- it never touches the
audio ledger, GPU, or disk. ``OTR_ShotLock`` calls
:func:`validate_execution_groups` at lock; the in-render resolver calls
:func:`prune_orphaned_groups` on a fallback-restamp.

Each execution group is a dict::

    {"group_id": str, "kind": "provider"|"consumer",
     "engine_id": str, "profile_id": str,
     "depends_on": [group_id, ...], "produces_base_for": [group_id, ...]}

``depends_on`` is the only edge source of truth for ordering; ``produces_base_for``
is provider-side bookkeeping (a provider lists the consumers it feeds).
"""
from __future__ import annotations

from typing import Iterable


class ResolverError(ValueError):
    """Raised for a structurally invalid execution-group set (cycle, dangling
    dependency, duplicate id, self-edge, or provider-after-consumer)."""


def _as_list(groups) -> list:
    return list(groups or [])


def _group_ids(groups) -> list:
    return [g.get("group_id") for g in groups]


def _validate_structure(groups) -> None:
    """Unique ids, every ``depends_on`` resolves, no self-edge."""
    ids = _group_ids(groups)
    if any(gid is None or gid == "" for gid in ids):
        raise ResolverError("every execution group needs a non-empty group_id")
    if len(ids) != len(set(ids)):
        dupes = sorted({gid for gid in ids if ids.count(gid) > 1})
        raise ResolverError(f"duplicate group_id(s): {dupes}")
    id_set = set(ids)
    for g in groups:
        gid = g.get("group_id")
        for dep in g.get("depends_on", []) or []:
            if dep == gid:
                raise ResolverError(f"group '{gid}' depends on itself")
            if dep not in id_set:
                raise ResolverError(
                    f"group '{gid}' depends on unknown group '{dep}'"
                )


def topological_order(groups) -> list:
    """Return ``group_id``s in dependency order (Kahn); raise on a cycle.

    Deterministic: ready nodes are popped in sorted order so the same input
    always yields the same order (stable plan -> stable cache keys).
    """
    groups = _as_list(groups)
    _validate_structure(groups)
    by_id = {g["group_id"]: g for g in groups}
    indeg = {gid: 0 for gid in by_id}
    for g in groups:
        for dep in g.get("depends_on", []) or []:
            # edge dep -> g: g depends on dep, so g's in-degree counts dep.
            indeg[g["group_id"]] += 1
    ready = sorted(gid for gid, d in indeg.items() if d == 0)
    order = []
    while ready:
        gid = ready.pop(0)
        order.append(gid)
        for other in groups:
            if gid in (other.get("depends_on", []) or []):
                oid = other["group_id"]
                indeg[oid] -= 1
                if indeg[oid] == 0:
                    ready.append(oid)
        ready.sort()
    if len(order) != len(by_id):
        remaining = sorted(set(by_id) - set(order))
        raise ResolverError(
            f"execution groups contain a cycle (unresolvable: {remaining})"
        )
    return order


def _assert_provider_before_consumer(groups) -> None:
    """A provider must be ordered strictly before every consumer that
    depends on it (kills the overlay double-load)."""
    order = topological_order(groups)
    pos = {gid: i for i, gid in enumerate(order)}
    by_id = {g["group_id"]: g for g in groups}
    for g in groups:
        gid = g["group_id"]
        for dep in g.get("depends_on", []) or []:
            if pos[dep] >= pos[gid]:
                raise ResolverError(
                    f"provider '{dep}' is not ordered before consumer '{gid}'"
                )
        # produces_base_for is bookkeeping; assert it is internally consistent.
        for consumer in g.get("produces_base_for", []) or []:
            if consumer not in by_id:
                raise ResolverError(
                    f"group '{gid}' produces_base_for unknown group "
                    f"'{consumer}'"
                )


def validate_execution_groups(
    groups: Iterable,
    *,
    prune_orphans: bool = False,
    pruned_group_ids: Iterable = (),
) -> list:
    """Validate (and optionally prune) an execution-group set; return it.

    Validation: unique ids, resolvable + acyclic ``depends_on``, no self-edge,
    provider-before-consumer. A cycle FAILS the lock -- never a partial order.

    When ``prune_orphans`` is True, ``pruned_group_ids`` (the groups being
    removed by a fallback-restamp) are dropped and any provider thereby orphaned
    is recursively removed (AS-2) BEFORE the final validation, so the returned
    set is the honest, runnable plan.
    """
    groups = _as_list(groups)
    if prune_orphans:
        groups = prune_orphaned_groups(groups, pruned_group_ids)
    _validate_structure(groups)
    _assert_provider_before_consumer(groups)
    return groups


def prune_orphaned_groups(groups: Iterable, pruned_group_ids: Iterable) -> list:
    """Remove ``pruned_group_ids`` + recursively any now-orphaned PROVIDER.

    "Orphaned provider" = a group of ``kind == "provider"`` that, after the
    removals, has no remaining group depending on it. This is the
    ``character_3d -> humo`` background case: prune the degraded consumer group,
    then the ``ltx_video`` background it fed (nothing else consumes it) drops
    too. Consumers are NEVER auto-pruned (only the explicitly pruned ones go);
    only dead providers cascade.

    Pure + order-preserving. Re-stamping the pruned set keeps the ledger honest
    (no dangling skip-edges, no group whose output is unused).
    """
    groups = _as_list(groups)
    remove = set(pruned_group_ids or ())
    remaining = {
        g["group_id"]: g for g in groups if g["group_id"] not in remove
    }

    changed = True
    while changed:
        changed = False
        for gid, g in list(remaining.items()):
            if g.get("kind") != "provider":
                continue
            consumed = any(
                gid in (other.get("depends_on", []) or [])
                for oid, other in remaining.items()
                if oid != gid
            )
            if not consumed:
                del remaining[gid]
                changed = True

    return [g for g in groups if g["group_id"] in remaining]
