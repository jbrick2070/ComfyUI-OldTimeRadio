"""The image producer must read the POST-AUDIO ledger, not the pre-audio one.

FOUND BY A LIVE LEG. `fastwan_8gb`, 45-word sweep, 2026-08-12:

    RenderError: still-spine handoff missing materialized scene still
    for shot shot_music_closing_001 beat music_closing_001 engine fastwan_8gb

THE MECHANISM, and my first diagnosis was WRONG. I read it as a still that was
planned and then failed to materialize. An independent audit refuted that: the
`music_closing_001` still was **never planned at all**. A stale
`still_shot_006_music` was generated instead.

Why: `derive_scene_still_targets` reserves a synthetic closing target ONLY when
it sees no existing `music_close` line. Reading the PRE-AUDIO ledger it does see
one -- the pre-audio sentinel, under its authored id (`shot_006_music`) -- so the
reservation is suppressed. `EpisodeAssembler` then mirrors the closing cue under
a DIFFERENT, deterministic id (`music_closing_001`), and the still for that id
was never minted. The suppression is correct logic reading the wrong ledger.

THE FIX IS WIRING, NOT CODE. `OTR_ShotLock` (node 90) has already imported the
exact `music_closing_001/002/003` mirror ids by the time it emits
`patched_ledger_json`. Retargeting canonical link 255 so
`OTR_MetaBriefImagePromptGen` (node 89) takes ITS `script_json` from node 90
output 0, instead of the pre-audio `OTR_LedgerFreezeCascade` (node 62) output 1,
makes the producer see the real closing beat id and mint a target for it.

WHY THIS FILE EXISTS AT ALL. Every one of the 95 still-spine helper tests and
all 538 workflow tests passed both BEFORE and AFTER the retarget, because they
exercise the helpers directly and nothing pinned the graph. The defect lived in
the wiring, so the guard has to live there too -- otherwise link 255 can be
reverted and the suite stays green while every scene-still engine breaks again.

Eight of thirteen heavy engines require a scene still (`fastwan_8gb`, `ltx_8gb`,
`ltx_audio_in`, `wan_i2v`, `wan_ti2v`, `mesh_stage`, `minimax_h3_video`,
`minimax_h3_audio_in`), so this is most of the video roster.
"""
from __future__ import annotations

import json
import pathlib

import pytest


CANONICAL = (pathlib.Path(__file__).resolve().parents[1]
             / "workflows" / "otr_canonical.json")


@pytest.fixture(scope="module")
def graph():
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def node_by_type(graph, type_name):
    hits = [n for n in graph["nodes"] if n.get("type") == type_name]
    assert len(hits) == 1, "expected exactly one %s, got %d" % (type_name, len(hits))
    return hits[0]


def test_the_image_producer_reads_SHOTLOCKS_patched_ledger(graph):
    """THE FIX. Read by TYPE, not by hard-coded node id, so renumbering the
    graph cannot silently retire this guard."""
    producer = node_by_type(graph, "OTR_MetaBriefImagePromptGen")
    shotlock = node_by_type(graph, "OTR_ShotLock")

    script_in = next(i for i in producer["inputs"] if i["name"] == "script_json")
    link_id = script_in["link"]
    link = next(l for l in graph["links"] if l[0] == link_id)

    src_node, src_slot = link[1], link[2]
    assert src_node == shotlock["id"], (
        "the image producer's script_json comes from node %s, not OTR_ShotLock. "
        "Reading the PRE-AUDIO ledger suppresses the closing-music still "
        "reservation and every scene-still engine fails at video dispatch."
        % src_node)
    assert shotlock["outputs"][src_slot]["name"] == "patched_ledger_json"


def test_it_does_NOT_read_the_pre_audio_freeze_cascade(graph):
    """The defective source, named explicitly so a revert is unmistakable."""
    producer = node_by_type(graph, "OTR_MetaBriefImagePromptGen")
    freeze = node_by_type(graph, "OTR_LedgerFreezeCascade")
    script_in = next(i for i in producer["inputs"] if i["name"] == "script_json")
    link = next(l for l in graph["links"] if l[0] == script_in["link"])
    assert link[1] != freeze["id"], (
        "regression: the image producer is back on the pre-audio ledger, which "
        "is how the music_closing_001 still went unplanned")


def test_the_link_is_recorded_on_BOTH_sides(graph):
    """litegraph stores a link in `links[]` AND on the source node's output
    `links` list. A half-moved link round-trips but drifts in the editor."""
    producer = node_by_type(graph, "OTR_MetaBriefImagePromptGen")
    shotlock = node_by_type(graph, "OTR_ShotLock")
    freeze = node_by_type(graph, "OTR_LedgerFreezeCascade")

    script_in = next(i for i in producer["inputs"] if i["name"] == "script_json")
    link_id = script_in["link"]

    assert link_id in (shotlock["outputs"][0].get("links") or []), (
        "the link is not recorded on OTR_ShotLock's output")
    for out in (freeze.get("outputs") or []):
        assert link_id not in (out.get("links") or []), (
            "the link is still recorded on OTR_LedgerFreezeCascade's output")


def test_shotlock_does_not_depend_on_the_image_producer(graph):
    """The retarget is only legal because these are siblings. If ShotLock ever
    consumes the producer's output this becomes a cycle and the graph will not
    execute."""
    producer = node_by_type(graph, "OTR_MetaBriefImagePromptGen")
    shotlock = node_by_type(graph, "OTR_ShotLock")
    for inp in (shotlock.get("inputs") or []):
        lid = inp.get("link")
        if lid is None:
            continue
        link = next(l for l in graph["links"] if l[0] == lid)
        assert link[1] != producer["id"], (
            "OTR_ShotLock now consumes the image producer -- that is a cycle")


def test_the_graph_is_ACYCLIC(graph):
    """Cheap whole-graph guard: any future rewire that closes a loop fails
    here rather than at submit time on a live leg."""
    from collections import defaultdict, deque

    ids = {n["id"] for n in graph["nodes"]}
    adj, indeg = defaultdict(set), defaultdict(int)
    for link in graph["links"]:
        if link[3] not in adj[link[1]]:
            adj[link[1]].add(link[3])
            indeg[link[3]] += 1
    queue = deque(n for n in ids if indeg[n] == 0)
    ordered = 0
    while queue:
        node = queue.popleft()
        ordered += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    assert ordered == len(ids), "the canonical workflow contains a cycle"


def test_every_link_still_resolves(graph):
    """Referential integrity, because this change edited `links[]` by hand."""
    ids = {n["id"] for n in graph["nodes"]}
    link_ids = {l[0] for l in graph["links"]}
    for link in graph["links"]:
        assert link[1] in ids and link[3] in ids, "dangling link %r" % (link,)
    for node in graph["nodes"]:
        for inp in (node.get("inputs") or []):
            lid = inp.get("link")
            assert lid is None or lid in link_ids, (
                "node %s input %r points at missing link %r"
                % (node["id"], inp.get("name"), lid))
