"""Wave 2b -- opt-in v2 audio workflow contract + link-migration gates.

Validates ``workflows/otr_scifi_16gb_audio_v2_optin.json`` (emitted by
``scripts/build_optin_audio_v2_workflow.py``) against:

  * the workflow contract validator (rogue sockets, positional widget drift,
    and the full link-table battery) for the four new v2 audio nodes;
  * the EXECUTION-PLAN Wave 2b BY-NAME migration spec -- the node-62
    script_json partition, the CastLock dedicated v2-ledger source, the theme
    cue fanout, and the done -> gate chain;
  * no legacy audio-generator INSTANCE on the active path, node 15 dropped with
    no orphan links, SceneSequencer sfx left optional / None, and the I-7
    audio_done -> first-video-loader gate preserved.

These are headless link-integrity gates. R0b (live load in ComfyUI Desktop with
stub engines) remains the operator acceptance gate. ASCII-only source, no em-dash.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_WF_PATH = _REPO / "workflows" / "otr_scifi_16gb_full.json"

# Registry key -> the node id the builder assigns (build order, ids 80-83).
NEW_NODE_IDS = {
    "OTR_CastLock": 80,
    "OTR_BatchCharacterVoices": 81,
    "OTR_AnnouncerVoice": 82,
    "OTR_StableAudioTheme": 83,
}
# The legacy single-purpose audio-generator node TYPES. None of these may have
# a live INSTANCE in the opt-in graph (registry nodes may still SELECT the
# legacy engine via raw delegation -- that is an engine widget, not an instance).
LEGACY_AUDIO_GEN_TYPES = frozenset({
    "OTR_BatchBarkGenerator",
    "OTR_KokoroAnnouncer",
    "OTR_MusicGenTheme",
    "OTR_BatchAudioGenGenerator",
})
DROPPED_NODE_IDS = frozenset({11, 13, 14, 15})


# --------------------------------------------------------------------------- #
# Fixtures + helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wf():
    assert _WF_PATH.exists(), (
        "opt-in workflow missing -- run scripts/build_optin_audio_v2_workflow.py"
    )
    with _WF_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def by_id(wf):
    return {n["id"]: n for n in wf["nodes"]}


@pytest.fixture(scope="module")
def links_by_id(wf):
    return {L[0]: L for L in wf["links"]}


def _new_class_mapping():
    """The four new node classes keyed by registry id (imports headless)."""
    from nodes._otr_class_registry import NEW_AUDIO_NODES

    mapping = {}
    for spec in NEW_AUDIO_NODES:
        mod = importlib.import_module(spec.module_path.lstrip("."))
        mapping[spec.key] = getattr(mod, spec.class_name)
    return mapping


def _derive_widget_defaults(cls):
    """ComfyUI widget-vector for a class: every widget-typed (non-forceInput)
    input default, in declared order (required then optional)."""
    it = cls.INPUT_TYPES()
    out = []
    for group in ("required", "optional"):
        for _name, spec in (it.get(group) or {}).items():
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            is_widget_type = isinstance(typ, list) or typ in (
                "STRING", "INT", "FLOAT", "BOOLEAN", "BOOL",
            )
            if is_widget_type and not opts.get("forceInput"):
                out.append(opts.get("default"))
    return out


def _in_link(node, name):
    for inp in node.get("inputs") or []:
        if inp.get("name") == name:
            return inp.get("link")
    raise AssertionError("node %s has no input %r" % (node.get("id"), name))


def _out_links(node, name):
    for o in node.get("outputs") or []:
        if o.get("name") == name:
            return list(o.get("links") or [])
    raise AssertionError("node %s has no output %r" % (node.get("id"), name))


def _src(links_by_id, link_id):
    """(src_node_id, src_slot) for a link id."""
    L = links_by_id[link_id]
    return (L[1], L[2])


def _dst_nodes(links_by_id, link_ids):
    return sorted(links_by_id[i][3] for i in link_ids)


def _dst(links_by_id, by_id, link_id):
    """(dst_node_id, dst_input_name) for a link id."""
    L = links_by_id[link_id]
    dn, ds = L[3], L[4]
    return (dn, by_id[dn]["inputs"][ds]["name"])


# --------------------------------------------------------------------------- #
# Contract validator + link-table battery
# --------------------------------------------------------------------------- #
def test_contract_validator_passes_for_new_nodes(wf):
    """The four new nodes must satisfy their declared INPUT_TYPES (no rogue
    sockets, no widget drift) and the link table must be well-formed."""
    from nodes._workflow_validation import validate_workflow_contract

    validate_workflow_contract(wf, _new_class_mapping())


def test_link_table_is_wellformed(wf):
    links = wf["links"]
    ids = [L[0] for L in links]
    assert all(isinstance(L, list) and len(L) >= 6 for L in links)
    assert len(ids) == len(set(ids)), "duplicate link ids"
    assert wf["last_link_id"] == max(ids)
    assert not (set(ids) & {111, 112}), "G5-reserved link ids present"
    node_ids = {n["id"] for n in wf["nodes"]}
    for L in links:
        assert L[1] in node_ids and L[3] in node_ids, "orphan link %s" % (L[0],)


def test_four_new_nodes_present(by_id):
    for key, nid in NEW_NODE_IDS.items():
        assert nid in by_id, "missing new node id %s (%s)" % (nid, key)
        assert by_id[nid]["type"] == key


def test_no_legacy_audio_generator_instances(wf):
    bad = [n["id"] for n in wf["nodes"] if n["type"] in LEGACY_AUDIO_GEN_TYPES]
    assert not bad, "legacy audio-generator instances still present: %s" % bad


def test_node15_dropped_with_no_orphan_links(wf, by_id):
    for nid in DROPPED_NODE_IDS:
        assert nid not in by_id, "dropped node %s still present" % nid
    for L in wf["links"]:
        assert L[1] not in DROPPED_NODE_IDS, "link %s sources dropped node" % L[0]
        assert L[3] not in DROPPED_NODE_IDS, "link %s targets dropped node" % L[0]


def test_widget_vectors_exact(by_id):
    """Each new node's widgets_values equals its INPUT_TYPES-derived defaults
    (the legacy engine per role -- the byte-safe defaults), EXCEPT CastLock (80).

    The canonical workflow selects IndexTTS2 for character voices (node 81), so
    it intentionally sets CastLock to auto_registry + allow_voice_reuse=True so
    every character is voiced on IndexTTS2 instead of falling back to bark. The
    byte-identical guarantee only applies to bark-SELECTED paths; this workflow
    is not one, so the deviation from the preserve_ledger default is deliberate.
    """
    mapping = _new_class_mapping()
    # CastLock (80): intentional index-workflow override (see docstring).
    assert by_id[80]["widgets_values"] == ["default", "auto_registry", "neutral", True]
    for key, nid in NEW_NODE_IDS.items():
        if nid == 80:
            continue
        assert by_id[nid]["widgets_values"] == _derive_widget_defaults(mapping[key])


def test_force_input_sockets_have_no_widget_key(by_id):
    """Native forceInput sockets must not carry a ``widget`` key (matches the
    node-22 gate_signal encoding; legacy widget-converted inputs do carry one)."""
    for nid in NEW_NODE_IDS.values():
        for inp in by_id[nid]["inputs"]:
            assert "widget" not in inp, (
                "node %s socket %r has a widget key" % (nid, inp.get("name"))
            )


# --------------------------------------------------------------------------- #
# BY-NAME migration spec
# --------------------------------------------------------------------------- #
def test_audio_nodes_script_json_is_raw_freeze_out(by_id, links_by_id):
    """The three audio nodes read the RAW FreezeCascade script_json (node 62
    out slot 1) for byte-identical batch delegation (I-3)."""
    for nid in (81, 82, 83):
        link = _in_link(by_id[nid], "script_json")
        assert link is not None, "node %s script_json unwired" % nid
        assert _src(links_by_id, link) == (62, 1)


def test_castlock_reads_dedicated_v2_ledger_port(by_id, links_by_id):
    """OTR_CastLock (80) sources its script_json from the dedicated
    v2_ledger_json port (node 62 out slot 6), NOT the raw script_json fan."""
    link = _in_link(by_id[80], "script_json")
    assert _src(links_by_id, link) == (62, 6)


def test_audio_nodes_ledger_from_castlock(by_id, links_by_id):
    """The three audio nodes' ledger_json comes from CastLock (80) out slot 0."""
    for nid in (81, 82, 83):
        link = _in_link(by_id[nid], "ledger_json")
        assert link is not None, "node %s ledger_json unwired" % nid
        assert _src(links_by_id, link) == (80, 0)


def test_humo_ledger_retargeted_to_castlock(by_id, links_by_id):
    """HuMo (node 51) ledger_json is retargeted from raw 62.out to CastLock."""
    link = _in_link(by_id[51], "ledger_json")
    assert _src(links_by_id, link) == (80, 0)


def test_freeze_script_json_partition(by_id, links_by_id):
    """Node 62 script_json (out slot 1) fans to exactly the raw legacy / scene /
    shot consumers + the three audio nodes -- never node 15/51/11/13/14."""
    fan = _out_links(by_id[62], "script_json")
    dst_nodes = {links_by_id[i][3] for i in fan}
    assert dst_nodes == {3, 12, 20, 52, 55, 59, 71, 81, 82, 83}
    assert 51 not in dst_nodes  # HuMo now sources the cast-locked ledger
    assert not (dst_nodes & DROPPED_NODE_IDS)


def test_theme_cue_fanout_by_name(by_id, links_by_id):
    """opening_theme_audio -> EpisodeAssembler; closing_theme_audio ->
    EpisodeAssembler AND SignalLost (mutation-rigor on the closing fanout)."""
    theme = by_id[83]
    opening = [_dst(links_by_id, by_id, i) for i in _out_links(theme, "opening_theme_audio")]
    assert opening == [(7, "opening_theme_audio")]
    closing = sorted(_dst(links_by_id, by_id, i) for i in _out_links(theme, "closing_theme_audio"))
    assert closing == [(7, "closing_theme_audio"), (12, "closing_audio")]


def test_scene_sequencer_audio_inputs(by_id, links_by_id):
    """SceneSequencer (3) tts_audio_clips <- CharacterVoices; announcer_audio_clips
    <- Announcer; sfx_audio_clips left optional / None (node 15 dropped)."""
    node = by_id[3]
    assert _src(links_by_id, _in_link(node, "tts_audio_clips")) == (81, 0)
    assert _src(links_by_id, _in_link(node, "announcer_audio_clips")) == (82, 0)
    assert _in_link(node, "sfx_audio_clips") is None


def test_done_gate_chain_orders_the_engines(by_id, links_by_id):
    """CharacterVoices.done -> Announcer.gate_in -> Theme.gate_in, so the three
    engines never co-reside (I-7 single-engine residency)."""
    char_done = [_dst(links_by_id, by_id, i) for i in _out_links(by_id[81], "done")]
    assert char_done == [(82, "gate_in")]
    ann_done = [_dst(links_by_id, by_id, i) for i in _out_links(by_id[82], "done")]
    assert ann_done == [(83, "gate_in")]


def test_audio_done_gate_to_first_video_loader_preserved(by_id, links_by_id):
    """The I-7 post-unload gate: EpisodeAssembler.audio_done (7 out slot 3) ->
    the FLUX deferred loader (node 22) gate_signal -- untouched by the migration."""
    link = _in_link(by_id[22], "gate_signal")
    assert _src(links_by_id, link) == (7, 3)
