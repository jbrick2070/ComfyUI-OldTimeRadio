"""Headless litegraph builder for the opt-in v2 audio workflow (Wave 2b).

Derives ``workflows/otr_scifi_16gb_audio_v2_optin.json`` from the canonical
``workflows/otr_scifi_16gb_full.json`` by applying the audio + voice-casting
migration spec (EXECUTION-PLAN Wave 2b / E.0-E.5):

  * Insert the four v2 nodes -- OTR_CastLock, OTR_BatchCharacterVoices,
    OTR_AnnouncerVoice, OTR_StableAudioTheme -- as fresh INSTANCES (sockets +
    widget vectors derived live from each class INPUT_TYPES / RETURN_TYPES, so
    the file can never drift from the node surface).
  * Repartition node-62 (OTR_LedgerFreezeCascade) out[1] (script_json) by an
    explicit BY-NAME wiring table (never raw slot index): the raw string stays
    for the legacy + scene/shot consumers and feeds the three audio nodes'
    script_json for byte-identical batch delegation; OTR_CastLock reads the
    dedicated v2_ledger_json port (out[6]); CastLock.ledger_json feeds the three
    audio nodes' ledger_json AND HuMo (node 51) ledger_json.
  * Map the theme cues BY NAME (opening_theme_audio -> EpisodeAssembler in[1];
    closing_theme_audio -> EpisodeAssembler in[2] + SignalLost in[3]).
  * Drop the SFX node (15); SceneSequencer sfx_audio_clips is already optional /
    None-safe so the input is left unwired (pure prepend, byte-identical).
  * Chain the done -> gate ordering CharacterVoices -> Announcer -> Theme so the
    three engines never co-reside (I-7). The post-unload audio_done -> FLUX
    loader gate (link 209) is preserved untouched.

The legacy workflow is NOT modified; this is the opt-in copy (I-10 / PD3). The
new nodes default to the legacy engines (bark / kokoro / musicgen) and CastLock
defaults to preserve_ledger, so the opt-in graph is byte-safe until promotion.

UTF-8, no BOM, ASCII-only source. Run with the ComfyUI venv python from the repo
root. R0b (live load in ComfyUI Desktop) is the operator acceptance gate.
"""
from __future__ import annotations

import io
import json
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "workflows", "otr_scifi_16gb_full.json")
OUT = os.path.join(REPO, "workflows", "otr_scifi_16gb_audio_v2_optin.json")


def _headless_stubs() -> None:
    """Mirror tests/conftest.py so the node package imports without a GPU or
    the ComfyUI runtime (folder_paths). Import-time is side-effect-free (C-5)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OTR_TEST_MODE", "1")
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_output_directory = lambda: os.getcwd()
        fp.get_filename_list = lambda *a, **k: []
        sys.modules["folder_paths"] = fp
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


# Build order: CastLock first (id 80) so the audio nodes' ledger source exists,
# then the three audio nodes. ids start at last_node_id+1 (79 -> 80).
NEW_NODES = (
    # id, registry key, module, class, title, pos, color, bgcolor
    (80, "OTR_CastLock", "nodes.cast_lock", "CastLock",
     "1c. Cast Lock (v2 ledger authority)", [1060, 470], "#2B3C5A", "#1C273C"),
    (81, "OTR_BatchCharacterVoices", "nodes.batch_character_voices",
     "BatchCharacterVoices", "3a. Character Voices (v2)", [1140, 80],
     "#234035", "#162821"),
    (82, "OTR_AnnouncerVoice", "nodes.announcer_voice", "AnnouncerVoice",
     "3b. Announcer Voice (v2)", [1140, 330], "#234035", "#162821"),
    (83, "OTR_StableAudioTheme", "nodes.stable_audio_theme", "StableAudioTheme",
     "3c. Theme Music (v2)", [1140, 580], "#234035", "#162821"),
)

# Explicit BY-NAME wiring table: (src_id, src_output_name, dst_id, dst_input_name).
# Resolved to slot indices by NAME at build time (never a raw slot literal).
NEW_WIRES = (
    # raw FreezeCascade script_json -> the three audio nodes (byte-identical
    # batch delegation path); the dedicated v2 ledger port -> CastLock.
    (62, "script_json", 81, "script_json"),
    (62, "script_json", 82, "script_json"),
    (62, "script_json", 83, "script_json"),
    (62, "v2_ledger_json", 80, "script_json"),
    # CastLock canonical ledger -> the three audio nodes + HuMo (node 51).
    (80, "ledger_json", 81, "ledger_json"),
    (80, "ledger_json", 82, "ledger_json"),
    (80, "ledger_json", 83, "ledger_json"),
    (80, "ledger_json", 51, "ledger_json"),
    # audio outputs -> the legacy SceneSequencer / EpisodeAssembler / SignalLost.
    (81, "character_audio", 3, "tts_audio_clips"),
    (82, "announcer_audio", 3, "announcer_audio_clips"),
    (83, "opening_theme_audio", 7, "opening_theme_audio"),
    (83, "closing_theme_audio", 7, "closing_theme_audio"),
    (83, "closing_theme_audio", 12, "closing_audio"),
    # done -> gate ordering so the three engines never co-reside (I-7).
    (81, "done", 82, "gate_in"),
    (82, "done", 83, "gate_in"),
)

REMOVED_NODE_IDS = frozenset({11, 13, 14, 15})


def _derive_surface(input_types):
    """ComfyUI's serialization rule: a widget-typed input (STRING/INT/FLOAT/
    BOOLEAN/combo-list) becomes a WIDGET unless it carries forceInput; a
    forceInput widget-typed input and every non-widget-typed input become an
    input SOCKET. Returns (sockets, widget_defaults) in declaration order."""
    sockets = []          # (name, type_str)
    widget_defaults = []  # default value, in declaration order
    for group_name in ("required", "optional"):
        for name, spec in (input_types.get(group_name) or {}).items():
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            is_force = bool(opts.get("forceInput"))
            is_widget_type = isinstance(typ, list) or typ in (
                "STRING", "INT", "FLOAT", "BOOLEAN", "BOOL",
            )
            if is_widget_type and not is_force:
                widget_defaults.append(opts.get("default"))
            else:
                sockets.append((name, typ if not isinstance(typ, list) else "COMBO"))
    return sockets, widget_defaults


def _introspect():
    """Import each new class and capture its serialized surface. The classes
    import headless (lazy engine libs; C-5), so this is cheap + GPU-free."""
    import importlib

    surface = {}
    for _id, key, modpath, clsname, *_rest in NEW_NODES:
        cls = getattr(importlib.import_module(modpath), clsname)
        sockets, widget_defaults = _derive_surface(cls.INPUT_TYPES())
        surface[key] = {
            "sockets": sockets,
            "widgets": widget_defaults,
            "return_types": list(cls.RETURN_TYPES),
            "return_names": list(cls.RETURN_NAMES),
        }
    return surface


def _build_node(spec, surface) -> dict:
    """Assemble one new node dict mirroring the litegraph schema of the existing
    nodes. Input sockets carry NO widget key (native forceInput; matches node 22
    gate_signal). Links are placeholders -- reconciled from the link table."""
    nid, key, _mod, _cls, title, pos, color, bgcolor = spec
    s = surface[key]
    inputs = [{"name": n, "type": t, "link": None} for (n, t) in s["sockets"]]
    outputs = [
        {"name": nm, "type": ty, "links": [], "slot_index": i}
        for i, (nm, ty) in enumerate(zip(s["return_names"], s["return_types"]))
    ]
    return {
        "id": nid,
        "type": key,
        "pos": list(pos),
        "size": [400, 230],
        "title": title,
        "properties": {"Node name for S&R": key},
        "widgets_values": list(s["widgets"]),
        "inputs": inputs,
        "outputs": outputs,
        "flags": {},
        "order": 40 + (nid - 80),
        "mode": 0,
        "color": color,
        "bgcolor": bgcolor,
    }


def _in_slot(node, name) -> int:
    for i, inp in enumerate(node.get("inputs") or []):
        if inp.get("name") == name:
            return i
    raise KeyError("node %s has no input socket %r" % (node.get("id"), name))


def _out_slot(node, name):
    for i, outp in enumerate(node.get("outputs") or []):
        if outp.get("name") == name:
            return i, outp.get("type")
    raise KeyError("node %s has no output socket %r" % (node.get("id"), name))


def _reconcile(node, links) -> None:
    """Rebuild a node's input.link and output.links from the authoritative link
    table (single source of truth -> the two redundant stores can't disagree)."""
    nid = node["id"]
    for i, inp in enumerate(node.get("inputs") or []):
        match = [L[0] for L in links if L[3] == nid and L[4] == i]
        inp["link"] = match[0] if match else None
    for i, outp in enumerate(node.get("outputs") or []):
        outp["links"] = [L[0] for L in links if L[1] == nid and L[2] == i]


def build() -> dict:
    _headless_stubs()
    surface = _introspect()
    with io.open(SRC, "r", encoding="utf-8") as f:
        wf = json.load(f)

    by_id = {n["id"]: n for n in wf["nodes"]}
    # Drop the legacy audio-generator instances.
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in REMOVED_NODE_IDS]
    # Add the four v2 nodes.
    new_nodes = [_build_node(spec, surface) for spec in NEW_NODES]
    wf["nodes"].extend(new_nodes)
    by_id = {n["id"]: n for n in wf["nodes"]}

    # Resolve the BY-NAME wiring table to concrete (slot, type) tuples.
    resolved = []
    for src_id, src_name, dst_id, dst_name in NEW_WIRES:
        ss, sty = _out_slot(by_id[src_id], src_name)
        ds = _in_slot(by_id[dst_id], dst_name)
        resolved.append((src_id, ss, dst_id, ds, sty))
    new_wire_dsts = {(dst_id, ds) for (_si, _ss, dst_id, ds, _ty) in resolved}

    # Keep every existing link that does not touch a removed node and whose dst
    # input is not being re-wired by a new link (a litegraph input holds one
    # link; this is what retargets HuMo 51.ledger_json from 62 to CastLock).
    kept = []
    for L in wf["links"]:
        _lid, sn, _ss, dn, ds, _ty = L
        if sn in REMOVED_NODE_IDS or dn in REMOVED_NODE_IDS:
            continue
        if (dn, ds) in new_wire_dsts:
            continue
        kept.append(L)

    nid = wf["last_link_id"]
    for (src_id, ss, dst_id, ds, sty) in resolved:
        nid += 1
        kept.append([nid, src_id, ss, dst_id, ds, sty])

    wf["links"] = kept
    wf["last_link_id"] = nid
    wf["last_node_id"] = max(n["id"] for n in wf["nodes"])

    # Reconcile every node that gained or lost a link (plus the new nodes).
    dirty = {3, 7, 12, 51, 62} | {spec[0] for spec in NEW_NODES}
    for n in wf["nodes"]:
        if n["id"] in dirty:
            _reconcile(n, kept)
    return wf


def main() -> int:
    wf = build()
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, ensure_ascii=True, indent=2)
        f.write("\n")
    print("WROTE %s : %d nodes, %d links, last_node_id=%s last_link_id=%s" % (
        os.path.basename(OUT), len(wf["nodes"]), len(wf["links"]),
        wf["last_node_id"], wf["last_link_id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
