"""
Regression tests for every workflow JSON under workflows/.

Catches common widget-drift + workflow-corruption issues WITHOUT loading
ComfyUI or any OTR node module (no torch, no transformers). Safe to run
in CI and as part of the standard regression suite.

Scope:
    workflows/*.json  (top-level only — any subdirectories are excluded
    on purpose, in case they ever hold non-workflow fixtures)

Checks, per JSON file:
    1.  Valid UTF-8, no BOM, parses as a JSON object.
    2.  Format is recognized (UI format with nodes+links, or API format
        with integer-keyed nodes carrying class_type).
    3.  Every node has a non-empty "type" (UI) or "class_type" (API).
    4.  widgets_values (UI) is a list of scalar values or [node_id, slot]
        link-refs. No dicts, no nested objects.
    5.  Links are well-formed 5-tuples with int IDs and no duplicates.
    6.  Every link references existing source + destination node IDs.
    7.  Every non-null input.link (UI) references an existing link ID.
    8.  API-format inputs are either scalars or [str_node_id, int_slot].
    9.  No known-stale dropdown literals appear in any widget / input.

Add a new entry to STALE_DROPDOWN_LITERALS when a dropdown string gets
renamed (e.g. BUG-011 renamed "Obsidian (Low VRAM/Fast)" to
"Obsidian (UNSTABLE/4GB)") so older JSONs on disk that still reference
the old literal are caught and migrated.

Run:  python -m pytest tests/test_workflow_json_guardrails.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"


# ---------------------------------------------------------------------------
# Known-stale dropdown strings that MUST NOT appear in any widgets_values
# or API inputs. Grow this set when a dropdown value is renamed and older
# on-disk workflows need to be migrated.
#
# Canonical tracked renames:
#   BUG-011  "Obsidian (Low VRAM/Fast)" -> "Obsidian (UNSTABLE/4GB)"
#   2026-05-10  "auto (LLM generates)" -> "let the story decide"
#               (OTR_LedgerScriptWriter style sentinel — clearer label;
#                pre-rename workflows would silently bind to a missing
#                dropdown entry and the auto-derive path would never fire)
# ---------------------------------------------------------------------------
STALE_DROPDOWN_LITERALS: frozenset[str] = frozenset({
    "Obsidian (Low VRAM/Fast)",
    "auto (LLM generates)",
})


# Allowed element types for a single widgets_values slot.
# Lists are also allowed when they are the two-element [node_id, slot]
# link reference shape that ComfyUI saves for some converted widgets.
_SCALAR_WIDGET_TYPES = (str, int, float, bool, type(None))


# ---------------------------------------------------------------------------
# Discovery + classification
# ---------------------------------------------------------------------------
def _discover_workflow_jsons() -> list[Path]:
    if not WORKFLOWS_DIR.is_dir():
        return []
    return sorted(p for p in WORKFLOWS_DIR.glob("*.json") if p.is_file())


WORKFLOW_FILES: list[Path] = _discover_workflow_jsons()
WORKFLOW_IDS: list[str] = [f.name for f in WORKFLOW_FILES]


def _load_json(path: Path):
    """Read + decode + parse, with no-BOM + valid-UTF-8 guarantees."""
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), (
        f"{path.name} starts with a UTF-8 BOM; strip it"
    )
    # .decode will raise UnicodeDecodeError on malformed UTF-8
    text = raw.decode("utf-8")
    return json.loads(text)


def _classify(doc) -> str:
    """Return 'ui', 'api', or 'unknown'."""
    if not isinstance(doc, dict) or not doc:
        return "unknown"
    if "nodes" in doc and "links" in doc:
        return "ui"
    # API format: top-level keys are stringified integers, values are
    # dicts with a class_type field.
    if all(isinstance(k, str) and k.lstrip("-").isdigit() for k in doc.keys()):
        if all(isinstance(v, dict) and "class_type" in v for v in doc.values()):
            return "api"
    return "unknown"


# ---------------------------------------------------------------------------
# Top-level sanity (not parametrized — fires once regardless of JSON count)
# ---------------------------------------------------------------------------
def test_workflows_directory_exists():
    assert WORKFLOWS_DIR.is_dir(), f"workflows/ not found at {WORKFLOWS_DIR}"


def test_at_least_one_workflow_json_discovered():
    # Sanity: if this ever hits zero, the test file silently passes with
    # no parametrize expansion, which is a trap. Fail loudly instead.
    assert WORKFLOW_FILES, (
        f"No *.json files found directly under {WORKFLOWS_DIR}. "
        "If workflows were moved, update _discover_workflow_jsons()."
    )


# ---------------------------------------------------------------------------
# Parametrized per-file guardrails
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("workflow_path", WORKFLOW_FILES, ids=WORKFLOW_IDS)
class TestWorkflowJson:
    # ---- Phase 1: file-level parse ----
    def test_parses_as_utf8_json_no_bom(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        assert isinstance(doc, dict), (
            f"{workflow_path.name} must decode to a JSON object, got "
            f"{type(doc).__name__}"
        )

    def test_format_recognized(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        fmt = _classify(doc)
        assert fmt in ("ui", "api"), (
            f"{workflow_path.name} has an unrecognized shape. Expected a "
            "ComfyUI UI workflow (with nodes + links) or API prompt "
            "(integer-keyed dict of class_type+inputs)."
        )

    # ---- Phase 2: node-level shape ----
    def test_every_node_has_non_empty_type(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        fmt = _classify(doc)
        if fmt == "ui":
            for n in doc.get("nodes", []):
                t = n.get("type")
                assert isinstance(t, str) and t.strip(), (
                    f"{workflow_path.name}: node id={n.get('id')!r} has "
                    f"missing or empty 'type' field"
                )
        else:  # api
            for nid, n in doc.items():
                t = n.get("class_type")
                assert isinstance(t, str) and t.strip(), (
                    f"{workflow_path.name}: node '{nid}' has missing or "
                    f"empty 'class_type' field"
                )

    def test_api_inputs_shape(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        if _classify(doc) != "api":
            pytest.skip("UI-format file; see UI-specific tests")
        node_ids = set(doc.keys())
        for nid, node in doc.items():
            inputs = node.get("inputs")
            assert isinstance(inputs, dict), (
                f"{workflow_path.name}: node '{nid}' ({node.get('class_type')}) "
                f"must have an 'inputs' dict, got {type(inputs).__name__}"
            )
            for name, val in inputs.items():
                if isinstance(val, list):
                    # Link reference: [src_node_id_str, src_slot_int]
                    assert len(val) == 2, (
                        f"{workflow_path.name}: node '{nid}' input {name!r} "
                        f"link-ref must be length 2, got {val!r}"
                    )
                    src_id, src_slot = val
                    assert isinstance(src_id, str) and src_id in node_ids, (
                        f"{workflow_path.name}: node '{nid}' input {name!r} "
                        f"links to non-existent source node {src_id!r}"
                    )
                    assert isinstance(src_slot, int) and src_slot >= 0, (
                        f"{workflow_path.name}: node '{nid}' input {name!r} "
                        f"has non-int or negative slot: {src_slot!r}"
                    )
                else:
                    assert isinstance(val, _SCALAR_WIDGET_TYPES), (
                        f"{workflow_path.name}: node '{nid}' input {name!r} "
                        f"is {type(val).__name__}, must be scalar or link-ref"
                    )

    # ---- Phase 3: UI widgets_values element typing ----
    def test_widgets_values_are_scalars_or_link_refs(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        if _classify(doc) != "ui":
            pytest.skip("API format has no widgets_values")
        for n in doc.get("nodes", []):
            wv = n.get("widgets_values")
            if wv is None:
                continue
            assert isinstance(wv, list), (
                f"{workflow_path.name}: node {n.get('id')} "
                f"({n.get('type')}) widgets_values must be a list, got "
                f"{type(wv).__name__}"
            )
            for idx, val in enumerate(wv):
                if isinstance(val, list):
                    # Two-element [node_id, slot] reference
                    assert len(val) == 2, (
                        f"{workflow_path.name}: node {n.get('id')} "
                        f"({n.get('type')}) widgets_values[{idx}] is a "
                        f"malformed link-ref (len!=2): {val!r}"
                    )
                    continue
                assert isinstance(val, _SCALAR_WIDGET_TYPES), (
                    f"{workflow_path.name}: node {n.get('id')} "
                    f"({n.get('type')}) widgets_values[{idx}] is "
                    f"{type(val).__name__}, must be str/int/float/bool/None"
                )

    # ---- Phase 4: UI links table integrity ----
    def test_links_are_well_formed_and_unique(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        if _classify(doc) != "ui":
            pytest.skip("API format has no top-level links table")
        seen: set[int] = set()
        for link in doc.get("links") or []:
            assert isinstance(link, list) and len(link) >= 5, (
                f"{workflow_path.name}: malformed link entry {link!r}"
            )
            link_id, src, src_slot, dst, dst_slot = link[:5]
            assert isinstance(link_id, int), (
                f"{workflow_path.name}: link_id must be int, got "
                f"{type(link_id).__name__} in {link!r}"
            )
            assert link_id not in seen, (
                f"{workflow_path.name}: duplicate link_id {link_id}"
            )
            seen.add(link_id)
            assert isinstance(src, int) and isinstance(dst, int), (
                f"{workflow_path.name}: link {link_id} has non-int node IDs "
                f"src={src!r} dst={dst!r}"
            )
            assert isinstance(src_slot, int) and isinstance(dst_slot, int), (
                f"{workflow_path.name}: link {link_id} has non-int slot IDs: "
                f"{link!r}"
            )

    def test_link_endpoints_exist(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        if _classify(doc) != "ui":
            pytest.skip("API format")
        node_ids = {n.get("id") for n in doc.get("nodes", [])}
        for link in doc.get("links") or []:
            if not isinstance(link, list) or len(link) < 5:
                continue
            link_id, src, _, dst, _ = link[:5]
            assert src in node_ids, (
                f"{workflow_path.name}: link {link_id} source node "
                f"{src} does not exist in nodes table"
            )
            assert dst in node_ids, (
                f"{workflow_path.name}: link {link_id} destination node "
                f"{dst} does not exist in nodes table"
            )

    def test_input_links_reference_real_links(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        if _classify(doc) != "ui":
            pytest.skip("API format")
        link_ids = {
            link[0]
            for link in (doc.get("links") or [])
            if isinstance(link, list) and link and isinstance(link[0], int)
        }
        for n in doc.get("nodes", []):
            for inp in n.get("inputs") or []:
                link = inp.get("link")
                if link is None:
                    continue
                assert link in link_ids, (
                    f"{workflow_path.name}: node {n.get('id')} "
                    f"({n.get('type')}) input {inp.get('name')!r} "
                    f"references missing link id {link}"
                )

    # ---- Phase 5: stale-literal deny-list ----
    def test_no_stale_dropdown_literals(self, workflow_path: Path):
        doc = _load_json(workflow_path)
        fmt = _classify(doc)
        offenders: list[str] = []

        if fmt == "ui":
            for n in doc.get("nodes", []):
                wv = n.get("widgets_values") or []
                if not isinstance(wv, list):
                    continue
                for idx, val in enumerate(wv):
                    if isinstance(val, str) and val in STALE_DROPDOWN_LITERALS:
                        offenders.append(
                            f"node {n.get('id')} ({n.get('type')}) "
                            f"widgets_values[{idx}] = {val!r}"
                        )
        elif fmt == "api":
            for nid, node in doc.items():
                for name, val in (node.get("inputs") or {}).items():
                    if isinstance(val, str) and val in STALE_DROPDOWN_LITERALS:
                        offenders.append(
                            f"node '{nid}' ({node.get('class_type')}) "
                            f"inputs[{name!r}] = {val!r}"
                        )

        assert not offenders, (
            f"{workflow_path.name} contains stale dropdown strings "
            "(update the workflow or remove from STALE_DROPDOWN_LITERALS "
            "if the rename was reverted):\n  " + "\n  ".join(offenders)
        )


# ---------------------------------------------------------------------------
# Saved-default binding: the OTR_LedgerScriptWriter style widget MUST be
# saved as the auto-derive sentinel so a fresh load runs the LLM-derives-
# style-from-news path with no user intervention.
#
# History (2026-05-10): the sentinel mechanism was authored in commit
# de34c95 but the saved workflow value drifted away from it through two
# realignments (7077e54 + the post-fix), leaving the auto path dormant
# for every production run. This test guards against re-drift.
#
# Slot layout (post-control_after_generate fix, 17-slot writer widget):
#   [11] style  (combo with auto-sentinel as the canonical default)
#
# Hardcoded for the canonical workflow + the canonical writer slot. If
# the writer's INPUT_TYPES order changes, this test must be updated in
# lockstep with the workflow JSON (Prime Directive 3 — wire every change
# into the workflow JSON).
# ---------------------------------------------------------------------------

# Mirror of nodes/OTR_LedgerScriptWriter._STYLE_AUTO_SENTINEL. Hardcoded
# (not imported) so this test stays free of torch / transformers and
# can run in any CI environment.
_WRITER_STYLE_SENTINEL = "let the story decide"
# Slot 9 of OTR_LedgerScriptWriter.widgets_values in the current
# widget order:
#   0 episode_title / 1 target_words / 2 num_characters / 3 seed
#   4 seed_mode / 5 model_id / 6 custom_premise / 7 include_act_breaks
#   8 act_count / 9 style  <- this slot / 10 style_custom
#   11 creativity / 12 optimization_profile / 13 perfect_run_spacesaver
#   14 min_p / 15 repetition_penalty / 16 max_new_tokens_cap
# If the writer's INPUT_TYPES order changes, update this constant
# in lockstep so the drift guard keeps catching the real regression.
_WRITER_STYLE_SLOT = 9
_CANONICAL_WORKFLOW = "otr_scifi_16gb_full.json"


class TestWriterStyleSentinelDefault:
    def test_writer_style_widget_saved_as_auto_sentinel(self):
        """The canonical workflow's OTR_LedgerScriptWriter node MUST
        bind its style widget to the auto-derive sentinel. Any other
        value silently disables the LLM-derives-style-from-news path
        and freezes every run to one preset.
        """
        wf_path = WORKFLOWS_DIR / _CANONICAL_WORKFLOW
        assert wf_path.is_file(), (
            f"Canonical workflow {_CANONICAL_WORKFLOW!r} is missing "
            f"from {WORKFLOWS_DIR}"
        )
        doc = _load_json(wf_path)
        writers = [
            n for n in doc.get("nodes", [])
            if n.get("type") == "OTR_LedgerScriptWriter"
        ]
        assert len(writers) == 1, (
            f"Expected exactly one OTR_LedgerScriptWriter node in "
            f"{_CANONICAL_WORKFLOW}; found {len(writers)}"
        )
        wv = writers[0].get("widgets_values") or []
        assert _WRITER_STYLE_SLOT < len(wv), (
            f"OTR_LedgerScriptWriter widgets_values has only {len(wv)} "
            f"entries; expected slot {_WRITER_STYLE_SLOT} for the "
            f"style widget. Widget surface drift — re-check the writer "
            f"INPUT_TYPES order vs the saved layout."
        )
        actual = wv[_WRITER_STYLE_SLOT]
        assert actual == _WRITER_STYLE_SENTINEL, (
            f"OTR_LedgerScriptWriter widgets_values[{_WRITER_STYLE_SLOT}] "
            f"is {actual!r}; expected the auto-derive sentinel "
            f"{_WRITER_STYLE_SENTINEL!r}. If you intentionally saved a "
            f"specific preset as the default, update _WRITER_STYLE_SENTINEL "
            f"in this test in lockstep — but be aware the auto-derive "
            f"path will not fire on default runs."
        )


class TestVoicePathCleanbreakWiring:
    """Voice-path-cleanbreak 2026-05-12 (P3) wiring guardrails.

    Pins the post-cleanbreak invariants for the canonical workflow JSON
    so a future hand-edit cannot silently re-introduce the Director-fed
    voice-node secondary paths.
    """

    # Voice-side nodes (the ones P2 pruned). These must NEVER have a
    # ``production_plan_json`` input socket post-cleanbreak.
    _VOICE_NODE_TYPES = frozenset({
        "OTR_BatchBarkGenerator",
        "OTR_KokoroAnnouncer",
        "OTR_BatchAudioGenGenerator",
        "OTR_BatchProceduralSFX",
        "OTR_SceneSequencer",
        "OTR_MusicGenTheme",
    })

    def _doc(self):
        wf_path = WORKFLOWS_DIR / _CANONICAL_WORKFLOW
        assert wf_path.is_file(), (
            f"Canonical workflow {_CANONICAL_WORKFLOW!r} missing"
        )
        return _load_json(wf_path)

    def test_no_production_plan_json_wires_to_voice_nodes(self):
        """No link's destination is a voice-node ``production_plan_json``
        slot. Video-side wires (link 17 -> SignalLostVideo,
        link 38 -> OTRVideoPlan) are intentionally still present until
        the deferred video-side cleanbreak sprint.
        """
        doc = self._doc()
        nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}
        offenders: list[str] = []
        for L in doc.get("links", []):
            if not (isinstance(L, list) and len(L) >= 6):
                continue
            _lid, _src, _src_slot, dst, dst_slot, _typ = (
                L[0], L[1], L[2], L[3], L[4], L[5]
            )
            dst_node = nodes_by_id.get(dst)
            if dst_node is None or dst_node.get("type") not in self._VOICE_NODE_TYPES:
                continue
            inputs = dst_node.get("inputs", [])
            if 0 <= dst_slot < len(inputs):
                in_name = inputs[dst_slot].get("name")
                if in_name == "production_plan_json":
                    offenders.append(
                        f"link {L[0]} -> {dst_node['type']}(id={dst})."
                        f"production_plan_json"
                    )
        assert not offenders, (
            "voice-path-cleanbreak violation: production_plan_json "
            "wires still reach voice nodes:\n  "
            + "\n  ".join(offenders)
        )

    def test_voice_nodes_have_no_production_plan_json_input_socket(self):
        """The voice nodes themselves no longer declare a
        ``production_plan_json`` input socket. Any saved workflow that
        still carries the socket has stale wiring."""
        doc = self._doc()
        offenders: list[str] = []
        for n in doc.get("nodes", []):
            if n.get("type") not in self._VOICE_NODE_TYPES:
                continue
            for inp in n.get("inputs", []):
                if inp.get("name") == "production_plan_json":
                    offenders.append(
                        f"{n['type']}(id={n['id']}) declares "
                        f"input.production_plan_json"
                    )
        assert not offenders, (
            "voice-path-cleanbreak violation: production_plan_json input "
            "socket still present on voice nodes:\n  "
            + "\n  ".join(offenders)
        )

    def test_musicgen_script_json_wired_from_freeze_cascade(self):
        """OTR_MusicGenTheme must read its ``script_json`` input from
        ``OTR_LedgerFreezeCascade.script_json``. That edge is what
        gives MusicGen access to meta.gen_params_initial.style and
        meta.news.script_brief for the deterministic palette."""
        doc = self._doc()
        nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}
        # Find the single MusicGen and FreezeCascade nodes.
        musicgens = [
            n for n in doc.get("nodes", [])
            if n.get("type") == "OTR_MusicGenTheme"
        ]
        cascades = [
            n for n in doc.get("nodes", [])
            if n.get("type") == "OTR_LedgerFreezeCascade"
        ]
        assert len(musicgens) == 1, (
            f"expected exactly one OTR_MusicGenTheme; got {len(musicgens)}"
        )
        assert len(cascades) == 1, (
            f"expected exactly one OTR_LedgerFreezeCascade; got {len(cascades)}"
        )
        mg = musicgens[0]
        cas = cascades[0]
        # Locate script_json input socket on MusicGen.
        script_input = next(
            (i for i in mg.get("inputs", []) if i.get("name") == "script_json"),
            None,
        )
        assert script_input is not None, (
            "OTR_MusicGenTheme is missing a script_json input socket"
        )
        link_id = script_input.get("link")
        assert link_id is not None, (
            "OTR_MusicGenTheme.script_json has no incoming link"
        )
        # Resolve the link and verify it originates at FreezeCascade.script_json.
        link_row = next(
            (L for L in doc.get("links", []) if L and L[0] == link_id),
            None,
        )
        assert link_row is not None, (
            f"link {link_id} referenced by MusicGen.script_json not found"
        )
        src_id, src_slot = link_row[1], link_row[2]
        assert src_id == cas["id"], (
            f"MusicGen.script_json wired from node id={src_id}; "
            f"expected FreezeCascade id={cas['id']}"
        )
        src_outputs = cas.get("outputs", [])
        assert 0 <= src_slot < len(src_outputs), (
            f"FreezeCascade has no output slot {src_slot}"
        )
        src_name = src_outputs[src_slot].get("name")
        assert src_name == "script_json", (
            f"MusicGen.script_json wired from "
            f"FreezeCascade.{src_name!r}; expected 'script_json'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
