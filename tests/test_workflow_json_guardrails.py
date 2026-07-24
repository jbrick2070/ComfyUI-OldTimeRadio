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
# Saved-default binding for the OTR_LedgerScriptWriter style widget was
# RETIRED (2026-07-05, style-engine consolidation): style/style_custom
# no longer exist as widgets at all -- the single deterministic engine
# call (build_story_contract()) supplies style, with no dropdown/
# sentinel/LLM-derive path left to drift. TestWriterStyleSentinelDefault
# and its _WRITER_STYLE_SENTINEL / _WRITER_STYLE_SLOT constants are
# deleted rather than repointed at a widget that no longer exists. See
# docs/2026-07-05-style-dropdown-blast-radius/RIP_OUT_PLAN.md.
# ---------------------------------------------------------------------------

_CANONICAL_WORKFLOW = "otr_canonical.json"


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

    def test_no_llm_director_in_workflow(self):
        """Sprint 2.5 (voice-path-cleanbreak). Hard guard: the workflow
        must NOT contain an OTR_LLMDirector node.

        Replaces the conditional Sprint 1.1 guardrail
        (test_no_director_output_links_to_voice_nodes). Sprint 2 deleted
        the Director class + workflow node + registration entirely. If
        someone re-adds the node to the workflow JSON or re-registers
        the class, this gate fails and the rebuild is blocked.

        The void left by Director is filled at writer time:
          - meta.visual_plan + meta.voice_assignments + meta.style are
            stamped by OTR_LedgerScriptWriter (K.5 block).
          - OTR_VideoPlan + OTR_SignalLostVideo read those fields from
            the L3 ledger via FreezeCascade.script_json.
        """
        doc = self._doc()
        director_nodes = [
            n for n in doc.get("nodes", [])
            if n.get("type") == "OTR_LLMDirector"
        ]
        assert not director_nodes, (
            "voice-path-cleanbreak Sprint 2 violation: OTR_LLMDirector "
            "node re-added to workflow. Director class was deleted; see "
            "docs/voice-path-cleanbreak-execution-plan.md Sprint 2."
        )

    def test_musicgen_script_json_wired_from_freeze_cascade(self):
        """OTR_MusicGenTheme must read its ``script_json`` input from
        ``OTR_LedgerFreezeCascade.script_json``. That edge is what
        gives MusicGen access to meta.gen_params_initial.style and
        meta.news.script_brief for the deterministic palette."""
        doc = self._doc()
        nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}
        # Find the single MusicGen and FreezeCascade nodes.
        # Wave 2b: OTR_StableAudioTheme (v2) replaced the legacy
        # OTR_MusicGenTheme instance and reads the raw FreezeCascade script_json
        # for byte-identical batch delegation + cue mood.
        musicgens = [
            n for n in doc.get("nodes", [])
            if n.get("type") == "OTR_StableAudioTheme"
        ]
        cascades = [
            n for n in doc.get("nodes", [])
            if n.get("type") == "OTR_LedgerFreezeCascade"
        ]
        assert len(musicgens) == 1, (
            f"expected exactly one OTR_StableAudioTheme; got {len(musicgens)}"
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


class TestWriterB2aSurface:
    """S30 B2a writer-surface guardrails. Pins the post-B2a invariants
    so a future commit cannot silently regress the two-widget surface
    or the two new output sockets.
    """

    def _doc(self):
        wf_path = WORKFLOWS_DIR / _CANONICAL_WORKFLOW
        assert wf_path.is_file(), (
            f"Canonical workflow {_CANONICAL_WORKFLOW!r} missing"
        )
        return _load_json(wf_path)

    def _writer(self):
        doc = self._doc()
        writers = [
            n for n in doc.get("nodes", [])
            if n.get("type") == "OTR_LedgerScriptWriter"
        ]
        assert len(writers) == 1
        return writers[0]

    def test_writer_output_slot_indexes_stable(self):
        """Every existing link's source slot still resolves to its
        original output name post-B2a. The two new outputs append at
        the end so existing slot_indices for links 106..109 are
        unchanged.
        """
        doc = self._doc()
        nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}
        writer = self._writer()
        expected = {
            0: "script_text",
            1: "script_json",
            2: "news_used",
            3: "estimated_minutes",
            4: "technical_model",
        }
        outputs = writer.get("outputs", [])
        for slot, expected_name in expected.items():
            assert slot < len(outputs), (
                f"writer output slot {slot} missing"
            )
            actual = outputs[slot].get("name")
            assert actual == expected_name, (
                f"writer output slot {slot} -> {actual!r}; "
                f"expected {expected_name!r}"
            )

        # Defense in depth: walk every link sourced at the writer and
        # verify the source slot still resolves to the right name on
        # the writer's outputs list.
        for L in doc.get("links", []):
            if not (isinstance(L, list) and len(L) >= 6):
                continue
            _lid, src, src_slot, _dst, _dst_slot, _typ = L[:6]
            if src != writer["id"]:
                continue
            assert 0 <= src_slot < len(outputs), (
                f"link {_lid} references writer output slot {src_slot} "
                f"out of range {len(outputs)}"
            )

    def test_writer_widget_migration_preserves_values(self):
        """Pin the writer's widgets_values layout. Post BUG-LOCAL-269/270
        the `seed` widget (and its control_after_generate companion) was
        removed. The 2026-07-05 style-engine consolidation then deleted
        the `style` / `style_custom` widgets outright (slots 8/9) -- style
        is no longer a widget at all, it comes from the single
        deterministic build_story_contract() engine call -- shifting
        every slot from `creativity` onward down by 2.
        """
        writer = self._writer()
        wv = writer.get("widgets_values", [])
        # Current writer widgets_values layout:
        #   0  episode_title              ""
        #   1  target_words               350
        #   2  num_characters             2
        #   3  creative_writing_model     catalog default repo_id
        #   4  technical_model            catalog default repo_id
        #   5  custom_premise             ""
        #   6  include_act_breaks         True
        #   7  act_count                  "auto"
        #   8  creativity                 "balanced"
        #   9  perfect_run_spacesaver     False
        #  10  min_p                      0.05
        #  11  repetition_penalty         1.03
        #  12  max_new_tokens_cap         200
        #  13  lemmy_cameo                "roll (~11% chance)"
        #  14  use_exchange                       True      (Build 4)
        #  15  enable_production_stage3_validators True     (Sprint 10B Wave 1 Agent B)
        #  16  news_briefs_required               True      (Sprint 2.2)
        #  17  openrouter_slot_a_model
        #  18  openrouter_slot_b_model
        #  19  comfy_slot_a_model
        #  20  comfy_slot_b_model
        #  21  refine_target_grade
        #  22  story_scaffold
        #  23  source_bank
        #  24  visual_style
        #  25  google_api_slot_a_model
        #  26  google_api_slot_b_model
        #  27  source_ref
        #  28  llm_device                  "cuda"
        #  29  llm_attn_impl               "sdpa"
        #  30  llm_quant_policy            "bnb_nf4"
        #  31  llm_vram_ceiling_gb         14.5
        #  32  gguf_n_ctx                  4096
        #  33  gguf_quant                  "Q8_0"
        # History: `seed` + companion removed 2026-05-25 (BUG-LOCAL-269/270);
        # the 2026-05-29 lean-down brought the vector to 19; S2 (2026-06-01)
        # appended the OpenRouter pair; Comfy Credits appended its sibling
        # pair; the refine loop (2026-06-23) appended refine_target_grade;
        # the story-scaffold toggle (2026-06-24) appended story_scaffold;
        # Stage 2C (2026-07-05) appended source_bank; Stage 3C (2026-07-06)
        # appended visual_style -- bringing the vector to 27. The 2026-07-05
        # style-engine consolidation then DELETED style + style_custom
        # (slots 8/9), dropping the vector to 25 and shifting every
        # subsequent slot down by 2. The Google BYO API direct LLM lane
        # appended two passive model pickers on 2026-07-08, bringing the
        # vector back to 27; Source Banks v2 then appended source_ref as
        # slot 27 without moving any prior slot.
        #
        # S5 platform-portability (2026-07-10): OLD pin 28 -> NEW pin 34.
        # The six explicit LLM runtime-policy widgets (llm_device ..
        # gguf_quant) were appended after source_ref at slots 28-33
        # (append-only, BUG-LOCAL-097). Defaults equal the nv50 16 GB
        # baseline the writer already resolved to, so an old 28-slot
        # workflow still resolves byte-identically. A gate_in forceInput
        # socket was also added (OTR_WorkflowValidator.validation_report,
        # link 279) but it is socket-only and consumes NO widgets_values
        # slot, so the vector ceiling is 34, not 35.
        assert len(wv) == 34, (
            f"writer widgets_values length drift: {len(wv)} (expected 34 "
            f"after appending the six S5 LLM runtime-policy widgets)"
        )
        # Slot 7: act_count must ship as "auto" so production derives the
        # act structure from target_words instead of freezing a brittle count.
        assert wv[7] == "auto", (
            f"act_count (slot 7) must ship 'auto' so it follows target_words; "
            f"got {wv[7]!r}"
        )
        # Slot 14: use_exchange -- the live grouped-exchange dialogue path
        # (ON in the shipped bake).
        assert wv[14] is True, (
            f"use_exchange (slot 14) must be ON in the shipped bake; "
            f"got {wv[14]!r}"
        )
        # Slot 15: enable_production_stage3_validators (ON in the shipped
        # bake).
        assert wv[15] is True, (
            f"enable_production_stage3_validators (slot 15); got {wv[15]!r}"
        )
        # Slot 16: news_briefs_required (ON in the shipped bake).
        assert wv[16] is True, (
            f"news_briefs_required (slot 16); got {wv[16]!r}"
        )
        # Slots 17/18: the S2 OpenRouter slot-slug pickers, appended at the
        # END. PRODUCTION RESTORE 2026-06-10: the shipped bake previously
        # pinned the recommended slugs, but ComfyUI's /prompt validator
        # REJECTS an out-of-catalog value whenever the remote lane is OFF
        # (value_not_in_list) -- the saved file would not queue AS-IS. The
        # bake now ships the DISABLED SENTINELS; an operator who enables a
        # lane picks a slug in the UI and the preservation rule keeps it.
        assert wv[17] == "(enable OpenRouter)", (
            f"openrouter_slot_a_model (slot 17) must ship the disabled "
            f"sentinel (the saved file must queue with lanes off); got "
            f"{wv[17]!r}"
        )
        assert wv[18] == "(enable OpenRouter)", (
            f"openrouter_slot_b_model (slot 18) must ship the disabled "
            f"sentinel; got {wv[18]!r}"
        )
        # Slots 19/20: the Comfy Credits slot-slug pickers -- same sentinel
        # rule as 17/18.
        assert wv[19] == "(enable Comfy Credits)", (
            f"comfy_slot_a_model (slot 19) must ship the disabled sentinel; "
            f"got {wv[19]!r}"
        )
        assert wv[20] == "(enable Comfy Credits)", (
            f"comfy_slot_b_model (slot 20) must ship the disabled sentinel; "
            f"got {wv[20]!r}"
        )
        # Slot 21: refine_target_grade (refine loop v1, 2026-06-23) -- APPENDED
        # at the END; ships "Off" (the loop is default-OFF / byte-identical).
        assert wv[21] == "Off", (
            f"refine_target_grade (slot 21) must ship 'Off' (the refine loop is "
            f"default-OFF in the shipped bake); got {wv[21]!r}"
        )
        # Slot 22: story_scaffold (scaffold toggle, 2026-06-24) -- ships "auto"
        # (follow OTR_ENABLE_STYLE_GRAMMAR / its default).
        assert wv[22] == "auto", (
            f"story_scaffold (slot 22) must ship 'auto' (follow the env/default "
            f"scaffold setting in the shipped bake); got {wv[22]!r}"
        )
        # Slot 23: source_bank (Stage 2C multi-modal story schema, 2026-07-05)
        # -- APPENDED at the END; ships the production story-bank surface AND
        # must be a REGISTERED bank id (cross-checked against the live routing
        # registry so a re-order / typo cannot ship silently). The roster trim
        # (2026-07-17) retired the science_news family; the default lane is now
        # scifi_news (the local-default sci-fi bank).
        assert wv[23] == "scifi_news", (
            f"source_bank (slot 23) must ship 'scifi_news' (the local default "
            f"story lane); got {wv[23]!r}"
        )
        from nodes import _otr_story_routing as _routing
        assert wv[23] in _routing.list_bank_ids(), (
            f"source_bank (slot 23) value {wv[23]!r} is not a registered "
            f"bank id: {_routing.list_bank_ids()!r}"
        )
        # Slot 24: visual_style (Stage 3C multi-modal story schema,
        # 2026-07-06) -- APPENDED at the END; ships the production look AND
        # must be a REGISTERED style id (live registry cross-check).
        assert wv[24] == "sci_fi_radio", (
            f"visual_style (slot 24) must ship 'sci_fi_radio' (the "
            f"production look); got {wv[24]!r}"
        )
        from nodes import _otr_visual_styles as _vstyles
        assert wv[24] in _vstyles.list_style_ids(), (
            f"visual_style (slot 24) value {wv[24]!r} is not a registered "
            f"style id: {_vstyles.list_style_ids()!r}"
        )
        assert wv[25] == "(select Google API model)", (
            f"google_api_slot_a_model (slot 25) must ship the unselected "
            f"sentinel; got {wv[25]!r}"
        )
        assert wv[26] == "(select Google API model)", (
            f"google_api_slot_b_model (slot 26) must ship the unselected "
            f"sentinel; got {wv[26]!r}"
        )
        assert wv[27] == "", (
            f"source_ref (slot 27) must ship blank/inert; got {wv[27]!r}"
        )
        # Slots 28-33: S5 platform-portability (2026-07-10) LLM
        # runtime-policy tail, APPENDED after source_ref. Defaults equal
        # the nv50 16 GB baseline so an old 28-slot workflow resolves
        # byte-identically.
        assert wv[28] == "cuda", (
            f"llm_device (slot 28) must ship 'cuda' (nv50 baseline); "
            f"got {wv[28]!r}"
        )
        assert wv[29] == "sdpa", (
            f"llm_attn_impl (slot 29) must ship 'sdpa'; got {wv[29]!r}"
        )
        assert wv[30] == "bnb_nf4", (
            f"llm_quant_policy (slot 30) must ship 'bnb_nf4'; got {wv[30]!r}"
        )
        assert wv[31] == 14.5, (
            f"llm_vram_ceiling_gb (slot 31) must ship 14.5; got {wv[31]!r}"
        )
        assert wv[32] == 4096, (
            f"gguf_n_ctx (slot 32) must ship 4096; got {wv[32]!r}"
        )
        assert wv[33] == "Q8_0", (
            f"gguf_quant (slot 33) must ship 'Q8_0'; got {wv[33]!r}"
        )
        # Creative + technical slots both bound to a non-empty repo id.
        assert isinstance(wv[3], str) and wv[3], (
            f"creative_writing_model widget value not a non-empty "
            f"string: {wv[3]!r}"
        )
        assert isinstance(wv[4], str) and wv[4], (
            f"technical_model widget value not a non-empty string: "
            f"{wv[4]!r}"
        )
        assert wv[8] == "balanced", (
            f"creativity widget drifted: {wv[8]!r}"
        )
        assert wv[9] is False, (
            f"perfect_run_spacesaver widget drifted from slot 9: {wv[9]!r}"
        )

    def test_writer_broadcasts_normalized_model_ids(self):
        """AST-walk OTR_LedgerScriptWriter source. Both new outputs
        must route through _strip_label_suffix before being broadcast.
        Stripping happens in _resolve_inputs and the writer returns
        resolved["creative_writing_model"] / ["technical_model"]
        unchanged -- so it is enough to verify _strip_label_suffix is
        invoked inside _resolve_inputs.
        """
        import ast

        writer_src = (
            PACK_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(writer_src)
        # Find _resolve_inputs.
        target = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_resolve_inputs"
            ):
                target = node
                break
        assert target is not None, "_resolve_inputs not found"
        # Count _strip_label_suffix calls inside _resolve_inputs.
        strip_calls = 0
        for sub in ast.walk(target):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr == "_strip_label_suffix"
                ):
                    strip_calls += 1
        assert strip_calls >= 2, (
            f"expected _resolve_inputs to call _strip_label_suffix at "
            f"least twice (once per slot); found {strip_calls}"
        )

    def test_writer_uses_slot_scheduler_not_direct_load_llm(self):
        """B2b clean-break: writer.run must NOT call `.load_llm(...)`
        directly. The slot scheduler routes every LLM acquisition
        through request_slot. AST walk over run() looking for
        forbidden direct-load patterns.
        """
        import ast

        writer_src = (
            PACK_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(writer_src)
        run_method = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "OTR_LedgerScriptWriter"
            ):
                for sub in node.body:
                    if (
                        isinstance(sub, ast.FunctionDef)
                        and sub.name == "run"
                    ):
                        run_method = sub
                        break
                break
        assert run_method is not None, (
            "OTR_LedgerScriptWriter.run not found"
        )
        bad_calls: list[str] = []
        for sub in ast.walk(run_method):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Attribute) and fn.attr == "load_llm":
                    bad_calls.append(
                        f"line {getattr(sub, 'lineno', '?')}: "
                        f".load_llm(...)"
                    )
                elif isinstance(fn, ast.Name) and fn.id == "load_llm":
                    bad_calls.append(
                        f"line {getattr(sub, 'lineno', '?')}: "
                        f"load_llm(...)"
                    )
        assert not bad_calls, (
            "B2b post-condition: writer.run must not call load_llm "
            "directly. Route through _SlotScheduler / request_slot:\n  "
            + "\n  ".join(bad_calls)
        )


class TestCascadeB3Surface:
    """S30 B3 cascade-surface guardrails. The cascade's local `model_id`
    widget + 6 phase-toggle widgets are deleted; a `technical_model`
    input socket replaces them, wired from the writer's broadcast
    output. Canonical workflow JSON is the only one with the cascade
    placed; other workflow JSONs are exempt.
    """

    _DELETED_PHASE_WIDGETS = frozenset({
        "enable_phase_3_polish",
        "polish_announcer_beats",
        "enable_phase_4_scene_coherence",
        "enable_phase_4_5_smart_suggestion",
        "enable_phase_5_voice_drift",
        "enable_phase_6_episode_arc",
    })

    def _cascade_class(self):
        # Lazy import: nodes/_otr_*.py do not import torch but pulling
        # the cascade class via `nodes.OTR_LedgerFreezeCascade` walks
        # the package's __init__.py which DOES import torch in some
        # paths. AST-walk the source file instead to keep this test
        # cheap.
        return None

    def test_cascade_has_no_local_model_widget(self):
        """AST scan of OTR_LedgerFreezeCascade.INPUT_TYPES: the
        `optional` block must NOT contain a `model_id` widget. The
        only model-related entry is `technical_model` and it MUST be
        a forceInput socket (no local widget).
        """
        import ast

        src = (
            PACK_ROOT / "nodes" / "OTR_LedgerFreezeCascade.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        # Find OTR_LedgerFreezeCascade.INPUT_TYPES classmethod.
        target = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "OTR_LedgerFreezeCascade"
            ):
                for sub in node.body:
                    if (
                        isinstance(sub, ast.FunctionDef)
                        and sub.name == "INPUT_TYPES"
                    ):
                        target = sub
                        break
                break
        assert target is not None, (
            "OTR_LedgerFreezeCascade.INPUT_TYPES not found"
        )
        # Walk the function body looking for any "model_id" Dict key.
        bad_keys: list[str] = []
        for sub in ast.walk(target):
            if not isinstance(sub, ast.Dict):
                continue
            for k in sub.keys:
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value == "model_id"
                ):
                    bad_keys.append(
                        f"line {getattr(k, 'lineno', '?')}: 'model_id' key"
                    )
        assert not bad_keys, (
            "cascade INPUT_TYPES must not declare a model_id widget "
            "post-B3:\n  " + "\n  ".join(bad_keys)
        )
        # Confirm `technical_model` is present (sanity).
        found_technical = False
        for sub in ast.walk(target):
            if not isinstance(sub, ast.Dict):
                continue
            for k in sub.keys:
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value == "technical_model"
                ):
                    found_technical = True
                    break
        assert found_technical, (
            "cascade INPUT_TYPES must declare a `technical_model` "
            "input socket (forceInput) post-B3"
        )

    def test_cascade_phase_toggles_extinct(self):
        """AST scan: none of the 6 deleted phase-toggle widgets
        (`enable_phase_3_polish`, `polish_announcer_beats`, etc.)
        may appear as a key inside cascade INPUT_TYPES.
        """
        import ast

        src = (
            PACK_ROOT / "nodes" / "OTR_LedgerFreezeCascade.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(src)
        target = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ClassDef)
                and node.name == "OTR_LedgerFreezeCascade"
            ):
                for sub in node.body:
                    if (
                        isinstance(sub, ast.FunctionDef)
                        and sub.name == "INPUT_TYPES"
                    ):
                        target = sub
                        break
                break
        assert target is not None
        bad_keys: list[str] = []
        for sub in ast.walk(target):
            if not isinstance(sub, ast.Dict):
                continue
            for k in sub.keys:
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value in self._DELETED_PHASE_WIDGETS
                ):
                    bad_keys.append(
                        f"line {getattr(k, 'lineno', '?')}: {k.value!r}"
                    )
        assert not bad_keys, (
            "cascade INPUT_TYPES must not carry the deleted phase-"
            "toggle widgets post-B3:\n  " + "\n  ".join(bad_keys)
        )

    def test_cascade_technical_socket_wired_in_canonical_json(self):
        """Canonical workflow JSON must have exactly one link from
        the writer's `technical_model` output (slot 4) to the
        cascade's `technical_model` input. last_link_id reflects the
        new link.
        """
        wf_path = WORKFLOWS_DIR / _CANONICAL_WORKFLOW
        doc = _load_json(wf_path)
        nodes_by_id = {n["id"]: n for n in doc.get("nodes", [])}
        writer = next(
            n for n in doc["nodes"]
            if n.get("type") == "OTR_LedgerScriptWriter"
        )
        cascade = next(
            n for n in doc["nodes"]
            if n.get("type") == "OTR_LedgerFreezeCascade"
        )

        # writer.outputs[5] must broadcast at least one link.
        writer_outputs = writer.get("outputs", [])
        assert len(writer_outputs) >= 5
        tm_output = writer_outputs[4]
        assert tm_output["name"] == "technical_model"
        assert tm_output.get("links"), (
            f"writer.technical_model output must have at least one "
            f"link; got {tm_output.get('links')}"
        )

        # Cascade must have technical_model input slot with a link id.
        cascade_inputs = cascade.get("inputs", [])
        tm_input = next(
            (i for i in cascade_inputs if i.get("name") == "technical_model"),
            None,
        )
        assert tm_input is not None, (
            "cascade must declare a `technical_model` input slot in "
            "the canonical workflow JSON post-B3"
        )
        link_id = tm_input.get("link")
        assert link_id is not None

        # Walk doc["links"] for the link with id == link_id; verify
        # source = writer's id at slot 5.
        link_row = next(
            (L for L in doc.get("links", []) if L and L[0] == link_id),
            None,
        )
        assert link_row is not None, (
            f"link id {link_id} missing from doc['links']"
        )
        _lid, src_id, src_slot, dst_id, _dst_slot, _typ = link_row[:6]
        assert src_id == writer["id"], (
            f"cascade.technical_model wired from node id={src_id}; "
            f"expected writer id={writer['id']}"
        )
        assert src_slot == 4, (
            f"cascade.technical_model wired from writer output slot "
            f"{src_slot}; expected slot 4 (writer's technical_model)"
        )
        assert dst_id == cascade["id"], (
            f"link {link_id} dst={dst_id}; expected cascade id "
            f"{cascade['id']}"
        )

        # last_link_id must reflect at least the new link id.
        assert doc["last_link_id"] >= link_id, (
            f"last_link_id={doc['last_link_id']} < new link "
            f"id={link_id}"
        )

    def test_cascade_widget_vector_trimmed_in_canonical_json(self):
        """The widgets deleted from cascade INPUT_TYPES (model_id +
        6 phase toggles + the ripped VRAM-ceiling widget) must also be
        trimmed from `widgets_values`.

        Sprint 6 (2026-05-25) added 4 critic-to-render coupling widgets;
        the 2026-07-03 VRAM rip then removed the ceiling widget, so the
        vector is 6: phase_7, phase_8, render_selection, render_max_n,
        protagonist_only, manual_line_ids (the operator's tier JSON owns
        the OOM budget now -- there is no hard-baked ceiling widget).
        """
        wf_path = WORKFLOWS_DIR / _CANONICAL_WORKFLOW
        doc = _load_json(wf_path)
        cascade = next(
            n for n in doc["nodes"]
            if n.get("type") == "OTR_LedgerFreezeCascade"
        )
        wv = cascade.get("widgets_values", [])
        assert len(wv) == 6, (
            f"cascade widgets_values length drift: {len(wv)} "
            f"(expected 6 post-VRAM-rip: phase_7, phase_8, "
            f"render_selection, render_max_n, protagonist_only, "
            f"manual_line_ids)"
        )
        # 0 = enable_phase_7_audio_readiness  (bool)
        # 1 = enable_phase_8_video_readiness  (bool)
        # 2 = render_selection                (str: "all"|"dramatic_peaks_only")
        # 3 = render_max_n                    (int)
        # 4 = protagonist_only                (bool)
        # 5 = manual_line_ids                 (str)
        assert isinstance(wv[0], bool)
        assert isinstance(wv[1], bool)
        assert isinstance(wv[2], str)
        assert wv[2] in ("all", "dramatic_peaks_only")
        assert isinstance(wv[3], (int, float))
        assert isinstance(wv[4], bool)
        assert isinstance(wv[5], str)


# ---------------------------------------------------------------------------
# story-ledger DRIFT chunk 3 (2026-06-25): CI drift guards.
#
# (a) WIDGET POSITIONAL/ORDER drift (BUG-LOCAL-097). The file already checks
#     widgets_values TYPING + link integrity + stale dropdown literals, and the
#     OTR_WorkflowValidator checks the widgets_values LENGTH vector
#     (BUG-LOCAL-293). NEITHER checks the saved widget ORDER against the LIVE
#     INPUT_TYPES order. A widget INSERTED mid-list (a non-append edit) leaves
#     every saved widgets_values slot bound to the wrong widget -- silent drift.
#     This zips each node's saved widget-name sequence (from inputs[].widget.name)
#     against the live INPUT_TYPES widget order and fails on any non-append
#     misalignment (a removed / renamed / reordered widget).
# (b) Vintage ledger SCHEMA-COMPAT. A field whose default changes SEMANTICS must
#     fail-loud or derive deterministically -- never silently default to wrong
#     (the sound_palette class). A frozen l3-2026-05-14 ledger must still read
#     cleanly through the new consistency guard with no false positive.
# ---------------------------------------------------------------------------


def _resolve_ncm() -> dict:
    """Resolve NODE_CLASS_MAPPINGS the way OTR_WorkflowValidator does (the
    registry lives on the custom-node ROOT package, not nodes/). Returns {} if
    the env cannot resolve it (the test then skips)."""
    import importlib
    import sys as _sys
    comfy_root = PACK_ROOT.parent.parent  # ...\ComfyUI
    if str(comfy_root) not in _sys.path:
        _sys.path.insert(0, str(comfy_root))
    for modname in ("custom_nodes.ComfyUI-OldTimeRadio", "ComfyUI-OldTimeRadio"):
        try:
            pkg = importlib.import_module(modname)
            ncm = getattr(pkg, "NODE_CLASS_MAPPINGS", {}) or {}
            if ncm:
                return ncm
        except Exception:  # noqa: BLE001
            continue
    return {}


def _live_widget_order(input_types: dict) -> list[str]:
    """Ordered NAMES of the inputs that occupy a widgets_values slot (widget-
    backed AND not forceInput), required then optional -- the canonical widget
    order the saved graph must agree with."""
    from nodes._otr_workflow_validator import (
        _wv_is_widget_backed, _wv_has_force_input,
    )
    req = (input_types.get("required") or {})
    opt = (input_types.get("optional") or {})
    out: list[str] = []
    for name, spec in list(req.items()) + list(opt.items()):
        if _wv_is_widget_backed(spec) and not _wv_has_force_input(spec):
            out.append(name)
    return out


def _is_ordered_subsequence(names, order) -> bool:
    """True iff ``names`` appears, in order, as a subsequence of ``order``."""
    it = iter(order)
    return all(n in it for n in names)


class TestWidgetOrderVsInputTypes:
    """BUG-LOCAL-097: the saved widget ORDER must match the live INPUT_TYPES."""

    def test_saved_widget_names_match_live_input_types_order(self):
        ncm = _resolve_ncm()
        if not ncm:
            pytest.skip("NODE_CLASS_MAPPINGS not resolvable in this env")
        if not WORKFLOW_FILES:
            pytest.skip("no workflow JSONs discovered")

        violations: list[str] = []
        checked = 0
        for path in WORKFLOW_FILES:
            doc = _load_json(path)
            if _classify(doc) != "ui":
                continue
            for node in (doc.get("nodes") or []):
                ntype = node.get("type")
                cls = ncm.get(ntype)
                if cls is None or not hasattr(cls, "INPUT_TYPES"):
                    continue  # non-OTR / unloadable node -- exempt
                try:
                    order = _live_widget_order(cls.INPUT_TYPES() or {})
                except Exception:  # noqa: BLE001
                    continue
                saved = [
                    inp["widget"]["name"]
                    for inp in (node.get("inputs") or [])
                    if isinstance(inp, dict)
                    and isinstance(inp.get("widget"), dict)
                    and inp["widget"].get("name")
                ]
                if not saved:
                    continue
                checked += 1
                if not _is_ordered_subsequence(saved, order):
                    missing = [w for w in saved if w not in order]
                    violations.append(
                        f"{path.name} node {node.get('id')} {ntype}: saved "
                        f"widget order {saved} is not an in-order subsequence "
                        f"of live INPUT_TYPES order {order}"
                        + (f" (removed/renamed: {missing})" if missing else
                           " (reordered)")
                    )
        assert not violations, (
            "widget ORDER drift vs live INPUT_TYPES (a non-append widget edit "
            "rebinds every later slot -- BUG-LOCAL-097):\n  "
            + "\n  ".join(violations)
        )
        assert checked > 0, (
            "the widget-order guard exercised ZERO nodes -- the resolver or the "
            "saved inputs[].widget.name shape changed; the guard is now blind"
        )


class TestVintageLedgerSchemaCompat:
    """A frozen l3-2026-05-14 ledger must read cleanly; a semantics-changing
    default must fail-loud or derive deterministically, never silently wrong."""

    def _vintage_l3_ledger(self) -> dict:
        # A minimal, FROZEN l3-2026-05-14 ledger as a pre-sound_palette /
        # pre-story_contract / pre-consistency_status writer would have emitted.
        return {
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_vintage_l3",
            "cast": [
                {"char_id": "c01", "name": "VEGA", "speaker_role": "character"},
                {"char_id": "c02", "name": "ANNOUNCER",
                 "speaker_role": "announcer"},
            ],
            "lines": [
                {"line_id": "l001", "beat_id": "b001", "char_id": "c01"},
                {"line_id": "l002", "beat_id": "b002", "char_id": "c01"},
            ],
            "meta": {"episode_title": "An Old Signal"},
        }

    def test_schema_version_pinned(self):
        # a SILENT schema bump is forbidden -- this fails loud on a rename.
        from nodes._otr_ledger_freeze import EXPECTED_SCHEMA_VERSION
        assert EXPECTED_SCHEMA_VERSION == "l3-2026-05-14"
        assert self._vintage_l3_ledger()["schema_version"] == EXPECTED_SCHEMA_VERSION

    def test_vintage_ledger_no_false_consistency_defect(self):
        # the new consistency guard (chunk 2) must NOT fire on a vintage ledger
        # that predates story_contract/canon: no source -> no silent-wrong defect.
        from nodes._otr_ledger_consistency import assert_ledger_consistency
        defects = assert_ledger_consistency(
            contract=None, outline=None, castlock=None, canon=None,
            ledger=self._vintage_l3_ledger(),
        )
        assert defects == [], f"vintage ledger tripped the guard: {defects}"

    def test_missing_sound_palette_default_is_deterministic_not_wrong(self):
        # an outline dict with NO sound_palette derives [] deterministically
        # (not garbage); and with no contract sound_world to source, that empty
        # palette is NOT a defect (the regression only fires when a source EXISTS).
        from nodes._otr_canon import episode_canon_from_outline_dict
        from nodes._otr_ledger_consistency import assert_ledger_consistency
        canon = episode_canon_from_outline_dict({
            "title": "An Old Signal", "premise": "A relay wakes.",
            "setting": "a relay", "time_of_day": "night",
        })
        assert canon.sound_palette == []
        # no contract -> the sound_palette row is skipped -> no false positive.
        assert assert_ledger_consistency(canon=canon, ledger={"meta": {}}) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
