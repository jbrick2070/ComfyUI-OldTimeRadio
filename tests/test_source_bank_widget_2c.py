"""tests/test_source_bank_widget_2c.py

Multi-modal story schema STAGE 2 CHUNK 2C -- the `source_bank` selector
widget on OTR_LedgerScriptWriter (kibitz-converged plan,
kibitz-runs/2026-07-05-multimodal-2c/r4/final.md).

Pins:
  1. Widget surface: source_bank sits immediately before source_ref and
     visual_style, in that order (the 2026-09-14 writer reorder grouped
     widgets by section rather than by historical append order, so the two
     Google API selectors now sit beside the other model-brain pickers --
     comfy_slot_a_model / comfy_slot_b_model -- not beside source_bank);
     choices come LIVE from the routing registry (exact list, registry
     order, including non-runnable custom banks -- the honest-error
     contract); canvas default is the roll command, same as every shipping
     JSON. source_bank is also the sole
     `required` entry as of the same reorder -- every other writer input,
     episode_title and num_characters included, is `optional`.
  2. Registration fail-loud: a broken registry RAISES out of INPUT_TYPES
     (deliberate exception to the "INPUT_TYPES must never raise"
     convention; no baked-in fallback choice list).
  3. Gate-first ordering: a non-runnable source_bank pick raises
     StoryBankNotRunnableError as the FIRST act of run() -- before the
     story-scaffold env mutation and before _resolve_inputs (RSS fetch).
  4. Threading: resolve_creative_system_prompt(source_bank_id=...) selects
     the pack; compose_line/compose_line_draft thread it end-to-end, and
     every recursive compose_line self-call forwards it (AST pin).
  5. _resolve_inputs carries source_bank as the one authoritative value.
  6. Headless surface: source_bank is on both CREATIVE_WHITELISTs and
     patch_widget_by_name lands it on the canonical workflow's own
     source_bank widget -- resolved by NAME from the node's descriptors
     (tests/_support/writer_slots.py), never by a hardcoded index. Four
     migrations have shifted these positions; each one left a stale numbered
     comment behind, and a drifted index mostly lands on a neighbouring ""
     or False and keeps passing while checking nothing.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nodes import _otr_story_routing as routing  # noqa: E402
from nodes import OTR_LedgerScriptWriter as W_mod  # noqa: E402
from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    OTR_LedgerScriptWriter,
    _resolve_inputs,
)
from nodes._otr_creative_prompt_router import (  # noqa: E402
    resolve_creative_system_prompt,
)
from tests._support.writer_slots import assert_relative_order, value  # noqa: E402

_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_canonical.json"
_PUBLIC_DOMAIN_BANK = "public_domain"
_NON_RUNNABLE_BANK = "custom_source_bank"


# ---------------------------------------------------------------------------
# 1. Widget surface
# ---------------------------------------------------------------------------
class TestWidgetSurface:
    def test_source_bank_positional_pin(self):
        # The claim is about the GROUP, not about where the group starts.
        # Stage 3C (2026-07-06) appended visual_style after source_bank;
        # Google API (2026-07-08) appended its selector pair after
        # visual_style; Source Banks v2 appended source_ref after those --
        # that used to be a run of five that stayed together. The 2026-09-14
        # writer reorder deliberately broke that up: widgets now group by
        # section ("what are we making?" vs. "which brain writes it?")
        # instead of by historical append order, so the two Google API
        # selectors moved next to comfy_slot_a_model/comfy_slot_b_model, and
        # source_ref moved in BETWEEN source_bank and visual_style. What
        # survives -- and what stays true across the next add or removal
        # earlier in the node -- is these two smaller groups.
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        order = list(spec["required"].keys()) + list(spec["optional"].keys())
        assert_relative_order(order, [
            "source_bank",
            "source_ref",
            "visual_style",
        ])
        assert_relative_order(order, [
            "comfy_slot_a_model",
            "comfy_slot_b_model",
            "google_api_slot_a_model",
            "google_api_slot_b_model",
        ])

    def test_choices_are_the_roll_sentinel_then_the_registry_in_order(self):
        """2026-07-31: the randomizer command is PREPENDED as choice 0.

        It is a UI command, not a registry row -- everything after it is
        still exactly the registry, in registry order.
        """
        from nodes import _otr_rolls as rolls

        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        # source_bank is the sole `required` entry as of the 2026-09-14
        # writer reorder -- it no longer lives in `optional`.
        choices, meta = spec["required"]["source_bank"]
        assert choices[0] == rolls.BANK_SENTINEL
        assert choices[1:] == list(routing.list_bank_ids())
        assert rolls.BANK_SENTINEL not in routing.list_bank_ids()
        assert meta["default"] == rolls.BANK_SENTINEL
        # The honest-error contract: non-runnable custom banks ARE listed.
        assert _NON_RUNNABLE_BANK in choices
        assert _PUBLIC_DOMAIN_BANK in choices

    def test_default_is_a_runnable_bank(self):
        bank = routing.require_runnable_bank("scifi_news_pro")
        assert bank.runnable is True


# ---------------------------------------------------------------------------
# 2. Registration fail-loud (no fallback choice list)
# ---------------------------------------------------------------------------
class TestRegistrationFailLoud:
    def test_broken_registry_raises_out_of_input_types(self, monkeypatch):
        def _boom():
            raise routing.RegistryValidationError(
                "test: banks.json unreadable")
        monkeypatch.setattr(W_mod._otr_story_routing, "list_bank_ids", _boom)
        with pytest.raises(routing.StoryRoutingError):
            OTR_LedgerScriptWriter.INPUT_TYPES()


# ---------------------------------------------------------------------------
# 3. Gate-first ordering
# ---------------------------------------------------------------------------
class TestGateFirst:
    def test_non_runnable_pick_raises_before_any_side_effect(
            self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            W_mod, "_apply_story_scaffold_env",
            lambda *_a, **_k: calls.append("scaffold_env") or "auto")
        monkeypatch.setattr(
            W_mod, "_resolve_inputs",
            lambda *_a, **_k: calls.append("resolve_inputs") or {})
        node = OTR_LedgerScriptWriter()
        with pytest.raises(routing.StoryBankNotRunnableError) as ei:
            node.run(source_bank=_NON_RUNNABLE_BANK)
        assert _NON_RUNNABLE_BANK in str(ei.value)
        assert calls == [], (
            f"run() performed side effects before the runnable gate: {calls}"
        )

    def test_unknown_bank_raises_unknown_bank_error(self):
        node = OTR_LedgerScriptWriter()
        with pytest.raises(routing.UnknownBankError):
            node.run(source_bank="no_such_bank")


# ---------------------------------------------------------------------------
# 4. Threading
# ---------------------------------------------------------------------------
# 5. Threading
# ---------------------------------------------------------------------------
class TestThreading:
    def test_resolver_routes_the_selected_bank(self):
        science = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407",
            phase="line_composer_system")
        other = resolve_creative_system_prompt(
            "mistralai/Mistral-Nemo-Instruct-2407",
            phase="line_composer_system",
            source_bank_id=_PUBLIC_DOMAIN_BANK)
        assert science != other, (
            "source_bank_id did not change the resolved prompt -- the "
            "widget would be dead"
        )
        # Cross-check against the lane pack on disk.
        pack_path = (_REPO / "nodes" / "story_packs" / _PUBLIC_DOMAIN_BANK /
                     "faithful_radio_adaptation.json")
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        assert other == pack["prompt_stages"]["line_composer_system"]

    def test_compose_line_threads_source_bank(self, monkeypatch):
        from nodes import _otr_line_composer as lc
        from nodes import _otr_creative_prompt_router as router
        seen = []
        real = router.resolve_creative_system_prompt

        def _spy(repo_id, phase, source_bank_id="media_archive"):
            seen.append(source_bank_id)
            return real(repo_id, phase, source_bank_id=source_bank_id)

        monkeypatch.setattr(
            router, "resolve_creative_system_prompt", _spy)
        req = lc.LineRequest(
            speaker="MARGOT",
            intent="steady the room",
            mood="calm",
            canon_header="",
            last_lines=[],
        )
        out = lc.compose_line(
            creative_fn=lambda *args, **kwargs: (
                "A quiet line about the machine."),
            req=req,
            creative_repo_id="mistralai/Mistral-Nemo-Instruct-2407",
            source_bank_id=_PUBLIC_DOMAIN_BANK,
        )
        assert out.text
        assert seen == [_PUBLIC_DOMAIN_BANK]


    def test_compose_line_forwards_bank_to_draft(self):
        src = (_REPO / "nodes" / "_otr_line_composer.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        fn = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "compose_line"
        )
        draft_calls = [
            call for call in ast.walk(fn)
            if isinstance(call, ast.Call)
            and getattr(call.func, "id", getattr(call.func, "attr", ""))
            == "compose_line_draft"
        ]
        assert len(draft_calls) == 1
        assert "source_bank_id" in {
            keyword.arg for keyword in draft_calls[0].keywords
        }


    def test_writer_call_sites_pass_the_resolved_bank(self):
        # AST pin: every _OTRLC.compose_line( call in the writer passes
        # source_bank_id.
        src = (_REPO / "nodes" / "OTR_LedgerScriptWriter.py").read_text(
            encoding="utf-8")
        tree = ast.parse(src)
        sites = 0
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            f = call.func
            if (isinstance(f, ast.Attribute) and f.attr == "compose_line"
                    and getattr(f.value, "id", "") == "_OTRLC"):
                kwarg_names = {k.arg for k in call.keywords}
                assert "source_bank_id" in kwarg_names, (
                    f"writer compose_line call at line {call.lineno} "
                    f"missing source_bank_id"
                )
                sites += 1
        assert sites == 2, f"expected 2 writer compose_line sites, {sites}"


# ---------------------------------------------------------------------------
# 6. Resolved surface
# ---------------------------------------------------------------------------
class TestResolvedSurface:
    def test_resolve_inputs_carries_source_bank(self):
        resolved = _resolve_inputs(custom_premise="test premise")
        assert resolved["source_bank"] == "scifi_news_pro"
        resolved2 = _resolve_inputs(
            custom_premise="test premise", source_bank=_PUBLIC_DOMAIN_BANK)
        assert resolved2["source_bank"] == _PUBLIC_DOMAIN_BANK


# ---------------------------------------------------------------------------
# 7. Headless surface
# ---------------------------------------------------------------------------
class TestHeadlessSurface:
    def test_source_bank_on_both_whitelists(self):
        from nodes._otr_workflow_apply import CREATIVE_WHITELIST as pkg_wl
        import otr_api
        assert "source_bank" in pkg_wl
        assert "source_bank" in otr_api.CREATIVE_WHITELIST

    def test_patch_widget_by_name_lands_on_source_bank(self):
        import otr_api
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        schemas = {
            "OTR_LedgerScriptWriter": {
                "input": {
                    "required": spec["required"],
                    "optional": spec["optional"],
                },
            },
        }
        workflow = otr_api.load_workflow(str(_CANONICAL_WORKFLOW))
        otr_api.patch_widget_by_name(
            workflow, 1, "source_bank", "scifi_news_pro", schemas)
        node1 = next(n for n in workflow["nodes"] if n["id"] == 1)
        # The patch must land on source_bank itself and leave its neighbours
        # -- the two Google API selectors, source_ref, and the llm
        # runtime-policy block -- exactly as the canonical saved them. Each
        # value is resolved from the node's own widget descriptors, so this
        # still reads the widget it names after the next migration moves it.
        assert len(node1["widgets_values"]) == 35
        assert value(node1, "source_bank") == "scifi_news_pro"
        assert value(node1, "google_api_slot_a_model") == (
            "(select Google API model)")
        assert value(node1, "google_api_slot_b_model") == (
            "(select Google API model)")
        assert value(node1, "source_ref") == ""
        # NOT "cuda": the canonical is retargeted to whichever machine is
        # under test (operator ruling 2026-09-07), so pin the WIDGET, not the
        # pick.
        _llm_device_options = spec["optional"]["llm_device"][0]
        assert value(node1, "llm_device") in _llm_device_options, (
            "llm_device holds %r, which is not one of %r"
            % (value(node1, "llm_device"), _llm_device_options))


# ---------------------------------------------------------------------------
# 8. Independent source banks v1, WAVE 7 -- a client bank reaches THIS widget
#
# Waves 1-6 proved a client bundle joins the routing registry
# (test_user_bank_admission). That is one hop short of the operator: the
# registry only matters if its rows reach the COMBO the Story Writer publishes
# to ComfyUI. The wave-7 question was whether that last hop needed a new
# widget, a new node, or a canonical-JSON change; these pins are the answer --
# it needs NONE, because the choices are read live from the registry and the
# pack comes from the row's own default_story_model. The surface was already
# right; what was missing was the proof and the signpost.
# ---------------------------------------------------------------------------
class TestClientBankReachesTheWidget:
    _CLIENT_ID = "client_widget_probe"
    _CLIENT_MODEL = "client_widget_probe_drama"
    _DONOR_PACK = (_REPO / "nodes" / "story_packs" / "media_archive"
                   / "media_restoration_adventure.json")

    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        routing._clear_caches()
        yield
        routing._clear_caches()

    @pytest.fixture
    def client_bank(self, tmp_path, monkeypatch):
        """An ACTIVATED client bundle, discovered from a temp repo base."""
        from nodes import _otr_user_banks as ub
        monkeypatch.setattr(
            ub, "user_banks_root",
            lambda root=None: tmp_path / "user_packs" / "source_banks")
        monkeypatch.setattr(
            ub, "snapshots_root",
            lambda root=None: tmp_path / "user_packs" / ".snapshots")
        root = ub.user_banks_root() / self._CLIENT_ID
        (root / ub.STORY_PACKS_DIRNAME).mkdir(parents=True)
        (root / ub.BANK_JSON_FILENAME).write_text(
            json.dumps({
                "schema_version": ub.RECEIPT_SCHEMA_VERSION,
                "bank": {
                    "source_bank_id": self._CLIENT_ID,
                    "label": "Client Widget Probe",
                    "source_kind": "archive_item",
                    "interpreter": "media_archive_interpreter",
                    "fetcher": "media_archive_rss",
                    "default_story_model": self._CLIENT_MODEL,
                    "default_story_pipeline": "legacy_many_pass",
                    "defaults": {"style_pool_class": "media"},
                    "required_seams": [],
                    "runnable": True,
                    "guide_ref": "",
                },
            }, indent=2),
            encoding="utf-8")
        (root / f"{self._CLIENT_ID}.py").write_text(
            "def fetch_source(**kwargs):\n    raise NotImplementedError\n",
            encoding="utf-8")
        pack = json.loads(self._DONOR_PACK.read_text(encoding="utf-8"))
        pack["source_bank_id"] = self._CLIENT_ID
        pack["story_model_id"] = self._CLIENT_MODEL
        (root / ub.STORY_PACKS_DIRNAME / f"{self._CLIENT_MODEL}.json").write_text(
            json.dumps(pack, indent=2), encoding="utf-8")
        digest = ub.bundle_digest(root)
        snapshot = ub.snapshot_dirname(self._CLIENT_ID, digest)
        (ub.snapshots_root() / snapshot).mkdir(parents=True, exist_ok=True)
        (root / ub.RECEIPT_FILENAME).write_text(
            json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                        "source_bank_id": self._CLIENT_ID, "digest": digest,
                        "snapshot": snapshot}), encoding="utf-8")
        return root

    def test_client_bank_is_a_choice_on_the_published_widget(self, client_bank):
        """The wave-7 pin: the operator can SELECT an activated client bank."""
        # source_bank is the sole `required` entry as of the 2026-09-14
        # writer reorder -- it no longer lives in `optional`.
        choices, meta = OTR_LedgerScriptWriter.INPUT_TYPES()["required"][
            "source_bank"]
        assert self._CLIENT_ID in choices
        # It joins as a peer of the shipped rows. The canvas default is roll.
        assert _PUBLIC_DOMAIN_BANK in choices
        assert _NON_RUNNABLE_BANK in choices
        from nodes import _otr_rolls as rolls
        assert meta["default"] == rolls.BANK_SENTINEL
        assert self._CLIENT_ID in rolls.eligible_bank_ids()
        # Choice 0 is the roll command (a UI command, never a registry row);
        # everything after it is exactly the registry, client row included.
        assert choices[0] == rolls.BANK_SENTINEL
        assert choices[1:] == list(routing.list_bank_ids())

    def test_widget_value_routes_to_the_clients_own_pack(self, client_bank):
        """No pack widget exists or is needed: the row's default resolves,
        inside the client's own bundle rather than any shipped directory."""
        resolved = _resolve_inputs(custom_premise="test premise",
                                   source_bank=self._CLIENT_ID)
        assert resolved["source_bank"] == self._CLIENT_ID
        pack = routing.resolve_story_pack(resolved["source_bank"])
        assert pack.source_bank_id == self._CLIENT_ID
        assert pack.story_model_id == self._CLIENT_MODEL
        bundle_root = routing.user_bank_bundle(self._CLIENT_ID).root
        assert bundle_root == client_bank

    def test_adding_a_bank_changes_no_canonical_widget_vector(self, client_bank):
        """Admitting a bank must not disturb the stored workflow: the canonical
        widget vector is positional (BUG-LOCAL-097) and a client bank adds a
        legal VALUE, never a slot."""
        workflow = json.loads(
            _CANONICAL_WORKFLOW.read_text(encoding="utf-8"))
        node1 = next(n for n in workflow["nodes"] if n["id"] == 1)
        assert len(node1["widgets_values"]) == 35
        # 2026-08-15 (operator): canonical ships the roll sentinel here. The
        # point of this assertion is that admitting a client bank does not
        # disturb the source_bank widget, whatever legal value it holds.
        assert value(node1, "source_bank") == "roll (any eligible bank)"
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        order = list(spec["required"].keys()) + list(spec["optional"].keys())
        # source_ref now sits between source_bank and visual_style (the
        # 2026-09-14 writer reorder) -- same claim, new neighbours.
        assert_relative_order(order, ["source_bank", "source_ref", "visual_style"])


# ---------------------------------------------------------------------------
# 9. The signpost row answers with the path, not a dead end (wave 7)
# ---------------------------------------------------------------------------
class TestAddYourOwnSignpost:
    def test_non_runnable_error_carries_the_rows_guide_ref(self):
        """`guide_ref` had no runtime consumer before wave 7. The one row that
        exists to advertise extensibility must not answer a click with a dead
        end -- JSON owns the words, require_runnable_bank raises them."""
        bank = routing.get_bank(_NON_RUNNABLE_BANK)
        assert bank.guide_ref, "the signpost row must carry a guide_ref"
        with pytest.raises(routing.StoryBankNotRunnableError) as ei:
            routing.require_runnable_bank(_NON_RUNNABLE_BANK)
        message = str(ei.value)
        assert bank.guide_ref in message
        # It names the landed path, not a runner that never shipped.
        assert "otr_check bank" in message
        # apple/EXTENDING.md, not docs/EXTENDING_OTR.md, and the move was a FIX
        # rather than churn: `.comfyignore` excludes the whole `docs/` tree from
        # the published bundle, so a registry user who clicked "+ Add Your Own"
        # was pointed at a file their install had never received. `apple/` is
        # not excluded and does ship, which is what makes the row's own promise
        # ("the contract is apple/EXTENDING.md, which ships with the pack")
        # true. Both files exist on GitHub; only one of them reaches a user.
        assert "apple/EXTENDING.md" in message
        assert "user_packs/source_banks/" in message

    def test_runnable_bank_message_does_not_misname_the_client_row_location(
            self):
        """A client bundle's row lives in its own bank.json, so the error may
        not tell every operator to edit banks.json (the 8c45172d defect)."""
        with pytest.raises(routing.StoryBankNotRunnableError) as ei:
            routing.require_runnable_bank(_NON_RUNNABLE_BANK)
        assert "banks.json" not in str(ei.value)
        assert "its bank row" in str(ei.value)
