"""Stage 2 story-path routing tests (multi-modal story schema).

Proves: (a) the shipped registries load fail-loud and route the science lane
byte-identically, (b) the full registry fail-loud matrix (unknown ids, sweep
violations, list-dup ids, precedence mismatch, required-seam absence), (c) the
cache SPLIT -- a routed permissive load never satisfies a strict load_pack,
(d) run gating on bank.runnable only, (e) LAZY import (zero file I/O at module
import time).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from nodes import _otr_story_pack as sp
from nodes import _otr_story_routing as routing

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fresh_caches():
    routing._clear_caches()
    yield
    routing._clear_caches()


# ---------------------------------------------------------------------------
# synthetic registry builder (tmp_path)
# ---------------------------------------------------------------------------

def _bank_row(**over):
    row = {
        "source_bank_id": "tbank", "label": "T Bank", "source_kind": "test",
        "interpreter": "", "fetcher": "", "default_story_model": "tmodel",
        "default_story_pipeline": "tpipe", "defaults": {},
        "required_seams": ["line_composer_system"], "runnable": False,
        "guide_ref": "",
    }
    row.update(over)
    return row


def _pipe_row(**over):
    row = {
        "story_pipeline_id": "tpipe", "label": "T Pipe", "executable": False,
        # Chunk 3: required bool -- synthetic rows default to the
        # non-contract lane (tests that need it flip it explicitly).
        "requires_source_contract": False,
        "declared_seams": [],
        "passes": [{"pass_id": "p1", "slot": "creative", "seam_refs": [],
                    "description": "d"}],
        "notes": [],
    }
    row.update(over)
    return row


def _pack_obj(**over):
    obj = {
        "source_bank_id": "tbank", "story_model_id": "tmodel",
        "story_pipeline_id": "tpipe", "schema_version": "v2.0",
        "prompt_stages": {"line_composer_system": "hello world"},
    }
    obj.update(over)
    return obj


def _mk_registry(root: Path, banks=None, pipelines=None, packs=None,
                 monkeypatch=None):
    """Write a synthetic story_packs tree and point routing at it."""
    packs_dir = root / "story_packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    (packs_dir / "banks.json").write_text(json.dumps(
        {"schema_version": "v2.0",
         "banks": banks if banks is not None else [_bank_row()]}),
        encoding="utf-8")
    (packs_dir / "pipelines.json").write_text(json.dumps(
        {"schema_version": "v2.0",
         "pipelines": pipelines if pipelines is not None else [_pipe_row()]}),
        encoding="utf-8")
    for rel, obj in (packs if packs is not None
                     else {"tbank/tmodel.json": _pack_obj()}).items():
        p = packs_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj) if isinstance(obj, dict) else obj,
                     encoding="utf-8")
    monkeypatch.setattr(routing, "_STORY_PACKS_ROOT", packs_dir)
    return packs_dir


# ---------------------------------------------------------------------------
# (a) shipped registries + science routing
# ---------------------------------------------------------------------------


def test_shipped_media_archive_is_runnable_gate():
    assert routing.require_runnable_bank("media_archive").runnable is True


def test_unknown_ids_raise():
    with pytest.raises(routing.UnknownBankError):
        routing.get_bank("nope")
    with pytest.raises(routing.UnknownPipelineError):
        routing.get_pipeline("nope")
    with pytest.raises(routing.UnknownStoryModelError):
        routing.resolve_story_pack("media_archive", "no_such_model")


# ---------------------------------------------------------------------------
# (a2) Chunk 2B: the shipped lane packs (exact key sets pinned)
# ---------------------------------------------------------------------------

LANE_PACK_KEYS = {
    ("media_archive", "media_restoration_adventure"):
        frozenset({
            "outline_macro_system", "outline_phase_system",
            "outline_beat_system", "line_composer_system",
            "exchange_system", "coda_system", "announcer_intro_system",
            "announcer_intro_safe_system", "announcer_outro_system",
        }),
    ("public_domain", "faithful_radio_adaptation"):
        frozenset({
            "outline_macro_system", "outline_phase_system",
            "outline_beat_system", "line_composer_system",
            "exchange_system", "coda_system", "announcer_intro_system",
            "announcer_intro_safe_system", "announcer_outro_system",
        }),
    ("custom_source_bank", "simple_4_prompt_experimental"):
        frozenset({"pass_1_creative_story", "pass_2_creative_ledger_fill",
                   "pass_3_technical_schema_cleanup",
                   "pass_4_technical_ledger_audit"}),
}


@pytest.mark.parametrize("bank_id,model_id", sorted(LANE_PACK_KEYS))
def test_lane_pack_resolves_with_exact_keys(bank_id, model_id):
    pack = routing.resolve_story_pack(bank_id, model_id)
    assert frozenset(pack.prompt_stages) == LANE_PACK_KEYS[(bank_id, model_id)]
    assert pack.source_bank_id == bank_id
    assert pack.story_model_id == model_id
    # also reachable as the bank default:
    assert routing.get_bank(bank_id).default_story_model == model_id


@pytest.mark.parametrize("bank_id", ["custom_source_bank"])
def test_lane_bank_not_runnable(bank_id):
    """The lane execution paths do not exist yet: run-intent must raise LOUD
    (never a silent fall-through to the science path)."""
    with pytest.raises(routing.StoryBankNotRunnableError):
        routing.require_runnable_bank(bank_id)


def test_public_domain_lane_is_runnable():
    """Public-domain now has fetcher, interpreter, story rules, default
    source_ref, and all production seams, so it can pass the run-intent gate.
    Roster trim 2026-07-17: the base id is retired; the surviving lane is
    public_domain."""
    assert routing.require_runnable_bank("public_domain").runnable is True


def test_media_archive_lane_is_runnable():
    """The media_archive RSS lane now has fetcher/interpreter/story-rules
    contracts and can pass the run-intent gate."""
    assert routing.require_runnable_bank("media_archive").runnable is True


def test_media_archive_drama_seeds_sidecar_is_not_a_story_pack():
    """The seed deck is JSON-owned config beside the media pack, not a
    story_model. The registry sweep should admit it without making it routable.
    """
    assert routing.require_runnable_bank("media_archive").runnable is True
    with pytest.raises(routing.UnknownStoryModelError):
        routing.resolve_story_pack("media_archive", "drama_seeds")


# ---------------------------------------------------------------------------
# (b) fail-loud matrix on a synthetic registry
# ---------------------------------------------------------------------------

def test_synthetic_registry_happy_path(tmp_path, monkeypatch):
    _mk_registry(tmp_path, monkeypatch=monkeypatch)
    pack = routing.resolve_story_pack("tbank")
    assert pack.prompt_stages["line_composer_system"] == "hello world"


def test_non_runnable_bank_run_intent_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path, monkeypatch=monkeypatch)
    with pytest.raises(routing.StoryBankNotRunnableError):
        routing.require_runnable_bank("tbank")
    # resolution (addressability) is still allowed:
    assert routing.resolve_story_pack("tbank") is not None


def test_missing_registry_files_raise(tmp_path, monkeypatch):
    empty = tmp_path / "story_packs"
    empty.mkdir()
    monkeypatch.setattr(routing, "_STORY_PACKS_ROOT", empty)
    with pytest.raises(routing.RegistryValidationError):
        routing.get_bank("anything")


def test_duplicate_bank_ids_raise(tmp_path, monkeypatch):
    _mk_registry(tmp_path, banks=[_bank_row(), _bank_row()],
                 monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError, match="duplicate id"):
        routing.get_bank("tbank")


def test_duplicate_pipeline_ids_raise(tmp_path, monkeypatch):
    _mk_registry(tmp_path, pipelines=[_pipe_row(), _pipe_row()],
                 monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError, match="duplicate id"):
        routing.get_bank("tbank")


def test_unknown_pack_directory_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path, packs={
        "tbank/tmodel.json": _pack_obj(),
        "ghost_bank/x.json": _pack_obj(source_bank_id="ghost_bank",
                                       story_model_id="x"),
    }, monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError,
                       match="not a registered source_bank_id"):
        routing.get_bank("tbank")


def test_unexpected_toplevel_file_raises(tmp_path, monkeypatch):
    root = _mk_registry(tmp_path, monkeypatch=monkeypatch)
    (root / "stray.json").write_text("{}", encoding="utf-8")
    with pytest.raises(routing.RegistryValidationError, match="unexpected file"):
        routing.get_bank("tbank")


def test_misfiled_pack_header_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path, packs={
        "tbank/tmodel.json": _pack_obj(),
        "tbank/other.json": _pack_obj(story_model_id="NOT_other"),
    }, monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError,
                       match="does not match path coordinates"):
        routing.get_bank("tbank")


def test_default_model_missing_pack_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path, banks=[_bank_row(default_story_model="absent")],
                 monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError, match="has no pack at"):
        routing.get_bank("tbank")


def test_pipeline_precedence_mismatch_raises(tmp_path, monkeypatch):
    _mk_registry(
        tmp_path,
        pipelines=[_pipe_row(), _pipe_row(story_pipeline_id="tpipe2")],
        banks=[_bank_row(default_story_pipeline="tpipe2")],
        monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError,
                       match="!= bank default_story_pipeline"):
        routing.get_bank("tbank")


def test_required_seam_absent_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path,
                 banks=[_bank_row(required_seams=["coda_system"])],
                 monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError,
                       match="required_seams"):
        routing.get_bank("tbank")


def test_non_production_required_seam_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path,
                 banks=[_bank_row(required_seams=["made_up_seam"])],
                 monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError,
                       match="non-production seam"):
        routing.get_bank("tbank")


def test_pack_pipeline_unregistered_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path, packs={
        "tbank/tmodel.json": _pack_obj(story_pipeline_id="ghost_pipe"),
    }, monkeypatch=monkeypatch)
    with pytest.raises(routing.UnknownPipelineError):
        routing.get_bank("tbank")


def test_declared_seams_shadowing_production_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path,
                 pipelines=[_pipe_row(declared_seams=["line_composer_system"])],
                 monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError, match="shadow"):
        routing.get_bank("tbank")


def test_pipeline_seam_ref_outside_universe_raises(tmp_path, monkeypatch):
    _mk_registry(
        tmp_path,
        pipelines=[_pipe_row(passes=[{
            "pass_id": "p1", "slot": "creative",
            "seam_refs": ["nope_seam"], "description": "d"}])],
        monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError, match="seam_refs"):
        routing.get_bank("tbank")


def test_unknown_bank_key_raises(tmp_path, monkeypatch):
    _mk_registry(tmp_path, banks=[_bank_row(default_visual_style="x")],
                 monkeypatch=monkeypatch)
    with pytest.raises(routing.RegistryValidationError, match="unknown key"):
        routing.get_bank("tbank")


def test_malformed_registry_json_raises(tmp_path, monkeypatch):
    root = _mk_registry(tmp_path, monkeypatch=monkeypatch)
    (root / "banks.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(routing.RegistryValidationError, match="malformed JSON"):
        routing.get_bank("tbank")


# ---------------------------------------------------------------------------
# (c) cache split: permissive routed load never satisfies strict load_pack
# ---------------------------------------------------------------------------


def test_load_pack_with_seams_rejects_non_frozenset():
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack_with_seams("whatever.json", ["list_not_frozenset"])


# ---------------------------------------------------------------------------
# (e) lazy import: zero file I/O at module import time
# ---------------------------------------------------------------------------

def test_module_import_performs_no_io(monkeypatch):
    """Importing _otr_story_routing must not read the registries (ComfyUI
    import isolation): a broken Path.read_text cannot break the import."""
    def _boom(self, *a, **k):
        raise AssertionError(f"import-time file I/O attempted: {self}")

    monkeypatch.setattr(Path, "read_text", _boom)
    try:
        # reload re-executes the module body IN PLACE (same module object,
        # same __dict__) -- proving zero import-time I/O without forking the
        # module identity other importers (the router) already hold.
        mod = importlib.reload(routing)
        assert mod is routing
        assert mod._REGISTRY is None  # untouched until first access
    finally:
        monkeypatch.undo()
