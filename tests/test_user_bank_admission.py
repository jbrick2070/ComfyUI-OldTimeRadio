"""Client banks admitted by the ONE routing authority, alongside the shipped six.

Independent source banks v1, wave 2. The asymmetry under test: a broken
SHIPPED seed still fails loud (node registration dies), while a broken CLIENT
bundle is quarantined -- absent from the dropdown, reported, boot survives.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_routing as ROUTING
from nodes import _otr_user_banks as ub


REPO = Path(__file__).resolve().parents[1]
DONOR_PACK = (REPO / "nodes" / "story_packs" / "media_archive"
              / "media_restoration_adventure.json")
CLIENT_ID = "client_noir"
CLIENT_MODEL = "client_noir_drama"


@pytest.fixture(autouse=True)
def _fresh_registry():
    ROUTING._clear_caches()
    yield
    ROUTING._clear_caches()


@pytest.fixture
def user_root(tmp_path, monkeypatch):
    """Point the client-bundle discovery at a temp repo base."""
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    monkeypatch.setattr(ub, "snapshots_root",
                        lambda root=None: tmp_path / "user_packs" / ".snapshots")
    return tmp_path


def _client_row(**overrides) -> dict:
    """A client bank row shaped like the shipped media_archive lane.

    It reuses the shipped RSS fetcher/interpreter ids on purpose: v1 admits a
    client bank that leans on a registered lane, and it keeps this test about
    ROUTING rather than about client Python."""
    row = {
        "source_bank_id": CLIENT_ID,
        "label": "Client Noir",
        "source_kind": "archive_item",
        "interpreter": "media_archive_interpreter",
        "fetcher": "media_archive_rss",
        "default_story_model": CLIENT_MODEL,
        "default_story_pipeline": "legacy_many_pass",
        "defaults": {"style_pool_class": "media"},
        "required_seams": [],
        "runnable": True,
        "guide_ref": "",
    }
    row.update(overrides)
    return row


def _build(user_root: Path, *, row=None, activate: bool = True,
           pack_patch=None) -> Path:
    bank_id = (row or _client_row())["source_bank_id"]
    root = ub.user_banks_root() / bank_id
    (root / ub.STORY_PACKS_DIRNAME).mkdir(parents=True)
    (root / ub.BANK_JSON_FILENAME).write_text(
        json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                    "bank": row or _client_row()}, indent=2),
        encoding="utf-8")
    (root / f"{bank_id}.py").write_text(
        "def fetch_source(**kwargs):\n    raise NotImplementedError\n",
        encoding="utf-8")
    pack = json.loads(DONOR_PACK.read_text(encoding="utf-8"))
    pack["source_bank_id"] = bank_id
    pack["story_model_id"] = CLIENT_MODEL
    if pack_patch is not None:
        pack_patch(pack)
    (root / ub.STORY_PACKS_DIRNAME / f"{CLIENT_MODEL}.json").write_text(
        json.dumps(pack, indent=2), encoding="utf-8")
    if activate:
        digest = ub.bundle_digest(root)
        snapshot = ub.snapshot_dirname(bank_id, digest)
        (ub.snapshots_root() / snapshot).mkdir(parents=True, exist_ok=True)
        (root / ub.RECEIPT_FILENAME).write_text(
            json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                        "source_bank_id": bank_id, "digest": digest,
                        "snapshot": snapshot}), encoding="utf-8")
    return root


SHIPPED_IDS = ("media_archive", "original", "scifi_news_pro", "public_domain",
               "shakespeare", "scifi_news", "custom_source_bank")


# -- (a) the stock install is unchanged -------------------------------------

def test_stock_install_lists_exactly_the_shipped_banks():
    assert ROUTING.list_bank_ids() == SHIPPED_IDS
    assert ROUTING.list_validation_issues() == ()


# -- (b) an activated client bank becomes a peer -----------------------------

def test_client_bank_joins_the_dropdown(user_root):
    _build(user_root)
    ids = ROUTING.list_bank_ids()
    assert CLIENT_ID in ids
    assert set(SHIPPED_IDS).issubset(ids)
    assert ROUTING.list_validation_issues() == ()


def test_client_bank_row_resolves_like_a_shipped_row(user_root):
    _build(user_root)
    bank = ROUTING.require_runnable_bank(CLIENT_ID)
    assert bank.source_bank_id == CLIENT_ID
    assert bank.default_story_pipeline == "legacy_many_pass"


def test_client_pack_resolves_from_the_clients_own_folder(user_root):
    root = _build(user_root)
    pack = ROUTING.resolve_story_pack(CLIENT_ID)
    assert pack.source_bank_id == CLIENT_ID
    assert pack.story_model_id == CLIENT_MODEL
    assert ROUTING.user_bank_bundle(CLIENT_ID).root == root
    # ...and a shipped bank still resolves from the shipped root.
    assert ROUTING.resolve_story_pack("media_archive").source_bank_id == "media_archive"


def test_shipped_banks_report_no_bundle(user_root):
    _build(user_root)
    assert ROUTING.user_bank_bundle("media_archive") is None


# -- (c) quarantine: the client never takes the boot down --------------------

def _issue(code_expected: str):
    issues = ROUTING.list_validation_issues()
    assert len(issues) == 1, issues
    assert issues[0].code == code_expected
    return issues[0]


def test_unactivated_bundle_is_absent_from_the_dropdown(user_root):
    _build(user_root, activate=False)
    assert CLIENT_ID not in ROUTING.list_bank_ids()
    assert ROUTING.list_bank_ids() == SHIPPED_IDS
    _issue("missing_receipt")


def test_unknown_pipeline_quarantines_instead_of_killing_boot(user_root):
    _build(user_root, row=_client_row(default_story_pipeline="no_such_pipeline"))
    assert CLIENT_ID not in ROUTING.list_bank_ids()
    issue = _issue("bad_bundle_contract")
    assert "no_such_pipeline" in issue.detail


def test_pack_header_mismatch_quarantines(user_root):
    _build(user_root, pack_patch=lambda p: p.update(story_model_id="wrong_model"))
    assert CLIENT_ID not in ROUTING.list_bank_ids()
    assert _issue("bad_bundle_contract")


def test_missing_required_seam_quarantines(user_root):
    """The row promises a seam its own default pack does not carry."""
    _build(user_root,
           row=_client_row(required_seams=["coda_system"]),
           pack_patch=lambda p: p["prompt_stages"].pop("coda_system", None))
    issue = _issue("bad_bundle_contract")
    assert "required_seams" in issue.detail
    assert "coda_system" in issue.detail
    assert CLIENT_ID not in ROUTING.list_bank_ids()


def test_runnable_without_a_lane_quarantines(user_root):
    _build(user_root, row=_client_row(fetcher="", interpreter=""))
    issue = _issue("bad_bundle_contract")
    assert CLIENT_ID not in ROUTING.list_bank_ids()
    assert "fetcher" in issue.detail or "runner to exist" in issue.detail


def test_unregistered_fetcher_quarantines(user_root):
    _build(user_root, row=_client_row(fetcher="not_a_real_fetcher"))
    issue = _issue("bad_bundle_contract")
    assert "not_a_real_fetcher" in issue.detail


def test_shipped_id_cannot_be_shadowed_by_a_client_bundle(user_root):
    _build(user_root, row=_client_row(source_bank_id="media_archive"))
    _issue("protected_id")
    # the SHIPPED media_archive is untouched
    assert ROUTING.get_bank("media_archive").label == "Media RSS / Archive"


def test_a_broken_bundle_does_not_block_a_healthy_one(user_root):
    _build(user_root)
    broken = ub.user_banks_root() / "client_broken"
    broken.mkdir(parents=True)
    (broken / ub.BANK_JSON_FILENAME).write_text("{oops", encoding="utf-8")
    ids = ROUTING.list_bank_ids()
    assert CLIENT_ID in ids and "client_broken" not in ids
    assert [i.bank_id for i in ROUTING.list_validation_issues()] == ["client_broken"]


# -- (d) the shipped seed keeps its fail-loud posture ------------------------

def test_shipped_seed_problem_still_raises(user_root, monkeypatch, tmp_path):
    """No client-quarantine leniency leaks onto the shipped registry."""
    stray = ROUTING._STORY_PACKS_ROOT / "not_a_registered_bank"
    stray.mkdir()
    try:
        with pytest.raises(ROUTING.RegistryValidationError):
            ROUTING._ensure_loaded()
    finally:
        stray.rmdir()


# -- (e) atomic publish + the two-way cache reset ----------------------------

def test_reentrant_load_fails_loud_instead_of_recursing(monkeypatch):
    monkeypatch.setattr(ROUTING, "_LOADING", True)
    with pytest.raises(ROUTING.RegistryValidationError, match="re-entered"):
        ROUTING._ensure_loaded()


def test_clear_caches_resets_the_loading_flag_and_the_registry():
    ROUTING._ensure_loaded()
    assert ROUTING._REGISTRY is not None
    ROUTING._clear_caches()
    assert ROUTING._REGISTRY is None
    assert ROUTING._LOADING is False


def test_a_client_bank_appears_only_after_a_cache_reset(user_root):
    assert CLIENT_ID not in ROUTING.list_bank_ids()
    _build(user_root)
    assert CLIENT_ID not in ROUTING.list_bank_ids(), "cached registry must be stable"
    ROUTING._clear_caches()
    assert CLIENT_ID in ROUTING.list_bank_ids()
