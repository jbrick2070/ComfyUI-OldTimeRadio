"""Client-owned fetch_source / interpret_source execution.

Independent source banks v1, wave 3. The laws under test:

  * a CLIENT row may route an entry point to its own bundle with the reserved
    self id; a SHIPPED row that says it is still an unregistered typo;
  * the shipped registries never learn the client id -- no shadowing, no
    extension, no replacement;
  * a bank may execute only its OWN bundle (owner identity, not mere presence);
  * loading or calling client Python fails LOUD -- discovery already
    quarantined the broken bundles, so at execution time there is nobody left
    to silently fall back to;
  * a client result still passes normalize_fetch_result /
    validate_interpreter_result unchanged -- client code never reaches the
    ledger.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_source_payload as osp
from nodes import _otr_story_routing as ROUTING
from nodes import _otr_user_banks as ub


REPO = Path(__file__).resolve().parents[1]
CLIENT_ID = "client_selfrun"

# A minimal bundle module that honours the shipped keyword contract exactly.
GOOD_MODULE = '''
def fetch_source(*, bank, technical_model, source_ref="",
                 load_config=None, policy=None):
    return {
        "headline": "Client Headline",
        "summary": "Client summary.",
        "full_text": "Client full text.",
        "source": "Client Bundle",
        "date": "2026-07-24",
        "link": "",
        "seed_text": "a client-authored seed",
    }


class _Briefs:
    casting_brief = "Cast the client's people."
    script_brief = "Tell the client's story."
    news_close_brief = "That was the client's story."
    key_terms = ["client", "bundle"]
    attempts = 1

    def model_dump(self):
        return {
            "casting_brief": self.casting_brief,
            "script_brief": self.script_brief,
            "news_close_brief": self.news_close_brief,
            "key_terms": list(self.key_terms),
            "attempts": self.attempts,
        }


def interpret_source(*, bank, payload, technical_fn, model_id):
    return _Briefs()


def check_compatibility(**kwargs):
    return True
'''


class _Row:
    """Duck-typed bank row -- resolution reads three attributes, nothing more."""

    def __init__(self, bank_id=CLIENT_ID, *, fetcher="", interpreter=""):
        self.source_bank_id = bank_id
        self.fetcher = fetcher
        self.interpreter = interpreter


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """An ADMITTED client bundle whose module is the good one above."""
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    monkeypatch.setattr(ub, "snapshots_root",
                        lambda root=None: tmp_path / "user_packs" / ".snapshots")
    return _build(CLIENT_ID, GOOD_MODULE)


def _build(bank_id: str, module_src: str) -> ub.UserBankBundle:
    root = ub.user_banks_root() / bank_id
    (root / ub.STORY_PACKS_DIRNAME).mkdir(parents=True)
    (root / f"{bank_id}.py").write_text(module_src, encoding="utf-8")
    (root / ub.BANK_JSON_FILENAME).write_text(
        json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                    "bank": {"source_bank_id": bank_id}}), encoding="utf-8")
    digest = ub.bundle_digest(root)
    return ub.UserBankBundle(
        bank_id=bank_id,
        root=root,
        module_path=root / f"{bank_id}.py",
        story_packs_dir=root / ub.STORY_PACKS_DIRNAME,
        fixtures_dir=root / ub.FIXTURES_DIRNAME,
        digest=digest,
        row=_Row(bank_id, fetcher=ub.SELF_ENTRY_POINT,
                 interpreter=ub.SELF_ENTRY_POINT),
    )


def _unload(bundle: ub.UserBankBundle) -> None:
    sys.modules.pop(ub.bundle_module_name(bundle.bank_id, bundle.digest), None)


# -- (a) module loading ------------------------------------------------------

def test_module_name_carries_the_activated_digest(bundle):
    name = ub.bundle_module_name(bundle.bank_id, bundle.digest)
    assert name.startswith("otr_user_bank_")
    assert bundle.bank_id in name
    assert bundle.digest[:16] in name


def test_edited_bytes_get_a_different_module_name(bundle):
    """A re-activated bundle must never be served from the old cache entry."""
    before = ub.bundle_module_name(bundle.bank_id, bundle.digest)
    (bundle.root / f"{bundle.bank_id}.py").write_text(
        GOOD_MODULE + "\nEXTRA = 1\n", encoding="utf-8")
    after = ub.bundle_module_name(bundle.bank_id, ub.bundle_digest(bundle.root))
    assert before != after


def test_load_bundle_module_imports_and_caches(bundle):
    _unload(bundle)
    try:
        module = ub.load_bundle_module(bundle)
        assert callable(module.fetch_source)
        assert ub.load_bundle_module(bundle) is module
    finally:
        _unload(bundle)


def test_broken_module_fails_loud_and_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    bad = _build("client_broken", "import nonexistent_third_party_lib\n")
    name = ub.bundle_module_name(bad.bank_id, bad.digest)
    with pytest.raises(ub.UserBankExecutionError) as err:
        ub.load_bundle_module(bad)
    assert "client_broken" in str(err.value)
    assert "--activate" in str(err.value)
    assert name not in sys.modules, "a half-executed module must never be cached"


def test_syntax_error_module_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    bad = _build("client_syntax", "def fetch_source(:\n")
    with pytest.raises(ub.UserBankExecutionError):
        ub.load_bundle_module(bad)


def test_load_bundle_module_rejects_a_non_bundle():
    """A caller error, not a bundle problem -- different class on purpose."""
    with pytest.raises(ub.UserBankLayoutError):
        ub.load_bundle_module("client_selfrun")


# -- (b) entry points --------------------------------------------------------

def test_entry_points_resolve_to_the_bundle_callables(bundle):
    _unload(bundle)
    try:
        fetch = ub.bundle_entry_point(bundle, ub.FETCH_ENTRY_ATTR)
        interpret = ub.bundle_entry_point(bundle, ub.INTERPRET_ENTRY_ATTR)
        assert fetch.__name__ == "fetch_source"
        assert interpret.__name__ == "interpret_source"
    finally:
        _unload(bundle)


def test_missing_entry_point_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    partial = _build("client_partial", "def fetch_source(**kwargs):\n    return {}\n")
    _unload(partial)
    try:
        with pytest.raises(ub.UserBankExecutionError) as err:
            ub.bundle_entry_point(partial, ub.INTERPRET_ENTRY_ATTR)
        assert "interpret_source" in str(err.value)
    finally:
        _unload(partial)


def test_non_callable_entry_point_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    wrong = _build("client_notcallable", "fetch_source = 'not a function'\n")
    _unload(wrong)
    try:
        with pytest.raises(ub.UserBankExecutionError) as err:
            ub.bundle_entry_point(wrong, ub.FETCH_ENTRY_ATTR)
        assert "not callable" in str(err.value)
    finally:
        _unload(wrong)


def test_unknown_entry_point_name_is_a_caller_error(bundle):
    with pytest.raises(ub.UserBankLayoutError):
        ub.bundle_entry_point(bundle, "run_everything")


# -- (c) the shipped registries never learn the client id --------------------

def test_self_id_is_never_a_registered_entry_point():
    assert ub.SELF_ENTRY_POINT not in osp.registered_fetcher_ids()
    assert ub.SELF_ENTRY_POINT not in osp.registered_interpreter_ids()
    assert osp.client_entry_point_id() == ub.SELF_ENTRY_POINT


def test_shipped_registry_ids_are_unchanged():
    """Wave 3 adds an execution route, never a registry row."""
    assert osp.registered_fetcher_ids() == frozenset({
        "science_rss", "media_archive_rss", "public_domain_source",
        "shakespeare_folger"})
    assert osp.registered_interpreter_ids() == frozenset({
        "news_interpreter", "media_archive_interpreter",
        "public_domain_interpreter", "shakespeare_interpreter"})


# -- (d) resolution with an owner bundle -------------------------------------

def test_resolve_fetcher_self_returns_the_bundle_callable(bundle):
    _unload(bundle)
    try:
        entry = osp.resolve_fetcher(bundle.row, owner=bundle)
        assert entry.fetch.__name__ == "fetch_source"
        assert entry.seed_source == (
            f"{osp.USER_BANK_SEED_SOURCE_PREFIX}{CLIENT_ID}")
    finally:
        _unload(bundle)


def test_resolve_interpreter_self_returns_the_bundle_callable(bundle):
    _unload(bundle)
    try:
        fn = osp.resolve_interpreter(bundle.row, owner=bundle)
        assert fn.__name__ == "interpret_source"
    finally:
        _unload(bundle)


def test_self_without_an_owner_is_loud(bundle):
    with pytest.raises(osp.ClientOwnerMissingError):
        osp.resolve_fetcher(bundle.row)
    with pytest.raises(osp.ClientOwnerMissingError):
        osp.resolve_interpreter(bundle.row)


def test_a_bank_may_not_execute_another_banks_bundle(bundle):
    """Owner IDENTITY, not mere presence -- else bank A runs bank B's code."""
    stranger = _Row("client_other", fetcher=ub.SELF_ENTRY_POINT,
                    interpreter=ub.SELF_ENTRY_POINT)
    with pytest.raises(osp.ClientOwnerMissingError) as err:
        osp.resolve_fetcher(stranger, owner=bundle)
    assert "does not own" in str(err.value)
    with pytest.raises(osp.ClientOwnerMissingError):
        osp.resolve_interpreter(stranger, owner=bundle)


def test_owner_must_be_an_admitted_bundle(bundle):
    with pytest.raises(osp.ClientOwnerMissingError):
        osp.resolve_fetcher(bundle.row, owner={"bank_id": CLIENT_ID})


def test_a_client_bank_may_reuse_a_shipped_entry_point(bundle):
    """Reuse is allowed; replacement is not."""
    reuser = _Row(CLIENT_ID, fetcher="science_rss",
                  interpreter="news_interpreter")
    assert osp.resolve_fetcher(reuser, owner=bundle).seed_source == "rss_fetch"
    assert callable(osp.resolve_interpreter(reuser, owner=bundle))


def test_empty_ids_still_fail_with_the_contract_missing_error(bundle):
    blank = _Row(CLIENT_ID)
    with pytest.raises(osp.SourceContractMissingError):
        osp.resolve_fetcher(blank, owner=bundle)
    with pytest.raises(osp.SourceContractMissingError):
        osp.resolve_interpreter(blank, owner=bundle)


def test_shipped_bank_resolution_is_unaffected():
    """No owner, shipped ids: byte-identical behaviour to wave 2."""
    shipped = _Row("scifi_news", fetcher="science_rss",
                   interpreter="news_interpreter")
    assert osp.resolve_fetcher(shipped).seed_source == "rss_fetch"
    assert callable(osp.resolve_interpreter(shipped))
    unknown = _Row("scifi_news", fetcher="not_a_fetcher")
    with pytest.raises(osp.UnknownFetcherError):
        osp.resolve_fetcher(unknown)


# -- (e) client output still crosses the shipped validators ------------------

def test_client_fetch_result_still_passes_normalize_fetch_result(bundle):
    _unload(bundle)
    try:
        entry = osp.resolve_fetcher(bundle.row, owner=bundle)
        payload, meta, rights = osp.normalize_fetch_result(
            entry.fetch(bank=bundle.row, technical_model="m", source_ref="",
                        load_config=None, policy=None),
            origin="wave3 client fetch")
        assert payload["seed_text"] == "a client-authored seed"
        assert (meta, rights) == ({}, {})
    finally:
        _unload(bundle)


def test_client_interpreter_result_still_passes_the_validator(bundle):
    _unload(bundle)
    try:
        fn = osp.resolve_interpreter(bundle.row, owner=bundle)
        briefs = fn(bank=bundle.row, payload={}, technical_fn=None, model_id="m")
        dump = osp.validate_interpreter_result(briefs, origin="wave3 client")
        assert dump["key_terms"] == ["client", "bundle"]
    finally:
        _unload(bundle)


def test_a_client_payload_that_breaks_the_contract_is_rejected(tmp_path,
                                                               monkeypatch):
    """The seven-key contract is not negotiable because the fetcher is a client."""
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    liar = _build("client_liar",
                  "def fetch_source(**kwargs):\n"
                  "    return {'seed_text': 'only one key'}\n")
    _unload(liar)
    try:
        entry = osp.resolve_fetcher(liar.row, owner=liar)
        with pytest.raises(osp.SourcePayloadContractError):
            osp.normalize_fetch_result(entry.fetch(), origin="wave3 liar")
    finally:
        _unload(liar)


def test_a_client_briefs_object_that_breaks_the_contract_is_rejected(
        tmp_path, monkeypatch):
    monkeypatch.setattr(ub, "user_banks_root",
                        lambda root=None: tmp_path / "user_packs" / "source_banks")
    liar = _build("client_briefliar",
                  "def interpret_source(**kwargs):\n"
                  "    return object()\n")
    _unload(liar)
    try:
        fn = osp.resolve_interpreter(liar.row, owner=liar)
        with pytest.raises(osp.SourcePayloadContractError):
            osp.validate_interpreter_result(fn(), origin="wave3 briefliar")
    finally:
        _unload(liar)


# -- (f) the registry sweep: self is a CLIENT-only privilege -----------------
#
# Driven against the REAL registry rows and packs, so the exemption is proven
# where it actually lives rather than against a synthetic stand-in.

@pytest.fixture
def live_registry():
    ROUTING._clear_caches()
    reg = ROUTING._ensure_loaded()
    yield reg
    ROUTING._clear_caches()


def _self_owned(bank):
    return dataclasses.replace(bank, fetcher=ub.SELF_ENTRY_POINT,
                               interpreter=ub.SELF_ENTRY_POINT)


def test_a_shipped_row_declaring_self_is_still_an_unregistered_typo(
        live_registry):
    """The exemption is unlocked by is_client ALONE -- never by the row."""
    row = _self_owned(live_registry.banks["media_archive"])
    with pytest.raises(ROUTING.RegistryValidationError) as err:
        ROUTING._crossref_bank(row, live_registry.pipelines,
                               live_registry.pack_dirs, is_client=False)
    message = str(err.value)
    assert "is not registered" in message
    # The refusal names the shipped registry only -- a shipped row is never
    # told about the client escape hatch, because it does not have one.
    assert "client bundle may also declare" not in message


def test_the_same_row_passes_as_a_client_bundle(live_registry):
    row = _self_owned(live_registry.banks["media_archive"])
    ROUTING._crossref_bank(row, live_registry.pipelines,
                           live_registry.pack_dirs,
                           origin="client bundle probe", is_client=True)


def test_a_client_row_still_cannot_invent_an_unregistered_id(live_registry):
    """Client rows get exactly ONE extra value, not a free-for-all."""
    row = dataclasses.replace(live_registry.banks["media_archive"],
                              fetcher="my_own_special_fetcher")
    with pytest.raises(ROUTING.RegistryValidationError):
        ROUTING._crossref_bank(row, live_registry.pipelines,
                               live_registry.pack_dirs,
                               origin="client bundle probe", is_client=True)


def test_the_client_refusal_names_the_self_escape_hatch(live_registry):
    row = dataclasses.replace(live_registry.banks["media_archive"],
                              interpreter="my_own_special_interpreter")
    with pytest.raises(ROUTING.RegistryValidationError) as err:
        ROUTING._crossref_bank(row, live_registry.pipelines,
                               live_registry.pack_dirs,
                               origin="client bundle probe", is_client=True)
    assert ub.SELF_ENTRY_POINT in str(err.value)


# -- (g) the writer wires the owner through ----------------------------------

def test_writer_passes_an_owner_to_both_resolvers():
    """Drift guard: a self-owned client bank is dead without owner=."""
    src = (REPO / "nodes" / "OTR_LedgerScriptWriter.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else None
        if name in ("resolve_fetcher", "resolve_interpreter"):
            seen.setdefault(name, []).append(
                {kw.arg for kw in node.keywords})
    assert set(seen) == {"resolve_fetcher", "resolve_interpreter"}, seen
    for name, calls in seen.items():
        for kwargs in calls:
            assert "owner" in kwargs, (
                f"{name} must be resolved with owner= so a client bank can "
                f"execute its own bundle")
