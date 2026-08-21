"""Lane-enablement chunk 3 -- source-payload fetcher/interpreter contracts.

Plan of record: docs/multimodal-story-schema/CHUNK3_SOURCE_PAYLOAD_SUBPLAN.md
(v5 FINAL, kibitz r1-r4 converged 2026-07-05). Proves:

(1) payload validator matrix (EXACT keys, str values, non-empty seed_text);
(2) registry resolution -- science resolves both contracts; an empty-id bank
    raises SourceContractMissingError naming the bank; synthetic unknown ids
    raise Unknown*Error (defense-in-depth, no registry load involved);
(3) routing sweep rules -- dangling ids fail load; runnable banks need a REAL
    lane (contract ids on a requires_source_contract pipeline, or an
    executable runner);
(4) science byte-identity -- the wrappers forward EXACT args (incl. the
    source->outlet / date->pub_date renames and seed=0); seed_source stamps;
(5) error translation -- NewsInterpreterError -> SourceInterpretError chained;
    anything else propagates untouched;
(6) fail-loud behavior -- SourceContractMissingError propagates un-degraded
    out of _resolve_inputs; run()'s D.2.5 resolve sits OUTSIDE the try and the
    handler catches ONLY SourceInterpretError;
(7) lazy/cycle import posture;
(8) interpreter-result contract pin (attrs + dump values + dual-surface
    coherence).
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nodes import _otr_public_domain_sources as _pd_sources
from nodes import _otr_source_payload as osp
from nodes import _otr_story_routing as routing

REPO = Path(__file__).resolve().parents[1]
WRITER_PATH = REPO / "nodes" / "OTR_LedgerScriptWriter.py"
MODULE_PATH = REPO / "nodes" / "_otr_source_payload.py"


@pytest.fixture(autouse=True)
def _fresh_caches():
    routing._clear_caches()
    yield
    routing._clear_caches()


def _payload(**over):
    p = {
        "headline": "h", "summary": "s", "full_text": "f",
        "source": "src", "date": "2026-07-05", "link": "l",
        "seed_text": "h s",
    }
    p.update(over)
    return p


# ---------------------------------------------------------------------------
# (1) payload validator matrix
# ---------------------------------------------------------------------------

def test_payload_validator_accepts_and_copies():
    src = _payload()
    out = osp.validate_source_payload(src, origin="t")
    assert out == src and out is not src  # shallow copy


def test_payload_validator_missing_key():
    p = _payload()
    del p["link"]
    with pytest.raises(osp.SourcePayloadContractError, match="missing"):
        osp.validate_source_payload(p, origin="t")


def test_payload_validator_unknown_key():
    with pytest.raises(osp.SourcePayloadContractError, match="unknown"):
        osp.validate_source_payload(_payload(extra="nope"), origin="t")


def test_payload_validator_non_str_value():
    with pytest.raises(osp.SourcePayloadContractError, match="must be str"):
        osp.validate_source_payload(_payload(date=None), origin="t")


def test_payload_validator_empty_seed_text():
    with pytest.raises(osp.SourcePayloadContractError, match="seed_text"):
        osp.validate_source_payload(_payload(seed_text="   "), origin="t")


def test_payload_validator_non_dict():
    with pytest.raises(osp.SourcePayloadContractError, match="dict"):
        osp.validate_source_payload("nope", origin="t")


def test_normalize_fetch_result_accepts_legacy_dict():
    src = _payload()
    payload, meta, rights = osp.normalize_fetch_result(src, origin="t")
    assert payload == src and payload is not src
    assert meta == {}
    assert rights == {}


def test_normalize_fetch_result_accepts_source_fetch_result_and_copies_sidecars():
    source_meta = {"title": "The Time Machine"}
    source_rights = {"license_status": "public_domain_us"}
    result = osp.SourceFetchResult(
        payload=_payload(),
        source_meta=source_meta,
        source_rights=source_rights,
    )

    payload, meta, rights = osp.normalize_fetch_result(result, origin="t")

    assert payload == _payload()
    assert meta == source_meta and meta is not source_meta
    assert rights == source_rights and rights is not source_rights


def test_normalize_fetch_result_none_sidecars_become_empty_dicts():
    result = osp.SourceFetchResult(payload=_payload())
    _payload_out, meta, rights = osp.normalize_fetch_result(result, origin="t")
    assert meta == {}
    assert rights == {}


def test_normalize_fetch_result_rejects_unknown_result_type():
    with pytest.raises(osp.SourcePayloadContractError, match="fetcher result"):
        osp.normalize_fetch_result(["bad"], origin="t")


@pytest.mark.parametrize("field", ["source_meta", "source_rights"])
def test_normalize_fetch_result_rejects_non_dict_sidecars(field):
    kwargs = {"payload": _payload(), field: ["bad"]}
    result = osp.SourceFetchResult(**kwargs)
    with pytest.raises(osp.SourcePayloadContractError, match=field):
        osp.normalize_fetch_result(result, origin="t")


# ---------------------------------------------------------------------------
# (2) registry resolution
# ---------------------------------------------------------------------------

def test_science_bank_resolves_both_contracts():
    # science_news lane removed (roster trim 2026-07-17); media_archive is the
    # kept legacy lane that resolves BOTH the fetcher and interpreter contracts.
    bank = routing.get_bank("media_archive")
    entry = osp.resolve_fetcher(bank)
    assert entry.seed_source == "media_archive_rss"
    assert callable(entry.fetch)
    assert callable(osp.resolve_interpreter(bank))


@pytest.mark.parametrize("bank_id", ["custom_source_bank"])
def test_empty_id_bank_raises_missing_contract(bank_id):
    bank = routing.get_bank(bank_id)
    with pytest.raises(osp.SourceContractMissingError, match=bank_id):
        osp.resolve_fetcher(bank)
    with pytest.raises(osp.SourceContractMissingError, match=bank_id):
        osp.resolve_interpreter(bank)


def test_public_domain_bank_resolves_both_contracts_and_is_runnable():
    bank = routing.get_bank("public_domain")
    entry = osp.resolve_fetcher(bank)
    assert entry.seed_source == "public_domain_source"
    assert callable(entry.fetch)
    assert callable(osp.resolve_interpreter(bank))
    assert bank.runnable is True
    # The bank no longer pins one book. It DEALS: source_ref is deliberately
    # blank and selection_mode carries the behaviour, which is the whole fix
    # for "every public_domain episode was The Time Machine".
    assert bank.defaults["selection_mode"] == "random"
    assert not str(bank.defaults.get("source_ref") or "").strip()
    assert bank.defaults["manifest_path"].endswith("manifest.sample.json")


def test_public_domain_fetcher_wrapper_returns_source_fetch_result():
    """Contract shape, not which book.

    This used to pin the headline to "The Time Machine", which was accurate
    only because the bank had exactly ONE vendored source and no selector. The
    bank now deals from a library, so a blank ref is deliberately
    nondeterministic; the wrapper contract is what this test owns.
    """
    bank = routing.get_bank("public_domain")
    entry = osp.resolve_fetcher(bank)
    result = entry.fetch(bank=bank, technical_model="ignored-technical")
    payload, meta, rights = osp.normalize_fetch_result(
        result, origin="public_domain_source")
    assert payload["headline"] and " - " in payload["headline"]
    assert payload["full_text"].strip()
    assert ":" in meta["source_ref"], meta["source_ref"]
    # The library is 64 public_domain_us works and one cc0 (cradle_protocol),
    # so pinning ONE licence value made this fail 1 run in 65 -- a blank ref
    # deals at random, which is this test's whole premise. Assert the licence
    # is a VALID declared status; the sibling test below owns exact values on
    # a pinned ref, which is where determinism belongs.
    assert rights["license_status"] in _pd_sources._LICENSE_STATUSES, (
        rights["license_status"], meta["source_ref"])


def test_public_domain_fetcher_wrapper_is_deterministic_for_a_pinned_ref():
    """The determinism the old test actually relied on, kept where it belongs:
    on an EXPLICIT ref rather than on the bank having only one book."""
    bank = routing.get_bank("public_domain")
    entry = osp.resolve_fetcher(bank)
    result = entry.fetch(bank=bank, technical_model="ignored-technical",
                         source_ref="time_machine:arrival")
    payload, meta, rights = osp.normalize_fetch_result(
        result, origin="public_domain_source")
    assert payload["headline"] == "The Time Machine - The Time Traveller returns"
    assert meta["source_ref"] == "time_machine:arrival"
    assert rights["license_status"] == "public_domain_us"


def test_media_archive_bank_resolves_both_contracts():
    bank = routing.get_bank("media_archive")
    entry = osp.resolve_fetcher(bank)
    assert entry.seed_source == "media_archive_rss"
    assert callable(entry.fetch)
    assert callable(osp.resolve_interpreter(bank))


def test_unknown_ids_raise_unknown_errors():
    fake = SimpleNamespace(source_bank_id="fake", fetcher="no_such",
                           interpreter="no_such")
    with pytest.raises(osp.UnknownFetcherError, match="no_such"):
        osp.resolve_fetcher(fake)
    with pytest.raises(osp.UnknownInterpreterError, match="no_such"):
        osp.resolve_interpreter(fake)


def test_error_hierarchy_single_base():
    for err in (osp.UnknownFetcherError, osp.UnknownInterpreterError,
                osp.SourceContractMissingError, osp.SourcePayloadContractError,
                osp.SourceInterpretError):
        assert issubclass(err, osp.SourcePayloadError)
    assert not issubclass(osp.SourcePayloadError, routing.StoryRoutingError)


# ---------------------------------------------------------------------------
# (3) sweep rules on a synthetic registry
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


def _mk_registry(root, monkeypatch, banks=None, pipelines=None):
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
    p = packs_dir / "tbank" / "tmodel.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_pack_obj()), encoding="utf-8")
    monkeypatch.setattr(routing, "_STORY_PACKS_ROOT", packs_dir)
    return packs_dir


def test_sweep_dangling_fetcher_id_fails_load(tmp_path, monkeypatch):
    _mk_registry(tmp_path, monkeypatch,
                 banks=[_bank_row(fetcher="no_such_fetcher")])
    with pytest.raises(routing.RegistryValidationError, match="no_such_fetcher"):
        routing.get_bank("tbank")


def test_sweep_dangling_interpreter_id_fails_load(tmp_path, monkeypatch):
    _mk_registry(tmp_path, monkeypatch,
                 banks=[_bank_row(interpreter="no_such_interp")])
    with pytest.raises(routing.RegistryValidationError, match="no_such_interp"):
        routing.get_bank("tbank")


def test_sweep_runnable_contract_pipeline_needs_both_ids(tmp_path, monkeypatch):
    _mk_registry(tmp_path, monkeypatch,
                 banks=[_bank_row(runnable=True, fetcher="science_rss",
                                  interpreter="")],
                 pipelines=[_pipe_row(requires_source_contract=True)])
    with pytest.raises(routing.RegistryValidationError,
                       match="BOTH fetcher and interpreter"):
        routing.get_bank("tbank")


def test_sweep_runnable_contract_pipeline_with_both_ids_loads(tmp_path, monkeypatch):
    _mk_registry(tmp_path, monkeypatch,
                 banks=[_bank_row(runnable=True, fetcher="science_rss",
                                  interpreter="news_interpreter")],
                 pipelines=[_pipe_row(requires_source_contract=True)])
    assert routing.get_bank("tbank").runnable is True


def test_sweep_runnable_runner_pipeline_loads(tmp_path, monkeypatch):
    # A future runner-backed bank: no contract ids, but the pipeline runner
    # exists (executable=true) -- the registry admits it.
    _mk_registry(tmp_path, monkeypatch,
                 banks=[_bank_row(runnable=True)],
                 pipelines=[_pipe_row(executable=True)])
    assert routing.get_bank("tbank").runnable is True


def test_sweep_runnable_without_any_lane_fails_load(tmp_path, monkeypatch):
    # runnable + no contract ids + no runner = a runtime-broken state; the
    # registry must reject it (kibitz r3 codex M3).
    _mk_registry(tmp_path, monkeypatch, banks=[_bank_row(runnable=True)])
    with pytest.raises(routing.RegistryValidationError, match="runner"):
        routing.get_bank("tbank")


def test_pipeline_missing_requires_source_contract_fails(tmp_path, monkeypatch):
    row = _pipe_row()
    del row["requires_source_contract"]
    _mk_registry(tmp_path, monkeypatch, pipelines=[row])
    with pytest.raises(routing.RegistryValidationError,
                       match="requires_source_contract"):
        routing.get_bank("tbank")


def test_shipped_pipeline_flags():
    assert routing.get_pipeline("legacy_many_pass").requires_source_contract is True
    pipe = routing.get_pipeline("simple_4_prompt_experimental")
    assert pipe.requires_source_contract is False
    assert pipe.executable is False  # runner not shipped; flip together


# ---------------------------------------------------------------------------
# (4) science byte-identity -- wrapper forwarding pins
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bank_id", ["scifi_news_pro"])
def test_science_rss_wrapper_forwards_runtime_policy(
        monkeypatch, bank_id):
    """Both science RSS banks share the same source-fetch contract."""
    calls = {}

    def _fake(
        model_id, *, load_config=None, policy=None, receipt_sink=None,
    ):
        calls["model_id"] = model_id
        calls["load_config"] = load_config
        calls["policy"] = policy
        calls["receipt_sink"] = receipt_sink
        if receipt_sink is not None:
            receipt_sink.update({
                "headline": "h",
                "source": "src",
                "url": "l",
                "date": "2026-07-05",
                "body_chars": 1,
                "body_source": "rss_full",
                "rss_content_index": 1,
                "rss_content_count": 2,
                "body_bytes_utf8": 1,
                "body_sha256": hashlib.sha256(b"f").hexdigest(),
                "selected_at": "2026-07-30T12:00:00",
            })
        return _payload()

    import nodes.OTR_LedgerScriptWriter as writer
    monkeypatch.setattr(writer, "_fetch_rss_seed_or_die", _fake)
    bank = routing.get_bank(bank_id)
    entry = osp.resolve_fetcher(bank)
    load_config = object()
    policy = object()
    out = entry.fetch(
        bank=bank,
        technical_model="tm-id",
        source_ref="ignored://source",
        load_config=load_config,
        policy=policy,
    )

    assert calls["model_id"] == "tm-id"
    assert calls["load_config"] is load_config
    assert calls["policy"] is policy
    assert isinstance(calls["receipt_sink"], dict)
    payload, meta, rights = osp.normalize_fetch_result(
        out, origin=f"{bank_id} wrapper test")
    assert payload == _payload()
    assert meta == {
        "kind": "science_rss",
        "source_ref": "l",
        "source_url": "l",
        "source_label": "src",
        "source_date": "2026-07-05",
        # PBUG-20260815-06 (durable-headline half, 2026-08-19): the SELECTED
        # post's headline is now stamped at selection time, for both RSS
        # lanes. This exact-equality pin is doing its job -- it is what caught
        # the new key on the full suite -- so the expectation moves with the
        # contract rather than the assertion being loosened to a subset check.
        "post_headline": "h",
        "_news_seed_receipt": {
            "headline": "h",
            "source": "src",
            "url": "l",
            "date": "2026-07-05",
            "body_chars": 1,
            "body_source": "rss_full",
            "rss_content_index": 1,
            "rss_content_count": 2,
            "body_bytes_utf8": 1,
            "body_sha256": hashlib.sha256(b"f").hexdigest(),
            "selected_at": "2026-07-30T12:00:00",
        },
    }
    assert rights == {
        "license_status": "unknown",
        "source_url": "l",
        "source_label": "src",
    }


def test_media_archive_wrapper_forwards_source_ref_keyword(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return _payload(source="Media History Archive")

    import nodes._otr_media_archive_sources as mas
    monkeypatch.setattr(mas, "fetch_media_archive_rss", _fake)
    bank = routing.get_bank("media_archive")
    entry = osp.resolve_fetcher(bank)
    out = entry.fetch(bank=bank, technical_model="tm-id",
                      source_ref="archive://item")

    payload, meta, rights = osp.normalize_fetch_result(
        out, origin="media archive wrapper test")
    assert payload["source"] == "Media History Archive"
    assert meta["kind"] == "media_archive_rss"
    assert meta["source_ref"] == payload["link"] == "l"
    assert rights == {
        "license_status": "unknown",
        "source_url": "l",
        "source_label": "Media History Archive",
    }
    assert seen == {
        "bank": bank,
        "technical_model": "tm-id",
        "source_ref": "archive://item",
    }


def test_news_interpreter_wrapper_forwards_exact_kwargs(monkeypatch):
    """Style-engine consolidation (2026-07-05): news interpretation is
    style-agnostic -- build_news_briefs no longer takes a style kwarg."""
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return "briefs-sentinel"

    import nodes.news_interpreter as ni
    monkeypatch.setattr(ni, "build_news_briefs", _fake)
    # science_news LANE removed, but the news_interpreter WRAPPER (_interpret_news)
    # is still registered; probe it directly via a bank that declares that id.
    bank = SimpleNamespace(source_bank_id="news_wrapper_probe",
                           interpreter="news_interpreter")
    interp = osp.resolve_interpreter(bank)
    payload = _payload()
    out = interp(bank=bank, payload=payload, technical_fn="TFN",
                 model_id="m-id")
    assert out == "briefs-sentinel"
    # EXACT kwarg set incl. the RENAME mapping (source->outlet, date->pub_date)
    # and the constant seed=0 (cache-key stability, BUG-LOCAL-269/270).
    assert seen == {
        "technical_fn": "TFN",
        "full_text": payload["full_text"],
        "headline": payload["headline"],
        "summary": payload["summary"],
        "outlet": payload["source"],
        "pub_date": payload["date"],
        "seed": 0,
        "model_id": "m-id",
    }


# ---------------------------------------------------------------------------
# (5) error translation scope
# ---------------------------------------------------------------------------

def test_wrapper_translates_only_news_interpreter_error(monkeypatch):
    import nodes.news_interpreter as ni
    # science_news LANE removed; the news_interpreter WRAPPER is still registered
    # and still owns the NewsInterpreterError -> SourceInterpretError translation.
    bank = SimpleNamespace(source_bank_id="news_wrapper_probe",
                           interpreter="news_interpreter")
    interp = osp.resolve_interpreter(bank)

    def _raise_nie(**kw):
        raise ni.NewsInterpreterError(attempts=3, reason="boom-nie")

    monkeypatch.setattr(ni, "build_news_briefs", _raise_nie)
    with pytest.raises(osp.SourceInterpretError, match="boom-nie") as ei:
        interp(bank=bank, payload=_payload(), technical_fn=None,
               model_id="m")
    assert isinstance(ei.value.__cause__, ni.NewsInterpreterError)

    def _raise_other(**kw):
        raise ValueError("not-a-nie")

    monkeypatch.setattr(ni, "build_news_briefs", _raise_other)
    with pytest.raises(ValueError, match="not-a-nie"):
        interp(bank=bank, payload=_payload(), technical_fn=None,
               model_id="m")


# ---------------------------------------------------------------------------
# (6) fail-loud behavior + structural pins in the writer
# ---------------------------------------------------------------------------

def test_resolve_inputs_missing_contract_propagates():
    """A bank without a fetcher contract fails _resolve_inputs LOUD --
    never a silent slide into the science path."""
    import nodes.OTR_LedgerScriptWriter as writer
    with pytest.raises(osp.SourceContractMissingError,
                       match="custom_source_bank"):
        writer._resolve_inputs(custom_premise="",
                               source_bank="custom_source_bank")


def test_resolve_inputs_passes_source_ref_and_returns_sidecars(monkeypatch):
    import nodes.OTR_LedgerScriptWriter as writer

    seen = {}

    def _fake_fetch(**kw):
        seen.update(kw)
        return osp.SourceFetchResult(
            payload=_payload(seed_text="seed from fetcher"),
            source_meta={"source_ref": kw["source_ref"]},
            source_rights={"license_status": "public_domain_us"},
        )

    monkeypatch.setitem(
        osp._FETCHERS,
        "science_rss",
        osp.FetcherEntry(fetch=_fake_fetch, seed_source="rss_fetch"),
    )

    out = writer._resolve_inputs(custom_premise="",
                                 source_bank="scifi_news_pro",
                                 source_ref="pd://fixture")

    assert seen["source_ref"] == "pd://fixture"
    assert out["news_seed"] == "seed from fetcher"
    assert out["source_ref"] == "pd://fixture"
    assert out["source_meta"] == {"source_ref": "pd://fixture"}
    assert out["source_rights"] == {"license_status": "public_domain_us"}


def test_resolve_inputs_custom_premise_returns_empty_sidecars():
    import nodes.OTR_LedgerScriptWriter as writer

    out = writer._resolve_inputs(custom_premise="a quiet custom seed")

    assert out["source_ref"] == ""
    assert out["source_meta"] == {}
    assert out["source_rights"] == {}


def test_resolve_inputs_uses_selected_link_not_differing_request(monkeypatch):
    """The fetcher's selected item owns provenance; the widget is a request."""
    import nodes.OTR_LedgerScriptWriter as writer

    def _fake_fetch(**_kw):
        return osp.SourceFetchResult(
            payload=_payload(link="https://selected.example/item"),
            source_meta={"source_ref": "https://selected.example/item"},
            source_rights={"license_status": "unknown"},
        )

    monkeypatch.setitem(
        osp._FETCHERS,
        "science_rss",
        osp.FetcherEntry(fetch=_fake_fetch, seed_source="rss_fetch"),
    )
    out = writer._resolve_inputs(
        custom_premise="",
        source_bank="scifi_news_pro",
        source_ref="https://requested.example/feed",
    )

    assert out["source_ref"] == "https://selected.example/item"
    assert out["source_meta"]["requested_source_ref"] == (
        "https://requested.example/feed")


def test_writer_stamps_source_sidecars_into_meta():
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    run_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    assignments = set()
    for node in ast.walk(run_fn):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "meta"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)):
                assignments.add(target.slice.value)

    assert "source_ref" in assignments
    assert "source_meta" in assignments
    assert "source_rights" in assignments


def test_writer_resolves_outside_try_and_catches_only_interpret_error():
    """Structural pin (chunk-2 resolve-outside-try pattern): in run(), the
    resolve_interpreter call sits OUTSIDE any try/except, and the D.2.5
    handler catches ONLY SourceInterpretError (contract errors propagate)."""
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    run_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_fn = node
            break
    assert run_fn is not None

    # Collect every Call whose terminal attr is resolve_interpreter, plus the
    # set of nodes inside Try bodies.
    def _calls(node, attr):
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == attr]

    resolve_calls = _calls(run_fn, "resolve_interpreter")
    assert resolve_calls, "run() must call resolve_interpreter"
    in_try: set = set()
    for t in ast.walk(run_fn):
        if isinstance(t, ast.Try):
            for stmt in t.body:
                for n in ast.walk(stmt):
                    in_try.add(id(n))
    for call in resolve_calls:
        assert id(call) not in in_try, (
            "resolve_interpreter must be resolved OUTSIDE any try body "
            "(a bank without the contract fails the episode LOUD)."
        )

    # The handler that catches SourceInterpretError must catch ONLY it.
    handlers = []
    for t in ast.walk(run_fn):
        if not isinstance(t, ast.Try):
            continue
        for h in t.handlers:
            names = []
            for n in ast.walk(h.type) if h.type is not None else []:
                if isinstance(n, ast.Attribute):
                    names.append(n.attr)
                elif isinstance(n, ast.Name):
                    names.append(n.id)
            if "SourceInterpretError" in names:
                handlers.append(names)
    assert handlers, "run() must catch SourceInterpretError in D.2.5"
    for names in handlers:
        assert "SourcePayloadContractError" not in names
        assert "SourcePayloadError" not in names
        assert "Exception" not in names


def test_writer_run_has_no_otrni_references():
    """Chunk 3 wiring pin: the writer's run() no longer references _OTRNI
    (the news_interpreter call moved into the contract wrapper)."""
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert "_OTRNI" not in names
            return
    raise AssertionError("run() not found")


def test_no_production_direct_calls_to_fetch_rss():
    """AST guard (A): no production CALL to _fetch_rss_seed_or_die outside
    _otr_source_payload.py (definition + tests exempt)."""
    for py in sorted((REPO / "nodes").rglob("*.py")):
        if py.name == "_otr_source_payload.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_fetch_rss_seed_or_die"):
                raise AssertionError(
                    f"{py.name}:{node.lineno} calls _fetch_rss_seed_or_die "
                    f"directly; the science_rss wrapper owns that call."
                )


# ---------------------------------------------------------------------------
# (7) lazy / cycle import posture
# ---------------------------------------------------------------------------

def test_module_import_is_lazy_and_cycle_free(monkeypatch):
    """Importing _otr_source_payload does no file I/O; module-level imports
    never touch the writer / news_interpreter / story_orchestrator / routing
    (three-edge cycle guard)."""
    def _boom(self, *a, **k):
        raise AssertionError(f"import-time file I/O attempted: {self}")

    monkeypatch.setattr(Path, "read_text", _boom)
    try:
        mod = importlib.reload(osp)
        assert mod is osp
    finally:
        monkeypatch.undo()

    # Static pin: no module-level import of the banned edges.
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    banned = {"OTR_LedgerScriptWriter", "news_interpreter",
              "story_orchestrator", "_otr_story_routing"}
    for node in tree.body:  # TOP-LEVEL statements only
        if isinstance(node, ast.Import):
            mods = {a.name.split(".")[-1] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods = {a.name for a in node.names} | {
                (node.module or "").split(".")[-1]}
        else:
            continue
        hit = mods & banned
        assert not hit, (
            f"_otr_source_payload.py must not import {sorted(hit)} at module "
            f"level (lazy wrapper imports only)."
        )


# ---------------------------------------------------------------------------
# (8) interpreter-result contract pin
# ---------------------------------------------------------------------------

class _GoodBriefs:
    casting_brief = "cb"
    script_brief = "sb"
    key_terms = ["ALPHA", "BETA"]
    attempts = 1

    def model_dump(self):
        return {
            "casting_brief": "cb", "script_brief": "sb",
            "news_close_brief": "ncb", "key_terms": ["ALPHA", "BETA"],
        }


def test_interpreter_result_contract_good():
    dump = osp.validate_interpreter_result(_GoodBriefs(), origin="t")
    assert dump["news_close_brief"] == "ncb"
    assert dump["key_terms"] == ["ALPHA", "BETA"]
    assert "upstream_identity_names" not in dump, (
        "the optional identity surface must not rewrite frozen four-key "
        "interpreter dumps"
    )


class _IdentityBriefs(_GoodBriefs):
    upstream_identity_names = ["Marta Vale", "Gil Neri"]

    def model_dump(self):
        dump = super().model_dump()
        dump["upstream_identity_names"] = list(self.upstream_identity_names)
        return dump


def test_interpreter_result_optional_identity_names_round_trip():
    brief = _IdentityBriefs()
    dump = osp.validate_interpreter_result(brief, origin="identity-test")
    assert dump["upstream_identity_names"] == ["Marta Vale", "Gil Neri"]


@pytest.mark.parametrize(
    "bad_names",
    [
        "Marta Vale",
        b"Marta Vale",
        ("Marta Vale",),
        [""],
        ["   "],
        ["Marta Vale", 42],
    ],
)
def test_interpreter_result_optional_identity_direct_surface_rejects_malformed(
        bad_names):
    brief = _IdentityBriefs()
    brief.upstream_identity_names = bad_names
    with pytest.raises(
        osp.SourcePayloadContractError, match="upstream_identity_names",
    ):
        osp.validate_interpreter_result(brief, origin="identity-test")


@pytest.mark.parametrize(
    "bad_names",
    ["Marta Vale", ("Marta Vale",), [""], ["Marta Vale", None]],
)
def test_interpreter_result_optional_identity_dump_surface_rejects_malformed(
        bad_names):
    class _BadDump(_IdentityBriefs):
        def model_dump(self):
            dump = super().model_dump()
            dump["upstream_identity_names"] = bad_names
            return dump

    with pytest.raises(
        osp.SourcePayloadContractError, match="upstream_identity_names",
    ):
        osp.validate_interpreter_result(_BadDump(), origin="identity-test")


def test_interpreter_result_optional_identity_split_brain_rejected():
    class _SplitIdentity(_IdentityBriefs):
        def model_dump(self):
            dump = super().model_dump()
            dump["upstream_identity_names"] = ["Someone Else"]
            return dump

    with pytest.raises(osp.SourcePayloadContractError, match="split-brain"):
        osp.validate_interpreter_result(_SplitIdentity(), origin="identity-test")


@pytest.mark.parametrize("direct_only", [True, False])
def test_interpreter_result_optional_identity_must_exist_on_both_surfaces(
        direct_only):
    if direct_only:
        class _OneSurface(_IdentityBriefs):
            def model_dump(self):
                return _GoodBriefs.model_dump(self)
    else:
        class _OneSurface(_GoodBriefs):
            def model_dump(self):
                dump = super().model_dump()
                dump["upstream_identity_names"] = ["Marta Vale"]
                return dump

    with pytest.raises(
        osp.SourcePayloadContractError, match="upstream_identity_names",
    ):
        osp.validate_interpreter_result(_OneSurface(), origin="identity-test")


@pytest.mark.parametrize("attr", ["model_dump", "casting_brief",
                                  "script_brief", "key_terms", "attempts"])
def test_interpreter_result_missing_attr(attr):
    stub = _GoodBriefs()
    obj = SimpleNamespace(**{
        a: getattr(stub, a) for a in
        ("casting_brief", "script_brief", "key_terms", "attempts")
        if a != attr
    })
    if attr != "model_dump":
        obj.model_dump = stub.model_dump
    with pytest.raises(osp.SourcePayloadContractError, match=attr):
        osp.validate_interpreter_result(obj, origin="t")


def test_interpreter_result_str_key_terms_rejected():
    stub = _GoodBriefs()
    stub.key_terms = "ALPHA"  # a bare str would char-split at tuple()
    with pytest.raises(osp.SourcePayloadContractError, match="NON-STRING"):
        osp.validate_interpreter_result(stub, origin="t")


def test_interpreter_result_dump_missing_news_close_brief():
    class _NoNCB(_GoodBriefs):
        def model_dump(self):
            d = super().model_dump()
            del d["news_close_brief"]
            return d
    with pytest.raises(osp.SourcePayloadContractError,
                       match="news_close_brief"):
        osp.validate_interpreter_result(_NoNCB(), origin="t")


def test_interpreter_result_dump_tuple_key_terms_rejected():
    """r3 codex OPT: dump key_terms as tuple (direct attr valid) must fail
    BEFORE ledger freeze (freeze expects a list)."""
    class _TupleDump(_GoodBriefs):
        def model_dump(self):
            d = super().model_dump()
            d["key_terms"] = ("ALPHA", "BETA")
            return d
    with pytest.raises(osp.SourcePayloadContractError, match="list"):
        osp.validate_interpreter_result(_TupleDump(), origin="t")


def test_interpreter_result_split_brain_rejected():
    """r4 codex M1: dump values must EQUAL direct attrs."""
    class _Split(_GoodBriefs):
        def model_dump(self):
            d = super().model_dump()
            d["script_brief"] = "DIFFERENT"
            return d
    with pytest.raises(osp.SourcePayloadContractError, match="split-brain"):
        osp.validate_interpreter_result(_Split(), origin="t")


def test_interpreter_result_bool_attempts_rejected():
    stub = _GoodBriefs()
    stub.attempts = True
    with pytest.raises(osp.SourcePayloadContractError, match="attempts"):
        osp.validate_interpreter_result(stub, origin="t")


def test_real_news_briefs_model_satisfies_contract():
    """The production NewsBriefs pydantic model passes the contract pin
    (science byte-identity: the validator admits the real object)."""
    import nodes.news_interpreter as ni
    briefs = ni.NewsBriefs(
        casting_brief="c" * 30,
        script_brief="s" * 30,
        news_close_brief="n" * 30,
        key_terms=["ALPHA"],
        attempts=1,
    )
    dump = osp.validate_interpreter_result(briefs, origin="t")
    assert dump == briefs.model_dump()


# ---------------------------------------------------------------------------
# (9) bounded source-interpreter execution + same-source fallback
# ---------------------------------------------------------------------------


class _TypedInterpreterFailure(RuntimeError):
    def __init__(self, *, attempts, reason):
        self.attempts = attempts
        self.reason = reason
        super().__init__(reason)


def _source_interpret_error(*, attempts, reason="all attempts failed"):
    cause = _TypedInterpreterFailure(attempts=attempts, reason=reason)
    err = osp.SourceInterpretError(str(cause))
    err.__cause__ = cause
    return err


def _rich_payload():
    return _payload(
        headline="Restored lunar television archive",
        summary="Archivists preserve rare broadcast recordings",
        full_text=(
            "Researchers catalog the lunar broadcast collection and restore "
            "the surviving television recordings for public study."
        ),
        seed_text="Restored lunar television archive recordings",
    )


def test_interpreter_exhaustion_classifier_excludes_runtime_failures():
    assert osp.interpreter_exhaustion_attempts(
        _source_interpret_error(attempts=3)) == 3
    assert osp.interpreter_exhaustion_attempts(
        _source_interpret_error(attempts=0, reason="missing seed deck")) == 0
    assert osp.interpreter_exhaustion_attempts(
        _source_interpret_error(
            attempts=[],
            reason="all 3 attempt(s) failed; last error: schema reject",
        )) == 3
    assert osp.interpreter_exhaustion_attempts(
        _source_interpret_error(
            attempts=[], reason="slot fn raised: RuntimeError: backend down",
        )) == 0


@pytest.mark.parametrize(
    "interpreter_id",
    [
        "media_archive_interpreter",
        "public_domain_interpreter",
        "shakespeare_interpreter",
        "news_interpreter",
    ],
)
def test_source_interpreter_fallback_preserves_source_without_term_fabrication(
        interpreter_id):
    bank = SimpleNamespace(interpreter=interpreter_id)
    payload = _rich_payload()
    fallback = osp.build_source_interpreter_fallback(
        bank=bank,
        payload=payload,
        source_meta={"cast_hints": ["Archivist Ada", "Curator Lee"]},
        attempts=3,
        failure_reason="schema remained malformed",
    )
    dump = osp.validate_interpreter_result(fallback, origin="fallback-test")
    assert dump["deterministic_fallback"] is True
    assert fallback.model_id == "deterministic"
    assert fallback.character_names == ["Archivist Ada", "Curator Lee"]
    assert fallback.attempts == 3
    assert fallback.key_terms == []
    assert payload["headline"] in fallback.script_brief
    assert payload["summary"] in fallback.script_brief
    assert payload["headline"] in fallback.news_close_brief


def test_source_interpreter_fallback_rejects_unknown_route():
    with pytest.raises(osp.UnknownInterpreterError, match="no bank-specific"):
        osp.build_source_interpreter_fallback(
            bank=SimpleNamespace(interpreter="not_registered"),
            payload=_rich_payload(),
            source_meta={},
            attempts=3,
            failure_reason="bad output",
        )


def test_source_interpreter_fallback_serves_a_client_owned_lane():
    """The reserved client lane gets a brief, not a router-error message.

    A client interpreter that exhausts its structured-output ladder used to
    reach `UnknownInterpreterError` naming 'self' -- OUR router complaining
    about THEIR failure. The brief is built from the client's own bank
    identity plus the validated payload; it borrows no shipped dramaturgy.
    """
    bank = SimpleNamespace(
        interpreter=osp.client_entry_point_id(),
        label="Lighthouse Logs",
        source_bank_id="lighthouse_logs",
    )
    payload = _rich_payload()
    fallback = osp.build_source_interpreter_fallback(
        bank=bank,
        payload=payload,
        source_meta={"cast_hints": ["Keeper Nan"]},
        attempts=2,
        failure_reason="client interpreter schema remained malformed",
    )
    dump = osp.validate_interpreter_result(
        fallback, origin="client-fallback-test")
    assert dump["deterministic_fallback"] is True
    assert fallback.model_id == "deterministic"
    assert fallback.attempts == 2
    assert fallback.key_terms == []
    assert fallback.character_names == ["Keeper Nan"]
    assert "Lighthouse Logs" in fallback.script_brief
    assert "Lighthouse Logs" in fallback.news_close_brief
    assert payload["headline"] in fallback.script_brief
    assert payload["summary"] in fallback.script_brief


def test_client_fallback_names_the_bank_id_when_the_row_has_no_label():
    bank = SimpleNamespace(
        interpreter=osp.client_entry_point_id(),
        label="",
        source_bank_id="lighthouse_logs",
    )
    fallback = osp.build_source_interpreter_fallback(
        bank=bank,
        payload=_rich_payload(),
        source_meta=None,
        attempts=1,
        failure_reason="exhausted",
    )
    osp.validate_interpreter_result(fallback, origin="client-fallback-unlabeled")
    assert "lighthouse_logs" in fallback.script_brief
    assert fallback.character_names == []


class _SlotSchedulerStub:
    transitions = 0
    calls_by_slot = {"creative": 0, "technical": 0}
    slot_calls_by_helper = {}
    slot_transitions_by_phase = []

    @contextmanager
    def helper_context(self, _name):
        yield


def test_writer_source_interpreter_accepts_one_bounded_result():
    from nodes import OTR_LedgerScriptWriter as writer

    calls = []

    def _interpreter(*, bank, payload, technical_fn, model_id):
        del bank, payload, technical_fn
        calls.append(model_id)
        brief = _GoodBriefs()
        brief.attempts = 2
        return brief

    meta = {}
    brief = writer._run_source_interpreter(
        interpreter=_interpreter,
        bank=SimpleNamespace(interpreter="media_archive_interpreter"),
        payload=_rich_payload(),
        source_meta={},
        technical_fn=lambda *args, **kwargs: "{}",
        technical_model_id="tech-model",
        slot_scheduler=_SlotSchedulerStub(),
        meta=meta,
    )

    assert isinstance(brief, _GoodBriefs)
    assert calls == ["tech-model"]
    assert meta["source_interpreter"] == {
        "schema_version": "source_interpreter_v2",
        "status": "accepted",
        "model_calls": 2,
        "model": "tech-model",
    }


def test_writer_source_interpreter_uses_same_source_fallback():
    from nodes import OTR_LedgerScriptWriter as writer

    calls = []

    def _interpreter(*, bank, payload, technical_fn, model_id):
        del bank, payload, technical_fn
        calls.append(model_id)
        raise _source_interpret_error(
            attempts=3, reason="all 3 attempts failed: malformed JSON")

    meta = {}
    fallback = writer._run_source_interpreter(
        interpreter=_interpreter,
        bank=SimpleNamespace(interpreter="media_archive_interpreter"),
        payload=_rich_payload(),
        source_meta={"cast_hints": ["Archivist Ada"]},
        technical_fn=lambda *args, **kwargs: "{}",
        technical_model_id="tech-model",
        slot_scheduler=_SlotSchedulerStub(),
        meta=meta,
    )

    assert calls == ["tech-model"]
    assert fallback.deterministic_fallback is True
    assert fallback.character_names == ["Archivist Ada"]
    assert meta["source_interpreter"]["status"] == (
        "deterministic_same_source_fallback")
    assert meta["source_interpreter"]["model_calls"] == 3
    assert meta["source_interpreter"]["model"] == "deterministic"


def test_writer_source_interpreter_propagates_non_exhaustion_failure():
    from nodes import OTR_LedgerScriptWriter as writer

    def _interpreter(**_kwargs):
        raise _source_interpret_error(
            attempts=0, reason="media archive seed deck missing")

    with pytest.raises(osp.SourceInterpretError, match="seed deck"):
        writer._run_source_interpreter(
            interpreter=_interpreter,
            bank=SimpleNamespace(interpreter="media_archive_interpreter"),
            payload=_rich_payload(),
            source_meta={},
            technical_fn=lambda *args, **kwargs: "{}",
            technical_model_id="tech-model",
            slot_scheduler=_SlotSchedulerStub(),
            meta={},
        )


@pytest.mark.parametrize(
    ("source_receipt", "expected"),
    [
        (
            {"status": "accepted", "model": "tech-model"},
            {"slot": "technical", "model": "tech-model"},
        ),
        (
            {
                "status": "deterministic_same_source_fallback",
                "model": "deterministic",
            },
            {"slot": "deterministic", "model": "deterministic"},
        ),
    ],
)
def test_final_slot_telemetry_uses_source_interpreter_receipt(
        source_receipt, expected):
    from nodes import OTR_LedgerScriptWriter as writer

    meta = {
        "news": {"script_brief": "brief", "key_terms": ["archive"]},
        "source_interpreter": source_receipt,
    }
    writer._stamp_final_slot_telemetry(
        meta=meta,
        resolved={
            "creative_writing_model": "creative-model",
            "technical_model": "technical-model",
        },
        slot_scheduler=_SlotSchedulerStub(),
        pipeline_id="legacy_many_pass",
        title_source="outline_fallback",
    )
    assert meta["gen_params_by_phase"]["news_interpreter"] == expected
