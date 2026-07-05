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
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


# ---------------------------------------------------------------------------
# (2) registry resolution
# ---------------------------------------------------------------------------

def test_science_bank_resolves_both_contracts():
    bank = routing.get_bank("science_news")
    entry = osp.resolve_fetcher(bank)
    assert entry.seed_source == "rss_fetch"
    assert callable(entry.fetch)
    assert callable(osp.resolve_interpreter(bank))


@pytest.mark.parametrize("bank_id", ["media_archive", "public_domain_story",
                                     "custom_source_bank"])
def test_empty_id_bank_raises_missing_contract(bank_id):
    bank = routing.get_bank(bank_id)
    with pytest.raises(osp.SourceContractMissingError, match=bank_id):
        osp.resolve_fetcher(bank)
    with pytest.raises(osp.SourceContractMissingError, match=bank_id):
        osp.resolve_interpreter(bank)


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

def test_science_rss_wrapper_forwards_exact_args(monkeypatch):
    calls = {}

    def _fake(style, model_id):
        calls["args"] = (style, model_id)
        return _payload()

    import nodes.OTR_LedgerScriptWriter as writer
    monkeypatch.setattr(writer, "_fetch_rss_seed_or_die", _fake)
    bank = routing.get_bank("science_news")
    entry = osp.resolve_fetcher(bank)
    out = entry.fetch(bank=bank, style_slug="mission_control_procedural",
                      technical_model="tm-id")
    assert calls["args"] == ("mission_control_procedural", "tm-id")
    assert out == _payload()


def test_news_interpreter_wrapper_forwards_exact_kwargs(monkeypatch):
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return "briefs-sentinel"

    import nodes.news_interpreter as ni
    monkeypatch.setattr(ni, "build_news_briefs", _fake)
    bank = routing.get_bank("science_news")
    interp = osp.resolve_interpreter(bank)
    payload = _payload()
    out = interp(bank=bank, payload=payload, technical_fn="TFN",
                 style="styleX", model_id="m-id")
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
        "style": "styleX",
        "seed": 0,
        "model_id": "m-id",
    }


# ---------------------------------------------------------------------------
# (5) error translation scope
# ---------------------------------------------------------------------------

def test_wrapper_translates_only_news_interpreter_error(monkeypatch):
    import nodes.news_interpreter as ni
    bank = routing.get_bank("science_news")
    interp = osp.resolve_interpreter(bank)

    def _raise_nie(**kw):
        raise ni.NewsInterpreterError(attempts=3, reason="boom-nie")

    monkeypatch.setattr(ni, "build_news_briefs", _raise_nie)
    with pytest.raises(osp.SourceInterpretError, match="boom-nie") as ei:
        interp(bank=bank, payload=_payload(), technical_fn=None,
               style="s", model_id="m")
    assert isinstance(ei.value.__cause__, ni.NewsInterpreterError)

    def _raise_other(**kw):
        raise ValueError("not-a-nie")

    monkeypatch.setattr(ni, "build_news_briefs", _raise_other)
    with pytest.raises(ValueError, match="not-a-nie"):
        interp(bank=bank, payload=_payload(), technical_fn=None,
               style="s", model_id="m")


# ---------------------------------------------------------------------------
# (6) fail-loud behavior + structural pins in the writer
# ---------------------------------------------------------------------------

def test_resolve_inputs_missing_contract_propagates():
    """A bank without a fetcher contract fails _resolve_inputs LOUD --
    never a silent slide into the science path."""
    import nodes.OTR_LedgerScriptWriter as writer
    with pytest.raises(osp.SourceContractMissingError, match="media_archive"):
        writer._resolve_inputs(custom_premise="", source_bank="media_archive")


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
