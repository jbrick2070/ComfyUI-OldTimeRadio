"""Sprint 10A step 3-B regression -- make_constrained_generate_fn factory.

The factory itself can be exercised without a real model + tokenizer:
  * Build the JsonSchemaParser from a pydantic schema.
  * Build the prefix_allowed_tokens_fn from a (minimal) tokenizer
    stand-in.
  * Verify each generation receives fresh parser/prefix state.
  * Verify the closure passes prefix_allowed_tokens_fn into
    model.generate() (mocked) so the constraint actually reaches the
    sampler.

Tests do NOT load Mistral-Nemo. The real model is exercised by the
operator soak gate (>=19/20 first-attempt valid plans).
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field
from typing import Literal


class _TinySchema(BaseModel):
    """Minimal schema for tests that don't depend on Stage1Plan shape."""

    color: Literal["red", "green", "blue"]
    count: int = Field(..., ge=0, le=99)


class _OtherTinySchema(BaseModel):
    status: Literal["ready"]


# ---------------------------------------------------------------------------
# Module import (smoke)
# ---------------------------------------------------------------------------


class TestImport:
    def test_module_imports_without_loading_model(self):
        # The compat shim must run as a side-effect of the package
        # import, even though no model has been loaded yet.
        from nodes import _otr_constrained_generate as cg

        assert hasattr(cg, "make_constrained_generate_fn")
        assert hasattr(cg, "ConstrainedGenerateFn")
        assert hasattr(cg, "get_cached_transformers_schema_constraint")

    def test_compat_shim_alias_in_place(self):
        # After calling the public helper, the v4 alias must be on
        # transformers.tokenization_utils so lmfe can find it.
        #
        # The module-import side-effect alone is not load-bearing
        # any more: some test fixtures elsewhere in the suite reload
        # transformers.tokenization_utils between this test and the
        # module's initial import, which would clobber a one-shot
        # alias. The helper is idempotent and is called at every
        # factory use site -- this test exercises that contract
        # directly so it stays green regardless of test ordering.
        # Skip when transformers is absent (base Python 3.11 sandbox).
        pytest.importorskip("transformers")
        from nodes import _otr_lmfe_compat

        _otr_lmfe_compat.ensure_lmfe_transformers_compat()
        import transformers
        assert hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase")


# ---------------------------------------------------------------------------
# Factory contract
# ---------------------------------------------------------------------------


class TestFactoryContract:
    def test_missing_keys_raises_model_loader_error(self):
        from nodes._otr_constrained_generate import make_constrained_generate_fn
        from nodes._otr_model_loader import ModelLoaderError

        with pytest.raises(ModelLoaderError, match="missing required keys"):
            make_constrained_generate_fn({"model": object()}, _TinySchema)

        with pytest.raises(ModelLoaderError, match="missing required keys"):
            make_constrained_generate_fn({"tokenizer": object()}, _TinySchema)

    def _make_minimal_cache_entry(self):
        """Build a cache_entry shape sufficient for the FACTORY to
        succeed (the closure itself needs a real model.generate, but
        the factory only needs model.device + tokenizer's vocab
        methods that lm-format-enforcer queries to build the prefix-
        fn).
        """
        # Minimal tokenizer stand-in: lm-format-enforcer queries
        # vocab via `tokenizer.get_vocab()` and `tokenizer.decode()`.
        # We supply a tiny ASCII-letter vocab plus a few JSON
        # structural tokens.
        tokenizer = MagicMock()
        vocab = {chr(c): c for c in range(32, 127)}  # 95 ASCII chars
        vocab.update({"<eos>": 200, "<bos>": 201, "<pad>": 202})
        tokenizer.get_vocab.return_value = vocab
        # vocab_size for any range-loop the integration might do
        tokenizer.vocab_size = len(vocab)
        tokenizer.__len__.return_value = 203
        tokenizer.all_special_ids = [200, 201, 202]
        tokenizer.encode.return_value = [48]
        # decode + convert_ids_to_tokens for lmfe's introspection
        tokenizer.decode = lambda ids, **kw: "".join(
            chr(i) if 32 <= i < 127 else "?" for i in (ids if hasattr(ids, "__iter__") else [ids])
        )
        tokenizer.convert_ids_to_tokens = lambda ids: [
            chr(i) if 32 <= i < 127 else f"<{i}>" for i in (ids if hasattr(ids, "__iter__") else [ids])
        ]
        tokenizer.eos_token_id = 200

        model = MagicMock()
        model.device = "cpu"

        return {"model": model, "tokenizer": tokenizer}

    def test_factory_returns_callable_with_schema_attributes(self):
        # Requires lmformatenforcer to build the JsonSchemaParser; skip in
        # the base Python 3.11 sandbox where it is not installed.
        pytest.importorskip("lmformatenforcer")
        from nodes._otr_constrained_generate import make_constrained_generate_fn

        cache_entry = self._make_minimal_cache_entry()
        fn = make_constrained_generate_fn(cache_entry, _TinySchema)

        assert callable(fn)
        # Only stable metadata survives on the closure, never mutable history.
        assert fn.schema_model is _TinySchema
        assert not hasattr(fn, "json_schema_parser")
        assert not hasattr(fn, "prefix_allowed_tokens_fn")

    def test_constraint_cache_reuses_only_tokenizer_scan(self, monkeypatch):
        pytest.importorskip("lmformatenforcer")
        from nodes._otr_constrained_generate import (
            get_cached_transformers_schema_constraint,
        )
        from lmformatenforcer.integrations import transformers as integration

        scan = MagicMock(wraps=integration.build_token_enforcer_tokenizer_data)
        monkeypatch.setattr(integration, "build_token_enforcer_tokenizer_data", scan)

        cache_entry = self._make_minimal_cache_entry()
        parser_1, prefix_1 = get_cached_transformers_schema_constraint(
            cache_entry, _TinySchema,
        )
        parser_1_again, prefix_1_again = (
            get_cached_transformers_schema_constraint(cache_entry, _TinySchema)
        )
        parser_2, prefix_2 = get_cached_transformers_schema_constraint(
            cache_entry, _OtherTinySchema,
        )

        assert parser_1_again is not parser_1
        assert prefix_1_again is not prefix_1
        assert parser_2 is not parser_1
        assert prefix_2 is not prefix_1
        internal = cache_entry["_otr_lmfe_constraint_cache"]
        assert internal["tokenizer"] is cache_entry["tokenizer"]
        assert set(internal) == {"tokenizer", "tokenizer_data"}
        assert scan.call_count == 1
        # Warm entries may still carry the old history-owning shape.
        internal["by_schema"] = {_TinySchema: (parser_1, prefix_1)}
        get_cached_transformers_schema_constraint(cache_entry, _TinySchema)
        assert "by_schema" not in internal and scan.call_count == 1
        cache_entry["tokenizer"] = self._make_minimal_cache_entry()["tokenizer"]
        get_cached_transformers_schema_constraint(cache_entry, _TinySchema)
        assert scan.call_count == 2


# ---------------------------------------------------------------------------
# Closure passes prefix_allowed_tokens_fn into generate
# ---------------------------------------------------------------------------


class TestClosurePassesConstraint:
    def test_generate_invoked_with_prefix_fn(self):
        # Requires lmformatenforcer + torch; skip in the base Python 3.11
        # sandbox where lmformatenforcer is not installed.
        pytest.importorskip("lmformatenforcer")
        pytest.importorskip("torch")
        from nodes._otr_constrained_generate import make_constrained_generate_fn

        # Build a fuller mock: tokenizer + model whose generate()
        # returns a single fake token row.
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {chr(c): c for c in range(32, 127)}
        tokenizer.vocab_size = 95
        tokenizer.__len__.return_value = 201
        tokenizer.all_special_ids = [200]
        tokenizer.encode.return_value = [48]
        tokenizer.decode = lambda ids, **kw: "out"
        tokenizer.convert_ids_to_tokens = lambda ids: ["x"]
        tokenizer.eos_token_id = 200
        tokenizer.apply_chat_template.return_value = "PROMPT"
        # The "() -> dict-with-input_ids tensor-like" shape pytorch
        # uses. .to(device) returns self.
        class _FakeInputs(dict):
            def to(self, device): return self
        # tensor shape proxy
        class _FakeTensor:
            def __init__(self, n): self._n = n
            @property
            def shape(self): return (1, self._n)
            def __getitem__(self, i):
                # out[0][prompt_len:] returns an iterable for decode()
                return [99]
        inputs = _FakeInputs(input_ids=_FakeTensor(3))
        tokenizer.return_value = inputs

        model = MagicMock()
        model.device = "cpu"
        # model.generate returns out, indexed as out[0][prompt_len:]
        # Match the shape inputs["input_ids"].shape[1] expects.
        gen_out = [[99, 99, 99, 99, 99]]  # prompt_len=3, suffix=2 tokens
        # Wrap so out[0] is subscriptable from prompt_len: onward
        class _GenOut:
            def __getitem__(self, i): return gen_out[i]
        model.generate.return_value = _GenOut()

        cache_entry = {"model": model, "tokenizer": tokenizer}

        # Patch torch.no_grad to a context manager passthrough so the
        # closure runs without a real torch install.
        with patch("nodes._otr_model_loader._normalize_messages_for_cache_entry",
                   side_effect=lambda ce, msgs: msgs):
            fn = make_constrained_generate_fn(cache_entry, _TinySchema)
            # Stub torch import inside the closure.
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *a, **kw):
                if name == "torch":
                    fake_torch = MagicMock()
                    fake_torch.no_grad.return_value.__enter__ = MagicMock()
                    fake_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
                    return fake_torch
                return real_import(name, *a, **kw)

            with patch("builtins.__import__", side_effect=fake_import):
                _ = fn(
                    [{"role": "user", "content": "hi"}],
                    temperature=0.5,
                    max_new_tokens=50,
                )

        # The factory's prefix_allowed_tokens_fn must have been passed
        # into model.generate as a kwarg.
        assert model.generate.called
        _, gkwargs = model.generate.call_args
        assert "prefix_allowed_tokens_fn" in gkwargs
        assert callable(gkwargs["prefix_allowed_tokens_fn"])
        assert not hasattr(fn, "prefix_allowed_tokens_fn")
        # And num_beams=1 must be set explicitly per the docstring rationale.
        assert gkwargs.get("num_beams") == 1
        # temperature + max_new_tokens passed through verbatim
        assert gkwargs.get("temperature") == 0.5
        assert gkwargs.get("max_new_tokens") == 50


def _real_constraint(schema):
    from nodes._otr_constrained_generate import get_cached_transformers_schema_constraint
    entry = TestFactoryContract()._make_minimal_cache_entry()
    return get_cached_transformers_schema_constraint(entry, schema)


def _feed_json(prefix, text):
    import torch
    ids = [201]
    for position, char in enumerate(text):
        assert ord(char) in prefix(0, torch.tensor(ids)), (position, text[:position], char)
        ids.append(ord(char))
    return prefix(0, torch.tensor(ids))


@pytest.mark.parametrize("nested", [False, True])
def test_real_lmfe_accepts_twenty_five_items_without_implicit_limit(nested):
    import json
    from pydantic import RootModel

    class Array(RootModel[list[int]]):
        pass

    class Nested(BaseModel):
        items: list[int]

    parser, prefix = _real_constraint(Nested if nested else Array)
    value = {"items": list(range(25))} if nested else list(range(25))
    assert 200 in _feed_json(prefix, json.dumps(value, separators=(",", ":")))
    assert parser.config.max_json_array_length == 0
    assert parser.config.alphabet, "the tokenizer-installed alphabet must survive"


@pytest.mark.parametrize("limit", [2, 25])
def test_real_lmfe_preserves_explicit_array_limit(limit):
    import json
    from pydantic import RootModel

    class Bounded(RootModel):
        root: list[int] = Field(max_length=limit)

    _, prefix = _real_constraint(Bounded)
    allowed = _feed_json(prefix, json.dumps(list(range(limit)), separators=(",", ":"))[:-1])
    assert ord("]") in allowed
    assert ord(",") not in allowed


def _reusable_native_closure(kind, monkeypatch, outcome):
    """Real LMFE/tensors, with a model retaining only weak generation references."""
    import weakref
    import torch
    from nodes import _otr_constrained_generate as constrained
    from nodes import OTR_LedgerScriptWriter as writer

    entry = TestFactoryContract()._make_minimal_cache_entry()
    tokenizer = entry["tokenizer"]
    tokenizer.apply_chat_template.return_value = "PROMPT"

    class Inputs(dict):
        def to(self, _device):
            return self

    tokenizer.return_value = Inputs(input_ids=torch.tensor([[201]]))
    refs, knobs = [], []

    class Model:
        device = "cpu"
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            prefix = kwargs["prefix_allowed_tokens_fn"]
            enforcer = prefix.token_enforcer
            assert not enforcer.prefix_states, "history leaked into a new generation"
            refs.append((weakref.ref(prefix), weakref.ref(enforcer), weakref.ref(enforcer.root_parser)))
            prefix(0, torch.tensor([201]))
            prefix(0, torch.tensor([201, ord("{")]))
            assert enforcer.prefix_states
            knobs.append({name: kwargs.get(name) for name in ("min_p", "top_p", "repetition_penalty", "num_beams")})
            knobs[-1]["open_string_bound"] = kwargs["stopping_criteria"][0].telemetry()["open_string_bound"]
            if self.calls == 1 and outcome == "min_p":
                raise TypeError("unsupported min_p")
            if self.calls == 1 and outcome == "error":
                raise RuntimeError("generation interrupted")
            if outcome == "open_string":
                output = torch.tensor([[201, ord('"'), *range(1000, 3050)]])
                assert kwargs["stopping_criteria"][0](output, None)
                return output
            return torch.tensor([[201, *map(ord, '{"color":"red","count":1}'), 200]])

    model = Model()
    entry.update(model=model, context_cap=8192)
    monkeypatch.setattr("nodes._otr_model_loader._normalize_messages_for_cache_entry", lambda _entry, messages: list(messages))
    monkeypatch.setattr(writer._OTRHB, "make_streamer", lambda *_args: None)
    if kind == "writer":
        fn = writer._build_truncating_generate_fn(entry, top_p=.88, min_p=.04,
                                                 repetition_penalty=1.03, schema_model=_TinySchema)
    else:
        fn = constrained.make_constrained_generate_fn(entry, _TinySchema)
    return fn, entry, model, refs, knobs


@pytest.mark.parametrize("kind,outcome", [
    ("writer", "success"), ("writer", "error"), ("writer", "min_p"),
    ("standalone", "success"), ("standalone", "error"),
])
def test_each_actual_generation_has_fresh_collectible_history(kind, outcome, monkeypatch):
    import gc
    from nodes import _vram_log
    observations = []
    monkeypatch.setattr(_vram_log, "memory_snapshot",
                        lambda phase, **kwargs: observations.append(phase))
    fn, entry, model, refs, knobs = _reusable_native_closure(kind, monkeypatch, outcome)
    messages = [{"role": "user", "content": "A structured reply."}]
    if outcome == "error":
        # Do not retain pytest's ExceptionInfo/traceback as a false history owner.
        try:
            fn(messages, temperature=.2, max_new_tokens=100)
        except RuntimeError as error:
            assert str(error) == "generation interrupted"
        else:
            pytest.fail("the model error must propagate")
    else:
        fn(messages, temperature=.2, max_new_tokens=100)
    fn(messages, temperature=.2, max_new_tokens=100)
    assert model.calls == (3 if outcome == "min_p" else 2)
    assert observations == [
        ("writer" if kind == "writer" else "constrained") + "_generation_returned"
    ] * (1 if outcome == "error" else 2)
    assert set(entry["_otr_lmfe_constraint_cache"]) == {"tokenizer", "tokenizer_data"}
    assert not hasattr(fn, "prefix_allowed_tokens_fn")
    gc.collect()
    assert all(ref() is None for group in refs for ref in group)
    if kind == "writer":
        assert all(k["top_p"] == .88 and k["repetition_penalty"] == 1.03 and k["num_beams"] == 1 for k in knobs)
        assert knobs[0]["min_p"] == .04
        if outcome == "min_p":
            assert all(k["min_p"] is None for k in knobs[1:])


def test_resident_tokenizer_cache_does_not_retain_dynamic_schema():
    import gc
    import weakref
    from nodes._otr_constrained_generate import get_cached_transformers_schema_constraint
    entry = TestFactoryContract()._make_minimal_cache_entry()

    def temporary_schema():
        class TemporarySchema(BaseModel):
            response: str
        get_cached_transformers_schema_constraint(entry, TemporarySchema)
        return weakref.ref(TemporarySchema)

    reference = temporary_schema()
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("kind", ["writer", "standalone"])
def test_capacity_marker_survives_normalization_without_changing_default_guard(kind, monkeypatch):
    from nodes import _otr_loader_backends as backends
    from nodes._otr_generation_budget import ProviderCapacityMessages
    from nodes._otr_decode_guard import MAX_OPEN_STRING_TOKENS
    fn, _entry, _model, _refs, knobs = _reusable_native_closure(kind, monkeypatch, "success")
    monkeypatch.setattr(backends, "tokenizer_supports_system_role", lambda _tokenizer: False)
    monkeypatch.setattr(backends, "normalize_messages_for_tokenizer", lambda _tokenizer, messages: list(messages))
    messages = [{"role": "user", "content": "A complete structured story."}]
    fn(ProviderCapacityMessages(messages), temperature=.2,
       max_new_tokens=None)
    fn(messages, temperature=.2, max_new_tokens=100)
    assert [entry["open_string_bound"] for entry in knobs] == [None, MAX_OPEN_STRING_TOKENS]


@pytest.mark.parametrize("kind", ["writer", "standalone"])
def test_open_string_halt_reports_its_actual_reason_and_evidence(kind, monkeypatch, caplog):
    from nodes._otr_generation_budget import GenerationDegeneracyError
    fn, _entry, _model, _refs, _knobs = _reusable_native_closure(kind, monkeypatch, "open_string")
    with pytest.raises(GenerationDegeneracyError) as caught:
        fn([{"role": "user", "content": "Structured reply."}],
           temperature=.2, max_new_tokens=5000)
    assert caught.value.halt_reason == "open_string"
    assert caught.value.open_string_tokens >= 2048
    assert caught.value.raw_completion.startswith('"')
    assert "open JSON string" in str(caught.value)
    halt_logs = [r.message for r in caplog.records if "DECODE HALTED" in r.message]
    assert halt_logs and all("open JSON string" in message and "repeated" not in message for message in halt_logs)
