"""Sprint 10A step 3-B regression -- make_constrained_generate_fn factory.

The factory itself can be exercised without a real model + tokenizer:
  * Build the JsonSchemaParser from a pydantic schema.
  * Build the prefix_allowed_tokens_fn from a (minimal) tokenizer
    stand-in.
  * Verify the closure carries the parser + prefix-fn for inspection.
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
        from nodes._otr_constrained_generate import make_constrained_generate_fn

        cache_entry = self._make_minimal_cache_entry()
        fn = make_constrained_generate_fn(cache_entry, _TinySchema)

        assert callable(fn)
        # Closure exposes the schema bindings for introspection.
        assert fn.schema_model is _TinySchema
        assert fn.json_schema_parser is not None
        assert fn.prefix_allowed_tokens_fn is not None


# ---------------------------------------------------------------------------
# Closure passes prefix_allowed_tokens_fn into generate
# ---------------------------------------------------------------------------


class TestClosurePassesConstraint:
    def test_generate_invoked_with_prefix_fn(self):
        from nodes._otr_constrained_generate import make_constrained_generate_fn

        # Build a fuller mock: tokenizer + model whose generate()
        # returns a single fake token row.
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = {chr(c): c for c in range(32, 127)}
        tokenizer.vocab_size = 95
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
        with patch("nodes._otr_constrained_generate._normalize_messages_for_cache_entry",
                   side_effect=lambda ce, msgs: msgs):
            fn = make_constrained_generate_fn(cache_entry, _TinySchema)
            # Stub torch import inside the closure.
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *a, **kw):
                if name == "torch":
                    fake_torch = MagicMock()
                    fake_torch.no_grad.return_value.__enter__ = MagicMock()
                    fake_torch.no_grad.return_value.__exit__ = MagicMock()
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
        assert gkwargs["prefix_allowed_tokens_fn"] is fn.prefix_allowed_tokens_fn
        # And num_beams=1 must be set explicitly per the docstring rationale.
        assert gkwargs.get("num_beams") == 1
        # temperature + max_new_tokens passed through verbatim
        assert gkwargs.get("temperature") == 0.5
        assert gkwargs.get("max_new_tokens") == 50
