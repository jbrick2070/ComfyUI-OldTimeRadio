import pytest
from pydantic import BaseModel

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_comfy_backend as comfy
from nodes import _otr_loader_backends as loader_backends
from nodes import _otr_model_loader as model_loader
from nodes import _otr_openrouter_backend as openrouter
from nodes._otr_generation_budget import (
    GenerationContextOverflowError,
    fit_output_tokens,
)


class _RequireFullMessages(list):
    _otr_require_full_output_budget = True
    _otr_strict_remote_output_budget = True


def test_720_word_script_request_is_clamped_to_remaining_context():
    assert fit_output_tokens(
        9520, context_cap=8192, prompt_tokens=3200,
    ) == 4992


def test_prod_length_script_fits_only_at_raised_context_cap():
    """The production-length context-cap wall (2026-07-19) and its fix, at the
    transport-arithmetic level. The source banks inflate the P5 prompt to
    ~4785 tokens; a 720w script needs ~6960 output tokens
    (_script_output_token_budget). At the false 8192 cap the request is CLAMPED
    (-> truncated JSON -> the live P5 JSONDecodeError); at Mistral-Nemo's raised
    16384 authoritative cap the full request is admitted unclamped.
    """
    prompt_tokens = 4785
    needed_output = 6960
    # The wall: 8192 clamps below what the 720w script needs.
    assert fit_output_tokens(
        needed_output, context_cap=8192, prompt_tokens=prompt_tokens,
    ) == 8192 - prompt_tokens
    assert 8192 - prompt_tokens < needed_output
    # The fix: 16384 admits the full request, no truncation.
    assert fit_output_tokens(
        needed_output, context_cap=16384, prompt_tokens=prompt_tokens,
    ) == needed_output


def test_context_budget_fails_when_prompt_leaves_no_viable_artifact_room():
    with pytest.raises(GenerationContextOverflowError, match="cannot fit"):
        fit_output_tokens(512, context_cap=8192, prompt_tokens=8150)


def test_complete_patch_budget_refuses_clamp_but_default_call_still_clamps():
    assert fit_output_tokens(
        2000, context_cap=8192, prompt_tokens=7000,
    ) == 1192
    with pytest.raises(
        GenerationContextOverflowError,
        match="complete requested output",
    ):
        fit_output_tokens(
            2000,
            context_cap=8192,
            prompt_tokens=7000,
            require_full=True,
        )


def test_local_transport_clamps_output_without_left_truncating_prompt():
    class Tensor:
        shape = (1, 7100)

        def __getitem__(self, _key):
            raise AssertionError("the prompt must not be sliced")

    class OutputTensor:
        shape = (1, 1092)

        def __getitem__(self, _key):
            return self

    class Inputs(dict):
        def __init__(self, input_ids):
            super().__init__(input_ids=input_ids)

        def to(self, _device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, _messages, **_kwargs):
            return "serialized prompt"

        def __call__(self, _prompt, *, return_tensors):
            assert return_tensors == "pt"
            return Inputs(Tensor())

        def decode(self, _tokens, *, skip_special_tokens):
            assert skip_special_tokens is True
            return "{}"

    class Model:
        device = "cpu"

        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [OutputTensor()]

    model = Model()
    generate = writer._build_truncating_generate_fn({
        "model": model,
        "tokenizer": Tokenizer(),
        "context_cap": 8192,
    })

    result = generate(
        [{"role": "user", "content": "720 words"}],
        temperature=.2,
        max_new_tokens=9520,
    )

    assert result == "{}"
    assert model.kwargs["max_new_tokens"] == 1092
    assert model.kwargs["input_ids"].shape == (1, 7100)


def test_writer_local_complete_patch_refuses_before_model_generate(
    monkeypatch,
):
    class Tensor:
        shape = (1, 7000)

    class Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=Tensor())

        def to(self, _device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, _messages, **_kwargs):
            return "serialized prompt"

        def __call__(self, _prompt, *, return_tensors):
            assert return_tensors == "pt"
            return Inputs()

    class Model:
        device = "cpu"

        def generate(self, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("capacity preflight must prevent generation")

    monkeypatch.setattr(
        loader_backends, "tokenizer_supports_system_role", lambda _tokenizer: False,
    )
    monkeypatch.setattr(
        loader_backends,
        "normalize_messages_for_tokenizer",
        lambda _tokenizer, messages: list(messages),
    )
    generate = writer._build_truncating_generate_fn({
        "model": Model(), "tokenizer": Tokenizer(), "context_cap": 8192,
    })
    with pytest.raises(
        writer.PromptContextOverflowError,
        match="complete requested output",
    ):
        generate(
            _RequireFullMessages([
                {"role": "user", "content": "compact patch"},
            ]),
            temperature=.2,
            max_new_tokens=2000,
        )


@pytest.mark.parametrize(
    "factory_name", ["make_generate_fn", "make_polish_generate_fn"],
)
def test_model_loader_captures_complete_patch_marker_before_normalization(
    monkeypatch, factory_name,
):
    class Tensor:
        shape = (1, 7000)

    class Inputs(dict):
        def __init__(self):
            super().__init__(input_ids=Tensor())

        def to(self, _device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, _messages, **_kwargs):
            return "serialized prompt"

        def __call__(self, _prompt, *, return_tensors):
            assert return_tensors == "pt"
            return Inputs()

    class Model:
        device = "cpu"

        def generate(self, **_kwargs):  # pragma: no cover - must never run
            raise AssertionError("capacity preflight must prevent generation")

    monkeypatch.setattr(
        model_loader,
        "_normalize_messages_for_cache_entry",
        lambda _entry, messages: list(messages),
    )
    generate = getattr(model_loader, factory_name)({
        "model": Model(),
        "tokenizer": Tokenizer(),
        "model_id": "local-test",
        "context_cap": 8192,
    })
    with pytest.raises(
        model_loader.ModelLoaderError,
        match="complete requested output",
    ):
        generate(
            _RequireFullMessages([
                {"role": "user", "content": "compact patch"},
            ]),
            temperature=.2,
            max_new_tokens=2000,
        )


def test_local_structured_transport_adds_prefix_without_losing_sampling(
    monkeypatch,
):
    class ResultSchema(BaseModel):
        status: str

    class Tensor:
        shape = (1, 20)

    class OutputTensor:
        shape = (1, 5)

        def __getitem__(self, _key):
            return self

    class Inputs(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, _messages, **_kwargs):
            return "serialized prompt"

        def __call__(self, _prompt, *, return_tensors):
            assert return_tensors == "pt"
            return Inputs(input_ids=Tensor())

        def decode(self, _tokens, *, skip_special_tokens):
            assert skip_special_tokens is True
            return '{"status":"ready"}'

    class Model:
        device = "cpu"

        def __init__(self):
            self.kwargs = None

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return [OutputTensor()]

    prefix = object()
    parser = object()
    monkeypatch.setattr(
        "nodes._otr_constrained_generate.get_cached_transformers_schema_constraint",
        lambda cache_entry, schema_model: (parser, prefix),
    )
    model = Model()
    generate = writer._build_truncating_generate_fn(
        {"model": model, "tokenizer": Tokenizer(), "context_cap": 8192},
        top_p=.88,
        min_p=.04,
        repetition_penalty=1.03,
        schema_model=ResultSchema,
    )

    assert generate(
        [{"role": "user", "content": "structured"}],
        temperature=.2,
        max_new_tokens=100,
    ) == '{"status":"ready"}'
    assert model.kwargs["prefix_allowed_tokens_fn"] is prefix
    assert model.kwargs["num_beams"] == 1
    assert model.kwargs["top_p"] == pytest.approx(.88)
    assert model.kwargs["min_p"] == pytest.approx(.04)
    assert model.kwargs["repetition_penalty"] == pytest.approx(1.03)
    assert generate.schema_model is ResultSchema


def test_openrouter_transport_subtracts_prompt_from_remote_output_budget(
        monkeypatch):
    seen = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        openrouter,
        "_post_chat_completion",
        lambda **kwargs: seen.update(kwargs) or {
            "status_code": 200,
            "json": {"choices": [{"message": {"content": "{}"}}]},
            "text": "",
        },
    )
    entry = {
        "slug": "test/model",
        "context_cap": 8192,
        "max_tokens_cap": 8192,
        "base_url": "https://example.invalid",
    }

    openrouter.OpenRouterBackend().generate(
        entry,
        [{"role": "user", "content": "x" * 28000}],
        temperature=.2,
        max_new_tokens=9520,
    )

    assert seen["payload"]["max_tokens"] == 1192


def test_openrouter_complete_patch_refuses_before_network(monkeypatch):
    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        openrouter,
        "_post_chat_completion",
        lambda **kwargs: calls.append(kwargs),
    )
    entry = {
        "slug": "test/model",
        "context_cap": 8192,
        "max_tokens_cap": 8192,
        "base_url": "https://example.invalid",
    }
    with pytest.raises(
        openrouter.OpenRouterConfigError,
        match="complete requested output",
    ):
        openrouter.OpenRouterBackend().generate(
            entry,
            _RequireFullMessages([
                {"role": "user", "content": "x" * 28000},
            ]),
            temperature=.2,
            max_new_tokens=2000,
        )
    assert calls == []


def test_openrouter_complete_patch_refuses_provider_cap_before_network(
    monkeypatch,
):
    calls = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(
        openrouter,
        "_post_chat_completion",
        lambda **kwargs: calls.append(kwargs),
    )
    with pytest.raises(
        openrouter.OpenRouterConfigError,
        match="provider_output_cap=1000",
    ):
        openrouter.OpenRouterBackend().generate(
            {
                "slug": "test/model",
                "context_cap": 8192,
                "max_tokens_cap": 1000,
                "base_url": "https://example.invalid",
            },
            _RequireFullMessages([
                {"role": "user", "content": "compact"},
            ]),
            temperature=.2,
            max_new_tokens=2000,
        )
    assert calls == []


def test_comfy_credits_complete_patch_refuses_before_network(monkeypatch):
    calls = []
    monkeypatch.setattr(comfy, "_bearer", lambda: "test-token")
    backend = comfy.ComfyCreditsBackend()
    monkeypatch.setattr(
        backend,
        "_post_with_retries",
        lambda **kwargs: calls.append(kwargs),
    )
    entry = {
        "slug": "test/model",
        "context_cap": 8192,
        "max_tokens_cap": 8192,
    }
    with pytest.raises(
        comfy.ComfyCreditsConfigError,
        match="complete requested output",
    ):
        backend.generate(
            entry,
            _RequireFullMessages([
                {"role": "user", "content": "x" * 28000},
            ]),
            temperature=.2,
            max_new_tokens=2000,
        )
    assert calls == []


def test_comfy_credits_strict_patch_keeps_exact_requested_budget(monkeypatch):
    seen = {}
    monkeypatch.setattr(comfy, "_bearer", lambda: "test-token")
    backend = comfy.ComfyCreditsBackend()

    def post(**kwargs):
        seen.update(kwargs)
        return "{}"

    monkeypatch.setattr(backend, "_post_with_retries", post)
    out = backend.generate(
        {
            "slug": "test/model",
            "context_cap": 8192,
            "max_tokens_cap": 8192,
        },
        _RequireFullMessages([
            {"role": "user", "content": "compact"},
        ]),
        temperature=.2,
        max_new_tokens=384,
    )
    assert out == "{}"
    assert seen["payload"]["max_tokens"] == 384
