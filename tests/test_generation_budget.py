import pytest

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_openrouter_backend as openrouter
from nodes._otr_generation_budget import (
    GenerationContextOverflowError,
    fit_output_tokens,
)


def test_720_word_script_request_is_clamped_to_remaining_context():
    assert fit_output_tokens(
        9520, context_cap=8192, prompt_tokens=3200,
    ) == 4992


def test_context_budget_fails_when_prompt_leaves_no_viable_artifact_room():
    with pytest.raises(GenerationContextOverflowError, match="cannot fit"):
        fit_output_tokens(512, context_cap=8192, prompt_tokens=8150)


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
