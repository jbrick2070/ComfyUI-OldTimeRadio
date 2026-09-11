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


def _exact_prompt_entry(monkeypatch, capacity=32768):
    torch = pytest.importorskip("torch")
    import weakref
    from nodes import _otr_constrained_generate as constrained
    runs, moves, refs, generated = [], [], [], []

    class Inputs(dict):
        def to(self, device):
            moves.append(device)
            return self

    class Tokenizer:
        eos_token_id = 20000

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False and kwargs["add_generation_prompt"] is True
            if any(m["role"] == "system" for m in messages):
                raise ValueError("System role not supported")
            assert kwargs.get("enable_thinking") is False
            return "<start>" + "|".join(m["role"] + ":" + m["content"] for m in messages) + "<assistant>"

        def __call__(self, prompt, *, return_tensors):
            assert return_tensors == "pt"
            ids = [31000, *map(ord, prompt), 31001]
            runs.append(ids)
            inputs = Inputs(input_ids=torch.tensor([ids]))
            refs.append(weakref.ref(inputs))
            return inputs

        def decode(self, tokens, **kwargs):
            return '{"value":"ok"}'

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            generated.append((kwargs["input_ids"].tolist()[0], kwargs["max_new_tokens"]))
            return torch.cat([kwargs["input_ids"], torch.tensor([[20000]])], dim=1)

    monkeypatch.setattr(writer._OTRHB, "make_streamer", lambda *args: None)
    monkeypatch.setattr(constrained, "get_cached_transformers_schema_constraint",
                        lambda *args: (None, lambda *args: []))
    entry = {"model": Model(), "tokenizer": Tokenizer(), "model_id": "Qwen/Qwen3.5-4B",
             "context_cap": capacity, "native_context_capacity": capacity,
             "context_capacity_source": "loaded decoder config"}
    return entry, runs, moves, refs, generated


@pytest.mark.parametrize("route", ["writer", "constrained", "base", "polish"])
def test_native_fit_and_generation_use_identical_cpu_prompt_without_truncation(route, monkeypatch):
    import gc
    from nodes import _otr_constrained_generate as constrained
    from nodes._otr_generation_budget import ProviderCapacityMessages
    entry, runs, moves, refs, generated = _exact_prompt_entry(monkeypatch)
    from nodes import _vram_log
    observed = []
    def observe(phase, **fields):
        # Actual generation has returned, while the output is still available.
        assert len(generated) == 1
        observed.append((phase, fields))
    monkeypatch.setattr(_vram_log, "memory_snapshot", observe)
    messages = ProviderCapacityMessages([
        {"role": "system", "content": "Keep the complete supplied source."},
        {"role": "user", "content": "Beginning " + "source text " * 850 + " END MARKER"},
    ])
    measured = model_loader.inspect_native_prompt_fit(entry, messages, max_new_tokens=None)
    assert measured["fits"] and measured["prompt_tokens"] > 8192
    assert measured["capacity_known"] and measured["context_cap"] == 32768
    assert not moves and not generated
    assert not observed  # Fit inspection must not pretend a generation returned.
    gc.collect()
    assert all(ref() is None for ref in refs)
    factories = {"writer": writer._build_truncating_generate_fn,
                 "constrained": lambda e: constrained.make_constrained_generate_fn(e, _FitSchema),
                 "base": model_loader.make_generate_fn, "polish": model_loader.make_polish_generate_fn}
    result = factories[route](entry)(messages, temperature=.2, max_new_tokens=None)
    assert result == '{"value":"ok"}'
    assert observed == [(route + "_generation_returned", {"model_id": entry["model_id"]})]
    assert runs[0] == runs[1] == generated[0][0]
    assert generated[0][1] == measured["effective_output_tokens"]
    assert moves == ["cpu"]
    assert all(m["role"] in ("system", "user") for m in messages)
    assert messages[0]["role"] == "system"  # no normalization mutation


class _FitSchema(BaseModel):
    value: str


@pytest.mark.parametrize("eos", [20000, [19999, 20000], (19999, 20000), {19999, 20000}])
def test_constrained_eos_at_exact_capacity_is_a_completed_reply(eos, monkeypatch):
    from nodes import _otr_constrained_generate as constrained
    from nodes._otr_generation_budget import ProviderCapacityMessages
    entry, runs, moves, refs, generated = _exact_prompt_entry(monkeypatch)
    messages = ProviderCapacityMessages([{"role": "user", "content": "Reply."}])
    prepared = model_loader.prepare_native_prompt(entry, messages)
    entry["context_cap"] = prepared["prompt_tokens"] + 1
    entry["tokenizer"].eos_token_id = eos
    result = constrained.make_constrained_generate_fn(entry, _FitSchema)(
        messages, temperature=.2, max_new_tokens=None)
    assert result == '{"value":"ok"}' and generated[-1][1] == 1


def test_unmarked_none_budget_is_a_programmer_error_before_prompt_preparation(monkeypatch):
    entry, runs, moves, refs, generated = _exact_prompt_entry(monkeypatch)
    with pytest.raises(TypeError, match="provider-capacity message contract"):
        model_loader.inspect_native_prompt_fit(entry, [{"role": "user", "content": "Reply."}],
                                              max_new_tokens=None)
    assert not runs and not moves and not generated


@pytest.mark.parametrize("bound", [False, True])
def test_scheduler_fit_includes_schema_and_does_not_count_as_generation(bound, monkeypatch):
    from nodes._otr_structured_call import inspect_structured_fit, structured_call
    from nodes._otr_generation_budget import ProviderCapacityMessages
    entry, runs, moves, refs, generated = _exact_prompt_entry(monkeypatch)
    monkeypatch.setattr(model_loader, "request_slot", lambda *args, **kwargs: entry)
    scheduler = writer._SlotScheduler(creative_id="Qwen/Qwen3.5-4B", technical_id="Qwen/Qwen3.5-4B",
                                      top_p=.92, min_p=0, repetition_penalty=1)
    slot = scheduler.for_slot("creative")
    if bound:
        slot = slot._otr_bind_schema(_FitSchema)
    source = ProviderCapacityMessages([{"role": "user", "content": "Return the value."}])
    measured = inspect_structured_fit(slot, source, _FitSchema, max_new_tokens=None)
    assert measured["fits"] and not generated and not moves
    assert scheduler.calls_by_slot == {"creative": 0, "technical": 0}
    assert scheduler.slot_calls_by_helper == {}
    assert len(runs[0]) > len(source[0]["content"]) + 100
    result = structured_call(helper_name="fit-proof", slot_fn=slot, prompt=source,
                             schema=_FitSchema, base_temperature=.2, structural_retry_temperature=.1,
                             max_new_tokens=None, max_attempts=1)
    assert result.value == "ok"
    assert generated[0][0] == runs[0] == runs[1]
    assert scheduler.calls_by_slot == {"creative": 1, "technical": 0}
    assert source[0]["content"] == "Return the value."


@pytest.mark.parametrize("route", ["writer", "constrained", "base", "polish"])
def test_fit_inspection_releases_inputs_and_refuses_atomic_budget_before_device_move(route, monkeypatch):
    import gc
    from nodes import _otr_constrained_generate as constrained
    entry, runs, moves, refs, generated = _exact_prompt_entry(monkeypatch, capacity=128)
    messages = _RequireFullMessages([{"role": "user", "content": "A small complete patch."}])
    measured = model_loader.inspect_native_prompt_fit(entry, messages, max_new_tokens=128)
    assert measured["fits"] is False and measured["phase"] == "prompt_no_room"
    assert not moves and not generated
    gc.collect()
    assert all(ref() is None for ref in refs)
    factories = {"writer": writer._build_truncating_generate_fn,
                 "constrained": lambda e: constrained.make_constrained_generate_fn(e, _FitSchema),
                 "base": model_loader.make_generate_fn, "polish": model_loader.make_polish_generate_fn}
    with pytest.raises(writer.PromptContextOverflowError, match="complete requested output"):
        factories[route](entry)(messages, temperature=.2, max_new_tokens=128)
    assert not moves and not generated


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


def test_context_budget_accepts_one_token_and_preserves_explicit_minimum():
    assert fit_output_tokens(512, context_cap=8192, prompt_tokens=8191) == 1
    assert fit_output_tokens(512, context_cap=8192, prompt_tokens=8150) == 42
    with pytest.raises(GenerationContextOverflowError, match="cannot fit"):
        fit_output_tokens(512, context_cap=8192, prompt_tokens=8192)
    with pytest.raises(GenerationContextOverflowError, match="at least 64"):
        fit_output_tokens(512, context_cap=8192, prompt_tokens=8150, min_output_tokens=64)


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
    # The TYPE changed on 2026-08-13 and the change is the point. This used to
    # expect a bare ModelLoaderError, which carries NO phase -- so the retry
    # ladder, which reads the phase to decide whether a failure is rerollable,
    # treated an identical capacity failure as terminal on this transport and
    # rerollable on OTR_LedgerScriptWriter's. Same condition, same marker, two
    # answers, decided by which transport the pass happened to take. The marker
    # match below is this test's actual subject and is unchanged.
    with pytest.raises(
        model_loader.PromptContextOverflowError,
        match="complete requested output",
    ) as excinfo:
        generate(
            _RequireFullMessages([
                {"role": "user", "content": "compact patch"},
            ]),
            temperature=.2,
            max_new_tokens=2000,
        )
    assert excinfo.value.phase, (
        "the whole reason this raise changed type is that it must carry a "
        "phase for the ladder to route on")


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
