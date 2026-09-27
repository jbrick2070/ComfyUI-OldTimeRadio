"""The Comfy-native Gemma 4 writer (plan row 0n): identity, weight, generation, wiring.

Slices A1-A3 of kibitz-runs/2026-09-26-comfy-gemma-writer. CPU only, no network, no
ComfyUI: every assertion calls the real function it is about.
"""
from __future__ import annotations

import pytest

from nodes import _otr_comfy_textgen_backend as native
from nodes import _otr_visual_assets as va


def _prompt(creative="Qwen/Qwen3.5-4B", technical="Qwen/Qwen3.5-4B", replay=""):
    return {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "1": {"class_type": "OTR_LedgerScriptWriter",
              "inputs": {"gate_in": ["63", 0], "replay_from": replay,
                         "creative_writing_model": creative,
                         "technical_model": technical}},
    }


def _plan(prompt):
    return va.plan_prompt(prompt, "63", resolve_video=lambda v: v,
                          freeze_video=lambda v: v, role_video_slots={})


def test_the_manifest_row_is_the_backend_pin():
    """One pin, two readers: the fetch table and the backend must name the same
    file at the same revision, size and hash, or the download verifies a file the
    loader then refuses."""
    spec = va.MANIFEST[(native.WEIGHT_CATEGORY, native.WEIGHT_TOKEN)]
    assert spec == {"repo_id": native.WEIGHT_REPO,
                    "filename": native.WEIGHT_FILENAME,
                    "revision": native.WEIGHT_REVISION,
                    "size": native.WEIGHT_SIZE,
                    "sha256": native.WEIGHT_SHA256}
    assert native.WEIGHT_FILENAME.rsplit("/", 1)[-1] == native.WEIGHT_TOKEN


def test_only_the_exact_native_id_needs_a_weight():
    assert native.native_writer_weights(native.MODEL_ID) == (
        (native.WEIGHT_CATEGORY, native.WEIGHT_TOKEN),)
    for other in ("google/gemma-4-E2B-it", "Qwen/Qwen3.5-4B", "comfy:slot-a",
                  "", None, native.MODEL_ID + " (5.2 GB download)"):
        assert native.native_writer_weights(other) == (), other
    assert native.is_native_writer(native.MODEL_ID)
    assert not native.is_native_writer("google/gemma-4-E2B-it")


@pytest.mark.parametrize("creative, technical", [
    (native.MODEL_ID, "Qwen/Qwen3.5-4B"),
    ("Qwen/Qwen3.5-4B", native.MODEL_ID + " (5.2 GB download)"),
    (native.MODEL_ID + " (5.2 GB download)", native.MODEL_ID),
])
def test_either_slot_plans_the_native_writer_once(creative, technical):
    plan = _plan(_prompt(creative, technical))
    assert plan["writer_models"] == {native.MODEL_ID}


def test_hf_and_cloud_writer_picks_plan_no_writer_weight():
    for creative, technical in (("Qwen/Qwen3.5-4B", "google/gemma-4-12b-it"),
                                ("comfy:slot-a", "comfy:slot-b"),
                                ("google_api:slot-a", "google_api:slot-b")):
        assert _plan(_prompt(creative, technical))["writer_models"] == set()


def test_a_replay_plans_no_writer_weight():
    """A replay's writer passes through frozen; nothing it names is fetched."""
    plan = _plan(_prompt(native.MODEL_ID, native.MODEL_ID, replay="C:/bundle"))
    assert plan["replay"] is True
    assert plan["writer_models"] == set()


def test_both_slots_request_the_pinned_weight_once():
    requests = va.native_requests(set(), folder_paths=va._NothingInstalled,
                                  writer_models={native.MODEL_ID})
    assert [(r["category"], r["token"]) for r in requests] == [
        (native.WEIGHT_CATEGORY, native.WEIGHT_TOKEN)]
    assert requests[0]["path"] is None
    assert requests[0]["spec"]["revision"] == native.WEIGHT_REVISION
    assert requests[0]["spec"]["sha256"] == native.WEIGHT_SHA256


def test_no_writer_models_requests_nothing_extra():
    assert va.native_requests(set(), folder_paths=va._NothingInstalled) == []


def test_a_linked_writer_widget_never_refuses_the_readiness_pass():
    """The video weights must still plan when the writer's model is wired from
    another node; the linked slot is noted, never a refusal."""
    prompt = _prompt()
    prompt["1"]["inputs"]["creative_writing_model"] = ["99", 0]
    del prompt["1"]["inputs"]["technical_model"]
    plan = _plan(prompt)
    assert plan["writer_models"] == set()
    assert any("creative_writing_model is linked" in n for n in plan["skipped"])


# ---------------------------------------------------------------------------
# SLICE A1: the generate adapter. CPU only: a fake ComfyUI CLIP with the real
# attribute path (cond_stage_model.gemma4.transformer.sample_token) and a loop
# shaped like comfy/text_encoders/llama.py BaseGenerate.generate, plus a REAL
# tiny fast tokenizer so the real lm-format-enforcer and the writer's real
# factory run against the adapter.
# ---------------------------------------------------------------------------
import os
from pathlib import Path
from types import SimpleNamespace

import torch

_SPECIALS = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>"]
_CHARS = list('{}[]":,. abcdefghijklmnopqrstuvwxyz0123456789') + ["\n"]
EOS = 1


def _raw_tokenizer():
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
    vocab = {t: i for i, t in enumerate(_SPECIALS + _CHARS)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    raw.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    raw.decoder = decoders.Fuse()
    raw.add_special_tokens([AddedToken(s, special=True) for s in _SPECIALS])
    return raw


@pytest.fixture(scope="module")
def tok():
    return native.build_native_tokenizer(_raw_tokenizer())


class _Decoder:
    """Plays ComfyUI's BaseGenerate: the loop looks sample_token up on the
    instance every step, passes an EMPTY history, appends only after it
    returns, and stops on its own stop ids."""

    def __init__(self, script, vocab, stop_tokens=(EOS,)):
        self.script = script            # step -> token id the logits favour
        self.vocab = vocab
        self.stop_tokens = set(stop_tokens)
        # ComfyUI's decoder exposes its stop ids here (gemma4.py stop_tokens).
        self.model = SimpleNamespace(config=SimpleNamespace(stop_tokens=list(stop_tokens)))
        self.histories = []
        self.finished = False
        self.seeds = []

    def sample_token(self, logits, temperature, top_k, top_p, min_p,
                     repetition_penalty, token_history, generator, do_sample=True,
                     presence_penalty=0.0, penalty_mask=None):
        self.histories.append(list(token_history))
        return torch.argmax(logits, dim=-1, keepdim=True)

    def loop(self, prompt, max_length, seed):
        self.seeds.append(seed)
        generated = []
        for step in range(max_length):
            logits = torch.zeros((1, self.vocab))
            logits[0, self.script(step)] = 5.0
            token = self.sample_token(logits, 1.0, 0, 1.0, 0.0, 1.0, [], None)
            tid = int(token.reshape(-1)[0])
            generated.append(tid)
            if tid in self.stop_tokens:
                break
        self.finished = True
        return generated


class _Clip:
    def __init__(self, decoder):
        self.resets = 0
        clip = self

        class _CSM:
            gemma4 = SimpleNamespace(transformer=decoder)

            def reset_clip_options(self_inner):
                assert "sample_token" not in decoder.__dict__, "reset ran before restore"
                clip.resets += 1

        self.cond_stage_model = _CSM()
        self.patcher = SimpleNamespace(load_device="cpu")
        self.calls = []

    # ComfyUI 0.34's CLIP.generate: no ``mtp`` (0.37 added one Gemma 4 ignores).
    # The adapter must run on both, so the fake takes only what 0.34 takes.
    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p,
                 min_p, repetition_penalty, seed, presence_penalty):
        assert isinstance(seed, int), "Comfy's sampler needs an int seed"
        self.calls.append(dict(do_sample=do_sample, max_length=max_length,
                               top_k=top_k, top_p=top_p, min_p=min_p,
                               repetition_penalty=repetition_penalty))
        prompt = [t[0] for t in tokens["gemma4"][0]]
        return self.cond_stage_model.gemma4.transformer.loop(prompt, max_length, seed)


def _adapter(script, vocab=64, stops=(EOS,)):
    decoder = _Decoder(script, vocab)
    clip = _Clip(decoder)
    return native.ComfyGemmaGenerateAdapter(clip, stop_token_ids=stops), clip, decoder


def _ids(*tokens):
    return torch.tensor([list(tokens)], dtype=torch.long)


def test_the_template_matches_googles_official_one_on_text_turns(tok):
    """The official E2B template renders these conversations the same way.
    Skipped where the HF snapshot is not installed; the renderings below are
    the ones it produced on 2026-09-26."""
    conversations = [
        [{"role": "user", "content": "U1"}],
        [{"role": "system", "content": " SYS "}, {"role": "user", "content": " U1\n"}],
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"},
         {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}],
        [{"role": "user", "content": "U1"},
         {"role": "assistant", "content": "<|channel>thought\nx<channel|>A1 "},
         {"role": "user", "content": "U2"}],
    ]
    ours = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations]
    assert ours[2] == ("<bos><|turn>system\nSYS<turn|>\n<|turn>user\nU1<turn|>\n"
                       "<|turn>model\nA1<turn|>\n<|turn>user\nU2<turn|>\n<|turn>model\n")
    snapshot = Path(os.environ.get("OTR_GEMMA4_E2B_HF_SNAPSHOT", (
        r"C:\ComfyUI-Models\huggingface\hub\models--google--gemma-4-E2B-it"
        r"\snapshots\6b7e72c67d3c4556f42b56d5a68b4b8e864c63b4")))
    if not (snapshot / "chat_template.jinja").is_file():
        pytest.skip("official Gemma 4 E2B template not installed here")
    from transformers import AutoTokenizer
    official = AutoTokenizer.from_pretrained(str(snapshot))
    for conversation, mine in zip(conversations, ours):
        assert mine == official.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True), conversation


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
    [{"role": "tool", "content": "x"}],
])
def test_the_template_refuses_what_it_does_not_render(tok, messages):
    with pytest.raises(Exception, match="text-only"):
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def test_special_tokens_are_marked_special(tok):
    assert {tok.convert_tokens_to_ids(s) for s in _SPECIALS} <= set(tok.all_special_ids)


def test_the_seed_is_the_writers_when_set_and_random_when_not():
    from nodes import _otr_sampling_seed as seed_leaf
    prompt = [2, 5, 6]
    assert native.native_seed(prompt, raw_env="7") == seed_leaf.prompt_keyed_seed(7, [prompt])
    unseeded = {native.native_seed(prompt, raw_env="") for _ in range(8)}
    assert all(isinstance(s, int) and s > 0 for s in unseeded)
    assert len(unseeded) > 1, "unseeded production must not be a constant"
    malformed = native.native_seed(prompt, raw_env="not-an-int")
    assert isinstance(malformed, int) and malformed > 0, "malformed -> random, never raises"


def test_generation_returns_prompt_plus_ids_through_a_real_eos():
    adapter, clip, decoder = _adapter(lambda step: [7, 8, EOS][min(step, 2)])
    out = adapter.generate(input_ids=_ids(2, 5), do_sample=False, max_new_tokens=10)
    assert out.tolist() == [[2, 5, 7, 8, EOS]]
    assert all(h == [] for h in decoder.histories), "the loop passes an empty history"
    assert "sample_token" not in decoder.__dict__ and clip.resets == 1
    assert clip.calls[0]["max_length"] == 10


def test_the_budget_is_new_tokens_and_is_honoured():
    adapter, _, _ = _adapter(lambda step: 9)
    out = adapter.generate(input_ids=_ids(2), do_sample=False, max_new_tokens=4)
    assert out.tolist() == [[2, 9, 9, 9, 9]]


def test_a_stopping_criterion_latches_and_the_stop_is_not_reported_as_eos():
    adapter, _, decoder = _adapter(lambda step: 11)
    seen = []

    def after_three(input_ids, scores):
        seen.append(tuple(input_ids.shape))
        return input_ids.shape[-1] >= 4

    out = adapter.generate(input_ids=_ids(2), do_sample=False, max_new_tokens=50,
                           stopping_criteria=[after_three])
    assert out.tolist() == [[2, 11, 11, 11]]
    assert decoder.finished, "the stock loop must close itself"
    assert seen[0] == (1, 2), "criteria see a [1, n] view of prompt + generated"


def test_an_extra_eos_the_loop_does_not_know_still_ends_the_loop():
    """Composer QA on 34e5e3a2: a caller's extra eos id (here 0) is not in the
    decoder's own stop list. The hook must halt on it AND hand the loop an id
    the loop stops on, or the loop runs forward passes to the budget."""
    adapter, _, decoder = _adapter(lambda step: [7, 0, 9][min(step, 2)])
    out = adapter.generate(input_ids=_ids(2), do_sample=False, max_new_tokens=50,
                           eos_token_id=[0])
    assert out.tolist() == [[2, 7, 0]]
    assert len(decoder.histories) == 2, "the loop ended on the halting step itself"


@pytest.mark.parametrize("loop_kwargs", [
    {"do_sample": True, "presence_penalty": 0.0},                       # ComfyUI 0.34
    {"do_sample": True, "presence_penalty": 0.0, "penalty_mask": None},  # ComfyUI 0.37
])
def test_the_hook_hands_the_sampler_exactly_what_the_installed_loop_passed(loop_kwargs):
    """Measured on the 4060's stock 0.34 portable: its sample_token has no
    penalty_mask, so a hook that always forwards one raises TypeError there."""
    seen = []

    def original(logits, temperature, top_k, top_p, min_p, repetition_penalty,
                 token_history, generator, **kwargs):
        seen.append(kwargs)
        return torch.tensor([[7]])

    run = native._Run([2], 4, {EOS}, None, None, None, loop_stop_id=EOS)
    token = run.hook(original)(torch.zeros((1, 16)), 1.0, 0, 1.0, 0.0, 1.0, [2], None,
                               **loop_kwargs)
    assert seen == [loop_kwargs] and int(token) == 7 and run.error is None


def test_a_callback_error_is_reraised_after_the_loop_closes_and_the_hook_is_gone():
    adapter, clip, decoder = _adapter(lambda step: 11)

    def broken(input_ids, scores):
        if input_ids.shape[-1] >= 3:
            raise KeyError("criterion blew up")
        return False

    with pytest.raises(KeyError, match="criterion blew up"):
        adapter.generate(input_ids=_ids(2), do_sample=False, max_new_tokens=50,
                         stopping_criteria=[broken])
    assert decoder.finished, "the loop exited normally; the error was latched"
    assert "sample_token" not in decoder.__dict__ and clip.resets == 1


def test_a_genuine_loop_exception_restores_the_hook():
    adapter, clip, decoder = _adapter(lambda step: 11)

    def boom(prompt, max_length, seed):
        raise RuntimeError("interrupted")

    decoder.loop = boom
    with pytest.raises(RuntimeError, match="interrupted"):
        adapter.generate(input_ids=_ids(2), do_sample=False, max_new_tokens=5)
    assert "sample_token" not in decoder.__dict__ and clip.resets == 1


def test_sampling_passes_an_int_seed_and_explicit_filters():
    adapter, clip, decoder = _adapter(lambda step: EOS)
    adapter.generate(input_ids=_ids(2), do_sample=True, temperature=0.7,
                     max_new_tokens=3, top_p=0.95, top_k=64, min_p=0.05,
                     repetition_penalty=1.03)
    call = clip.calls[0]
    assert isinstance(decoder.seeds[0], int)
    assert (call["top_k"], call["top_p"], call["min_p"], call["repetition_penalty"]) == (
        64, 0.95, 0.05, 1.03)
    adapter.generate(input_ids=_ids(2), do_sample=True, temperature=0.7, max_new_tokens=3)
    call = clip.calls[1]
    assert (call["top_k"], call["top_p"], call["min_p"]) == (0, 1.0, 0.0), (
        "an unset filter is disabled deliberately, never ComfyUI's UI default")


@pytest.mark.parametrize("kwargs, error", [
    ({"input_ids": _ids(2), "max_new_tokens": 3, "logits_processor": []}, TypeError),
    ({"input_ids": torch.tensor([[2], [3]]), "max_new_tokens": 3}, ValueError),
    ({"input_ids": _ids(2, 3), "attention_mask": torch.tensor([[0, 1]]),
      "max_new_tokens": 3}, ValueError),
    ({"input_ids": _ids(2), "max_new_tokens": 3, "num_beams": 2}, ValueError),
])
def test_unsupported_generate_arguments_are_refused_not_ignored(kwargs, error):
    adapter, _, decoder = _adapter(lambda step: EOS)
    with pytest.raises(error):
        adapter.generate(**kwargs)
    assert "sample_token" not in decoder.__dict__


def test_the_streamer_sees_the_prompt_then_each_real_token_then_end():
    events = []

    class _Streamer:
        def put(self, value):
            events.append(("put", value.tolist()))

        def end(self):
            events.append(("end",))

    adapter, _, _ = _adapter(lambda step: [7, EOS][min(step, 1)])
    adapter.generate(input_ids=_ids(2, 5), do_sample=False, max_new_tokens=5,
                     streamer=_Streamer())
    assert events == [("put", [2, 5]), ("put", [7]), ("put", [EOS]), ("end",)]


def test_the_grammar_masks_a_forbidden_favourite_on_the_same_logits(tok):
    """The favourite token is illegal JSON at the start; the real
    lm-format-enforcer callback masks it and greedy picks a legal one."""
    from nodes import _otr_lmfe_compat
    _otr_lmfe_compat.ensure_lmfe_transformers_compat()
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
    data = build_token_enforcer_tokenizer_data(tok)
    allowed_fn = build_transformers_prefix_allowed_tokens_fn(
        data, JsonSchemaParser({"type": "object", "properties": {"a": {"type": "string"}},
                                "required": ["a"]}))
    x = tok.convert_tokens_to_ids("x")
    adapter, _, _ = _adapter(lambda step: x, vocab=len(tok))
    out = adapter.generate(input_ids=_ids(2), do_sample=False, max_new_tokens=1,
                           prefix_allowed_tokens_fn=allowed_fn)
    first = out[0, 1].item()
    assert first != x and tok.convert_ids_to_tokens(first) in ("{", " ", "\n")


def test_the_writers_real_factory_runs_unchanged_on_a_native_entry(tok, monkeypatch):
    """THE POINT OF THE ADAPTER: OTR_LedgerScriptWriter._build_truncating_
    generate_fn -- prompt preparation, budget fitting, the liveness guard, the
    substring stop, decoding -- runs against the native entry with no branch."""
    from nodes import OTR_LedgerScriptWriter as writer
    reply = [tok.convert_tokens_to_ids(c) for c in "hello end"] + [EOS]
    adapter, _, _ = _adapter(lambda step: reply[min(step, len(reply) - 1)],
                             vocab=len(tok))
    entry = {"provider": "comfy_native", "model": adapter, "tokenizer": tok,
             "model_id": native.MODEL_ID, "context_cap": 512,
             "native_context_capacity": 512,
             "context_capacity_source": "native working cap (test)"}
    generate_fn = writer._build_truncating_generate_fn(entry)
    text = generate_fn([{"role": "user", "content": "say hi"}], temperature=0.0,
                       max_new_tokens=40)
    assert text.strip() == "hello end"
    text = generate_fn([{"role": "user", "content": "say hi"}], temperature=0.0,
                       max_new_tokens=40, stop=[" end"])
    assert text.strip() == "hello"


# ---------------------------------------------------------------------------
# SLICE A3: the dropdown row, request_slot, load and release. Fakes stand in
# for ComfyUI's CLIP loader and model management; the OTR code under test is
# the real code.
# ---------------------------------------------------------------------------

import sys  # noqa: E402
import types  # noqa: E402

from nodes import _otr_model_catalog as catalog  # noqa: E402
from nodes import _otr_model_loader as loader  # noqa: E402


def _row():
    return catalog._by_repo_id()[native.MODEL_ID]


def test_the_catalog_row_names_this_backend_and_its_pin():
    row = _row()
    assert (row.provider, row.loader_backend) == (native.PROVIDER, native.LOADER_BACKEND)
    assert row.hf_repo_id == native.WEIGHT_REPO
    assert row.approx_safetensors_gb == round(native.WEIGHT_SIZE / 2**30, 2)
    assert row.context_window == native.WORKING_CONTEXT_CAP
    assert row.implied_quant_policy == "none" and row.requires_auth is False


def test_the_dropdown_offers_it_with_a_download_badge_and_no_fit_claim():
    """No machine class is claimed until an episode is proven on it."""
    label = catalog.dropdown_choices()
    mine = [choice for choice in label if choice.startswith(native.MODEL_ID)]
    assert mine == [native.MODEL_ID + " (4.8 GB download)"]
    assert catalog.validate_model_id(mine[0]) == native.MODEL_ID
    assert catalog.fit_tags_for(native.MODEL_ID) == ()


def test_on_disk_comes_from_comfys_folder_lookup(monkeypatch, tmp_path):
    weight = tmp_path / native.WEIGHT_TOKEN
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda category, token: (
        str(weight) if (category, token) == (native.WEIGHT_CATEGORY, native.WEIGHT_TOKEN)
        and weight.exists() else None)
    monkeypatch.setitem(sys.modules, "folder_paths", fake)

    def on_disk():
        entry, = [e for e in catalog.build_dropdown_choices() if e.repo_id == native.MODEL_ID]
        return entry.on_disk

    assert on_disk() is False
    weight.write_bytes(b"x")
    assert on_disk() is True


def test_sampling_is_the_e2b_baseline_not_a_cloud_none():
    assert catalog.sampling_baseline(native.MODEL_ID) == (1.0, 0.95, 64)
    assert catalog.sampling_baseline(native.MODEL_ID) == catalog.sampling_baseline(
        "google/gemma-4-E2B-it")


def test_the_context_cap_is_the_rows_working_cap_without_any_hf_scan(monkeypatch):
    def no_scan(*_a, **_k):
        raise AssertionError("a native id must not scan an HF config")

    monkeypatch.setattr(catalog, "_read_config_context", no_scan)
    verdict = catalog.resolve_context_cap(native.MODEL_ID, context_pin=None)
    assert (verdict.value, verdict.native_capacity, verdict.explicit_pin) == (
        native.WORKING_CONTEXT_CAP, None, None)
    tighter = catalog.resolve_context_cap(native.MODEL_ID, context_pin=4096)
    assert (tighter.value, tighter.explicit_pin) == (4096, 4096)
    looser = catalog.resolve_context_cap(native.MODEL_ID, context_pin=65536)
    assert looser.value == native.WORKING_CONTEXT_CAP, "a pin never widens the cap"


def test_the_lane_is_the_local_in_process_one_and_a_policy_can_refuse_it():
    from nodes._otr_shared import llm_policy
    assert llm_policy.lane_for_row(_row()) == llm_policy.LANE_TRANSFORMERS
    remote_only = llm_policy.BASELINE_POLICY.__class__(
        **{**llm_policy.BASELINE_POLICY.__dict__,
           "lane_allowlist": (llm_policy.LANE_OPENROUTER,)})
    with pytest.raises(loader.ModelLoaderError, match="NO FALLBACK"):
        loader.request_slot("creative", native.MODEL_ID, policy=remote_only)


def test_the_writer_binds_a_schema_on_a_native_slot():
    from nodes import OTR_LedgerScriptWriter as writer
    scheduler = writer._SlotScheduler.__new__(writer._SlotScheduler)
    scheduler.ids = {"creative": native.MODEL_ID}
    markers = scheduler._slot_transport_markers("creative")
    assert markers["_otr_local_schema_binding"] is True
    assert not any(markers[k] for k in ("_otr_openrouter", "_otr_comfy_credits",
                                        "_otr_google_api", "_otr_supports_json_object"))


# -- request_slot -------------------------------------------------------------

@pytest.fixture
def slot_env(monkeypatch):
    """request_slot with the native load and the shared ensure faked, and every
    transformers-only step wired to fail if the native arm ever reaches it."""
    monkeypatch.delenv("OTR_HARD_VRAM_CONTEXT_LIMIT", raising=False)
    loader.LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})
    calls = {"ensure": [], "load": [], "release": []}

    def forbidden(name):
        def _raise(*_a, **_k):
            raise AssertionError("the native arm reached " + name)
        return _raise

    from nodes import _otr_hf_env
    monkeypatch.setattr(_otr_hf_env, "ensure_hf_home", forbidden("ensure_hf_home"))
    monkeypatch.setattr(catalog, "auto_download_if_missing", forbidden("auto_download"))
    monkeypatch.setattr(loader, "load_llm", forbidden("load_llm"))
    monkeypatch.setattr(loader, "_require_transformers_model_support",
                        forbidden("the transformers support gate"))
    monkeypatch.setattr(loader, "_assert_policy_admits_vram",
                        forbidden("the VRAM estimate"))
    monkeypatch.setattr(va, "ensure_writer_weights",
                        lambda mid: calls["ensure"].append(mid) or Path("w.safetensors"))

    def fake_load(model_id, *, policy=None, context_verdict=None):
        calls["load"].append((model_id, context_verdict.value))
        return {"provider": "comfy_native", "model": object(), "tokenizer": object(),
                "model_id": model_id, "device": "cpu",
                "context_cap": context_verdict.value}

    monkeypatch.setattr(native, "load_native_writer", fake_load)
    monkeypatch.setattr(native, "release_native_entry",
                        lambda entry: calls["release"].append(entry["model_id"]))
    try:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except AttributeError:
        pass
    yield calls
    loader.LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})


def test_request_slot_loads_the_native_writer_without_any_hf_work(slot_env):
    entry = loader.request_slot("creative", native.MODEL_ID + " (4.8 GB download)")
    assert slot_env["ensure"] == [native.MODEL_ID]
    assert slot_env["load"] == [(native.MODEL_ID, native.WORKING_CONTEXT_CAP)]
    assert loader.LLM_CACHE["cache_entry"] is entry
    assert loader.has_local_resident_llm() is True, "native is local, not remote"


def test_both_slots_reuse_one_native_entry(slot_env):
    first = loader.request_slot("creative", native.MODEL_ID)
    second = loader.request_slot("technical", native.MODEL_ID)
    assert first is second and len(slot_env["load"]) == 1


def test_switching_to_a_transformers_model_releases_the_native_one_first(
        slot_env, monkeypatch, tmp_path):
    loader.request_slot("creative", native.MODEL_ID)
    order = []
    monkeypatch.setattr(native, "release_native_entry",
                        lambda entry: order.append(("release", entry["model_id"])))
    from nodes import _otr_hf_env
    monkeypatch.setattr(_otr_hf_env, "ensure_hf_home", lambda: str(tmp_path))
    monkeypatch.setattr(catalog, "auto_download_if_missing", lambda *a, **k: None)
    monkeypatch.setattr(loader, "_require_transformers_model_support", lambda *_a: None)
    monkeypatch.setattr(loader, "_assert_policy_admits_vram", lambda *_a, **_k: None)

    def hf_load(model_id, **_kw):
        order.append(("load", model_id))
        return {"model": None, "tokenizer": None, "model_id": model_id,
                "device": "cpu", "context_cap": 8192}

    monkeypatch.setattr(loader, "load_llm", hf_load)
    loader.request_slot("technical", "google/gemma-4-E2B-it")
    assert order == [("release", native.MODEL_ID), ("load", "google/gemma-4-E2B-it")]


def test_load_llm_refuses_a_native_id_before_any_hf_work(monkeypatch):
    from nodes import _otr_hf_env
    monkeypatch.setattr(_otr_hf_env, "ensure_hf_home",
                        lambda: (_ for _ in ()).throw(AssertionError("HF work")))
    with pytest.raises(loader.ModelLoaderError, match="request_slot"):
        loader.load_llm(native.MODEL_ID)


# -- load and release -----------------------------------------------------------

class _LoadedClip:
    """What comfy.sd.load_clip hands back for the Gemma 4 E2B file: a CLIP with
    a patcher, the Gemma tokenizer chain and the decoder."""

    def __init__(self, load_device, native_window=131072):
        decoder = _Decoder(lambda step: EOS, 64)
        decoder.model.config.max_position_embeddings = native_window
        self.cond_stage_model = SimpleNamespace(gemma4=SimpleNamespace(transformer=decoder))
        self.tokenizer = SimpleNamespace(gemma4=SimpleNamespace(
            tokenizer=SimpleNamespace(tokenizer=_raw_tokenizer())))
        self.patcher = SimpleNamespace(load_device=load_device)


@pytest.fixture
def comfy_env(monkeypatch, tmp_path):
    """Fake ``folder_paths``, ``comfy.sd`` and ``comfy.model_management``."""
    weight = tmp_path / native.WEIGHT_TOKEN
    weight.write_bytes(b"x")
    state = {"load_clip": [], "unloaded": [], "clip_factory": None}
    folder = types.ModuleType("folder_paths")
    folder.get_full_path = lambda c, t: str(weight) if t == native.WEIGHT_TOKEN else None
    folder.get_full_path_or_raise = lambda c, t: folder.get_full_path(c, t)
    folder.get_folder_paths = lambda c: [str(tmp_path)]
    sd = types.ModuleType("comfy.sd")
    sd.CLIPType = SimpleNamespace(STABLE_DIFFUSION="stable_diffusion")

    def load_clip(*, ckpt_paths, embedding_directory, clip_type, model_options):
        state["load_clip"].append(dict(ckpt_paths=ckpt_paths, clip_type=clip_type,
                                       model_options=dict(model_options)))
        device = model_options.get("load_device", "cuda:0")
        return (state["clip_factory"] or _LoadedClip)(str(device))

    sd.load_clip = load_clip
    mm = types.ModuleType("comfy.model_management")
    mm.unload_model_and_clones = lambda patcher, all_devices=False: state["unloaded"].append(
        (patcher, all_devices))
    mm.throw_exception_if_processing_interrupted = lambda: None
    pkg = types.ModuleType("comfy")
    pkg.sd, pkg.model_management = sd, mm
    for name, module in (("folder_paths", folder), ("comfy", pkg), ("comfy.sd", sd),
                         ("comfy.model_management", mm)):
        monkeypatch.setitem(sys.modules, name, module)
    state["weight"] = weight
    return state


def _verdict(value=8192):
    return catalog.ContextCapVerdict("PASS", value, "comfy-native working cap %d" % value,
                                     None, None)


@pytest.mark.parametrize("device, options", [
    ("cuda", {}),
    ("cuda:1", {"load_device": torch.device("cuda:1")}),
    ("cpu", {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}),
])
def test_the_load_is_the_stock_clip_loader_with_the_policys_device(comfy_env, device,
                                                                   options):
    entry = native.load_native_writer(native.MODEL_ID, policy=SimpleNamespace(device=device),
                                      context_verdict=_verdict())
    call, = comfy_env["load_clip"]
    assert call["ckpt_paths"] == [str(comfy_env["weight"])]
    assert call["clip_type"] == "stable_diffusion"
    assert call["model_options"] == options
    assert isinstance(entry["model"], native.ComfyGemmaGenerateAdapter)
    assert entry["tokenizer"].chat_template == native.GEMMA4_CHAT_TEMPLATE
    assert (entry["provider"], entry["model_id"], entry["context_cap"],
            entry["native_context_capacity"]) == (
        "comfy_native", native.MODEL_ID, 8192, 131072)


def test_the_cap_never_exceeds_the_decoders_native_window(comfy_env):
    comfy_env["clip_factory"] = lambda device: _LoadedClip(device, native_window=4096)
    entry = native.load_native_writer(native.MODEL_ID, context_verdict=_verdict(8192))
    assert entry["context_cap"] == 4096
    assert "native window 4096" in entry["context_capacity_source"]


def test_a_failure_after_the_clip_exists_hands_it_back(comfy_env):
    class _NotGemma(_LoadedClip):
        def __init__(self, device):
            super().__init__(device)
            self.cond_stage_model = SimpleNamespace()

    comfy_env["clip_factory"] = _NotGemma
    with pytest.raises(RuntimeError, match="not a ComfyUI Gemma 4 decoder"):
        native.load_native_writer(native.MODEL_ID, context_verdict=_verdict())
    (patcher, all_devices), = comfy_env["unloaded"]
    assert all_devices is True


def test_teardown_unloads_through_comfy_and_retires_the_adapter(comfy_env, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    entry = native.load_native_writer(native.MODEL_ID, context_verdict=_verdict())
    patcher, adapter = entry["clip"].patcher, entry["model"]
    loader._teardown_gpu_for_entry(entry)
    assert comfy_env["unloaded"] == [(patcher, True)]
    assert entry["clip"] is None and adapter.clip is None
    with pytest.raises(RuntimeError, match="unloaded"):
        adapter.generate(input_ids=_ids(2), max_new_tokens=4)


def test_the_shared_ensure_finds_a_present_weight_without_a_download(comfy_env, monkeypatch):
    def no_fetch(*_a, **_k):
        raise AssertionError("a present weight must not be fetched")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", no_fetch)
    assert va.ensure_writer_weights(native.MODEL_ID) == comfy_env["weight"]
