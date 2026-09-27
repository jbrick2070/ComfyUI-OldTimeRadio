"""The Comfy-native Gemma 4 writer (plan row 0n): identity, pinned weight, planning.

Slice A2 of kibitz-runs/2026-09-26-comfy-gemma-writer. CPU only, no network, no
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

    def generate(self, tokens, do_sample, max_length, temperature, top_k, top_p,
                 min_p, repetition_penalty, seed, presence_penalty, mtp):
        assert isinstance(seed, int), "Comfy's sampler needs an int seed"
        self.calls.append(dict(do_sample=do_sample, max_length=max_length,
                               top_k=top_k, top_p=top_p, min_p=min_p,
                               repetition_penalty=repetition_penalty, mtp=mtp))
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
    assert clip.calls[0]["max_length"] == 10 and clip.calls[0]["mtp"] is False


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
