"""A Comfy-native Gemma 4 writer: ComfyUI's own text model writing the script.

WHAT IT IS (plan row 0n; design campaign kibitz-runs/2026-09-26-comfy-gemma-writer).
ComfyUI ships an official ``llm_gemma4_text_gen`` template -- the stock CLIPLoader
loading Comfy-Org's ``gemma4_e2b_it_int8_convrot`` into the core TextGenerate node.
The model then lives inside ComfyUI's own model management, which loads and unloads
it like any other model (operator, 2026-09-26: "unload models after they are used").
Measured the same day: 82-125 tok/s on the RTX 5080 and a 5.1 GB peak on the 8 GB
RTX 4060, against 16-18 tok/s for today's 12B NF4 transformers writer.

BUILT IN SLICES, and this file grows with them. Slice A2 lands the identity and the
one weight the writer needs, so the queue-time preflight can fetch that weight before
the writer runs. Generation (A1) and the ``request_slot`` wiring (A3) follow. Until A3
lands, no dropdown offers :data:`MODEL_ID`, so nothing reaches this module at runtime
except the preflight's lookup, which answers "no weights" for every other id.

IMPORT-LIGHT ON PURPOSE: stdlib only at module scope. The preflight and the catalog
ask it questions while building a queue plan; neither may pay for torch or ComfyUI.
"""
from __future__ import annotations

#: The dropdown id. Virtual (not a Hugging Face repo id): the weight comes from
#: Comfy-Org as one ComfyUI text-encoder file, not a transformers snapshot, and the
#: HF ``google/gemma-4-E2B-it`` row stays a separate, unchanged choice.
MODEL_ID = "comfy_native:gemma4-e2b-it-int8-convrot"

#: The ComfyUI model category the stock CLIPLoader reads, and the file in it.
WEIGHT_CATEGORY = "text_encoders"
WEIGHT_TOKEN = "gemma4_e2b_it_int8_convrot.safetensors"

#: The exact upstream file, read from the Hub API on 2026-09-26 without a token
#: (the repo is ungated). ``_otr_visual_assets._PINNED_SOURCES`` carries the same
#: row; tests/test_comfy_textgen_backend.py holds the two in agreement.
WEIGHT_REPO = "Comfy-Org/gemma-4"
WEIGHT_FILENAME = "text_encoders/gemma4_e2b_it_int8_convrot.safetensors"
WEIGHT_REVISION = "63d0f7c476756b88910170c1df75e2384ea1af31"
WEIGHT_SIZE = 5_199_997_904
WEIGHT_SHA256 = "efeca0fcad2f863e5ed0a75e3af952b72bc963604c1dda6d20aee87a32b17566"

#: The working context this writer is qualified at. The decoder advertises a much
#: larger native window (below); the working cap is what an 8 GB card is asked to
#: hold, and a later probe decides whether it can rise. Kept separate on purpose so
#: neither number is ever read as the other.
WORKING_CONTEXT_CAP = 8192
NATIVE_CONTEXT_CAPACITY = 131072

_WEIGHTS_BY_MODEL = {MODEL_ID: ((WEIGHT_CATEGORY, WEIGHT_TOKEN),)}


def native_writer_weights(model_id) -> tuple:
    """``((category, token), ...)`` a Comfy-native writer id needs, else ``()``.

    Exact match on the normalized id only: a label, a badge or an HF repo id that
    merely resembles this one needs nothing from here."""
    return _WEIGHTS_BY_MODEL.get(str(model_id or "").strip(), ())


def is_native_writer(model_id) -> bool:
    return bool(native_writer_weights(model_id))


# ---------------------------------------------------------------------------
# SLICE A1: generation. An explicit adapter that speaks the transformers
# `generate()` subset OTR's four generation factories pass, on top of ComfyUI's
# own CLIP.generate. Design r3 (kibitz-runs/2026-09-26-comfy-gemma-writer/r3):
# ONE copy of the writer's halt classification, budget fitting and stop
# trimming -- the factories run unchanged against a real tokenizer and this
# adapter, instead of a parallel native path that would drift.
# ---------------------------------------------------------------------------

#: Gemma 4's stop ids as ComfyUI declares them (comfy/text_encoders/gemma4.py,
#: ``stop_tokens = [1, 50, 106]``): <eos>, and the two turn/channel closers.
GEMMA4_STOP_TOKEN_IDS = (1, 106, 50)

_BOS, _EOS, _PAD = "<bos>", "<eos>", "<pad>"

#: The text-only subset of Google's official Gemma 4 chat template (the E2B
#: snapshot's ``chat_template.jinja``): a leading system/developer turn, then
#: user/model turns, content trimmed, model turns passed through the same
#: ``strip_thinking`` the official template uses, thinking off, no tools. Anything
#: else (list content, a tool turn) is refused rather than rendered wrong.
#: tests/test_comfy_textgen_backend.py compares it to the official template.
GEMMA4_CHAT_TEMPLATE = (
    "{%- macro strip_thinking(text) -%}"
    "{%- set ns = namespace(result='') -%}"
    "{%- for part in text.split('<channel|>') -%}"
    "{%- if '<|channel>' in part -%}"
    "{%- set ns.result = ns.result + part.split('<|channel>')[0] -%}"
    "{%- else -%}{%- set ns.result = ns.result + part -%}{%- endif -%}"
    "{%- endfor -%}"
    "{{- ns.result | trim -}}"
    "{%- endmacro -%}"
    "{{- bos_token -}}"
    "{%- set loop_messages = messages -%}"
    "{%- if messages and messages[0]['role'] in ['system', 'developer'] -%}"
    "{%- if messages[0]['content'] is not string -%}"
    "{{- raise_exception('the native Gemma writer takes text-only messages') -}}"
    "{%- endif -%}"
    "{{- '<|turn>system\\n' + (messages[0]['content'] | trim) + '<turn|>\\n' -}}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
    "{%- if message['role'] not in ['user', 'assistant'] or "
    "message['content'] is not string -%}"
    "{{- raise_exception('the native Gemma writer takes text-only user and "
    "assistant turns after an optional system turn') -}}"
    "{%- endif -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else 'user' -%}"
    "{{- '<|turn>' + role + '\\n' -}}"
    "{%- if role == 'model' -%}{{- strip_thinking(message['content']) -}}"
    "{%- else -%}{{- message['content'] | trim -}}{%- endif -%}"
    "{{- '<turn|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|turn>model\\n' -}}{%- endif -%}"
)


def build_native_tokenizer(raw_tokenizer):
    """A transformers fast tokenizer over the ``tokenizers.Tokenizer`` ComfyUI
    embeds in the weight file -- the writer's real tokenizer, not a stand-in.

    Gate F4 (2026-09-26): identical ids to the HF ``google/gemma-4-E2B-it``
    tokenizer, and identical lm-format-enforcer allowed sets at every prefix
    ONCE the model's special tokens are marked special. Without that step the
    grammar let ``<|turn>`` and 19 other control tokens into JSON strings."""
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
                                        bos_token=_BOS, eos_token=_EOS,
                                        pad_token=_PAD)
    specials = sorted(
        {added.content for added in raw_tokenizer.get_added_tokens_decoder().values()
         if added.special and added.content not in (_BOS, _EOS, _PAD)})
    if specials:
        tokenizer.add_special_tokens({"additional_special_tokens": specials})
    tokenizer.chat_template = GEMMA4_CHAT_TEMPLATE
    return tokenizer


def native_seed(prompt_ids, raw_env=None) -> int:
    """The int ComfyUI's sampler needs, never None and never a constant.

    ``OTR_WRITER_SEED`` set: the prompt-keyed seed the transformers writer uses
    (``_otr_sampling_seed``), so a seeded A/B compares like with like. Unset or
    malformed: a fresh random seed per call, so production stays unseeded --
    ComfyUI's ``manual_seed(None)`` would fail, and a constant would make every
    episode's sampling identical."""
    import random

    try:
        from ._otr_sampling_seed import WRITER_SEED_ENV, parse_writer_seed, prompt_keyed_seed
    except ImportError:  # pragma: no cover -- flat import
        from _otr_sampling_seed import (  # type: ignore
            WRITER_SEED_ENV, parse_writer_seed, prompt_keyed_seed)
    if raw_env is None:
        try:
            from ._otr_shared import env as otr_env
        except ImportError:  # pragma: no cover -- flat import
            from _otr_shared import env as otr_env  # type: ignore
        raw_env = otr_env.get(WRITER_SEED_ENV, "")
    base = parse_writer_seed(raw_env)
    if base is not None:
        return prompt_keyed_seed(base, [list(prompt_ids)])
    return random.SystemRandom().randrange(1, 0x7FFF_FFFF)


def sampling_owner(clip):
    """The object whose ``sample_token`` the per-token hook wraps:
    ``clip.cond_stage_model.gemma4.transformer`` (the Gemma 4 decoder, which
    inherits ``sample_token`` from ComfyUI's BaseGenerate)."""
    inner = getattr(getattr(clip, "cond_stage_model", None), "gemma4", None)
    owner = getattr(inner, "transformer", None)
    if owner is None or not callable(getattr(owner, "sample_token", None)):
        raise RuntimeError("the loaded text model is not a ComfyUI Gemma 4 decoder "
                           "(no cond_stage_model.gemma4.transformer.sample_token)")
    return owner


def _truthy(verdict) -> bool:
    """A stopping criterion's answer: a bool, or a bool tensor per batch row."""
    try:
        return bool(verdict.any())
    except AttributeError:
        return bool(verdict)


class _Run:
    """One generate() call's state: the real sampled ids, the halt latch, the
    grammar mask buffers and the criteria. Never outlives its call."""

    def __init__(self, prompt, max_new_tokens, stop_ids, allowed_fn, criteria,
                 streamer):
        import torch

        self.prompt_len = len(prompt)
        self.buf = torch.empty(self.prompt_len + max_new_tokens + 1, dtype=torch.long)
        self.buf[: self.prompt_len] = torch.as_tensor(prompt, dtype=torch.long)
        self.n = self.prompt_len
        self.ids: list = []
        self.stop_ids = frozenset(int(t) for t in stop_ids)
        self.allowed_fn = allowed_fn
        self.criteria = list(criteria or ())
        self.streamer = streamer
        self.halted = False
        self.halted_by_criteria = False
        self.error: "BaseException | None" = None
        self.stop_tensor = None
        self.cpu_mask = None
        self.dev_mask = None

    def _prepare(self, logits):
        """First call only -- ComfyUI's step 0, before its allocation recording
        begins -- so decode steps never allocate a new device tensor here."""
        import torch

        stop_id = min(self.stop_ids)
        self.stop_tensor = torch.full((logits.shape[0], 1), stop_id,
                                      dtype=torch.long, device=logits.device)
        if self.allowed_fn is not None:
            vocab = int(logits.shape[-1])
            self.cpu_mask = torch.ones(vocab, dtype=torch.bool)
            self.dev_mask = torch.ones(vocab, dtype=torch.bool, device=logits.device)

    def _mask(self, logits):
        import torch

        allowed = self.allowed_fn(0, self.buf[: self.n])
        vocab = self.cpu_mask.shape[0]
        keep = [int(t) for t in allowed if 0 <= int(t) < vocab]
        if not keep:
            raise RuntimeError("the grammar allows no token at this position")
        self.cpu_mask.fill_(True)
        self.cpu_mask[torch.as_tensor(keep, dtype=torch.long)] = False
        self.dev_mask.copy_(self.cpu_mask)
        logits.masked_fill_(self.dev_mask.unsqueeze(0), torch.finfo(logits.dtype).min)

    def _record(self, token_id):
        import torch

        self.buf[self.n] = token_id
        self.n += 1
        self.ids.append(token_id)
        if self.streamer is not None:
            self.streamer.put(torch.tensor([token_id], dtype=torch.long))

    def hook(self, original):
        run = self

        def sample_token(logits, temperature, top_k, top_p, min_p,
                         repetition_penalty, token_history, generator,
                         do_sample=True, presence_penalty=0.0, penalty_mask=None):
            if run.stop_tensor is None:
                run._prepare(logits)
            if run.halted:
                return run.stop_tensor
            try:
                if run.allowed_fn is not None:
                    run._mask(logits)
                token = original(logits, temperature, top_k, top_p, min_p,
                                 repetition_penalty, token_history, generator,
                                 do_sample=do_sample, presence_penalty=presence_penalty,
                                 penalty_mask=penalty_mask)
                token_id = int(token.reshape(-1)[0])
                run._record(token_id)
                if token_id in run.stop_ids:
                    run.halted = True
                    return run.stop_tensor
                view = run.buf[: run.n].unsqueeze(0)
                for criterion in run.criteria:
                    if _truthy(criterion(view, logits)):
                        run.halted = True
                        run.halted_by_criteria = True
                        return run.stop_tensor
                return token
            except Exception as exc:  # noqa: BLE001 -- latched, re-raised after
                # Raising here would skip ComfyUI's malloc_graph_end and progress
                # update (llama.py after sample_token). Latch it, hand the loop a
                # stop token so it closes its own recording, and re-raise the
                # ORIGINAL exception once the hook is restored.
                run.error = exc
                run.halted = True
                return run.stop_tensor

        return sample_token


def _abandon_allocation_recording():
    """After a genuine exception out of ComfyUI's loop, close any allocation
    recording it left open. The prompt executor does the same in its own
    finally; doing it here too keeps a caller that catches and continues safe."""
    try:
        import comfy.memory_management as _mm
        import comfy.model_prefetch as _mp
    except Exception:  # noqa: BLE001 -- outside ComfyUI: nothing was recorded
        return
    if getattr(_mm, "aimdo_enabled", False):
        try:
            _mp.cleanup_prefetch_queues()
        except Exception:  # noqa: BLE001 -- best effort; the original error wins
            pass


class ComfyGemmaGenerateAdapter:
    """The transformers ``generate()`` subset OTR's factories pass, run by
    ComfyUI's own Gemma 4 decoder.

    Accepted: input_ids, attention_mask, do_sample, temperature, max_new_tokens,
    top_p, top_k, min_p, repetition_penalty, pad_token_id, eos_token_id,
    num_beams (1 only), stopping_criteria, streamer, prefix_allowed_tokens_fn.
    Anything else is REFUSED with a TypeError naming it -- accepting and
    ignoring an argument would advertise behaviour this adapter does not have.
    Returns ``[[*prompt_ids, *generated_ids]]``, the shape every factory slices
    with ``out[0][prompt_len:]``. A real EOS is included; a latched stop (a
    stopping criterion, a callback error) is not, so the writer's own halt
    classification reads the truth."""

    def __init__(self, clip, *, stop_token_ids=GEMMA4_STOP_TOKEN_IDS):
        from types import SimpleNamespace

        self.clip = clip
        self.stop_token_ids = tuple(int(t) for t in stop_token_ids)
        self.generation_config = SimpleNamespace(eos_token_id=list(self.stop_token_ids))
        self.config = SimpleNamespace(eos_token_id=list(self.stop_token_ids),
                                      text_config=None)

    @property
    def device(self):
        return getattr(getattr(self.clip, "patcher", None), "load_device", "cpu")

    def generate(self, input_ids=None, attention_mask=None, *, do_sample=True,
                 temperature=1.0, max_new_tokens=None, top_p=None, top_k=None,
                 min_p=None, repetition_penalty=None, pad_token_id=None,
                 eos_token_id=None, num_beams=1, stopping_criteria=None,
                 streamer=None, prefix_allowed_tokens_fn=None, **unsupported):
        import torch

        if unsupported:
            raise TypeError("ComfyGemmaGenerateAdapter.generate() does not support "
                            "%s" % ", ".join(sorted(unsupported)))
        if input_ids is None:
            raise TypeError("input_ids is required")
        ids = torch.as_tensor(input_ids).detach().to("cpu")
        if ids.ndim != 2 or int(ids.shape[0]) != 1:
            raise ValueError("the native Gemma writer takes exactly one prompt "
                             "(input_ids shape [1, n]); got %s" % (tuple(ids.shape),))
        if attention_mask is not None and not bool(
                torch.as_tensor(attention_mask).detach().to("cpu").bool().all()):
            raise ValueError("a padded attention_mask is not supported; the native "
                             "writer takes one unpadded prompt")
        if int(num_beams or 1) != 1:
            raise ValueError("beam search is not supported (num_beams=%r)" % num_beams)
        if max_new_tokens is None or int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be a positive int")
        budget = int(max_new_tokens)
        prompt = [int(t) for t in ids[0].tolist()]
        extra_stops = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else (
            () if eos_token_id is None else (eos_token_id,))
        run = _Run(prompt, budget, set(self.stop_token_ids) | {int(t) for t in extra_stops},
                   prefix_allowed_tokens_fn, stopping_criteria, streamer)
        sampling = bool(do_sample) and float(temperature or 0.0) > 0.0
        owner = sampling_owner(self.clip)
        if "sample_token" in owner.__dict__:
            raise RuntimeError("this Gemma decoder is already generating (its "
                               "sample_token is wrapped); calls must not overlap")
        if streamer is not None:
            streamer.put(torch.as_tensor(prompt, dtype=torch.long))
        owner.sample_token = run.hook(owner.sample_token)
        completed = False
        try:
            self.clip.generate(
                {"gemma4": [[(token, 1.0) for token in prompt]]},
                do_sample=sampling, max_length=budget,
                temperature=float(temperature) if sampling else 1.0,
                top_k=int(top_k) if top_k else 0,
                top_p=float(top_p) if top_p is not None else 1.0,
                min_p=float(min_p or 0.0),
                repetition_penalty=float(repetition_penalty or 1.0),
                presence_penalty=0.0,
                seed=native_seed(prompt) if sampling else 0,
                mtp=False)
            completed = True
        except BaseException:
            _abandon_allocation_recording()
            raise
        finally:
            owner.__dict__.pop("sample_token", None)
            reset = getattr(getattr(self.clip, "cond_stage_model", None),
                            "reset_clip_options", None)
            if callable(reset):
                reset()
            if completed and streamer is not None:
                streamer.end()
        if run.error is not None:
            raise run.error
        return torch.tensor([prompt + run.ids], dtype=torch.long)
