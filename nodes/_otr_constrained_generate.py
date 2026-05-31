"""Sprint 10A step 3-B -- Grammar-constrained generate function.

Wraps a model + tokenizer cache_entry into a closure that runs
transformers.model.generate() with an lm-format-enforcer
prefix_allowed_tokens_fn derived from a pydantic schema. The LLM is
constrained at the token-sampling layer: it CANNOT emit a token
sequence that would produce invalid JSON or violate the schema's
field-level constraints (regex, Literal, length bounds, etc.).

Sibling of `_otr_model_loader.make_generate_fn` / make_polish_generate_fn
-- same cache_entry shape, same chat-template normalization, same
prompt-prefix-strip post-process. The only delta: a
prefix_allowed_tokens_fn keyed off the pydantic schema is passed into
generate().

This module is dormant until 3-C wires Stage 1 into the writer.

Slot tag: technical -- structured JSON pass.

PD3 (workflow JSON): N/A; this module adds no node surface.
PD6 (LLM-slot tagging): the consumer (3-C wiring) will tag the call
site as 'technical' since structured-JSON passes belong to the
technical slot per the project rule.
"""
# LLM slot: technical
# Reason: this is a structured JSON pass; per project rule 6, all
# JSON-schema-constrained calls route to the technical model slot.

# COMPAT SHIM MUST RUN FIRST -- lm-format-enforcer 0.11.3 imports
# transformers.tokenization_utils.PreTrainedTokenizerBase (v4 path)
# which transformers v5 moved. See _otr_lmfe_compat docstring.
from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel

from . import _otr_lmfe_compat  # compat shim; ensure_lmfe_transformers_compat() called inside factory below
from ._otr_model_loader import (
    ModelLoaderError,
    _normalize_messages_for_cache_entry,
)


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Heartbeat streamer (opt-in live visibility)
# ---------------------------------------------------------------------------
#
# A constrained-decode pass with a large max_new_tokens budget (e.g. the
# Editor pass at 4096 tokens) runs as a single blocking model.generate()
# call: the console prints the pass header, then goes silent for the whole
# decode (100-200s at NF4 decode speed), then prints the verdict. That
# silence is indistinguishable from a hang.
#
# _HeartbeatStreamer is a read-only transformers BaseStreamer. generate()
# hands it each newly-sampled token id; the streamer NEVER feeds anything
# back, so attaching it does not change the sampled tokens -- output stays
# identical with or without it. Every `every` tokens it logs token count,
# tok/s, elapsed, and a short decoded tail so the operator can watch the
# JSON forming live instead of staring at a frozen console.
try:
    from transformers.generation.streamers import BaseStreamer as _BaseStreamer
except Exception:  # pragma: no cover - transformers missing in some test envs
    class _BaseStreamer:  # type: ignore[no-redef]
        def put(self, value: Any) -> None: ...
        def end(self) -> None: ...


class _HeartbeatStreamer(_BaseStreamer):
    """Read-only generate() observer that logs a live tok/s heartbeat.

    Attaching this to model.generate(streamer=...) cannot alter the
    generated tokens; it only reads them. Safe on any pass.
    """

    def __init__(self, tokenizer: Any, label: str, every: int = 32) -> None:
        self._tokenizer = tokenizer
        self._label = label
        self._every = max(1, int(every))
        self._ids: List[int] = []
        self._count = 0
        self._last_emit = 0
        self._t0: Optional[float] = None
        self._prompt_seen = False

    def put(self, value: Any) -> None:
        # The first put() carries the prompt input_ids -- start the clock
        # there and skip it for the new-token count.
        if not self._prompt_seen:
            self._prompt_seen = True
            self._t0 = time.monotonic()
            return
        try:
            raw = value.tolist() if hasattr(value, "tolist") else list(value)
        except Exception:
            return
        flat: List[int] = []
        for item in raw:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        if not flat:
            return
        self._ids.extend(flat)
        self._count += len(flat)
        if self._count - self._last_emit >= self._every:
            self._last_emit = self._count
            self._emit()

    def _emit(self) -> None:
        elapsed = (time.monotonic() - self._t0) if self._t0 else 0.0
        tps = (self._count / elapsed) if elapsed > 0 else 0.0
        tail = ""
        try:
            text = self._tokenizer.decode(self._ids, skip_special_tokens=True)
            tail = text[-80:].replace("\n", " ").replace("\r", " ")
        except Exception:
            tail = ""
        log.info(
            "[%s] heartbeat: %d tok | %.1f tok/s | %.1fs | ...%s",
            self._label, self._count, tps, elapsed, tail,
        )

    def end(self) -> None:
        # Final pulse so the operator sees the closing token count even if
        # the last chunk was shorter than the heartbeat interval.
        if self._count > self._last_emit:
            self._emit()


# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------


ConstrainedGenerateFn = Callable[..., str]
"""Closure signature returned by make_constrained_generate_fn:
    (messages, *, temperature, max_new_tokens) -> str

The returned string is the model's raw output AFTER the prompt prefix
has been stripped. Under constrained decoding it is guaranteed to be
a token sequence the lm-format-enforcer parser accepted -- i.e. a
well-formed JSON object conforming to the bound schema. Callers
should still json.loads() + pydantic-validate it (belt-and-braces)
since the parser may stop early on max_new_tokens before the JSON
closes.
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_constrained_generate_fn(
    cache_entry: dict[str, Any],
    schema_model: Type[BaseModel],
    heartbeat_label: Optional[str] = None,
) -> ConstrainedGenerateFn:
    """Wrap a cache_entry into a grammar-constrained generate closure.

    Args:
        cache_entry: dict produced by _otr_model_loader.load_llm. Must
            contain `model` and `tokenizer` keys.
        schema_model: a pydantic BaseModel subclass. The lm-format-
            enforcer JsonSchemaParser binds to its
            model_json_schema(); generate() will only emit token
            sequences that the parser accepts.
        heartbeat_label: opt-in live-visibility label. When set (e.g.
            "EditorPass"), the closure attaches a read-only
            _HeartbeatStreamer to model.generate() that logs token
            count, tok/s, elapsed, and a decoded tail every ~32 tokens
            so long blocking passes are visible in real time. The
            streamer only observes tokens -- it never feeds any back,
            so generated output is identical with or without it.
            Default None -> no streamer -> byte-identical to the
            prior behaviour for every existing caller.

    Returns:
        A callable (messages, *, temperature, max_new_tokens) -> str.

    Raises:
        ModelLoaderError on cache_entry missing required keys.

    The closure is independent of any specific call site -- it just
    knows the schema. Stage 1, Stage 3 LLM validators (step 5), and
    the whole-episode critic (step 7) each bind their own schema and
    get their own closure.
    """
    # [OpenRouter S4] Remote branch: a provider-tagged remote entry has
    # no tokenizer to bind a grammar parser to. Map the call's Pydantic
    # schema -> OpenRouter response_format (json_schema, strict) and
    # return the remote generate_fn. Integrity is enforced FAIL-CLOSED
    # (C4) by the SAME downstream the local path uses -- either the
    # structured_call validate + bounded-repair ladder (raises
    # StructuredCallFailedError on exhaustion) or the call site's direct
    # _parse_and_validate -- so malformed remote output can never reach
    # the ledger. A model that lacks json_schema support returns a 4xx,
    # which the backend surfaces as OpenRouterCallFailedError (also
    # fail-closed). Zero NEW validation logic; reuses the existing path.
    if cache_entry.get("provider") == "openrouter":
        from . import _otr_openrouter_backend as _orb
        response_format = _orb.schema_to_response_format(
            schema_model, name=getattr(schema_model, "__name__", "otr_schema"),
        )
        return _orb.make_openrouter_generate_fn(
            cache_entry, response_format=response_format,
        )
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )

    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]

    # Build the JsonSchemaParser + prefix_allowed_tokens_fn ONCE per
    # closure. The parser is stateless across calls; the prefix-fn
    # closes over the tokenizer + parser and is reusable across many
    # generate() invocations with different messages but the same
    # schema.
    #
    # Re-apply the lm-format-enforcer / transformers v5 compat shim
    # at factory time. Some test fixtures elsewhere in the suite
    # reload transformers.tokenization_utils and strip the v4 alias
    # we wired at module-import time; this call re-establishes it
    # before the lmformatenforcer import below. Cheap (single
    # hasattr check) and idempotent.
    _otr_lmfe_compat.ensure_lmfe_transformers_compat()
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    schema_dict = schema_model.model_json_schema()
    parser = JsonSchemaParser(schema_dict)
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    def constrained_generate_fn(
        messages: List[dict],
        *,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        # Lazy torch import keeps the module importable in test
        # environments where torch may be slow / partial.
        try:
            import torch
        except ImportError as exc:
            raise ModelLoaderError("torch not available") from exc

        messages = _normalize_messages_for_cache_entry(cache_entry, messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Opt-in live heartbeat (read-only; does not alter sampled tokens).
        streamer = (
            _HeartbeatStreamer(tokenizer, heartbeat_label)
            if heartbeat_label
            else None
        )

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                # The schema-binding argument. transformers passes
                # this hook into the logits-processing path; lm-
                # format-enforcer reuses it to keep the sampler in
                # the JSON-schema-valid subset.
                prefix_allowed_tokens_fn=prefix_fn,
                # num_beams=1 keeps memory + latency manageable.
                # Constrained sampling does not need beams to land
                # a valid object; beams would multiply the parser
                # state cost without quality gain on structured
                # output.
                num_beams=1,
                # Read-only observer for live tok/s visibility on long
                # passes; None when heartbeat_label is unset.
                streamer=streamer,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            out[0][prompt_len:],
            skip_special_tokens=True,
        )

    # Expose the parser + prefix-fn on the closure for tests that want
    # to assert the binding without invoking generate().
    constrained_generate_fn.schema_model = schema_model       # type: ignore[attr-defined]
    constrained_generate_fn.json_schema_parser = parser       # type: ignore[attr-defined]
    constrained_generate_fn.prefix_allowed_tokens_fn = prefix_fn  # type: ignore[attr-defined]

    return constrained_generate_fn
