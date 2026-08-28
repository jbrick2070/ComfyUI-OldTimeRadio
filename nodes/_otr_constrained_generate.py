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

THIS MODULE IS LIVE (banner corrected 2026-08-28 -- it claimed dormancy long after 3-C landed): OTR_LedgerScriptWriter constructs make_constrained_generate_fn for SlotJobFields, and 1,645 of 2,001 corpus ledgers carry the resulting meta['slot_drama_contracts'].

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
from . import _otr_writer_heartbeat as _OTRHB
from ._otr_generation_budget import GenerationDegeneracyError
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
# The implementation moved DOWN to the leaf module `_otr_writer_heartbeat` so
# the other two generate transports could reach it -- this module imports FROM
# `_otr_model_loader`, so a streamer living here was unreachable from there
# without a cycle, and those transports ran blind as a result. The name is kept
# as an alias because this module's own call site and its tests use it.
_HeartbeatStreamer = _OTRHB.WriterHeartbeatStreamer


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


def get_cached_transformers_schema_constraint(
    cache_entry: dict[str, Any],
    schema_model: Type[BaseModel],
) -> tuple[Any, Any]:
    """Return ``(JsonSchemaParser, prefix_fn)`` for one resident model.

    LMFE's expensive step is tokenizer-wide and schema-independent: it
    decodes/scans every token to build ``TokenEnforcerTokenizerData``. Gemma 4
    has a roughly 256K-token vocabulary, so rebuilding that data for every
    P0-P9 pass or retry is material. Cache it on the resident ``cache_entry``
    and cache the cheap schema-specific prefix functions beside it. Unloading
    the model drops the cache naturally with the tokenizer.
    """
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )
    tokenizer = cache_entry["tokenizer"]

    _otr_lmfe_compat.ensure_lmfe_transformers_compat()
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )

    cache = cache_entry.get("_otr_lmfe_constraint_cache")
    if not isinstance(cache, dict) or cache.get("tokenizer") is not tokenizer:
        cache = {
            "tokenizer": tokenizer,
            "tokenizer_data": build_token_enforcer_tokenizer_data(tokenizer),
            "by_schema": {},
        }
        cache_entry["_otr_lmfe_constraint_cache"] = cache

    by_schema = cache["by_schema"]
    cached = by_schema.get(schema_model)
    if cached is None:
        parser = JsonSchemaParser(schema_model.model_json_schema())
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(
            cache["tokenizer_data"], parser,
        )
        cached = (parser, prefix_fn)
        by_schema[schema_model] = cached
    return cached


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
    # BUG-LOCAL-299: Comfy Credits sibling. The Comfy lane is "OpenRouter over
    # Comfy's proxy", so the json_schema response_format is byte-identical --
    # reuse the OpenRouter schema mapper, then hand it to the Comfy generate_fn.
    # Same fail-closed downstream (structured_call validate + repair ladder).
    if cache_entry.get("provider") == "comfy_credits":
        from . import _otr_openrouter_backend as _orb
        from . import _otr_comfy_backend as _occ
        response_format = _orb.schema_to_response_format(
            schema_model, name=getattr(schema_model, "__name__", "otr_schema"),
        )
        return _occ.make_comfy_credits_generate_fn(
            cache_entry, response_format=response_format,
        )
    # Native GGUF lane. It accepts llama-cpp-python response_format, so map the
    # existing OpenRouter-style json_schema wrapper at the backend boundary.
    if cache_entry.get("provider") == "gguf_native":
        from . import _otr_openrouter_backend as _orb
        from . import _otr_gguf_backend as _gguf
        response_format = _orb.schema_to_response_format(
            schema_model, name=getattr(schema_model, "__name__", "otr_schema"),
        )
        return _gguf.make_gguf_generate_fn(
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

    # Build once per resident tokenizer + schema; see the cache helper above.
    parser, prefix_fn = get_cached_transformers_schema_constraint(
        cache_entry, schema_model,
    )

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

        # THE LIVENESS GUARD (2026-08-13). This route is LIVE -- the writer
        # builds it for the slot-drama contract and calls it once per voiced
        # beat -- and it was missed when the guard shipped, because the guard
        # was installed per-WRAPPER in OTR_LedgerScriptWriter rather than at
        # every local generate(). A six-agent audit of every `model.generate`
        # in nodes/ found this and `_otr_model_loader.make_generate_fn`
        # unprotected. A unit test can pass while a production route runs bare.
        #
        # Cost here is bounded today (192 output tokens, two attempts per slot),
        # so this is prophylaxis rather than an emergency -- but "bounded by
        # whatever the caller happened to pass" is not a liveness contract.
        from transformers import StoppingCriteriaList  # noqa: I001
        try:
            from ._otr_decode_guard import make_degeneracy_criterion
        except ImportError:  # pragma: no cover - flat/standalone import path
            from _otr_decode_guard import (  # type: ignore
                make_degeneracy_criterion,
            )
        # Tokenizer supplied: this route is ALWAYS schema-bound (it exists to
        # run lm-format-enforcer), so the open-string spiral signal applies and
        # a quote here is structure, never dialogue.
        _guard = make_degeneracy_criterion(
            inputs["input_ids"].shape[1], tokenizer=tokenizer,
        )

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([_guard]),
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
        decoded = tokenizer.decode(
            out[0][prompt_len:],
            skip_special_tokens=True,
        )
        if getattr(_guard, "hit", False):
            # A halted decode is NOT a short answer. Returning the truncated
            # text would hand the caller a fragment that parses as a real
            # reply -- the silent-truncation trap. Raise the same rerollable
            # phase the writer transport raises, so a caller that has a retry
            # path uses it and one that does not fails loudly instead of
            # quietly accepting half a JSON object.
            telemetry = _guard.telemetry()
            log.error(
                "[%s] DECODE HALTED (%s): repeated a %s-token run verbatim %s "
                "times. Rerollable. Telemetry: %s",
                heartbeat_label or "constrained-generate",
                _guard.reason, telemetry.get("cycle_tokens"),
                telemetry.get("required_repeats"), telemetry,
            )
            raise GenerationDegeneracyError(
                "constrained generation was halted by the liveness guard: the "
                "output repeated a run of tokens verbatim",
                halt_reason=_guard.reason,
                repetition=telemetry,
                raw_completion=decoded,
                prompt_tokens=prompt_len,
            )
        return decoded

    # Expose the parser + prefix-fn on the closure for tests that want
    # to assert the binding without invoking generate().
    constrained_generate_fn.schema_model = schema_model       # type: ignore[attr-defined]
    constrained_generate_fn.json_schema_parser = parser       # type: ignore[attr-defined]
    constrained_generate_fn.prefix_allowed_tokens_fn = prefix_fn  # type: ignore[attr-defined]

    return constrained_generate_fn
