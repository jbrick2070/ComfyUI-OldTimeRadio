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

from typing import Any, Callable, List, Type

from pydantic import BaseModel

from . import _otr_lmfe_compat  # compat shim; ensure_lmfe_transformers_compat() called inside factory below
from ._otr_model_loader import (
    ModelLoaderError,
    _normalize_messages_for_cache_entry,
)


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
) -> ConstrainedGenerateFn:
    """Wrap a cache_entry into a grammar-constrained generate closure.

    Args:
        cache_entry: dict produced by _otr_model_loader.load_llm. Must
            contain `model` and `tokenizer` keys.
        schema_model: a pydantic BaseModel subclass. The lm-format-
            enforcer JsonSchemaParser binds to its
            model_json_schema(); generate() will only emit token
            sequences that the parser accepts.

    Returns:
        A callable (messages, *, temperature, max_new_tokens) -> str.

    Raises:
        ModelLoaderError on cache_entry missing required keys.

    The closure is independent of any specific call site -- it just
    knows the schema. Stage 1, Stage 3 LLM validators (step 5), and
    the whole-episode critic (step 7) each bind their own schema and
    get their own closure.
    """
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
