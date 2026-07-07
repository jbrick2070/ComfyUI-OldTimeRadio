"""Sprint D D1b -- loader-backend dispatch + concrete adapter scaffolds.

Three concrete adapters implementing the `LoaderBackend` protocol
from `nodes/_otr_loader_backends`:

  TransformersSafetensorsBackend          (existing path, delegates)
  TransformersMultimodalTextOnlyBackend   (existing path, delegates)
  TransformersGPTQInt4Backend             (parked for a future period
                                           model; D1c reserves the
                                           runtime execution path)

Dispatch table `BACKENDS_BY_KEY` keys on the `loader_backend` Literal
value declared on the catalog row. `get_backend_for_row(row)` looks
up the right adapter.

The two existing backends are CURRENTLY thin wrappers around
`_otr_model_loader.load_llm` / `unload_llm` / `make_generate_fn` --
the legacy monolithic loader handles both text-only and multimodal
internally via in-body branching. D1b does NOT refactor that body;
the adapter classes exist so future per-backend hooks (chat-template
dispatch in D2c, GPTQ inference in D1c) land cleanly without
restructuring the big loader function.

The GPTQ adapter is a SCAFFOLD only at D1b. `load()` raises
NotImplementedError until D1c wires the actual `AutoGPTQForCausalLM`
path. This is intentional: structural pytest at D1b verifies the
adapter is constructable and registered in the dispatch table;
runtime smoke at D1c verifies the load path under
`OTR_REGRESSION_RUNTIME=1`.
"""
from __future__ import annotations

from typing import Any

from . import _otr_comfy_backend
from . import _otr_gguf_backend
from . import _otr_loader_backends
from . import _otr_model_catalog
from . import _otr_model_loader
from . import _otr_openrouter_backend


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------


class _LegacyTransformersBackendBase:
    """Shared delegate to the legacy `_otr_model_loader` machinery.

    Both `transformers_safetensors` and `transformers_multimodal_text_only`
    backends route through the same legacy entry points today. The
    legacy `load_llm` internally branches on multimodal vs text-only
    via heuristics on the model_id. D1b preserves that behavior by
    delegating; D-future may split the body and have each adapter own
    its own load implementation.
    """

    def load(self, repo_id: str, row: Any) -> dict[str, Any]:
        # Sprint D D4: precondition gate fires before the legacy
        # load delegate. Rows whose context_window is below
        # HARD_VRAM_CONTEXT_LIMIT raise here instead of silently
        # truncating mid-generation downstream. All 6 curated rows
        # have context_window=8192 which matches the default
        # HARD_VRAM_CONTEXT_LIMIT so this is a no-op for them; a
        # sub-limit period model would trip the precondition.
        _otr_loader_backends.check_context_window(row)
        return _otr_model_loader.load_llm(repo_id)

    def generate(
        self, model: Any, messages: list[dict], **kwargs: Any,
    ) -> str:
        # model here is the cache_entry dict returned by load();
        # legacy make_generate_fn returns a callable that takes
        # (messages, **kwargs) and returns the decoded string.
        gen_fn = _otr_model_loader.make_generate_fn(model)
        return gen_fn(messages, **kwargs)

    def unload(self, model: Any) -> None:  # noqa: ARG002
        # Legacy unload_llm operates on the singleton cache, not on
        # the cache_entry argument. Signature here keeps the protocol
        # uniform across adapters that DO take a model arg (GPTQ).
        _otr_model_loader.unload_llm()


class TransformersSafetensorsBackend(_LegacyTransformersBackendBase):
    """Adapter for the `transformers_safetensors` backend literal.

    Covers Mistral-Nemo, Qwen, Captain-Eris, MN-12B-Mag-Mell at D1b.
    Delegates entirely to the legacy load_llm path. Distinct class
    (not a singleton instance of the base) so D-future per-backend
    hooks have a stable identity to extend.
    """


class TransformersMultimodalTextOnlyBackend(_LegacyTransformersBackendBase):
    """Adapter for the `transformers_multimodal_text_only` backend literal.

    Covers Gemma-4-E2B-it and Gemma-4-E4B-it at D1b. Delegates to the
    legacy load_llm path which internally detects the Gemma-4 family
    via model_id and routes through the multimodal-text-only code
    branch in nodes/_otr_model_loader.py.
    """


class TransformersGPTQInt4Backend:
    """Adapter scaffold for the `transformers_gptq_int4` backend literal.

    Sprint D D1b ships this class as a constructable stub so the
    dispatch table at this commit boundary includes all three
    backends. The `AutoGPTQForCausalLM` runtime load path is
    reserved for a future period model behind
    `OTR_REGRESSION_RUNTIME=1`; no curated row uses this backend at
    present (the talkie row that did was removed 2026-05-22).

    Until D1c lands, every callable raises a clear NotImplementedError
    so a structural-pytest invocation does not accidentally try to
    pull weights or execute generation.
    """

    def load(self, repo_id: str, row: Any) -> dict[str, Any]:  # noqa: ARG002
        # Sprint D D4: precondition fires FIRST. A row whose
        # context_window is below HARD_VRAM_CONTEXT_LIMIT trips this
        # before the NotImplementedError, surfacing the context-
        # window mismatch as the dominant error rather than the
        # deferred-feature one.
        _otr_loader_backends.check_context_window(row)
        raise NotImplementedError(
            "TransformersGPTQInt4Backend.load is a D1b scaffold; the "
            "AutoGPTQForCausalLM runtime path lands in D1c behind "
            "OTR_REGRESSION_RUNTIME=1. Use the dispatch table for "
            "structural pytest only at D1b."
        )

    def generate(
        self, model: Any, messages: list[dict], **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            "TransformersGPTQInt4Backend.generate is a D1b scaffold; "
            "lands at D1c with the runtime smoke."
        )

    def unload(self, model: Any) -> None:
        raise NotImplementedError(
            "TransformersGPTQInt4Backend.unload is a D1b scaffold; "
            "lands at D1c."
        )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


BACKENDS_BY_KEY: dict[str, Any] = {
    "transformers_safetensors":          TransformersSafetensorsBackend(),
    "transformers_multimodal_text_only": TransformersMultimodalTextOnlyBackend(),
    "transformers_gptq_int4":            TransformersGPTQInt4Backend(),
    # Remote, opt-in, default-off (S1). Registered here so the dormant
    # dispatch table knows the adapter; S3 wires the LIVE call into
    # request_slot + the generate-fn factory. A remote call uses zero
    # local VRAM and must never evict the resident local model (C2).
    _otr_openrouter_backend.OPENROUTER_BACKEND_KEY:
        _otr_openrouter_backend.OpenRouterBackend(),
    # Comfy Credits remote, opt-in, default-off (2026-06-01). The credit-
    # billed sibling of the OpenRouter lane: ComfyUI proxies the call and
    # bills prepaid account credits. Zero local VRAM; never evicts the
    # resident local model (C2).
    _otr_comfy_backend.COMFY_BACKEND_KEY:
        _otr_comfy_backend.ComfyCreditsBackend(),
    # Native GGUF Gemma 4 12B path: in-process llama-cpp-python, no sidecar.
    _otr_gguf_backend.GGUF_BACKEND_KEY:
        _otr_gguf_backend.GGUFNativeBackend(),
}


def get_backend_for_row(row: Any) -> Any:
    """Return the loader-backend instance for a CuratedModel row.

    Looks up `row.loader_backend` (Literal value) against the
    BACKENDS_BY_KEY dispatch table. Raises KeyError with a clear
    message if the row carries an unrecognized backend literal --
    catches schema drift between the catalog and the dispatch table.
    """
    key = str(row.loader_backend)
    if key not in BACKENDS_BY_KEY:
        raise KeyError(
            f"no loader-backend adapter registered for "
            f"loader_backend={key!r} (row.repo_id={row.repo_id!r}). "
            f"Available keys: {sorted(BACKENDS_BY_KEY)}"
        )
    return BACKENDS_BY_KEY[key]


__all__ = [
    "TransformersSafetensorsBackend",
    "TransformersMultimodalTextOnlyBackend",
    "TransformersGPTQInt4Backend",
    "BACKENDS_BY_KEY",
    "get_backend_for_row",
]
