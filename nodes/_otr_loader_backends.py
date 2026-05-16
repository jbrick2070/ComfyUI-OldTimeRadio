"""Sprint D D1b -- loader-backend duck-typed protocol + helpers.

Introduces the abstraction layer that lets the period-LLM CATEGORY
land in D1c onward. The existing `load_llm` / `unload_llm` /
`make_generate_fn` machinery in `nodes/_otr_model_loader.py` stays
the implementation surface; this module is the dispatch layer that
routes a CuratedModel row's `loader_backend` Literal value to the
right concrete adapter.

Protocol shape (duck-typed):
    LoaderBackend
        load(repo_id, row) -> dict   # cache_entry the rest of the
                                       # writer pipeline already consumes
        generate(model, messages, **kwargs) -> str
        unload(model) -> None

Three concrete adapters in `nodes/_otr_model_runtime.py`:
    transformers_safetensors        -> existing load_llm path
    transformers_multimodal_text_only -> same load_llm path
    transformers_gptq_int4          -> NEW for talkie; D1c lands
                                       the runtime-gated execution
                                       path; D1b ships the scaffold

Why duck-typed not ABC: Cowork plan v3 picked duck-typing for shape
flexibility. The protocol acts as documentation + type-check hint;
concrete adapters are plain classes that implement the three
callables. No ABC inheritance chain. Adding a new backend means
adding a class with the three callables and registering it in the
BACKENDS_BY_KEY dispatch table.

`compute_effective_context_limit(row)` is the helper D2c will use to
cap prompt budget per backend. Mirror of CURATED_CONTEXT_OVERRIDES /
HARD_VRAM_CONTEXT_LIMIT but reads from the new row-level
context_window field (D1a) so per-row variance (talkie at 4096) is
respected without touching the legacy override dict.
"""
from __future__ import annotations

from typing import Any, Protocol

from . import _otr_model_catalog


class LoaderBackend(Protocol):
    """Duck-typed loader-backend protocol.

    Concrete adapters implement the three callables. The runtime
    cache_entry shape returned by `load` must match the legacy
    `load_llm` return shape (see `nodes/_otr_model_loader.load_llm`
    docstring) so the rest of the writer pipeline (`request_slot`,
    `make_generate_fn`, etc) continues to consume it unchanged.
    """

    def load(self, repo_id: str, row: Any) -> dict[str, Any]:
        """Load the model and return a cache_entry dict.

        Args:
            repo_id: canonical HuggingFace repo_id (no UI suffix).
            row: the CuratedModel row from _otr_model_catalog.

        Returns:
            cache_entry dict matching legacy load_llm shape.

        Raises:
            ModelLoaderError on any underlying load failure.
        """
        ...

    def generate(
        self, model: Any, messages: list[dict], **kwargs: Any,
    ) -> str:
        """Run one inference pass and return the decoded string."""
        ...

    def unload(self, model: Any) -> None:
        """Evict the model from VRAM and free its resources."""
        ...


def compute_effective_context_limit(row: Any) -> int:
    """Return the effective context-window cap for a row.

    `min(HARD_VRAM_CONTEXT_LIMIT, row.context_window)`.

    The HARD_VRAM_CONTEXT_LIMIT is the system-wide ceiling for VRAM
    safety (overridable via OTR_HARD_VRAM_CONTEXT_LIMIT env). The
    row.context_window field (D1a) carries the per-model native
    context window. Effective limit is the smaller of the two so:
      * Mistral-Nemo at 8192 native + 8192 limit = 8192 (no clamp).
      * Talkie at 4096 native + 8192 limit = 4096 (row wins, smaller).
      * Hypothetical 16384-native model + 8192 limit = 8192 (limit wins).

    Always returns int >= 0. Reading the field on a row constructed
    pre-D1a (no context_window) raises AttributeError -- by design;
    every catalog row carries the field after D1a.
    """
    return min(
        int(_otr_model_catalog.HARD_VRAM_CONTEXT_LIMIT),
        int(row.context_window),
    )


__all__ = [
    "LoaderBackend",
    "compute_effective_context_limit",
]
