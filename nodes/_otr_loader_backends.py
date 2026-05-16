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


def check_context_window(row: Any) -> None:
    """Sprint D D4 -- precondition gate for adapter load.

    Raises RuntimeError if `row.context_window < HARD_VRAM_CONTEXT_LIMIT`.
    The hard limit is the system-wide ceiling; a row whose native
    window is BELOW it means the loaded model would refuse prompts
    that are within the system budget but exceed the model's native
    window. Surface that mismatch loud at load time rather than
    silently truncating mid-generation.

    Talkie at context_window=4096 trips this by design under the
    default HARD_VRAM_CONTEXT_LIMIT=8192. The G5 operator gate
    covers whether to relax (loosen the precondition to a warning),
    accept (research-lane catalog visibility but load-blocked), or
    land a D-future compact-mode binding.

    Existing 6 catalog rows all have context_window=8192 which
    matches the default HARD_VRAM_CONTEXT_LIMIT so they DO NOT
    trip the precondition under any production setting.
    """
    if int(row.context_window) < int(_otr_model_catalog.HARD_VRAM_CONTEXT_LIMIT):
        raise RuntimeError(
            f"context_window {row.context_window} for "
            f"{getattr(row, 'repo_id', '?')!r} is below "
            f"HARD_VRAM_CONTEXT_LIMIT "
            f"{_otr_model_catalog.HARD_VRAM_CONTEXT_LIMIT}. Pick a "
            f"larger-window variant or land a compact-mode binding "
            f"in D-future. Operator gate G5 covers whether to "
            f"relax this precondition for research-lane models."
        )


def encode_messages_for_row(tokenizer, messages: list[dict], row: Any):
    """Sprint D D2c -- per-backend message encoding dispatch.

    Dispatches on the catalog row's `chat_template_kind` Literal:

      "transformers_default"  -> tokenizer.apply_chat_template(
                                     messages, return_tensors="pt",
                                     add_generation_prompt=True,
                                 )
      "raw_completion"        -> tokenizer(
                                     "\\n".join(m["content"] for m in messages),
                                     return_tensors="pt",
                                 )
      "manual"                -> NotImplementedError. The row-level
                                 manual template field is not in
                                 the v1 schema; deferred to D-future.

    Returns the encoded inputs ready to pass into model.generate().
    Raises ValueError on an unknown chat_template_kind.

    The dispatch is metadata-driven only. No `repo_id` substring
    matching, no per-row special cases. Adding a new tokenizer
    family means classifying it via chat_template_kind on the
    catalog row -- no edits here.
    """
    kind = row.chat_template_kind
    if kind == "transformers_default":
        return tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    if kind == "raw_completion":
        joined = "\n".join(m.get("content", "") for m in messages)
        return tokenizer(joined, return_tensors="pt")
    if kind == "manual":
        raise NotImplementedError(
            f"chat_template_kind='manual' requires a row-level "
            f"manual_chat_template field not present in the v1 "
            f"CuratedModel schema (D1a). Deferred to D-future for "
            f"tokenizers that ship without chat_template AND need "
            f"a non-raw template. Affected row: "
            f"repo_id={getattr(row, 'repo_id', '?')!r}"
        )
    raise ValueError(
        f"unknown chat_template_kind {kind!r}; expected one of "
        f"('transformers_default', 'raw_completion', 'manual')"
    )


def stop_strings_for_row(row: Any) -> list[str]:
    """Return the row's stop_tokens as a list (the shape generate()
    expects via stop_strings= kwarg). Empty tuple -> empty list,
    meaning "use the tokenizer's default EOS handling".
    """
    return list(getattr(row, "stop_tokens", ()) or ())


def generate_kwargs_for_row(row: Any) -> dict[str, Any]:
    """Return per-row kwargs to thread into the model.generate()
    call. At D2c this is just:

        {
            "max_new_tokens": <caller-controlled, NOT set here>,
            "stop_strings": list(row.stop_tokens),
        }

    The effective context limit (D1b helper) caps the PROMPT side
    of the budget. The caller decides max_new_tokens for the
    OUTPUT side. This helper assembles the stop_strings half so
    every adapter dispatches stop tokens identically.
    """
    return {
        "stop_strings": stop_strings_for_row(row),
    }


__all__ = [
    "LoaderBackend",
    "check_context_window",
    "compute_effective_context_limit",
    "encode_messages_for_row",
    "stop_strings_for_row",
    "generate_kwargs_for_row",
]
