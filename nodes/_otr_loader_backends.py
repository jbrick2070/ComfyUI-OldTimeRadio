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
    transformers_gptq_int4          -> parked for a future period
                                       model; D1b ships the scaffold,
                                       D1c reserves the runtime path

Why duck-typed not ABC: Cowork plan v3 picked duck-typing for shape
flexibility. The protocol acts as documentation + type-check hint;
concrete adapters are plain classes that implement the three
callables. No ABC inheritance chain. Adding a new backend means
adding a class with the three callables and registering it in the
BACKENDS_BY_KEY dispatch table.

`compute_effective_context_limit(row)` is the helper D2c will use to
cap prompt budget per backend. Mirror of CURATED_CONTEXT_OVERRIDES /
HARD_VRAM_CONTEXT_LIMIT but reads from the new row-level
context_window field (D1a) so per-row variance (a sub-limit native
window) is respected without touching the legacy override dict.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from . import _otr_model_catalog

log = logging.getLogger(__name__)


class LoaderBackend(Protocol):
    """Duck-typed loader-backend protocol.

    Concrete adapters implement the three callables. The runtime
    cache_entry shape returned by `load` must match the legacy
    `load_llm` return shape (see `nodes/_otr_model_loader.load_llm`
    docstring) so the rest of the writer pipeline (`request_slot`,
    `make_generate_fn`, etc) continues to consume it unchanged.
    """

    def load(self, repo_id: str, row: Any, policy: Any = None) -> dict[str, Any]:
        """Load the model and return a cache_entry dict.

        Args:
            repo_id: canonical HuggingFace repo_id (no UI suffix).
            row: the CuratedModel row from _otr_model_catalog.
            policy: the S1 LLMRuntimePolicy (None = nv50 baseline).

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
      * A 4096-native row + 8192 limit = 4096 (row wins, smaller).
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

    A row at context_window=4096 would trip this under the default
    HARD_VRAM_CONTEXT_LIMIT=8192. The G5 operator gate covers
    whether to relax (loosen the precondition to a warning), accept
    (research-lane catalog visibility but load-blocked), or land a
    D-future compact-mode binding.

    All 6 curated catalog rows have context_window=8192 which
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
        # Fold system messages into the first user turn for tokenizers
        # whose chat template rejects the system role (BUG-LOCAL-262)
        # so the dispatch path and the live writer path agree.
        normalized = normalize_messages_for_tokenizer(tokenizer, messages)
        return tokenizer.apply_chat_template(
            normalized,
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


# ---------------------------------------------------------------------------
# System-role normalization (BUG-LOCAL-262)
# ---------------------------------------------------------------------------
#
# Some chat templates hard-reject the `system` role. Gemma-2's Jinja
# template contains a literal raise_exception("System role not
# supported") -- Gemma-2 has no system role by design. Passing a
# message list with a {"role": "system", ...} entry straight into
# tokenizer.apply_chat_template throws TemplateError and aborts the
# run (the style picker's chooser pass was the first call to surface
# it; see BUG_LOG.md BUG-LOCAL-262).
#
# `tokenizer_supports_system_role` probes the tokenizer once;
# `normalize_messages_for_tokenizer` folds system content into the
# first following user turn when the probe says the role is
# unsupported. This is the standard Gemma pattern and works for any
# system-role-incompatible model, not just Gemma-2.


def tokenizer_supports_system_role(tokenizer: Any) -> bool:
    """Probe whether `tokenizer`'s chat template accepts a system role.

    Renders a minimal 2-message probe list through
    ``tokenizer.apply_chat_template(..., tokenize=False)``. If the
    render raises for any reason, the system role is treated as
    unsupported and the failure is logged once at debug level.

    This is a pure capability probe -- no generation, no model. It
    is cheap but not free (a Jinja render), so callers should cache
    the result per model residency rather than probe per generate
    call. See `normalize_messages_for_tokenizer` for the per-call
    entrypoint and `_build_truncating_generate_fn` for the caching
    idiom.

    Returns True if the probe rendered cleanly, False otherwise.
    """
    probe = [
        {"role": "system", "content": "probe"},
        {"role": "user", "content": "probe"},
    ]
    try:
        tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True,
        )
    except Exception as exc:  # noqa: BLE001 -- any render failure -> unsupported
        log.debug(
            "[_otr_loader_backends] tokenizer chat template rejects "
            "the system role; system messages will be folded into "
            "the first user turn (probe error: %s)",
            exc,
        )
        return False
    return True


def normalize_messages_for_tokenizer(
    tokenizer: Any, messages: list[dict],
) -> list[dict]:
    """Return a message list safe to feed `tokenizer.apply_chat_template`.

    If the tokenizer's chat template accepts a system role, `messages`
    is returned unchanged. If it does not (Gemma-2 and similar), every
    system message's content is folded into the first following user
    message as ``system_text + "\\n\\n" + user_text`` and the system
    entries are dropped -- the standard Gemma pattern.

    Edge cases:
      * A system message with no following user message is converted
        to a user message (its content is preserved, not lost).
      * Multiple system messages are concatenated in order before
        being prepended to the first user turn.

    The input list is never mutated; a new list is returned.
    """
    if not messages:
        return messages
    if tokenizer_supports_system_role(tokenizer):
        return messages
    if not any(m.get("role") == "system" for m in messages):
        return messages

    out: list[dict] = []
    pending_system: list[str] = []
    folded = False
    for msg in messages:
        if msg.get("role") == "system":
            pending_system.append(str(msg.get("content", "")))
            continue
        if pending_system and msg.get("role") == "user" and not folded:
            system_text = "\n\n".join(s for s in pending_system if s)
            user_text = str(msg.get("content", ""))
            merged = (
                f"{system_text}\n\n{user_text}"
                if system_text and user_text
                else (system_text or user_text)
            )
            new_msg = dict(msg)
            new_msg["content"] = merged
            out.append(new_msg)
            pending_system = []
            folded = True
            continue
        out.append(dict(msg))

    # System message(s) with no following user turn -> convert the
    # concatenated system text into a standalone user message so its
    # content survives into the prompt.
    if pending_system:
        system_text = "\n\n".join(s for s in pending_system if s)
        if system_text:
            out.insert(0, {"role": "user", "content": system_text})
    return out


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
    "normalize_messages_for_tokenizer",
    "stop_strings_for_row",
    "tokenizer_supports_system_role",
    "generate_kwargs_for_row",
]
