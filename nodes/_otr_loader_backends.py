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


#: Qwen rows whose chat template opens a think block unless told not to.
#: EXACT Hugging Face ids, never a prefix: a startswith("Qwen") test would
#: sweep up any future Qwen row whose template does not take the argument,
#: and apply_chat_template raises on an unknown kwarg. Both entries were read
#: from the published chat_template.jinja -- with add_generation_prompt each
#: emits a CLOSED think envelope when enable_thinking is false and an OPEN
#: one otherwise, so the writer reasons out loud if this never fires.
THINKING_SUPPRESSED_IDS = frozenset({
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.8-27B",
})


def chat_template_kwargs(model_id: str) -> dict:
    """Use Qwen's official direct-response switch on every native surface.

    Other models keep their exact previous template arguments. This controls
    template formatting, not sampling or a synthetic prompt rewrite.
    """
    stripped = str(model_id or "").split(" ", 1)[0]
    if _otr_model_catalog.hf_weights_id(stripped) in THINKING_SUPPRESSED_IDS:
        return {"enable_thinking": False}
    return {}


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


__all__ = [
    "LoaderBackend",
    "normalize_messages_for_tokenizer",
    "tokenizer_supports_system_role",
]
