"""Source interpreter bindings: named, allowlisted, declared in banks.json.

Protocol (kibitz r1):

    interpret_source(packet, bank, llm_fns) -> StoryInputPacket

Fixture interpreters are DETERMINISTIC PLUMBING: they validate and assemble
content that lives in the source packet JSON (fixture_* fields). They never
author prose. At transplant, the science bank's production interpreter wraps
news_interpreter.build_news_briefs verbatim and maps NewsBriefs ->
StoryInputPacket; that binding is registered production-side, not here.
"""

from __future__ import annotations

from typing import Any, Callable

from .contracts import SourceBankSpec, SourceMaterialPacket, StoryInputPacket


class InterpreterError(ValueError):
    """Fail-loud interpreter problem (missing content, wrong bank)."""


#: Optional LLM callables a production interpreter may use. Fixture
#: interpreters ignore them (network/model-free by design).
LlmFns = dict[str, Callable[..., Any]]


def _require(packet: SourceMaterialPacket, field: str) -> str:
    value = str(getattr(packet, field, "")).strip()
    if not value:
        raise InterpreterError(
            f"source packet for bank {packet.source_bank_id!r} is missing "
            f"required fixture content field {field!r} - author it in the "
            "packet JSON (JSON owns content; no prose is invented here)."
        )
    return value


def _base_kwargs(packet: SourceMaterialPacket, story_model_id: str) -> dict[str, Any]:
    return {
        "source_bank_id": packet.source_bank_id,
        "story_model_id": story_model_id,
        "source_label": packet.source_label,
        "casting_brief": _require(packet, "fixture_casting_brief"),
        "script_brief": (
            packet.fixture_script_brief.strip()
            or packet.source_summary.strip()
            or _require(packet, "fixture_script_brief")
        ),
        "close_brief": _require(packet, "fixture_close_brief"),
        "key_terms": list(packet.fixture_key_terms),
        "source_fidelity_rules": list(packet.fixture_fidelity_rules),
        "source_material": packet,
    }


def interpret_fixture_science_news(packet: SourceMaterialPacket,
                                   bank: SourceBankSpec,
                                   llm_fns: LlmFns | None = None,
                                   *, story_model_id: str) -> StoryInputPacket:
    if packet.source_bank_id != "science_news":
        raise InterpreterError(
            f"science interpreter got bank {packet.source_bank_id!r}"
        )
    kwargs = _base_kwargs(packet, story_model_id)
    if not kwargs["key_terms"]:
        raise InterpreterError(
            "science fixture packet must carry fixture_key_terms (>=1); the "
            "production lane guarantees key_terms via NewsBriefs"
        )
    kwargs["adaptation_trace"] = {"fixture": True, "story_model_id": story_model_id}
    return StoryInputPacket(**kwargs)


def interpret_fixture_media_archive(packet: SourceMaterialPacket,
                                    bank: SourceBankSpec,
                                    llm_fns: LlmFns | None = None,
                                    *, story_model_id: str) -> StoryInputPacket:
    if packet.source_bank_id != "media_archive":
        raise InterpreterError(
            f"media_archive interpreter got bank {packet.source_bank_id!r}"
        )
    kwargs = _base_kwargs(packet, story_model_id)
    kwargs["adaptation_trace"] = {"fixture": True, "story_model_id": story_model_id}
    return StoryInputPacket(**kwargs)


def interpret_fixture_public_domain(packet: SourceMaterialPacket,
                                    bank: SourceBankSpec,
                                    llm_fns: LlmFns | None = None,
                                    *, story_model_id: str) -> StoryInputPacket:
    if packet.source_bank_id != "public_domain_story":
        raise InterpreterError(
            f"public_domain interpreter got bank {packet.source_bank_id!r}"
        )
    if packet.rights_status != "public_domain":
        raise InterpreterError(
            f"public_domain packet has rights_status {packet.rights_status!r}; "
            "only 'public_domain' is allowed in this lane"
        )
    kwargs = _base_kwargs(packet, story_model_id)
    if not (packet.raw_text.strip() or packet.source_text_ref.strip()):
        raise InterpreterError(
            "public_domain packet must carry raw_text or source_text_ref"
        )
    kwargs["script_brief"] = packet.raw_text.strip() or kwargs["script_brief"]
    kwargs["adaptation_trace"] = {
        "fixture": True,
        "story_model_id": story_model_id,
        "source_kind": packet.source_kind,
    }
    return StoryInputPacket(**kwargs)


#: The explicit binding allowlist. banks.json names one of these; an unknown
#: binding is a hard error naming the bank (no hidden engines).
INTERPRETER_BINDINGS: dict[str, Callable[..., StoryInputPacket]] = {
    "fixture_science_news": interpret_fixture_science_news,
    "fixture_media_archive": interpret_fixture_media_archive,
    "fixture_public_domain": interpret_fixture_public_domain,
}


def resolve_interpreter(bank: SourceBankSpec) -> Callable[..., StoryInputPacket]:
    binding = (bank.interpreter or "").strip()
    try:
        return INTERPRETER_BINDINGS[binding]
    except KeyError as exc:
        raise InterpreterError(
            f"bank {bank.source_bank_id!r} declares unknown interpreter binding "
            f"{binding!r}; allowlist: {sorted(INTERPRETER_BINDINGS)}"
        ) from exc
