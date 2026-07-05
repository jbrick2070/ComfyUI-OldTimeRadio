"""Source interpreter facade for the OTR upstream transplant.

DESTINATION: ComfyUI-OldTimeRadio/nodes/_otr_source_interpreter.py
STATUS: staged in the lab (transplant_work/); NOT installed yet.

Declared contract (kibitz r1/r2):

    interpret_source(bank_id, *, article=None, bridge_story_input=None,
                     news_briefs_builder=None, **builder_kwargs)
        -> {"casting_brief", "script_brief", "close_brief", "key_terms"}

- science_news REQUIRES news_briefs_builder (production wires
  news_interpreter.build_news_briefs verbatim; this facade never re-implements
  the science brain and adds no science prose).
- media_archive / public_domain_story are PACKET-DRIVEN in v1: they REQUIRE
  bridge_story_input (the validated bridge artifact's story_input block).
  Live archive fetching is a later declared binding, not v1.
- Anything else: loud error. No lane ever borrows another lane's brain.
"""

from __future__ import annotations

from typing import Any, Callable

PACKET_DRIVEN_BANKS = ("media_archive", "public_domain_story")
LOGICAL_FIELDS = ("casting_brief", "script_brief", "close_brief", "key_terms")


class SourceInterpreterError(ValueError):
    """No usable source brain for this bank/input combination."""


def _from_bridge_story_input(bank_id: str, story_input: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(story_input, dict):
        raise SourceInterpreterError(
            f"{bank_id}: bridge_story_input must be an object"
        )
    if story_input.get("source_bank_id") != bank_id:
        raise SourceInterpreterError(
            f"bridge story_input bank {story_input.get('source_bank_id')!r} "
            f"!= requested bank {bank_id!r} - refusing cross-lane input"
        )
    out: dict[str, Any] = {}
    for field in LOGICAL_FIELDS:
        value = story_input.get(field)
        if field == "key_terms":
            if not isinstance(value, list):
                raise SourceInterpreterError(
                    f"{bank_id}: story_input.key_terms must be a list"
                )
            out[field] = list(value)
        else:
            if not str(value or "").strip():
                raise SourceInterpreterError(
                    f"{bank_id}: story_input.{field} is required and empty "
                    "(the lab authors this content in JSON; nothing is invented here)"
                )
            out[field] = str(value)
    return out


def interpret_source(
    bank_id: str,
    *,
    article: dict[str, Any] | None = None,
    bridge_story_input: dict[str, Any] | None = None,
    news_briefs_builder: Callable[..., Any] | None = None,
    **builder_kwargs: Any,
) -> dict[str, Any]:
    bank = (bank_id or "").strip()
    if bank == "science_news":
        if news_briefs_builder is None:
            raise SourceInterpreterError(
                "science_news requires news_briefs_builder "
                "(wire news_interpreter.build_news_briefs); the facade never "
                "re-implements the science brain"
            )
        # Map the article dict onto build_news_briefs' REAL keyword set
        # (kibitz r4, Codex M1; signature at news_interpreter.py:731-746 -
        # headline/summary/full_text/outlet/pub_date; technical_fn/style/seed
        # arrive via builder_kwargs from the writer call site). No `article`
        # kwarg exists in production.
        call_kwargs = dict(builder_kwargs)
        if article is not None:
            call_kwargs.setdefault("headline", str(article.get("headline", "")))
            call_kwargs.setdefault("summary", str(article.get("summary", "")))
            call_kwargs.setdefault("full_text", str(article.get("full_text", "")))
            call_kwargs.setdefault("outlet", str(article.get("source", "")))
            call_kwargs.setdefault("pub_date", str(article.get("date", "")))
        briefs = news_briefs_builder(**call_kwargs)
        data = briefs.model_dump() if hasattr(briefs, "model_dump") else dict(briefs)
        return {
            "casting_brief": data["casting_brief"],
            "script_brief": data["script_brief"],
            "close_brief": data["news_close_brief"],
            "key_terms": list(data["key_terms"]),
        }
    if bank in PACKET_DRIVEN_BANKS:
        if bridge_story_input is None:
            raise SourceInterpreterError(
                f"{bank} is packet-driven in v1 and requires bridge_story_input "
                "(a validated bridge artifact's story_input block); there is no "
                "live fetcher for this lane yet and no fallback to science RSS"
            )
        return _from_bridge_story_input(bank, bridge_story_input)
    if bank == "custom_source_bank":
        raise SourceInterpreterError(
            "custom_source_bank is visible but not runnable: create and "
            "validate a custom schema, source packet, and story pack first "
            "(see upstream lab CUSTOM_SOURCE_BANK_GUIDE.md)"
        )
    raise SourceInterpreterError(f"unknown source bank {bank!r}")
