"""Translate a planned verbatim passage into the episode language.

Operator ruling 2026-09-18: every lane is eligible for the episode language,
and for the fidelity lane "if it is an English source, translation is
necessary". The plan step (``_otr_verbatim_lane``) stays model-free; this
module takes its finished plan and returns one with the same speakers, the
same cut and the same order, and texts in the episode language.

What is fixed and what is not:

- TEXTS ONLY. The model never sees a speaker as something it may return;
  speakers are copied from the plan, because the executor and the outline
  compare them by exact string and a renamed speaker dies loud downstream.
- STRUCTURAL VALIDATION ONLY. Count and non-empty. No prose judgement (story
  quality is closed), no language detector, no back-translation.
- BATCHED FOR THE FLOOR. The smallest shipped writer context is 2048 tokens
  (``otr_4060_floor.json``), so consecutive entries are grouped under a
  source-word budget and ``max_new_tokens`` is sized to each batch rather
  than inherited from a stage that returns one short intent.
- LOUD ON FAILURE. Two attempts per batch through the shared ladder, then
  ``RuntimeError``: an English verbatim row on a native episode is the defect
  this module exists to remove, so a silent fallback is not an option.

Rows keep ``VERBATIM_SOURCE_FLAG`` downstream: the translation is the
performed text and no later rewriter may paraphrase it either.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

try:
    from . import _otr_episode_languages as _EPLANG
    from ._otr_repair_prompts import make_dispatching_repair_factory
    from ._otr_structured_call import StructuredCallFailedError, structured_call
    from ._otr_text_metrics import canonical_word_count
except ImportError:  # pragma: no cover -- flat test/standalone load
    import _otr_episode_languages as _EPLANG  # type: ignore
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore
    from _otr_structured_call import StructuredCallFailedError, structured_call  # type: ignore
    from _otr_text_metrics import canonical_word_count  # type: ignore

RECEIPT_VERSION = "otr_verbatim_translation_v1"

#: Source words per call. Fourteen shipped scenes carry a 300-word budget, so
#: a passage is three calls at most on the floor and one on a 4096+ context.
DEFAULT_MAX_SOURCE_WORDS = 120

#: Output allowance per SOURCE word, then a floor. CJK targets tokenise
#: denser per glyph than Latin; four covers the measured worst case with room.
_OUTPUT_TOKENS_PER_SOURCE_WORD = 4
_OUTPUT_TOKENS_FLOOR = 96

_SYSTEM = (
    "You translate a fixed passage of stage dialogue for a radio "
    "performance. Return JSON only: {\"texts\": [\"...\", ...]} with EXACTLY "
    "one string per numbered line, in the same order. Translate each line "
    "faithfully into the episode language: keep its sense, its register and "
    "roughly its length; keep every proper name as written; add nothing, "
    "drop nothing, merge nothing, and write no notes or speaker labels."
)


class TranslatedTexts(BaseModel):
    texts: list[str] = Field(min_length=1)


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def batch_entries(entries: Sequence[Any], *, max_source_words: int) -> list[list[int]]:
    """Consecutive index groups whose source words stay under the budget.

    Every batch holds at least one entry, so a single speech longer than the
    budget is still one call rather than an impossible zero-length batch.
    """
    budget = max(1, int(max_source_words))
    batches: list[list[int]] = []
    current: list[int] = []
    words = 0
    for index, entry in enumerate(entries):
        cost = max(1, canonical_word_count(str(entry.text)))
        if current and words + cost > budget:
            batches.append(current)
            current, words = [], 0
        current.append(index)
        words += cost
    if current:
        batches.append(current)
    return batches


def _user_prompt(entries: Sequence[Any], indices: Sequence[int]) -> str:
    rows = [
        f"{n}. {entries[i].speaker}: {' '.join(str(entries[i].text).split())}"
        for n, i in enumerate(indices, 1)
    ]
    return (
        f"Translate these {len(indices)} lines. Return exactly {len(indices)} "
        "strings in \"texts\", one per line, in order:\n\n" + "\n".join(rows)
    )


def _output_budget(entries: Sequence[Any], indices: Sequence[int]) -> int:
    source_words = sum(max(1, canonical_word_count(str(entries[i].text)))
                       for i in indices)
    return max(_OUTPUT_TOKENS_FLOOR,
               source_words * _OUTPUT_TOKENS_PER_SOURCE_WORD)


def translate_entries(
    entries: Sequence[Any],
    *,
    language_instruction: str,
    creative_fn: Callable[..., str],
    max_source_words: int = DEFAULT_MAX_SOURCE_WORDS,
    max_attempts: int = 2,
) -> tuple[tuple[Any, ...], dict]:
    """Return ``(translated_entries, receipt)``; raises on an exhausted batch.

    ``entries`` are ``BeatPlanEntry`` rows (``speaker``, ``text`` and the cut
    fields); every field but ``text`` is copied through untouched.
    """
    language_instruction = str(language_instruction or "").strip()
    if not language_instruction:
        raise ValueError("translate_entries needs the row's native instruction")
    entries = tuple(entries)
    if not entries:
        return (), {"receipt_version": RECEIPT_VERSION, "entries": 0, "batches": 0,
                    "attempts": 0}
    system = _EPLANG.lead_system(_SYSTEM, language_instruction)
    batches = batch_entries(entries, max_source_words=max_source_words)
    translated: list[Any] = list(entries)
    attempts_total = 0
    for indices in batches:
        want = len(indices)

        def _check(result: TranslatedTexts, want: int = want) -> str | None:
            if len(result.texts) != want:
                return (f"returned {len(result.texts)} strings for {want} "
                        "numbered lines; return exactly one per line")
            for n, text in enumerate(result.texts, 1):
                if not str(text or "").strip():
                    return f"line {n} came back empty"
            return None

        attempts = 0

        def _count(number: int, raw: str, error: BaseException | None) -> None:
            nonlocal attempts
            attempts = number

        try:
            # LLM slot: creative -- performing a fixed passage in the episode
            # language is voice work, not extraction.
            result = structured_call(
                prompt=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": _user_prompt(entries, indices)},
                ],
                schema=TranslatedTexts,
                slot_fn=creative_fn,
                base_temperature=0.2,
                structural_retry_temperature=0.1,
                repair_prompt_factory=make_dispatching_repair_factory(),
                post_validator=_check,
                max_new_tokens=_output_budget(entries, indices),
                max_attempts=max_attempts,
                helper_name="verbatim_translation",
                on_attempt_complete=_count,
            )
        except StructuredCallFailedError as exc:
            raise RuntimeError(
                "[verbatim_translation] the passage could not be translated "
                f"(batch of {want} line(s) starting at entry {indices[0]}; "
                f"{exc.attempts} attempt(s); last error: {exc.last_error}). "
                "An English verbatim row cannot ship on a native episode."
            ) from exc
        attempts_total += attempts or 1
        for i, text in zip(indices, result.texts):
            translated[i] = replace(entries[i], text=" ".join(str(text).split()))
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "entries": len(entries),
        "batches": len(batches),
        "attempts": attempts_total,
        "max_source_words": int(max_source_words),
        "source_sha256": _sha256("\n".join(e.text for e in entries)),
        "text_sha256": _sha256("\n".join(e.text for e in translated)),
    }
    return tuple(translated), receipt


def translate_plan(plan: Any, *, language_instruction: str,
                   creative_fn: Callable[..., str], iso: str = "",
                   model_id: str = "", **kwargs: Any) -> tuple[Any, dict]:
    """A ``VerbatimPlan`` with translated entries and a receipt naming the row.

    ``passage_text`` is re-rendered from the entries so the plan stays
    self-consistent; the interpreter already read the English projection at
    resolve time and the announcer chrome is row-owned, so nothing upstream
    is re-projected.
    """
    entries, receipt = translate_entries(
        plan.entries, language_instruction=language_instruction,
        creative_fn=creative_fn, **kwargs)
    receipt["iso"] = str(iso or "")
    receipt["model_id"] = str(model_id or "")
    passage_text = "\n\n".join(f"{e.speaker}\n{e.text}" for e in entries)
    return replace(plan, entries=entries, passage_text=passage_text), receipt


__all__ = [
    "DEFAULT_MAX_SOURCE_WORDS",
    "RECEIPT_VERSION",
    "TranslatedTexts",
    "batch_entries",
    "translate_entries",
    "translate_plan",
]
