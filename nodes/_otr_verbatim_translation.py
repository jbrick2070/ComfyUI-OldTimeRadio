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
import re
import unicodedata
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

#: Output-budget rungs, tried in order while the failure looks like the reply
#: RAN OUT OF ROOM. The first rung is the budget as it has always been, so
#: every language that already fit behaves byte-identically and pays nothing.
#: See the long note in `translate_entries` -- this grows rather than predicts
#: because the true tokens-per-source-word ratio depends on the writer's
#: tokenizer and the target script, and a GGUF writer ships no tokenizer file
#: to measure against.
_OUTPUT_BUDGET_GROWTH = (1, 3, 6)

#: What a reply cut off mid-JSON leaves behind. A batch that came back WHOLE
#: and merely wrong -- the wrong number of lines, an untranslated row, a bare
#: speaker label -- is not short of room, and re-asking with a bigger budget
#: would just spend three times as long arriving at the same refusal.
_TRUNCATION_SIGNATURES = (
    "no decodable top-level json object",
    "unterminated",
    "expecting ',' delimiter",
    "expecting ':' delimiter",
    "expecting value",
    "expecting property name",
    "unexpected end of",
)


def _looks_out_of_room(error: Any) -> bool:
    """Is this failure the shape of a reply that was cut off mid-JSON?"""
    text = str(error or "").lower()
    return any(signature in text for signature in _TRUNCATION_SIGNATURES)

_SYSTEM = (
    "You translate a fixed passage of stage dialogue for a radio "
    "performance. Return JSON only: {\"texts\": [\"...\", ...]} with EXACTLY "
    "one string per numbered line, in the same order. Translate each line "
    "faithfully into the episode language: keep its sense, its register and "
    "roughly its length; keep every proper name as written; add nothing, "
    "drop nothing, merge nothing, and write no notes. Each string is the "
    "SPOKEN WORDS ONLY -- never repeat the speaker's name or a colon at the "
    "start of a line; the name is already known and repeating it makes the "
    "voice read it aloud."
)


class TranslatedTexts(BaseModel):
    texts: list[str] = Field(min_length=1)


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


#: A speaker label the model echoed back into the line it was asked to
#: translate. The user prompt shows `N. SPEAKER: text` and says to return the
#: texts alone; live proof 2026-09-18 (el_pico_de_hierro_es) shows the model
#: returning "ANTÍFON DE EFESO: Ve, vete..." anyway -- and it TRANSLATED the
#: name, so matching the plan's speaker string is not enough. TTS reads the
#: label aloud and the caption burns it, so the strip is deterministic and
#: never left to prompt wording.
_ECHOED_LABEL = re.compile(r"^\s*([^\n:：]{1,40})[:：][ \t　]*")


def _fold(text: str) -> str:
    stripped = unicodedata.normalize("NFD", str(text or "").casefold())
    return "".join(c for c in stripped if not unicodedata.combining(c)).strip()


def _is_label_shaped(text: str, speaker: str = "") -> bool:
    """Does ``text`` OPEN with something shaped like a speaker label?

    True for the plan's own speaker (folded, so an accented or case-shifted
    spelling matches) and for a short prefix before a colon carrying NO
    lowercase -- "no lowercase" rather than "mostly uppercase" because
    Chinese, Japanese and Devanagari have no case at all, and an uppercase
    RATIO scores those labels zero.
    """
    match = _ECHOED_LABEL.match(str(text or ""))
    if not match:
        return False
    prefix = match.group(1).strip()
    if not any(c.isalpha() for c in prefix):
        return False
    return (not any(c.islower() for c in prefix)
            or _fold(prefix) == _fold(speaker))


def strip_echoed_label(text: str, speaker: str = "", source_text: str = "") -> str:
    """Remove a leading speaker label from one translated line.

    Strips when the prefix is the plan's own speaker (folded, so an accented
    or case-shifted spelling still matches) OR is shaped like a label -- a
    short run before the colon carrying NO lowercase. "No lowercase" rather
    than "mostly uppercase" because Chinese, Japanese and Devanagari have no
    case at all, and an uppercase RATIO scores those labels zero.

    A line that merely contains a colon later is untouched, and so is
    ordinary sentence-case prose before one.

    ``source_text`` is the ENGLISH line this one translates, and it is the
    discriminator that keeps a real salutation: if the source itself opens
    with a LABEL-SHAPED prefix ("MY LORD: ..."), the translation is entitled
    to one too and nothing is stripped. Only a prefix the model ADDED goes.
    The source is judged by the same shape rule as the translation, so an
    ordinary "Look: here they come" in the source does not disable the strip.
    """
    raw = str(text or "")
    match = _ECHOED_LABEL.match(raw)
    if not match:
        return raw.strip()
    if source_text and _is_label_shaped(str(source_text), speaker):
        return raw.strip()
    if not _is_label_shaped(raw, speaker):
        return raw.strip()
    # An empty remainder means the model returned ONLY a label. Returning the
    # raw label here shipped "ANA:" as the whole spoken line, because the
    # structural check had already run on the UNSTRIPPED text and passed it.
    # The strip now happens inside that check (see `translate_entries`), so
    # "" is the honest answer and the validator turns it into a retry.
    return raw[match.end():].strip()


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

        def _check(result: TranslatedTexts, want: int = want,
                   indices: Sequence[int] = indices) -> str | None:
            if len(result.texts) != want:
                return (f"returned {len(result.texts)} strings for {want} "
                        "numbered lines; return exactly one per line")
            # Validate what will actually SHIP -- the stripped line. Checking
            # the raw reply let "ANA: " pass as non-empty and then ship the
            # bare label as the whole spoken row.
            for n, (index, text) in enumerate(zip(indices, result.texts), 1):
                shipped = strip_echoed_label(
                    str(text), entries[index].speaker, entries[index].text)
                if not shipped.strip():
                    return (f"line {n} came back empty (or as a bare speaker "
                            "label); return the spoken words")
            return None

        attempts = 0

        def _count(number: int, raw: str, error: BaseException | None) -> None:
            nonlocal attempts
            attempts = number

        # THE BUDGET IS COUNTED IN ENGLISH SOURCE WORDS, AND THE OUTPUT IS NOT
        # ENGLISH (PBUG-20260918-08). A flat tokens-per-source-word multiplier
        # assumes the target costs what the source costs. Devanagari does not:
        # the Hindi leg died at t=69s with "no decodable top-level JSON object
        # found: line 1 column 1 (char 0)" after both attempts -- the shape of
        # a generation that ran out of room before it closed its JSON, not of
        # a model that cannot translate.
        #
        # The honest fix is to GROW rather than to predict. The true ratio
        # depends on the writer's tokenizer and the target script, and this
        # box has no tokenizer file to measure (the writers are GGUF), so any
        # per-language constant would be a guess dressed as a measurement.
        # Each rung retries the whole batch with more room; a script that fits
        # never pays for the later rungs, and English is byte-identical.
        budget = _output_budget(entries, indices)
        last_exc: "StructuredCallFailedError | None" = None
        result = None
        rungs_tried = 0
        for growth in _OUTPUT_BUDGET_GROWTH:
            rungs_tried += 1
            try:
                # LLM slot: creative -- performing a fixed passage in the
                # episode language is voice work, not extraction.
                result = structured_call(
                    prompt=[
                        {"role": "system", "content": system},
                        {"role": "user",
                         "content": _user_prompt(entries, indices)},
                    ],
                    schema=TranslatedTexts,
                    slot_fn=creative_fn,
                    base_temperature=0.2,
                    structural_retry_temperature=0.1,
                    repair_prompt_factory=make_dispatching_repair_factory(),
                    post_validator=_check,
                    max_new_tokens=budget * growth,
                    max_attempts=max_attempts,
                    helper_name="verbatim_translation",
                    on_attempt_complete=_count,
                )
                break
            except StructuredCallFailedError as exc:
                last_exc = exc
                attempts_total += attempts or 1
                if not _looks_out_of_room(exc.last_error):
                    # Whole reply, wrong content. More room cannot help, and
                    # climbing would triple the time to the same refusal.
                    break
        if result is None:
            raise RuntimeError(
                "[verbatim_translation] the passage could not be translated "
                f"(batch of {want} line(s) starting at entry {indices[0]}; "
                f"{last_exc.attempts} attempt(s) at each of "
                f"{rungs_tried} output budget(s) tried, the last of them "
                f"{budget * _OUTPUT_BUDGET_GROWTH[rungs_tried - 1]} tokens; "
                f"last error: {last_exc.last_error}). "
                "An English verbatim row cannot ship on a native episode."
            ) from last_exc
        attempts_total += attempts or 1
        for i, text in zip(indices, result.texts):
            cleaned = strip_echoed_label(str(text), entries[i].speaker,
                                         entries[i].text)
            translated[i] = replace(entries[i], text=" ".join(cleaned.split()))
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
    "strip_echoed_label",
    "translate_entries",
    "translate_plan",
]
