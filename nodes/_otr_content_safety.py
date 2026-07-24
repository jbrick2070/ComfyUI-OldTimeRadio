"""Narrow, shared spoken-content safety policy.

This module is intentionally not a general prose classifier. It recognizes only
three operator-authorized terminal categories with whole-word matching:
profanity, explicit guns/knives/weapons, and explicit sexual/nudity terms.
Smoking, style, genre, visual vocabulary, and ordinary English are out of
scope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import re
from typing import Any, Iterable, Mapping, MutableMapping

from pydantic import BaseModel, Field, model_validator

try:
    from ._otr_script_prep import clean_spoken_text
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_script_prep import clean_spoken_text  # type: ignore


PROFANITY_TERMS: tuple[str, ...] = (
    "damn",
    "damnit",
    "dammit",
    "hell",
    "goddamn",
    "godammit",
    "goddammit",
    "shit",
    "bullshit",
    "bitch",
    "bastard",
    "ass",
    "asshole",
    "fuck",
    "fucking",
    "fucker",
    "piss",
    "screw you",
    "screw this",
    "screw that",
)

EXPLICIT_WEAPON_TERMS: tuple[str, ...] = (
    "gun",
    "guns",
    "handgun",
    "handguns",
    "pistol",
    "pistols",
    "revolver",
    "revolvers",
    "rifle",
    "rifles",
    "shotgun",
    "shotguns",
    "firearm",
    "firearms",
    "knife",
    "knives",
    "dagger",
    "daggers",
    "switchblade",
    "switchblades",
    "weapon",
    "weapons",
)

EXPLICIT_NUDITY_TERMS: tuple[str, ...] = (
    "nude",
    "nudity",
    "naked",
    "porn",
    "pornographic",
    "pornography",
    "explicit sex",
    "sexual intercourse",
)

SPOKEN_ROLES = frozenset({"character", "announcer"})


class _SafetyLinePatch(BaseModel):
    line_id: str = Field(min_length=1)
    replacement_text: str = Field(min_length=1)


class _SafetyPatchSet(BaseModel):
    patches: list[_SafetyLinePatch]

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [patch.line_id for patch in self.patches]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate line_id in safety patch set")
        return self


@dataclass(frozen=True)
class SafetyHit:
    line_id: str
    line_index: int
    category: str
    term: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profanity_terms() -> tuple[str, ...]:
    """Return the global narrow profanity vocabulary."""
    return PROFANITY_TERMS


@lru_cache(maxsize=16)
def _pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted(set(terms), key=len, reverse=True)
    if not ordered:
        return re.compile(r"(?!x)x")
    body = "|".join(re.escape(term) for term in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def find_text_hits(text: Any) -> tuple[tuple[str, str], ...]:
    """Return unique category/matched-term pairs in stable order."""
    surface = str(text or "")
    if not surface:
        return ()
    categories = (
        ("profanity", profanity_terms()),
        ("weapon", EXPLICIT_WEAPON_TERMS),
        ("sexual_nudity", EXPLICIT_NUDITY_TERMS),
    )
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for category, terms in categories:
        for match in _pattern(tuple(terms)).finditer(surface):
            row = (category, match.group(0).casefold())
            if row not in seen:
                seen.add(row)
                found.append(row)
    return tuple(found)


def scan_spoken_ledger(ledger_data: Mapping[str, Any]) -> tuple[SafetyHit, ...]:
    """Scan the exact non-skipped character/announcer projection."""
    hits: list[SafetyHit] = []
    lines = ledger_data.get("lines") if isinstance(ledger_data, Mapping) else ()
    for index, row in enumerate(lines or ()):
        if not isinstance(row, Mapping):
            continue
        if bool(row.get("skip")):
            continue
        if str(row.get("speaker_role") or "") not in SPOKEN_ROLES:
            continue
        line_id = str(row.get("line_id") or f"lines[{index}]")
        for category, term in find_text_hits(row.get("text")):
            hits.append(SafetyHit(line_id, index, category, term))
    return tuple(hits)


def format_safety_hits(hits: Iterable[SafetyHit], *, limit: int = 8) -> str:
    rows = list(hits)
    sample = rows[: max(1, int(limit))]
    rendered = "; ".join(
        f"{hit.line_id}: {hit.category}={hit.term!r}"
        for hit in sample
    )
    if len(rows) > len(sample):
        rendered += f" (+{len(rows) - len(sample)} more)"
    return rendered


def propose_safety_patches(
    ledger_data: Mapping[str, Any],
    slot_fn,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Generate one atomic same-story patch set for current safety hits."""
    hits = scan_spoken_ledger(ledger_data)
    if not hits:
        return {}, {"status": "clean", "hits": [], "patch_count": 0}

    try:
        from ._otr_structured_call import structured_call
    except ImportError:  # pragma: no cover
        from _otr_structured_call import structured_call  # type: ignore

    hit_ids = {hit.line_id for hit in hits}
    lines_by_id = {
        str(row.get("line_id") or ""): row
        for row in (ledger_data.get("lines") or ())
        if isinstance(row, Mapping)
    }
    payload = [
        {
            "line_id": line_id,
            "speaker_role": str(lines_by_id[line_id].get("speaker_role") or ""),
            "text": str(lines_by_id[line_id].get("text") or ""),
            "hits": [
                {"category": hit.category, "term": hit.term}
                for hit in hits
                if hit.line_id == line_id
            ],
        }
        for line_id in sorted(hit_ids)
    ]
    prompt = [
        {
            "role": "system",
            "content": (
                "You repair explicit safety terms in an accepted radio story. "
                "Return JSON only. Preserve the exact story event, meaning, "
                "speaker, cast, line identity, order, canon, and tone. Change "
                "only the listed line text. Do not add stage directions, role "
                "labels, profanity, explicit guns/knives/weapons, or explicit "
                "sexual/nudity language."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return {\"patches\":[{\"line_id\":...,"
                "\"replacement_text\":...}]} with exactly one patch for "
                "each listed line_id. Lines: "
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
        },
    ]

    def _validate(result: _SafetyPatchSet) -> str | None:
        patches = {patch.line_id: patch.replacement_text for patch in result.patches}
        if set(patches) != hit_ids:
            return (
                "patch line_ids must equal the exact requested set: "
                + ", ".join(sorted(hit_ids))
            )
        for line_id, text in patches.items():
            if not str(text or "").strip():
                return f"replacement for {line_id} is empty"
            if not clean_spoken_text(str(text)):
                return (
                    f"replacement for {line_id} has no spoken surface after "
                    "stage-direction cleanup"
                )
            residual = find_text_hits(text)
            if residual:
                return f"replacement for {line_id} remains unsafe: {residual}"
        return None

    try:
        # LLM slot: technical -- atomic replacement of explicitly unsafe prose.
        result = structured_call(
            prompt=prompt,
            schema=_SafetyPatchSet,
            slot_fn=slot_fn,
            base_temperature=0.25,
            structural_retry_temperature=0.1,
            post_validator=_validate,
            max_new_tokens=max(256, min(2048, 128 * len(hit_ids))),
            max_attempts=3,
            helper_name="spoken_safety_cleanup",
        )
    except Exception as exc:  # residual validation remains fail-closed
        return {}, {
            "status": "cleanup_failed",
            "hits": [hit.to_dict() for hit in hits],
            "patch_count": 0,
            "error": f"{type(exc).__name__}: {exc}"[:600],
        }
    patches = {
        patch.line_id: patch.replacement_text.strip()
        for patch in result.patches
    }
    return patches, {
        "status": "patched",
        "hits": [hit.to_dict() for hit in hits],
        "patch_count": len(patches),
    }


def apply_safety_cleanup(
    ledger_data: MutableMapping[str, Any],
    slot_fn,
) -> dict[str, Any]:
    """Atomically apply a validated safety-only patch set to ledger rows."""
    patches, receipt = propose_safety_patches(ledger_data, slot_fn)
    if not patches:
        return receipt
    try:
        from ._otr_text_metrics import set_line_text_metrics
    except ImportError:  # pragma: no cover
        from _otr_text_metrics import set_line_text_metrics  # type: ignore
    rows = {
        str(row.get("line_id") or ""): row
        for row in (ledger_data.get("lines") or ())
        if isinstance(row, MutableMapping)
    }
    if set(patches) - set(rows):
        return {
            **receipt,
            "status": "apply_failed",
            "error": "validated line_id disappeared before atomic apply",
        }
    empty_spoken = [
        line_id for line_id, replacement in patches.items()
        if not clean_spoken_text(str(replacement))
    ]
    if empty_spoken:
        return {
            **receipt,
            "status": "apply_failed",
            "error": (
                "replacement has no spoken surface after stage-direction "
                "cleanup: " + ", ".join(sorted(empty_spoken))
            ),
        }
    for line_id, replacement in patches.items():
        set_line_text_metrics(rows[line_id], replacement)
    empty_rows = [
        line_id for line_id in patches
        if not clean_spoken_text(str(rows[line_id].get("text") or ""))
    ]
    if empty_rows:
        raise RuntimeError(
            "safety cleanup invariant failed after atomic apply: empty spoken "
            "surface in " + ", ".join(sorted(empty_rows))
        )
    residual = scan_spoken_ledger(ledger_data)
    if residual:
        raise RuntimeError(
            "safety cleanup invariant failed after validated atomic apply: "
            + format_safety_hits(residual)
        )
    return receipt


__all__ = [
    "EXPLICIT_NUDITY_TERMS",
    "PROFANITY_TERMS",
    "EXPLICIT_WEAPON_TERMS",
    "SPOKEN_ROLES",
    "SafetyHit",
    "apply_safety_cleanup",
    "find_text_hits",
    "format_safety_hits",
    "profanity_terms",
    "propose_safety_patches",
    "scan_spoken_ledger",
]
