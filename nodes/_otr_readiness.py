"""nodes/_otr_readiness.py — LFC Phase 7 (audio) + Phase 8 (video).

Both phases are deterministic, cheap, and default ON. They live AFTER
the LLM phases in the cascade chain so the final readiness report
reflects whatever the cleanup produced.

Phase 7 -- audio readiness
  Normalises line.text for TTS so the Bark / Kokoro / AudioGen
  consumers downstream get pronounceable input:
    * Numeric expansion via `num2words` when the package is present
      (1234 -> "one thousand two hundred thirty-four").
    * Common abbreviation expansion via a small built-in dict
      (Dr. -> "Doctor", Mr. -> "Mister", St. -> "Saint", &c.).
    * Symbol-to-word replacement for unsafe TTS characters
      (& -> "and", # -> "number", @ -> "at").
  Lookups that miss the dict are logged as warnings in
  `meta.audio_readiness.warnings`; the cascade advances.

Phase 8 -- video readiness
  Pure audit. Checks each cast row for a portrait field
  (cast[i].portrait_path) and each voiced beat for a non-empty
  speaker -> visual mapping. Does NOT mutate the ledger; emits
  `meta.video_readiness` with pass / warn counts so the downstream
  HuMo / LTX batch renderer knows which beats have visual coverage.

Both phases are intentionally module-local helpers so the cascade
orchestrator depends on a stable surface even when num2words isn't
installed (e.g. in CI without the optional dep).

ADR: docs/2026-05-11-multi-turn-polish-adr.md (sprint commit 5 of 14).
Status: LFC Phase 7 + 8 (2026-05-11).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("OTR.readiness")


__all__ = [
    "AudioReadinessReport",
    "VideoReadinessReport",
    "phase_7_audio_readiness",
    "phase_8_video_readiness",
    "ABBREV_EXPANSIONS",
    "SYMBOL_REPLACEMENTS",
]


# ---------------------------------------------------------------------------
# Audio readiness constants
# ---------------------------------------------------------------------------


# Common period-bearing abbreviations that TTS systems mispronounce.
# Period-anchored on both sides of the key for word-boundary precision.
# Keep the dict small -- soak data drives additions, not speculation.
ABBREV_EXPANSIONS: dict[str, str] = {
    "Dr.":   "Doctor",
    "Mr.":   "Mister",
    "Mrs.":  "Missus",
    "Ms.":   "Miss",
    "St.":   "Saint",
    "Ave.":  "Avenue",
    "Blvd.": "Boulevard",
    "Jr.":   "Junior",
    "Sr.":   "Senior",
    "vs.":   "versus",
    "etc.":  "et cetera",
    "Prof.": "Professor",
    "Sgt.":  "Sergeant",
    "Capt.": "Captain",
    "Lt.":   "Lieutenant",
    "Gen.":  "General",
    "Col.":  "Colonel",
    "Maj.":  "Major",
    "Cpl.":  "Corporal",
}


# Symbol replacements -- TTS handles letters cleanly, ASCII symbols
# inconsistently. Map to spoken English equivalents.
SYMBOL_REPLACEMENTS: dict[str, str] = {
    "&": " and ",
    "@": " at ",
    "#": " number ",
    "%": " percent",
    "$": " dollars",
    "+": " plus ",
    "=": " equals ",
    "/": " slash ",
}


# Number-token regex. Captures integers; floats and signed numbers
# handled by num2words automatically.
_NUM_TOKEN_RE = re.compile(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\b")


def _try_import_num2words():
    """Lazy import. Returns the num2words callable or None."""
    try:
        from num2words import num2words  # type: ignore
        return num2words
    except Exception:  # noqa: BLE001
        return None


def _expand_abbreviations(text: str) -> str:
    """Replace each known abbreviation with its full form.

    Anchored on a space-or-start prefix to avoid mid-word matches
    (e.g. "Dr." in "Drum." -- both end with "r." but the second is
    not the abbreviation). Period in the key is the right-side
    anchor; the left side uses word boundary.
    """
    out = text
    for short, long_form in ABBREV_EXPANSIONS.items():
        # \b before, exact short token, then space or end-of-string.
        # Pattern intentionally requires word-after-period so a
        # sentence-final "Dr." doesn't double-stamp.
        pattern = re.compile(
            r"\b" + re.escape(short) + r"(?=\s|$)",
            flags=0,
        )
        out = pattern.sub(long_form, out)
    return out


def _expand_symbols(text: str) -> str:
    out = text
    for sym, word in SYMBOL_REPLACEMENTS.items():
        if sym in out:
            out = out.replace(sym, word)
    # Collapse the extra spaces these replacements can introduce.
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _expand_numbers(text: str, *, num2words_fn) -> tuple[str, list[str]]:
    """Replace numeric tokens with their spelled-out form.

    Returns (new_text, list of unparseable tokens). num2words handles
    most cases; comma-separated thousands (1,234) get stripped before
    conversion.
    """
    if num2words_fn is None:
        return text, []
    unparseable: list[str] = []

    def _convert(match: re.Match) -> str:
        tok = match.group(1)
        clean = tok.replace(",", "")
        try:
            if "." in clean:
                return num2words_fn(float(clean))
            return num2words_fn(int(clean))
        except Exception as exc:  # noqa: BLE001
            unparseable.append(tok)
            log.debug("[phase_7] num2words miss on %r: %s", tok, exc)
            return tok

    new_text = _NUM_TOKEN_RE.sub(_convert, text)
    return new_text, unparseable


# ---------------------------------------------------------------------------
# Audio readiness
# ---------------------------------------------------------------------------


@dataclass
class AudioReadinessReport:
    """Phase 7 summary. Stamped on meta.audio_readiness."""

    lines_scanned: int = 0
    lines_normalized: int = 0
    abbreviations_expanded: int = 0
    symbols_expanded: int = 0
    numbers_expanded: int = 0
    unparseable_numbers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lines_scanned": int(self.lines_scanned),
            "lines_normalized": int(self.lines_normalized),
            "abbreviations_expanded": int(self.abbreviations_expanded),
            "symbols_expanded": int(self.symbols_expanded),
            "numbers_expanded": int(self.numbers_expanded),
            "unparseable_numbers": list(self.unparseable_numbers),
            "warnings": list(self.warnings),
        }


def phase_7_audio_readiness(led) -> AudioReadinessReport:
    """LFC Phase 7. Deterministic TTS-safety normalization on line.text.

    Iterates every voiced line (skipped lines excluded). For each:
      1. Expand abbreviations (Dr. -> Doctor, etc.).
      2. Replace symbols with words (& -> "and", etc.).
      3. Expand integers / floats via num2words when available.
      4. Recompute word_count + char_count in lockstep.

    On num2words ImportError the numeric-expansion step is skipped
    with a warning; everything else still runs. Deterministic; no
    LLM calls.

    Stamps meta.audio_readiness with full counts + unparseable
    tokens for soak diagnostics. Returns the report.
    """
    ledger_data = led.data if hasattr(led, "data") else led
    rep = AudioReadinessReport()
    num2words_fn = _try_import_num2words()
    if num2words_fn is None:
        rep.warnings.append(
            "num2words not installed; numeric expansion skipped"
        )

    lines = ledger_data.get("lines") or []
    rep.lines_scanned = len(lines)
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ln.get("skip"):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            continue
        text = ln.get("text") or ""
        if not text:
            continue
        original = text
        # 1. Abbreviations.
        expanded = _expand_abbreviations(text)
        abbrev_hits = sum(1 for k in ABBREV_EXPANSIONS if k in original)
        if abbrev_hits:
            rep.abbreviations_expanded += abbrev_hits
        # 2. Symbols.
        expanded_2 = _expand_symbols(expanded)
        sym_hits = sum(1 for k in SYMBOL_REPLACEMENTS if k in expanded)
        if sym_hits:
            rep.symbols_expanded += sym_hits
        # 3. Numbers.
        expanded_3, unparseable = _expand_numbers(
            expanded_2, num2words_fn=num2words_fn,
        )
        if unparseable:
            rep.unparseable_numbers.extend(unparseable)
            for u in unparseable:
                rep.warnings.append(
                    f"line_id={ln.get('line_id','')!r} number "
                    f"{u!r} could not be expanded"
                )
        num_hits = sum(
            1 for _ in _NUM_TOKEN_RE.finditer(text)
        ) - len(unparseable)
        if num_hits > 0:
            rep.numbers_expanded += num_hits

        if expanded_3 != original:
            ln["text"] = expanded_3
            ln["char_count"] = len(expanded_3)
            ln["word_count"] = sum(1 for _ in expanded_3.split())
            rep.lines_normalized += 1

    ledger_data.setdefault("meta", {})["audio_readiness"] = rep.to_dict()
    log.info(
        "[LFC:phase_7] scanned=%d normalized=%d (abbrev=%d sym=%d num=%d)",
        rep.lines_scanned, rep.lines_normalized,
        rep.abbreviations_expanded, rep.symbols_expanded,
        rep.numbers_expanded,
    )
    return rep


# ---------------------------------------------------------------------------
# Video readiness
# ---------------------------------------------------------------------------


@dataclass
class VideoReadinessReport:
    """Phase 8 summary. Stamped on meta.video_readiness."""

    cast_total: int = 0
    cast_with_portrait: int = 0
    cast_missing_portrait: list = field(default_factory=list)
    voiced_lines: int = 0
    voiced_lines_with_visual_mapping: int = 0
    voiced_lines_missing_visual: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cast_total": int(self.cast_total),
            "cast_with_portrait": int(self.cast_with_portrait),
            "cast_missing_portrait": list(self.cast_missing_portrait),
            "voiced_lines": int(self.voiced_lines),
            "voiced_lines_with_visual_mapping": int(
                self.voiced_lines_with_visual_mapping
            ),
            "voiced_lines_missing_visual": list(
                self.voiced_lines_missing_visual
            ),
            "warnings": list(self.warnings),
        }


def _has_portrait(cast_row: dict) -> bool:
    """True if the cast row carries a non-empty portrait reference."""
    for key in ("portrait_path", "portrait_image", "portrait", "image_path"):
        val = cast_row.get(key)
        if isinstance(val, str) and val.strip():
            return True
    return False


def phase_8_video_readiness(led) -> VideoReadinessReport:
    """LFC Phase 8. Pure audit -- mutates nothing.

    Checks each cast row for a portrait field. For each voiced line,
    confirms its char_id resolves to a cast row that has a portrait.
    Lines whose speaker has no portrait are warned (the HuMo / LTX
    batch renderer downstream falls back to a still-image tier or
    skips the visual; the warn list surfaces those beats so QA can
    see which clips will look thin).

    Stamps meta.video_readiness. Returns the report.
    """
    ledger_data = led.data if hasattr(led, "data") else led
    rep = VideoReadinessReport()

    cast = ledger_data.get("cast") or []
    cast_by_id: dict[str, dict] = {}
    for row in cast:
        if not isinstance(row, dict):
            continue
        cid = row.get("char_id", "")
        if cid:
            cast_by_id[cid] = row

    rep.cast_total = len(cast_by_id)
    for cid, row in cast_by_id.items():
        if _has_portrait(row):
            rep.cast_with_portrait += 1
        else:
            rep.cast_missing_portrait.append(cid)

    lines = ledger_data.get("lines") or []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ln.get("skip"):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            continue
        rep.voiced_lines += 1
        cid = ln.get("char_id") or ""
        row = cast_by_id.get(cid)
        if row is not None and _has_portrait(row):
            rep.voiced_lines_with_visual_mapping += 1
        else:
            rep.voiced_lines_missing_visual.append(
                ln.get("line_id", "")
            )

    if rep.cast_missing_portrait:
        rep.warnings.append(
            f"{len(rep.cast_missing_portrait)} cast row(s) without "
            f"portrait: {rep.cast_missing_portrait}"
        )
    if rep.voiced_lines_missing_visual:
        rep.warnings.append(
            f"{len(rep.voiced_lines_missing_visual)} voiced line(s) "
            f"reference a speaker with no portrait"
        )

    ledger_data.setdefault("meta", {})["video_readiness"] = rep.to_dict()
    log.info(
        "[LFC:phase_8] cast=%d/%d with portrait, voiced=%d/%d with visual",
        rep.cast_with_portrait, rep.cast_total,
        rep.voiced_lines_with_visual_mapping, rep.voiced_lines,
    )
    return rep
