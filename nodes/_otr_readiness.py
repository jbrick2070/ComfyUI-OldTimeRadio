"""nodes/_otr_readiness.py — LFC Phase 7 (audio) + Phase 8 (video).

Both phases are deterministic, cheap, and default ON. They live AFTER
the LLM phases in the cascade chain so the final readiness report
reflects whatever the cleanup produced.

Phase 7 -- audio readiness
  Projects canonical line text into a separate text_for_tts field so
  Bark / Kokoro / AudioGen consumers get pronounceable input without
  changing the authored ledger:
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

import hashlib
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("OTR.readiness")


__all__ = [
    "AudioReadinessReport",
    "VideoReadinessReport",
    "phase_7_audio_readiness",
    "phase_8_video_readiness",
    "ABBREV_EXPANSIONS",
    "SYMBOL_REPLACEMENTS",
    "normalize_for_delivery",
    "text_for_tts_source_sha256",
    "stamp_text_for_tts_delivery",
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
    """LFC Phase 7. Deterministic, non-destructive TTS normalization.

    Every non-skipped voiced line receives a complete text_for_tts
    projection, a canonical-source hash, and a normalization receipt.
    Canonical text, word counts, and proof ownership are never changed.
    """
    ledger_data = led.data if hasattr(led, "data") else led
    rep = AudioReadinessReport()
    if _try_import_num2words() is None:
        rep.warnings.append(
            "num2words not installed; numeric expansion skipped"
        )

    lines = ledger_data.get("lines") or []
    rep.lines_scanned = len(lines)
    for ln in lines:
        if not isinstance(ln, dict) or ln.get("skip"):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            continue
        canonical = str(ln.get("text") or "")
        if not canonical:
            continue

        delivery, receipt = normalize_for_delivery(canonical)
        rep.abbreviations_expanded += int(receipt["abbreviations_expanded"])
        rep.symbols_expanded += int(receipt["symbols_expanded"])
        rep.numbers_expanded += int(receipt["numbers_expanded"])
        unparseable = list(receipt["unparseable_numbers"])
        rep.unparseable_numbers.extend(unparseable)
        for token in unparseable:
            rep.warnings.append(
                f"line_id={ln.get('line_id','')!r} number "
                f"{token!r} could not be expanded"
            )

        ln["text_for_tts"] = delivery
        ln["text_for_tts_source_sha256"] = text_for_tts_source_sha256(
            canonical
        )
        ln["text_for_tts_receipt"] = receipt
        if receipt["changed"]:
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
# Content-owned delivery stamping (720-bakeoff C2 / S2 P1.3)
# ---------------------------------------------------------------------------
#
# Legacy lanes (science_news, public_domain, ...) get their pronunciation
# normalization (Dr. -> Doctor, digits -> words) from
# `phase_7_audio_readiness`. The freeze cascade SKIPS Phase 7 under
# `content_owned_readonly` (r2 P0.1), which switched that normalization
# OFF for scifi_news_pro; C2 restores it on the content-owned boundary instead --
# the SAME normalization stamped onto a separate `text_for_tts` delivery
# field (+ a source-text sha for staleness detection + a receipt), leaving
# `text`/counts/proof pristine.
#
# CORRECTION (2026-08-15): this note used to say Phase 7 "REWRITES
# canonical `line.text` in place", and offered that as the reason it is
# skipped. It does not. `:242-246` writes only `text_for_tts`, its source
# hash and its receipt -- exactly what `stamp_text_for_tts_delivery` below
# writes. NEITHER function mutates canonical text, and the distinction is
# load-bearing rather than tidy: the D2 clean window reconciles the
# acceptance proof immediately BEFORE this stamp, and that ordering is
# only safe because nothing here can move a hash the proof covers. A
# reader who believes the old claim goes hunting for a mutator that does
# not exist.
#
# The voice node speaks `text_for_tts` via the delivery resolver
# (`_otr_text_delivery`); the C1 durable-identity merge keeps the stamp
# alive across incremental saves only while the canonical text is
# unchanged (a changed line invalidates it -> the resolver then fails loud
# rather than speaking stale delivery text).


def text_for_tts_source_sha256(canonical_text: str) -> str:
    """SHA256 hex of the canonical line text -- the staleness key stored
    as ``text_for_tts_source_sha256`` and re-checked by the delivery
    resolver. Raw-UTF8 bytes so the stamp side and the resolver side are
    trivially identical."""
    return hashlib.sha256((canonical_text or "").encode("utf-8")).hexdigest()


def normalize_for_delivery(text: str) -> "tuple[str, dict]":
    """Compute the TTS DELIVERY text from a canonical line WITHOUT
    mutating anything: the same deterministic normalization
    `phase_7_audio_readiness` applies (abbreviations -> full words,
    symbols -> words, digits -> num2words), returned as a value plus a
    per-line receipt. Pure + deterministic -> C7-safe. When num2words is
    absent the numeric step is skipped (receipt records it), matching the
    legacy phase-7 degradation exactly."""
    original = text or ""
    num2words_fn = _try_import_num2words()
    expanded = _expand_abbreviations(original)
    abbrev_hits = sum(1 for k in ABBREV_EXPANSIONS if k in original)
    expanded_2 = _expand_symbols(expanded)
    sym_hits = sum(1 for k in SYMBOL_REPLACEMENTS if k in expanded)
    expanded_3, unparseable = _expand_numbers(
        expanded_2, num2words_fn=num2words_fn,
    )
    num_hits = sum(1 for _ in _NUM_TOKEN_RE.finditer(original)) - len(unparseable)
    receipt = {
        "abbreviations_expanded": int(abbrev_hits),
        "symbols_expanded": int(sym_hits),
        "numbers_expanded": int(max(0, num_hits)),
        "unparseable_numbers": list(unparseable),
        "num2words_available": num2words_fn is not None,
        "changed": expanded_3 != original,
    }
    return expanded_3, receipt


def stamp_text_for_tts_delivery(led) -> dict:
    """Stamp a complete, non-destructive TTS projection on voiced lines.

    The projection expands pronunciation-only abbreviations, symbols, and
    numbers. It never judges prose, substitutes content, or changes canonical
    text and therefore cannot reject an episode.
    """
    ledger_data = led.data if hasattr(led, "data") else led
    stamped = 0
    changed = 0
    for ln in ledger_data.get("lines") or []:
        if not isinstance(ln, dict) or ln.get("skip"):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            continue
        canonical = str(ln.get("text") or "")
        if not canonical.strip():
            continue
        delivery, receipt = normalize_for_delivery(canonical)
        ln["text_for_tts"] = delivery
        ln["text_for_tts_source_sha256"] = text_for_tts_source_sha256(
            canonical
        )
        ln["text_for_tts_receipt"] = receipt
        stamped += 1
        if receipt["changed"]:
            changed += 1

    summary = {
        "mode": "content_owned_delivery_stamp",
        "lines_stamped": int(stamped),
        "lines_delivery_differs": int(changed),
    }
    ledger_data.setdefault("meta", {})["text_for_tts_delivery"] = summary
    log.info(
        "[delivery-stamp] stamped=%d (delivery-differs=%d)", stamped, changed,
    )
    return summary


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
