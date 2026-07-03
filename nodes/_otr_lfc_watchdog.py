"""nodes/_otr_lfc_watchdog.py — VRAM watchdog + stats helpers (ADR 6.5 / 6.8).

Cheap module-local helpers that the LFC orchestrator uses to gate
LLM calls and detect voice drift.

ADR section 6.5  Stats-based voice drift detection
  Per-speaker stats computed in O(n) over the ledger lines:
      mean_line_length  -- avg word_count per speaker
      vocab_diversity   -- unique_tokens / total_tokens (Type-Token
                            Ratio; classic linguistic measure)
  flag_line_drift returns True when a line is OUT-of-character by
  either the 40% word-count deviation rule OR the 60% vocab-
  diversity collapse rule (ADR thresholds).

ADR section 6.8  VRAM telemetry
  `_torch_vram_allocated_gb` reads current CUDA-allocated VRAM as a
  forensic sample stamped on meta at cascade entry. TELEMETRY ONLY --
  no ceiling policy (the operator's tier JSON owns the OOM budget) and
  no quantization / weight-streaming / FA chasing (per Jeffrey's
  feedback_no_vram_dragons memory).

Both helpers are stateless. Tests cover the math without needing
torch / a real GPU; the reader has a `_torch_vram_allocated_gb`
internal that returns 0.0 when torch is unimportable.

Status: LFC sprint commit 12 of 14 (2026-05-11).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional

log = logging.getLogger("OTR.lfc_watchdog")


__all__ = [
    "SpeakerStats",
    "DRIFT_WORD_COUNT_THRESHOLD",
    "DRIFT_VOCAB_DIVERSITY_THRESHOLD",
    "compute_speaker_stats",
    "flag_line_drift",
    "_torch_vram_allocated_gb",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# ADR section 6.5 thresholds.
DRIFT_WORD_COUNT_THRESHOLD: float = 0.40  # >40% deviation
DRIFT_VOCAB_DIVERSITY_THRESHOLD: float = 0.60  # <60% of mean diversity


# Tokenization regex for Type-Token Ratio. Words = alphanumeric runs.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


# ---------------------------------------------------------------------------
# Speaker stats
# ---------------------------------------------------------------------------


@dataclass
class SpeakerStats:
    char_id: str
    line_count: int
    total_words: int
    mean_line_length: float
    vocab_diversity: float  # TTR: unique tokens / total tokens

    def to_dict(self) -> dict:
        return {
            "char_id": self.char_id,
            "line_count": self.line_count,
            "total_words": self.total_words,
            "mean_line_length": self.mean_line_length,
            "vocab_diversity": self.vocab_diversity,
        }


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def compute_speaker_stats(
    lines: Iterable[dict],
) -> dict[str, SpeakerStats]:
    """Per-speaker stats over the ledger lines.

    Returns a dict keyed by char_id. Only non-skipped voiced beats
    (character + announcer) contribute. Speakers with zero qualifying
    lines are not included.
    """
    by_speaker: dict[str, list[str]] = {}
    for ln in lines or []:
        if not isinstance(ln, dict):
            continue
        if ln.get("skip"):
            continue
        role = (ln.get("speaker_role") or "").strip().lower()
        if role not in ("character", "announcer"):
            continue
        cid = ln.get("char_id") or ""
        if not cid:
            continue
        text = ln.get("text") or ""
        if not text:
            continue
        by_speaker.setdefault(cid, []).append(text)

    out: dict[str, SpeakerStats] = {}
    for cid, texts in by_speaker.items():
        all_tokens: list[str] = []
        line_word_counts: list[int] = []
        for t in texts:
            toks = _tokenize(t)
            all_tokens.extend(toks)
            line_word_counts.append(len(toks))
        total_words = sum(line_word_counts)
        mean_len = (
            total_words / len(line_word_counts)
            if line_word_counts else 0.0
        )
        ttr = (
            len(set(all_tokens)) / total_words
            if total_words else 0.0
        )
        out[cid] = SpeakerStats(
            char_id=cid,
            line_count=len(line_word_counts),
            total_words=total_words,
            mean_line_length=mean_len,
            vocab_diversity=ttr,
        )
    return out


def flag_line_drift(
    line_text: str,
    speaker_stats: SpeakerStats,
    *,
    word_count_threshold: float = DRIFT_WORD_COUNT_THRESHOLD,
    vocab_diversity_threshold: float = DRIFT_VOCAB_DIVERSITY_THRESHOLD,
) -> tuple[bool, str]:
    """Return (drifted, reason).

    Per ADR section 6.5: flag a line when EITHER:
      * abs(line.word_count - mean_line_length) / mean_line_length
        > word_count_threshold, OR
      * line_diversity < speaker_mean_diversity *
        vocab_diversity_threshold.

    Returns (False, "") when stats are unavailable / mean is zero so
    the caller advances cleanly. Reason text is the human-readable
    diagnostic stamped on meta.
    """
    if speaker_stats.mean_line_length <= 0:
        return False, ""
    toks = _tokenize(line_text)
    if not toks:
        return False, ""
    wc = len(toks)
    deviation = (
        abs(wc - speaker_stats.mean_line_length)
        / speaker_stats.mean_line_length
    )
    if deviation > word_count_threshold:
        return True, (
            f"word_count={wc} deviates "
            f"{deviation:.0%} from speaker mean "
            f"{speaker_stats.mean_line_length:.1f} "
            f"(threshold {word_count_threshold:.0%})"
        )
    if speaker_stats.vocab_diversity > 0:
        line_diversity = (
            len(set(toks)) / max(1, wc)
        )
        floor = (
            speaker_stats.vocab_diversity * vocab_diversity_threshold
        )
        if line_diversity < floor:
            return True, (
                f"line_diversity={line_diversity:.2f} "
                f"below speaker mean "
                f"{speaker_stats.vocab_diversity:.2f} * "
                f"{vocab_diversity_threshold:.0%} "
                f"= floor {floor:.2f}"
            )
    return False, ""


# ---------------------------------------------------------------------------
# VRAM watchdog
# ---------------------------------------------------------------------------


def _torch_vram_allocated_gb() -> float:
    """Return CUDA-allocated VRAM in GB, or 0.0 when unavailable.

    Defensive: any failure (torch unimportable, no CUDA device,
    driver error) returns 0.0 so the watchdog never blocks on the
    measurement itself.
    """
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return 0.0
        return float(torch.cuda.memory_allocated()) / 1024**3
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "[LFC:watchdog] VRAM allocation read failed: %s; "
            "returning 0.0", exc,
        )
        return 0.0
