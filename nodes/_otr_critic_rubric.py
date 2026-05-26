r"""Sprint 10A step 7 -- Whole-episode critic rubric loader.

Parses docs/2026-05-26-sprint-10a-whole-episode-critic-rubric.md
(landed in step 1, commit c5cb09c) into structured Python objects
that the critic prompt builder can consume.

Per the rubric doc's 'Programmatic-consumption contract':
  * '## Axes' is the header that begins the axis list.
  * Each axis is an H3 (### N. <axis_name>). The integer prefix is
    the axis ordinal; the name (lowercased, spaces -> underscores,
    ASCII letters only) is the audit-trail JSON key.
  * The first paragraph after the H3 starting '**What it checks:**'
    is the axis definition.
  * Lines starting with '- **N -- <label>.**' are the anchor
    descriptions for scores 1-5.
  * '## Ship / discard threshold' defines the ship rule; the loader
    extracts the three threshold conditions as a structured list
    (no LLM call -- pure regex / markdown parse).

This module is PURE: no LLM, no I/O beyond reading the rubric
markdown file once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default path to the rubric markdown. Resolves relative to this file
# so the loader works in test + production contexts identically.
_DEFAULT_RUBRIC_PATH: Path = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "2026-05-26-sprint-10a-whole-episode-critic-rubric.md"
)


# ---------------------------------------------------------------------------
# Structured types
# ---------------------------------------------------------------------------


@dataclass
class RubricAxis:
    """One scoring axis."""

    ordinal: int                # 1..10
    name: str                   # original heading text (mixed case)
    json_key: str               # lowercased + underscored ASCII slug
    definition: str             # 'What it checks' paragraph
    anchors: List[str] = field(default_factory=list)  # exactly 5 entries


@dataclass
class ShipThreshold:
    """Ship/discard threshold rule.

    Per the rubric doc, the verdict is 'ship' IFF:
      1. No axis scores below min_axis_score.
      2. Mean of all axes scores >= min_mean_score.
      3. The SFW axis specifically is >= sfw_axis_min.
    """

    min_axis_score: int = 3
    min_mean_score: float = 3.5
    sfw_axis_key: str = "sfw_adherence"
    sfw_axis_min: int = 4


@dataclass
class Rubric:
    """The complete rubric, parsed from the markdown source."""

    axes: List[RubricAxis] = field(default_factory=list)
    threshold: ShipThreshold = field(default_factory=ShipThreshold)
    source_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers -- markdown parsing
# ---------------------------------------------------------------------------


_AXIS_HEADER_RE = re.compile(
    r"^###\s+(\d+)\.\s+(.+?)\s*$",
)
"""Matches: '### 1. Premise clarity' -> (ordinal=1, name='Premise clarity')."""

# Anchor lines look like: '- **1 -- Incoherent.** No discernible...'
# We accept both em-dash + ASCII double-dash variants because earlier
# tooling has been known to rewrite one to the other.
_ANCHOR_LINE_RE = re.compile(
    r"^-\s+\*\*(\d+)\s*[—\-]+\s*(.+?)\*\*\s*(.*?)$",
)


def _slugify(name: str) -> str:
    """Lowercase + ASCII-letters + underscore slug.

    Word-boundary characters (space + hyphen) become underscores.
    Other non-letters drop. So:
      'Audio-readiness'        -> 'audio_readiness'
      'Character distinctiveness' -> 'character_distinctiveness'
      'SFW adherence'          -> 'sfw_adherence'
    """
    keep = []
    for ch in name:
        if ch.isalpha():
            keep.append(ch)
        elif ch in (" ", "-"):
            keep.append("_")
    s = "".join(keep).strip("_").lower()
    # Collapse consecutive underscores ('Audio - readiness' -> 'audio_readiness')
    while "__" in s:
        s = s.replace("__", "_")
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_rubric(rubric_path: Optional[Path] = None) -> Rubric:
    """Parse the rubric markdown and return a Rubric object.

    Args:
        rubric_path: optional override (test fixtures pass tmp paths).
            Defaults to the project's canonical
            docs/2026-05-26-sprint-10a-whole-episode-critic-rubric.md.

    Returns:
        Rubric with all 10 axes parsed and a populated ShipThreshold.

    Raises:
        FileNotFoundError if the rubric file does not exist.
        ValueError if the parsed axis count is not the expected 10
            (loud failure -- the critic prompt assumes 10 axes; a
            silent drift to 8 or 12 would mis-shape downstream).
    """
    path = rubric_path if rubric_path is not None else _DEFAULT_RUBRIC_PATH
    text = path.read_text(encoding="utf-8")

    axes: List[RubricAxis] = []
    current: Optional[RubricAxis] = None
    in_axes_section = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Enter the axes section.
        if line.startswith("## Axes"):
            in_axes_section = True
            continue
        # Exit the axes section at the next H2.
        if in_axes_section and line.startswith("## ") and "Axes" not in line:
            # Finalize the in-flight axis and stop.
            if current is not None:
                axes.append(current)
                current = None
            in_axes_section = False
            continue

        if not in_axes_section:
            continue

        # New axis header.
        m = _AXIS_HEADER_RE.match(line)
        if m:
            # Finalize the previous axis before starting a new one.
            if current is not None:
                axes.append(current)
            ordinal = int(m.group(1))
            name = m.group(2).strip()
            current = RubricAxis(
                ordinal=ordinal,
                name=name,
                json_key=_slugify(name),
                definition="",
                anchors=[],
            )
            continue

        if current is None:
            continue

        # Definition paragraph after the H3: first line that starts
        # with '**What it checks:**'.
        stripped = line.lstrip()
        if stripped.startswith("**What it checks:**"):
            body = stripped[len("**What it checks:**"):].strip()
            current.definition = body
            continue

        # Anchor line: '- **N -- Label.** body'
        anchor_m = _ANCHOR_LINE_RE.match(line)
        if anchor_m:
            ordinal = int(anchor_m.group(1))
            label = anchor_m.group(2).strip()
            body = anchor_m.group(3).strip()
            text_anchor = f"{ordinal} -- {label}. {body}".strip()
            current.anchors.append(text_anchor)
            continue

    # Finalize the last axis if we ended mid-stream.
    if current is not None:
        axes.append(current)

    if len(axes) != 10:
        raise ValueError(
            f"_otr_critic_rubric: expected to parse 10 axes from "
            f"{path.name}, got {len(axes)}. The rubric format may "
            f"have drifted -- check the 'Programmatic-consumption "
            f"contract' section of the markdown."
        )

    # Each axis must carry exactly 5 anchors (scores 1-5).
    for axis in axes:
        if len(axis.anchors) != 5:
            raise ValueError(
                f"_otr_critic_rubric: axis '{axis.name}' has "
                f"{len(axis.anchors)} anchors, expected 5. The "
                f"rubric format may have drifted."
            )

    return Rubric(
        axes=axes,
        threshold=ShipThreshold(),
        source_path=str(path),
    )
