"""Where the shipped graphs live, answered once for the whole suite.

Since 2026-09-25 the machine variants sit BESIDE the canonical in
``workflows/`` (operator: "we can't store the variants in a subfolder").
ComfyUI's template gallery globs one directory level (``*/workflows/*.json``),
so the 24 graphs that lived in ``workflows/variants/`` shipped but never
appeared in the menu.

The move changes two things every graph-walking test used to assume: a glob of
``workflows/variants/`` now finds nothing, and a glob of ``workflows/*.json``
now finds the canonical AND the variants. This module is the one place that
tells them apart, so no test re-derives the rule and counts the canonical twice.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / "workflows"
CANONICAL = WORKFLOWS / "otr_canonical.json"


def variant_paths() -> list:
    """Every generated variant graph, sorted: each ``workflows/otr_*.json``
    except the canonical and any ``*.env.json`` sidecar."""
    return sorted(p for p in WORKFLOWS.glob("otr_*.json")
                  if p != CANONICAL and not p.name.endswith(".env.json"))


def variant_path(stem: str) -> Path:
    """The graph for one variant, by stem (``otr_8gb_video``)."""
    return WORKFLOWS / ("%s.json" % stem)


def shipped_graphs() -> list:
    """The canonical first, then every variant."""
    return [CANONICAL] + variant_paths()
