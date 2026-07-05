from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from upstream_story_lab.registry import Registry  # noqa: E402


@pytest.fixture()
def registry() -> Registry:
    """Fresh, uncached registry over the real fixtures."""

    return Registry(ROOT)


@pytest.fixture()
def mirror_nodes() -> Path:
    return ROOT / "production_mirror" / "nodes"
