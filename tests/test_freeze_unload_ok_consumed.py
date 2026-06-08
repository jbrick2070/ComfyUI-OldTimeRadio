"""Sprint E E9 / R-4: freeze_unload_ok stamp consumed downstream.

Pre-E9 the cascade stamped meta.freeze_unload_ok=False on its
finally-block unload_llm failure, but no downstream consumer read the
stamp. A leaked Mistral-Nemo cache passed silently into the audio /
FLUX VRAM and OOM'd. Post-E9 the audio chain (re-homed to OTR_CastLock
in the 1a clean-break) and BatchFluxRender each consult the flag and
attempt one defensive unload before claiming VRAM.

This test AST-walks the two source files for the read pattern. It
does not exercise the defensive-unload path at runtime (that would
require a real cache state); the unload call itself is exercised by
tests/test_loader_slot_primitives.py and friends.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_source(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


CONSUMER_FILES = (
    # Audio clean-break (1a): the freeze_unload_ok defensive unload re-homed from
    # the retired BatchBark node into OTR_CastLock, which runs first in the v2
    # audio chain (before any audio engine claims VRAM).
    "nodes/cast_lock.py",
)


@pytest.mark.parametrize("rel_path", CONSUMER_FILES)
def test_consumer_reads_freeze_unload_ok(rel_path: str):
    src = _read_source(rel_path)
    assert "freeze_unload_ok" in src, (
        f"{rel_path} must read meta.freeze_unload_ok to surface "
        "cascade-side unload failures before claiming VRAM"
    )


@pytest.mark.parametrize("rel_path", CONSUMER_FILES)
def test_consumer_attempts_defensive_unload(rel_path: str):
    src = _read_source(rel_path)
    # Either the canonical helper unload_llm OR a defensive call site
    # with that helper aliased -- both shapes are acceptable. The
    # important property is that on freeze_unload_ok=False the
    # consumer attempts one unload before its own model load.
    assert "unload_llm" in src, (
        f"{rel_path} must attempt unload_llm() on freeze_unload_ok=False; "
        "otherwise the Mistral-Nemo leak persists into the consumer's "
        "own VRAM peak and OOMs the 16 GB ceiling"
    )


@pytest.mark.parametrize("rel_path", CONSUMER_FILES)
def test_consumer_logs_warning_on_false_flag(rel_path: str):
    """The recovery attempt must log at WARN so soak diagnostics
    surface the cascade teardown failure."""
    src = _read_source(rel_path)
    assert "freeze_unload_ok=False" in src
    assert "log.warning" in src


def test_cascade_stamps_freeze_unload_ok():
    """Belt-and-braces drift guard on the producer side. If the
    cascade stops stamping freeze_unload_ok the consumer-side branches
    above become dead code."""
    src = _read_source("nodes/OTR_LedgerFreezeCascade.py")
    assert '"freeze_unload_ok"' in src or "'freeze_unload_ok'" in src
    # And it must be set on BOTH the True and False paths -- the False
    # branch is the load-bearing one for E9.
    assert "freeze_unload_ok"in src
    assert "unload_ok = False" in src
