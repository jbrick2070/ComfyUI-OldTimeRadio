"""Two custom_nodes folders must not crash ComfyUI boot.

A first-time registry install and a later Manager update are one folder
and must keep loading every node.
"""
from __future__ import annotations

import sys
from pathlib import Path

from nodes._otr_pack_singleton import (
    PACK_GUARD,
    PRESTARTUP_GUARD,
    claim,
)

REPO = Path(__file__).resolve().parent.parent


def test_second_folder_is_duplicate_same_folder_is_not():
    key = PACK_GUARD + "_test_claim"
    sys.modules.pop(key, None)
    try:
        assert claim(key, "C:/packs/one") is None
        assert claim(key, "C:/packs/one") is None
        assert claim(key, r"C:\packs\one") is None
        assert claim(key, "C:/packs/two") == "C:/packs/one"
    finally:
        sys.modules.pop(key, None)


def test_pack_and_prestartup_guards_are_distinct():
    assert PACK_GUARD != PRESTARTUP_GUARD
    assert PACK_GUARD.startswith("_comfyui_old_time_radio_")
    assert PRESTARTUP_GUARD.startswith("_comfyui_old_time_radio_")


def test_init_single_pack_fails_open():
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "_otr_dup = None" in src
    assert "Fail OPEN" in src
    assert "if _otr_dup is not None:" in src


def test_init_skips_nodes_on_duplicate():
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "DUPLICATE PACK skipped" in src
    claim_at = src.index("_otr_dup = _otr_claim_pack")
    skip_at = src.index("if _otr_dup is None:")
    assert claim_at < skip_at


def test_prestartup_uses_the_same_prestartup_guard_string():
    src = (REPO / "prestartup_script.py").read_text(encoding="utf-8")
    assert PRESTARTUP_GUARD in src
    assert "prestartup skipped" in src
    assert PACK_GUARD not in src


def test_prestartup_clears_leaked_otr_test_mode_on_live_boot():
    """Wiring: a live Comfy boot must unpin pytest's OTR_TEST_MODE.
    A test that only calls the helper never proves the boot path."""
    src = (REPO / "prestartup_script.py").read_text(encoding="utf-8")
    assert "OTR_TEST_MODE" in src
    assert "cleared leaked OTR_TEST_MODE=1" in src
    assert 'environ.pop("OTR_TEST_MODE"' in src or "environ.pop('OTR_TEST_MODE'" in src
