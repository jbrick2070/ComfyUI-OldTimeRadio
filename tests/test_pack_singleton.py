"""Two custom_nodes folders must not crash ComfyUI boot.

A first-time registry install and a later Manager update are one folder
and must keep loading nodes plus GET /otr/latest_ledger.
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


def test_init_single_pack_still_registers_the_ledger_route():
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "_otr_dup = None" in src
    assert "Fail OPEN" in src or "fail open" in src.lower() or "Fail OPEN" in src
    assert "if _otr_dup is not None:" in src
    assert "raise _OTRDuplicateRoute()" in src
    # The raise is the duplicate-only skip. It must run BEFORE any
    # PromptServer decorator, and it must NOT sit between the scrub
    # helper and the GET decorator (that block is exec'd by the ledger
    # privacy tests as a first-install stand-in).
    raise_at = src.index("raise _OTRDuplicateRoute()")
    re_at = src.index("import re as _otr_re")
    route_at = src.index('@_otr_PromptServer.instance.routes.get("/otr/latest_ledger")')
    assert raise_at < re_at < route_at
    prefix = src[re_at:route_at]
    assert "_otr_dup" not in prefix
    assert "for rd in _otr_PromptServer" not in src
    assert "except _OTRDuplicateRoute:" in src


def test_init_skips_nodes_and_routes_on_duplicate():
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert "DUPLICATE PACK skipped" in src
    assert "method HEAD is already registered" in src
    claim_at = src.index("_otr_dup = _otr_claim_pack")
    skip_at = src.index("if _otr_dup is None:")
    route_at = src.index('@_otr_PromptServer.instance.routes.get("/otr/latest_ledger")')
    assert claim_at < skip_at < route_at


def test_prestartup_uses_the_same_prestartup_guard_string():
    src = (REPO / "prestartup_script.py").read_text(encoding="utf-8")
    assert PRESTARTUP_GUARD in src
    assert "prestartup skipped" in src
    assert PACK_GUARD not in src
