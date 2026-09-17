"""Local OpenRouter health: key present, 8188/8000 slots, no secrets."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import winreg

NAMES = ("OPENROUTER_API_KEY", "OTR_OPENROUTER_SLOT_A_DEFAULT", "OPENROUTER_MODEL_A")


def _hkcu(name: str) -> tuple[bool, int]:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        try:
            value, _ = winreg.QueryValueEx(key, name)
        finally:
            key.Close()
        text = str(value or "").strip()
        return bool(text), len(text)
    except OSError:
        return False, 0


def _peek(port: int) -> None:
    base = f"http://127.0.0.1:{port}"
    print(f"=== :{port} ===")
    try:
        with urllib.request.urlopen(f"{base}/queue", timeout=4) as resp:
            q = json.loads(resp.read().decode("utf-8"))
        print("queue running", len(q.get("queue_running") or []),
              "pending", len(q.get("queue_pending") or []))
    except Exception as exc:
        print("queue", type(exc).__name__, str(exc)[:120])
        return
    try:
        with urllib.request.urlopen(
            f"{base}/object_info/OTR_LedgerScriptWriter", timeout=12
        ) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        block = info["OTR_LedgerScriptWriter"]["input"]
        req = block.get("required") or {}
        opt = block.get("optional") or {}
        slot = (req.get("openrouter_slot_a_model") or opt.get("openrouter_slot_a_model") or [[]])[0]
        print("slot_a_n", len(slot) if isinstance(slot, list) else type(slot).__name__)
        print("has_enable_sentinel", "(enable OpenRouter)" in (slot or []))
        print("has_gpt_latest", "~openai/gpt-latest" in (slot or []))
        print("slot_a_head", list(slot)[:8] if isinstance(slot, list) else slot)
    except Exception as exc:
        print("object_info", type(exc).__name__, str(exc)[:200])


print("name hkcu_set hkcu_len proc_set proc_len")
for n in NAMES:
    hs, hl = _hkcu(n)
    p = (os.environ.get(n) or "").strip()
    print(n, int(hs), hl, int(bool(p)), len(p))
_peek(8188)
_peek(8000)
