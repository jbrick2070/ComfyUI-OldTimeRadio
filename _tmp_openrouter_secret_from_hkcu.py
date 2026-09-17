"""Write OPENROUTER_API_KEY to the ignored secret file. Never print the key."""
from __future__ import annotations

import sys
import winreg
from pathlib import Path

DEST = Path(__file__).with_name("_tmp_openrouter.secret")


def _hkcu(name: str) -> str:
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        try:
            value, _ = winreg.QueryValueEx(key, name)
        finally:
            key.Close()
        return str(value or "").strip()
    except OSError:
        return ""


def main() -> int:
    token = _hkcu("OPENROUTER_API_KEY")
    if not token:
        if DEST.is_file():
            existing = DEST.read_text(encoding="utf-8").strip()
            has = "OPENROUTER_API_KEY=" in existing and len(existing) > 40
            print(f"HKCU_EMPTY existing_secret={int(has)} chars={len(existing)}")
            return 0 if has else 2
        print("NO_OPENROUTER_TOKEN")
        return 2
    DEST.write_text(f"OPENROUTER_API_KEY={token}\n", encoding="utf-8", newline="\n")
    print(f"WROTE_SECRET chars={len(token)} path_ok={DEST.is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
