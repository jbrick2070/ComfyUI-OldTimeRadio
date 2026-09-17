"""Write OpenRouter + Google keys to the ignored secret file. Never print them."""
from __future__ import annotations

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
    lines = []
    or_key = _hkcu("OPENROUTER_API_KEY")
    google = (
        _hkcu("OTR_GOOGLE_API_KEY")
        or _hkcu("GEMINI_API_KEY")
        or _hkcu("GOOGLE_API_KEY")
    )
    if not or_key:
        print("NO_OPENROUTER")
        return 2
    lines.append(f"OPENROUTER_API_KEY={or_key}")
    print(f"OPENROUTER_CHARS={len(or_key)}")
    if google:
        lines.append(f"OTR_GOOGLE_API_KEY={google}")
        lines.append(f"GOOGLE_API_KEY={google}")
        lines.append(f"GEMINI_API_KEY={google}")
        print(f"GOOGLE_CHARS={len(google)}")
    else:
        print("GOOGLE_MISSING")
    DEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"WROTE {DEST.name} lines={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
