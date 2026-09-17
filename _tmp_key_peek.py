import os
import winreg

def _hkcu(name):
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        try:
            value, _ = winreg.QueryValueEx(key, name)
        finally:
            key.Close()
        text = str(value or "").strip()
        return bool(text), len(text), (text[:4] + "..." if text else "")
    except OSError:
        return False, 0, ""

def _proc(name):
    text = (os.environ.get(name) or "").strip()
    return bool(text), len(text)

names = (
    "OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY", "OTR_ENABLE_OPENROUTER", "OTR_ENABLE_COMFY_CREDITS",
    "OTR_COMFY_API_KEY",
)
print("name hkcu_set hkcu_len proc_set proc_len")
for n in names:
    hs, hl, _pref = _hkcu(n)
    ps, pl = _proc(n)
    print(n, int(hs), hl, int(ps), pl)
