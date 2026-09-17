"""Wait until :8188 /object_info contains OTR_CastLock."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8188/object_info"
deadline = time.time() + 90
while time.time() < deadline:
    try:
        with urllib.request.urlopen(URL, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if "OTR_CastLock" in data and "OTR_BatchCharacterVoices" in data:
            bark = "bark" in str(
                data["OTR_BatchCharacterVoices"]
                .get("input", {})
                .get("required", {})
                .get("engine", [])
            )
            print("READY nodes", len(data), "bark_in_char_engine", bark)
            raise SystemExit(0)
        print("waiting nodes", len(data))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print("waiting", type(exc).__name__)
    time.sleep(2)
print("TIMEOUT object_info")
raise SystemExit(2)
