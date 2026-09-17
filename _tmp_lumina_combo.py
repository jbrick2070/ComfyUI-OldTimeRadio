"""Confirm lumina_image is a live VideoDirector combo value."""
from __future__ import annotations

import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=15) as resp:
    info = json.loads(resp.read().decode("utf-8"))
node = info["OTR_VideoDirector"]["input"]["required"]
for key in (
    "announcer_image_model",
    "music_image_model",
    "character_image_model",
):
    spec = node.get(key) or info["OTR_VideoDirector"]["input"].get("optional", {}).get(key)
    opts = spec[0] if isinstance(spec, list) else spec
    print(key, "lumina_image" in opts, "z_image_turbo" in (opts or []))
