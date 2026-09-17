"""Print image-model widgets from every shipping 16gb variant."""
from __future__ import annotations

import json
from pathlib import Path

root = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
want = (
    "announcer_image_model",
    "music_image_model",
    "character_image_model",
)
for path in sorted((root / "workflows" / "variants").glob("otr_16gb*.json")):
    wf = json.loads(path.read_text(encoding="utf-8"))
    for n in wf["nodes"]:
        if n.get("type") != "OTR_VideoDirector":
            continue
        names = [i.get("name") for i in n.get("inputs") or [] if i.get("widget")]
        vals = n.get("widgets_values") or []
        paired = dict(zip(names, vals))
        imgs = {k: paired.get(k) for k in want}
        print(path.name, imgs)
        break
    else:
        print(path.name, "NO VideoDirector")
