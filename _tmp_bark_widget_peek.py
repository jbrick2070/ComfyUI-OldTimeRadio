"""Print CastLock and voice-node widgets from otr_16gb_low."""
from __future__ import annotations

import json
from pathlib import Path

p = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\variants\otr_16gb_low.json")
wf = json.loads(p.read_text(encoding="utf-8"))
want = {
    "OTR_CastLock",
    "OTR_BatchCharacterVoices",
    "OTR_AnnouncerVoice",
    "OTR_LedgerScriptWriter",
}
for n in wf["nodes"]:
    t = n.get("type")
    if t in want:
        print(f"id={n['id']} type={t}")
        print("  widgets_values=", n.get("widgets_values"))
        names = [i.get("name") for i in n.get("inputs") or [] if i.get("widget")]
        print("  widget_inputs=", names)
        print()
