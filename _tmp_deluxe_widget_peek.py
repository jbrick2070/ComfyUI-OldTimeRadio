import json
from pathlib import Path

p = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\variants\otr_cloud_deluxe_7act.json")
g = json.loads(p.read_text(encoding="utf-8"))
want = (
    "OTR_VideoRenderBatch", "OTR_LedgerScriptWriter",
    "OTR_ImageGenDispatcher", "OTR_VoiceCast",
)
for n in g.get("nodes") or []:
    t = n.get("type")
    if t in want or "video" in str(t).lower() or "engine" in str(t).lower():
        wv = n.get("widgets_values")
        print(n.get("id"), t)
        if isinstance(wv, list):
            for i, v in enumerate(wv):
                s = repr(v)
                if len(s) > 120:
                    s = s[:117] + "..."
                print(" ", i, s)
