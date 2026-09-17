import json
from pathlib import Path
WF = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\variants\otr_cloud_deluxe_7act.json")
wf = json.loads(WF.read_text(encoding="utf-8"))
for n in wf["nodes"]:
    if n.get("type") != "OTR_VideoDirector":
        continue
    names = []
    for inp in n.get("inputs") or []:
        w = (inp.get("widget") or {}).get("name")
        if w:
            names.append(w)
    vals = n.get("widgets_values") or []
    for i, name in enumerate(names):
        print("%s=%r" % (name, vals[i] if i < len(vals) else "MISSING"))
    print("TYPES", sorted({x.get("type") for x in wf["nodes"] if isinstance(x, dict)}))
    break
# also print any node whose widgets mention wan
for n in wf["nodes"]:
    vals = n.get("widgets_values") or []
    blob = json.dumps(vals)
    if "wan" in blob.lower() or "vidu" in blob.lower():
        print("NODE", n.get("type"), "id", n.get("id"), blob[:400])
