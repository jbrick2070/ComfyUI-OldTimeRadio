"""Print VideoDirector engine widgets from a saved graph."""
from __future__ import annotations

import json
import sys
from pathlib import Path

wf = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in wf.get("nodes") or []:
    if node.get("type") != "OTR_VideoDirector":
        continue
    names = [i.get("name") for i in (node.get("inputs") or []) if i.get("widget")]
    vals = node.get("widgets_values") or []
    print("id", node.get("id"))
    for i, name in enumerate(names):
        if i < len(vals) and ("visual" in str(name) or "video" in str(name) or "engine" in str(name)):
            print("%s=%r" % (name, vals[i]))
