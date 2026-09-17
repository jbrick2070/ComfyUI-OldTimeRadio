"""Count QA quality_status on Bowl vs Understudy."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

files = {
    "fourth_bowl": Path(
        r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
        r"\signal_lost_the_fourth_bowl_20260916_174517"
        r"\signal_lost_the_fourth_bowl_20260916_174517_silent.mp4.qa.json"
    ),
    "understudy": Path(
        r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
        r"\signal_lost_signal_lost_the_understudy_20260916_181706"
        r"\signal_lost_signal_lost_the_understudy_20260916_181706_silent.mp4.qa.json"
    ),
}

for name, path in files.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    beats = data.get("beats") or []
    counts = Counter(str(b.get("quality_status") or "") for b in beats)
    print("===", name, "beats", len(beats), "===")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    holes = [
        (b.get("beat_id"), b.get("shot_id"), b.get("quality_status"), b.get("engine_id"))
        for b in beats
        if b.get("quality_status") not in ("ok", None)
    ]
    print("  non-ok:")
    for row in holes:
        print("   ", row)
    print()
