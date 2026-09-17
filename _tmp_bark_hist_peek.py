"""Dump history status + CastLock/voice inputs for the dead Bark prompt."""
from __future__ import annotations

import json
import urllib.request

PID = "10c19edc-e7ef-4652-a2ea-df54253fddbf"
url = f"http://127.0.0.1:8188/history/{PID}"
try:
    with urllib.request.urlopen(url, timeout=8) as resp:
        hist = json.loads(resp.read().decode("utf-8"))
except Exception as exc:
    print("history_err", type(exc).__name__, exc)
    raise SystemExit(0)

if not hist:
    print("history_empty")
    raise SystemExit(0)
entry = hist.get(PID) or next(iter(hist.values()), {})
status = (entry.get("status") or {})
print("status_str", status.get("status_str"))
msgs = status.get("messages") or []
for m in msgs[-6:]:
    print("msg", m[0] if isinstance(m, list) and m else m)

prompt = (entry.get("prompt") or [None, None, {}])
graph = prompt[2] if len(prompt) > 2 and isinstance(prompt[2], dict) else {}
# prompt layout can be [number, id, extra, outputs] vs [number, id, prompt]
if "class_type" in str(graph)[:80]:
    nodes = graph
else:
    nodes = prompt[2] if isinstance(prompt[2], dict) else {}
    # some history: prompt is [n, pid, prompt_dict, extra]
    if nodes and not any(isinstance(v, dict) and "class_type" in v for v in nodes.values()):
        for item in prompt:
            if isinstance(item, dict) and any(
                isinstance(v, dict) and v.get("class_type") for v in item.values()
            ):
                nodes = item
                break

want = {
    "OTR_CastLock", "OTR_BatchCharacterVoices", "OTR_AnnouncerVoice",
    "OTR_LedgerScriptWriter", "OTR_VideoDirector",
}
for nid, node in nodes.items():
    if not isinstance(node, dict):
        continue
    ct = node.get("class_type")
    if ct not in want:
        continue
    inp = node.get("inputs") or {}
    print(f"--- {ct} id={nid}")
    keys = [
        "source_bank", "act_count", "visual_style",
        "creative_writing_model", "technical_model",
        "voice_bank", "char_voice_engine", "announcer_voice_engine",
        "engine",
        "announcer_image_model", "music_image_model", "character_image_model",
        "announcer_video_model", "character_video_model",
    ]
    for k in keys:
        if k in inp:
            print(f"  {k}={inp[k]!r}")
