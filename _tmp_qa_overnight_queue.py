"""QA the live :8000 overnight queue against the intended pins."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000"
STATUS = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\tmp\mystory_cloud_queue.json")
IDEA = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_story_input.py")

EXPECT = {
    "c4cbde6d-ece8-47b0-87cd-961e06e1a942": {
        "label": "cheap 1-act Vidu",
        "act_count": "1",
        "video": "cloud_vidu_q2_pro_fast_720p",
        "image": "cloud_luma_photon_flash",
    },
    "1d26bda7-7470-4ebd-ac89-1880bcd520fa": {
        "label": "deluxe Foley 1-act",
        "act_count": "1",
        "video": "cloud_ltx25_foley_plus",
        "image": "cloud_luma_photon_flash",
    },
    "32d8d400-9abc-4bd0-af38-5289e35b5b8e": {
        "label": "deluxe Foley 5-act",
        "act_count": "5",
        "video": "cloud_ltx25_foley_plus",
        "image": "cloud_luma_photon_flash",
    },
    "cf7b4b63-9878-4eb9-a739-108864611bcc": {
        "label": "deluxe audio-in 5-act",
        "act_count": "5",
        "video": "cloud_ltx25_audio_in",
        "image": "cloud_luma_photon_flash",
    },
}


def get(path: str):
    with urllib.request.urlopen(URL + path, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _node_class(prompt: dict, class_type: str) -> dict:
    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node.get("inputs") or {}
    return {}


def _qa_prompt(pid: str, prompt: dict, slot: str) -> list[str]:
    bad = []
    want = EXPECT.get(pid, {})
    label = want.get("label") or pid[:8]
    writer = _node_class(prompt, "OTR_LedgerScriptWriter")
    vdir = _node_class(prompt, "OTR_VideoDirector")
    img = _node_class(prompt, "OTR_ImageGenDispatcher")
    voice = _node_class(prompt, "OTR_CharacterVoice")
    print("==== %s %s pid=%s" % (slot, label, pid))
    print("  source_bank", writer.get("source_bank"))
    print("  visual_style", writer.get("visual_style"))
    print("  act_count", writer.get("act_count"))
    print("  source_ref", repr(writer.get("source_ref")))
    print("  creative", writer.get("creative_model"))
    print("  technical", writer.get("technical_model"))
    print("  video announcer", vdir.get("announcer_video_model") or vdir.get("video_render_engine"))
    print("  video character", vdir.get("character_visual") or vdir.get("character_video_model"))
    # dump a few video-ish keys
    video_keys = [k for k in vdir if "video" in k.lower() or "visual" in k.lower() or "engine" in k.lower()]
    for k in sorted(video_keys)[:12]:
        print("  vdir.%s=%s" % (k, vdir.get(k)))
    img_keys = [k for k in img if "image" in k.lower() or "engine" in k.lower() or "model" in k.lower()]
    for k in sorted(img_keys)[:8]:
        print("  img.%s=%s" % (k, img.get(k)))
    if writer.get("source_bank") != "my_story":
        bad.append("%s source_bank=%r" % (label, writer.get("source_bank")))
    if writer.get("visual_style") != "recur_frac":
        bad.append("%s visual_style=%r" % (label, writer.get("visual_style")))
    if want.get("act_count") and str(writer.get("act_count")) != str(want["act_count"]):
        bad.append("%s act_count=%r want=%s" % (label, writer.get("act_count"), want["act_count"]))
    blob = " ".join(str(v) for v in list(vdir.values())[:40])
    if want.get("video") and want["video"] not in blob:
        # also search all vdir values
        allv = " ".join(str(v) for v in vdir.values())
        if want["video"] not in allv:
            bad.append("%s missing engine %s" % (label, want["video"]))
        else:
            print("  engine %s found in vdir values" % want["video"])
    else:
        print("  engine pin OK %s" % want.get("video"))
    return bad


def main() -> int:
    q = get("/queue")
    running = q.get("queue_running") or []
    pending = q.get("queue_pending") or []
    print("queue running=%d pending=%d" % (len(running), len(pending)))
    seen = []
    bad = []
    rows = []
    for row in running:
        if isinstance(row, list) and len(row) > 1:
            rows.append(("RUN", row[1], row[2] if len(row) > 2 else None))
    for row in pending:
        if isinstance(row, list) and len(row) > 1:
            rows.append(("PEND", row[1], row[2] if len(row) > 2 else None))
    for slot, pid, prompt in rows:
        seen.append(pid)
        if not isinstance(prompt, dict):
            print("==== %s %s HAS NO PROMPT BODY" % (slot, pid))
            bad.append("%s missing prompt body" % pid)
            continue
        bad.extend(_qa_prompt(pid, prompt, slot))
    expected_ids = list(EXPECT)
    for pid in expected_ids:
        if pid not in seen:
            bad.append("MISSING FROM QUEUE %s %s" % (EXPECT[pid]["label"], pid))
    extra = [p for p in seen if p not in EXPECT]
    for pid in extra:
        bad.append("UNEXPECTED JOB %s" % pid)

    text = IDEA.read_text(encoding="utf-8")
    print("==== house idea")
    print("  never says a word", "never says a word" in text)
    print("  toy cat statue", "toy cat statue" in text)
    print("  Whiskers talks", "Whiskers talks" in text)

    print("==== env from status file")
    if STATUS.is_file():
        print(STATUS.read_text(encoding="utf-8"))

    print("==== VERDICT")
    if bad:
        print("BLOCKERS")
        for b in bad:
            print(" -", b)
        return 1
    print("ALL GOOD -- 4 jobs, my_story, recur_frac, act pins, engines present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
