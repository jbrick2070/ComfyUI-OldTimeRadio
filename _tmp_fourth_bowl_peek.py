"""Inspect the_fourth_bowl and recent _shared beat shots."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

EP = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\signal_lost_the_fourth_bowl_20260915_225143")
SHARED = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\_shared\tmp")


def fmt(p: Path) -> str:
    st = p.stat()
    ts = datetime.fromtimestamp(st.st_mtime).strftime("%H:%M:%S")
    return "%s %8d %s" % (ts, st.st_size, p.name)


def main() -> int:
    print("EP", EP)
    print("exists", EP.is_dir())
    if EP.is_dir():
        for p in sorted(EP.iterdir(), key=lambda x: x.name):
            if p.is_dir():
                n = sum(1 for _ in p.rglob("*") if _.is_file())
                print(" subdir", p.name, "files", n)
            else:
                print(" file", fmt(p))
        for sub in ("video", "videos", "stills", "audio", "render", "shots"):
            d = EP / sub
            if not d.is_dir():
                continue
            files = [p for p in d.rglob("*") if p.is_file()]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            print("---", sub, "n=", len(files), "---")
            for p in files[:25]:
                print(" ", fmt(p), "rel", str(p.relative_to(EP)))

        for name in (
            "episode_canon.json",
            "stills/stills_manifest.json",
            "render_trace.json",
            "video_trace.json",
        ):
            p = EP / name
            if p.is_file():
                print("FOUND", name, "bytes", p.stat().st_size)

        ledger = list((EP / "audio").glob("*ledger.json")) if (EP / "audio").is_dir() else []
        for p in ledger[:2]:
            data = json.loads(p.read_text(encoding="utf-8"))
            meta = data.get("meta") or {}
            print("LEDGER", p.name)
            print(" title", meta.get("episode_title") or data.get("episode_title"))
            print(" source_bank", (meta.get("source_bank") or meta.get("story_input") or {}))
            if isinstance(meta.get("story_input"), dict):
                print(" bank", meta["story_input"].get("source_bank"))
                print(" style", meta["story_input"].get("visual_style"))
            print(" act_count", meta.get("act_count"))
            shots = data.get("shots") or data.get("section", {}).get("shots") if isinstance(data.get("section"), dict) else data.get("shots")
            if isinstance(data.get("video"), dict):
                print(" video_keys", list(data["video"].keys())[:12])
            engines = set()
            for shot in (data.get("shots") or []):
                if isinstance(shot, dict) and shot.get("engine_id"):
                    engines.add(shot.get("engine_id"))
            print(" shot_engines", sorted(engines)[:12], "n_shots", len(data.get("shots") or []))
            print(" n_lines", len(data.get("lines") or []))

        man = EP / "stills" / "stills_manifest.json"
        if man.is_file():
            m = json.loads(man.read_text(encoding="utf-8"))
            images = m.get("images") or m.get("rows") or []
            if isinstance(m, list):
                images = m
            print("STILLS rows", len(images) if isinstance(images, list) else type(m))
            if isinstance(images, list):
                kinds = {}
                for row in images:
                    if not isinstance(row, dict):
                        continue
                    k = str(row.get("kind") or "?")
                    kinds[k] = kinds.get(k, 0) + 1
                print(" still_kinds", kinds)

    print("--- shared tmp newest 20 media ---")
    if SHARED.is_dir():
        files = [p for p in SHARED.iterdir() if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files[:20]:
            print(" ", fmt(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
