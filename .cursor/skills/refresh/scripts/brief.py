"""Dump live shipping video pins for /refresh. Stdlib only. Run from repo root
or anywhere: python .cursor/skills/refresh/scripts/brief.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "workflows" / "otr_canonical.json").is_file():
            return p
    raise SystemExit("cannot find repo root (workflows/otr_canonical.json)")


def _shipping_set(root: Path) -> tuple[str, ...]:
    src = (root / "scripts" / "build_variants.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "SHIPPING_SET":
                    return tuple(ast.literal_eval(node.value))
    raise SystemExit("SHIPPING_SET not found")


def _node(graph: dict, ntype: str) -> dict | None:
    hits = [n for n in graph.get("nodes") or [] if n.get("type") == ntype]
    return hits[0] if hits else None


def _director_video(graph: dict) -> list[str]:
    node = _node(graph, "OTR_VideoDirector")
    wv = (node or {}).get("widgets_values") or []
    return [str(v) for v in wv[:3]]


def _profile(root: Path, pid: str) -> dict:
    path = root / "config" / "profiles" / ("%s.json" % pid)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = _repo_root()
    shipping = _shipping_set(root)
    local = [p for p in shipping if not p.startswith("otr_cloud_")]
    cloud = [p for p in shipping if p.startswith("otr_cloud_")]

    canon = json.loads(
        (root / "workflows" / "otr_canonical.json").read_text(encoding="utf-8")
    )
    print("CANONICAL")
    print("  video", " | ".join(_director_video(canon)))
    print("LOCAL", len(local))
    for pid in local:
        prof = _profile(root, pid)
        roles = prof.get("role_overrides") or {}
        print(
            "  %s  writer=%s  video=%s/%s/%s  image=%s"
            % (
                pid,
                (prof.get("llm") or {}).get("creative_model"),
                roles.get("announcer_visual"),
                roles.get("music_visual"),
                roles.get("character_visual"),
                roles.get("character_image"),
            )
        )
    print("CLOUD", len(cloud))
    for pid in cloud:
        prof = _profile(root, pid)
        roles = prof.get("role_overrides") or {}
        llm = prof.get("llm") or {}
        feat = prof.get("features") or {}
        print(
            "  %s  act=%s  writer=%s  comfy=%s  or=%s  video=%s  image=%s"
            % (
                pid,
                feat.get("act_count"),
                llm.get("creative_model"),
                llm.get("comfy_slot_a_model"),
                llm.get("openrouter_slot_a_model"),
                roles.get("character_visual"),
                roles.get("character_image"),
            )
        )
    print("COUNTS canonical=1 local=%d cloud=%d shipping=%d" % (
        len(local), len(cloud), len(shipping)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
