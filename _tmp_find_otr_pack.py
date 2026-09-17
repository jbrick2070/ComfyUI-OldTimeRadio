"""List custom_nodes dirs the :8000 install can see. No secrets."""
from __future__ import annotations

from pathlib import Path

CANDIDATES = [
    Path(r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"),
    Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes"),
    Path(r"C:\Users\jeffr\AppData\Roaming\ComfyUI\custom_nodes"),
]


def main() -> int:
    for root in CANDIDATES:
        print(f"DIR {root} exists={root.exists()}")
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if "old" in child.name.lower() or "otr" in child.name.lower() or "radio" in child.name.lower():
                target = None
                if child.is_symlink():
                    target = str(child.readlink())
                print(f"  {child.name} symlink={child.is_symlink()} target={target}")
                rd = child / "nodes" / "_otr_video_engines" / "render_driver.py"
                if rd.exists():
                    text = rd.read_text(encoding="utf-8", errors="replace")
                    print(f"    render_driver fanout={'def cloud_video_fanout_workers' in text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
