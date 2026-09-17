"""Compare install-pack vs workspace OTR render_driver. No secrets."""
from __future__ import annotations

import hashlib
from pathlib import Path

A = Path(
    r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"
    r"\comfyui-old-time-radio\nodes\_otr_video_engines\render_driver.py"
)
B = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
    r"\nodes\_otr_video_engines\render_driver.py"
)
C = Path(
    r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"
    r"\comfyui-old-time-radio"
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> int:
    print(f"install exists={A.exists()} sha={_sha(A) if A.exists() else '-'}")
    print(f"workspace exists={B.exists()} sha={_sha(B) if B.exists() else '-'}")
    print(f"same={A.exists() and B.exists() and _sha(A) == _sha(B)}")
    print(f"install_pack_is_symlink={C.is_symlink()} resolve={C.resolve()}")
    cmb_a = A.parent.parent / "_otr_shared" / "cloud_media_backend.py"
    cmb_b = B.parent.parent / "_otr_shared" / "cloud_media_backend.py"
    print(f"backend same={cmb_a.exists() and cmb_b.exists() and _sha(cmb_a) == _sha(cmb_b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
