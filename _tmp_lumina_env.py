"""Resolve Lumina env paths from User env and disk."""
from __future__ import annotations

import os
from pathlib import Path

keys = (
    "OTR_LUMINA_CKPT",
    "OTR_LUMINA_CLIP",
    "OTR_LUMINA_VAE",
    "OTR_LUMINA_TEXT_ENCODER",
)
print("process:")
for k in keys:
    v = os.environ.get(k, "")
    print(f"  {k}={v!r} exists={Path(v).is_file() if v else False}")

print("user:")
import subprocess
for k in keys:
    out = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"[Environment]::GetEnvironmentVariable('{k}','User')",
        ],
        text=True,
    ).strip()
    print(f"  {k}={out!r} exists={Path(out).is_file() if out else False}")
