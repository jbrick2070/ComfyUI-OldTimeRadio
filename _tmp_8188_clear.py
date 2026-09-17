"""Clear GPU :8188 only. Do not touch CPU :8000."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["COMFYUI_URL"] = "http://127.0.0.1:8188"

import otr_api  # noqa: E402

otr_api.COMFYUI_URL = "http://127.0.0.1:8188"
ok = otr_api.cancel_queue()
running, pending = otr_api.queue_snapshot()
print(f"cancel_ok={ok} running={running} pending={pending}")
