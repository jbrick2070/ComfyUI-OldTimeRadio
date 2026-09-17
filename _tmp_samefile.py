"""Confirm Documents OTR and the :8000 install pack are the same files."""
from __future__ import annotations

import os

a = r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_comfy_backend.py"
b = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes\comfyui-old-time-radio\nodes\_otr_comfy_backend.py"
print("samefile", os.path.samefile(a, b))
print("a", os.path.realpath(a))
print("b", os.path.realpath(b))
with open(a, encoding="utf-8") as fh:
    blob = fh.read()
print("has_remote_default", "DEFAULT_REMOTE_CONTEXT_WINDOW" in blob)
print("cap16384", "DEFAULT_OUTPUT_TOKENS_CAP = 16384" in blob)
