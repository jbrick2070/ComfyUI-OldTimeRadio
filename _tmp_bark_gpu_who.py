"""List GPU processes without killing anything."""
from __future__ import annotations

import subprocess

out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv",
    ],
    text=True,
)
print(out)
print("---")
out2 = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv",
    ],
    text=True,
)
print(out2)
