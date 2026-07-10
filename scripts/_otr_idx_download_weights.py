"""Download + validate the IndexTTS2 weights (Path B, default char voice).

This is the script named by eng_indextts2.py's fail-closed "weights
incomplete" error. Run it with the ISOLATED index-tts venv python (its
transformers pin ships huggingface_hub); the install wrapper
scripts/_otr_indextts2_install.ps1 does exactly that:

    <ComfyUI>/index-tts/.venv/Scripts/python.exe scripts/_otr_idx_download_weights.py

Downloads IndexTeam/IndexTTS-2 into <index-tts>/checkpoints (env override
OTR_INDEXTTS2_DIR) and FAILS LOUD (exit 1) if any expected artifact is
missing or either large checkpoint has the wrong byte size (truncated or
upstream-replaced download). Re-run safe: snapshot_download resumes.
"""
from __future__ import annotations

import os
import sys

_REPO_ID = "IndexTeam/IndexTTS-2"

# Byte sizes pinned from hf://models/IndexTeam/IndexTTS-2 (verified
# 2026-07-10; identical to the working install on the nv50 box). If upstream
# ever republishes different weights this validation fails LOUD -- re-derive
# the pins deliberately, never by accident.
_EXPECTED = {
    "config.yaml": None,
    "bpe.model": 475_997,
    "gpt.pth": 3_484_663_079,
    "s2mel.pth": 1_202_198_223,
    "feat1.pt": 57_170,
    "feat2.pt": 374_866,
    "wav2vec2bert_stats.pt": 9_343,
}


def _default_model_dir() -> str:
    env = os.environ.get("OTR_INDEXTTS2_DIR")
    if env:
        return env
    # realpath: junction-safe, same resolution the engine adapter uses.
    here = os.path.realpath(__file__)                 # <repo>/scripts/this.py
    repo_root = os.path.dirname(os.path.dirname(here))
    comfy_root = os.path.dirname(os.path.dirname(repo_root))
    return os.path.join(comfy_root, "index-tts", "checkpoints")


def main() -> int:
    model_dir = _default_model_dir()
    print(f"[idx-weights] repo:   {_REPO_ID}")
    print(f"[idx-weights] target: {model_dir}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[idx-weights] huggingface_hub is missing -- run this with the "
              "isolated index-tts venv python "
              "(see scripts/_otr_indextts2_install.ps1)")
        return 2

    snapshot_download(_REPO_ID, local_dir=model_dir)

    failures = []
    for name, size in _EXPECTED.items():
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            failures.append(f"missing: {name}")
        elif size is not None and os.path.getsize(path) != size:
            failures.append(
                f"size mismatch: {name} ({os.path.getsize(path)} != {size})")
    if not os.path.isdir(os.path.join(model_dir, "qwen0.6bemo4-merge")):
        failures.append("missing dir: qwen0.6bemo4-merge")

    if failures:
        print("[idx-weights] FAILED validation:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("[idx-weights] OK -- all expected artifacts present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
