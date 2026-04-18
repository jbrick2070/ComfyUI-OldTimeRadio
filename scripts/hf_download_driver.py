"""
hf_download_driver.py
=====================

Tiny child process invoked by scripts/download_video_stack_weights.ps1.
Reads a JSON spec from a file path passed as argv[1] (path approach
avoids cmd/PowerShell quote-stripping when passing JSON on the command
line).  Runs the matching huggingface_hub call and prints an OK/FAIL line.

Spec keys:
    repo   (str)              -- HF repo id
    target (str | "HF_CACHE") -- local dir (or the literal HF_CACHE sentinel
                                 to dump into the default transformers cache)
    kind   (str)              -- one of:
                                 "snapshot"           -> snapshot_download to target
                                 "snapshot_to_cache"  -> snapshot_download into HF cache
                                 "file"               -> hf_hub_download of spec["file"]
    file   (str, optional)    -- required when kind=="file"

HF_TOKEN is picked up from the environment for gated repos
(e.g. black-forest-labs/FLUX.1-dev).  Public repos don't need it.

Exit codes:
    0  success
    1  failure during download (traceback printed)
    3  unknown kind
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("FAIL hf_download_driver: missing spec-file path argument")
        return 2

    spec_path = argv[1]
    try:
        # utf-8-sig tolerates files with or without a BOM (defense in depth:
        # PowerShell emits no-BOM intentionally but some tools resave with BOM)
        with open(spec_path, encoding="utf-8-sig") as fh:
            spec = json.load(fh)
    except FileNotFoundError:
        print(f"FAIL hf_download_driver: spec file not found: {spec_path}")
        return 2
    except json.JSONDecodeError as exc:
        print(f"FAIL hf_download_driver: bad JSON in {spec_path}: {exc}")
        return 2

    repo = spec["repo"]
    target = spec["target"]
    kind = spec["kind"]
    token = os.environ.get("HF_TOKEN") or None

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError as exc:
        print(f"FAIL hf_download_driver: huggingface_hub missing: {exc}")
        return 2

    try:
        if kind == "snapshot":
            os.makedirs(target, exist_ok=True)
            out = snapshot_download(
                repo_id=repo,
                local_dir=target,
                local_dir_use_symlinks=False,
                token=token,
                allow_patterns=None,
                ignore_patterns=[
                    "*.gguf",
                    "*.onnx",
                    "*_fp4.safetensors",
                    "*.msgpack",
                ],
            )
            print(f"OK snapshot {repo} -> {out}")
            return 0

        if kind == "snapshot_to_cache":
            out = snapshot_download(repo_id=repo, token=token)
            print(f"OK cache {repo} -> {out}")
            return 0

        if kind == "file":
            os.makedirs(target, exist_ok=True)
            fname = spec["file"]
            out = hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=target,
                local_dir_use_symlinks=False,
                token=token,
            )
            print(f"OK file {repo}::{fname} -> {out}")
            return 0

        print(f"SKIP unknown kind {kind}")
        return 3

    except Exception as exc:
        print(f"FAIL {repo}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
