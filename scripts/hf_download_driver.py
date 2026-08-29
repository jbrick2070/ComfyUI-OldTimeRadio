"""
hf_download_driver.py
=====================

Tiny child process invoked by downloader wrappers (currently
scripts/download_ltx_0_9_8.ps1 and the 4060 model fetcher).
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
    revision        (str, optional) -- pin an exact commit/tag/branch for kind=="file"
                                       (and the snapshot kinds); passed straight to
                                       huggingface_hub so a repo re-tag cannot swap
                                       the weight under us.
    expected_size   (int, optional) -- byte length the materialized "file" MUST match.
    expected_sha256 (str, optional) -- lowercase hex SHA-256 the materialized "file"
                                       MUST match. Either check failing => exit 1
                                       (a wrong-revision / truncated / corrupt weight
                                       is rejected LOUD, never accepted silently).

HF_TOKEN is picked up from the environment for gated repos
(e.g. black-forest-labs/FLUX.1-dev).  Public repos don't need it.

Exit codes:
    0  success
    1  failure during download (traceback printed)
    3  unknown kind
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback


def _verify_materialized(path: str, expected_size, expected_sha256) -> str:
    """Return an error string if ``path`` fails the size / SHA-256 checks, else "".

    A wrong-revision or truncated/corrupt weight is caught HERE (fail loud) rather
    than surfacing later as an opaque runtime load/decode error. Both checks are
    optional; a check that is not requested (None/empty) is skipped."""
    try:
        actual_size = os.path.getsize(path)
    except OSError as exc:
        return "cannot stat materialized file %r: %s" % (path, exc)
    if expected_size is not None:
        if int(actual_size) != int(expected_size):
            return ("size mismatch: got %d bytes, expected %d"
                    % (actual_size, int(expected_size)))
    if expected_sha256:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        actual = h.hexdigest().lower()
        if actual != str(expected_sha256).strip().lower():
            return ("SHA-256 mismatch: got %s, expected %s"
                    % (actual, str(expected_sha256).strip().lower()))
    return ""


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
    revision = spec.get("revision") or None
    expected_size = spec.get("expected_size")
    expected_sha256 = spec.get("expected_sha256")

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError as exc:
        print(f"FAIL hf_download_driver: huggingface_hub missing: {exc}")
        return 2

    # Optional spec keys (PowerShell-side override for repos that need
    # a narrower filter — e.g. PuLID where we only want pulid_flux*.safetensors
    # and not the whole eva_clip / facexlib dir).
    allow_patterns = spec.get("allow_patterns")  # list[str] or None
    ignore_patterns = spec.get("ignore_patterns") or [
        "*.gguf",
        "*.onnx",
        "*_fp4.safetensors",
        "*.msgpack",
    ]

    try:
        if kind == "snapshot":
            os.makedirs(target, exist_ok=True)
            out = snapshot_download(
                repo_id=repo,
                local_dir=target,
                local_dir_use_symlinks=False,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            print(f"OK snapshot {repo} -> {out}")
            return 0

        if kind == "snapshot_to_cache":
            out = snapshot_download(repo_id=repo, token=token, revision=revision)
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
                revision=revision,
            )
            err = _verify_materialized(out, expected_size, expected_sha256)
            if err:
                print(f"FAIL {repo}::{fname} integrity: {err}")
                return 1
            _rev = f" @ {revision}" if revision else ""
            print(f"OK file {repo}::{fname}{_rev} -> {out}")
            return 0

        print(f"SKIP unknown kind {kind}")
        return 3

    except Exception as exc:
        print(f"FAIL {repo}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
