"""Ensure the pinned upscale-engine checkpoints are present under
``models/upscale_models/``. Idempotent; safe to run every session.

Currently manages ONE asset: Real-ESRGAN x2plus (BSD-3-Clause), used by the
shipped ``spandrel_esrgan`` upscale engine.

Behavior:
* If the file exists at ``<comfy_root>/models/upscale_models/<filename>`` AND
  its SHA-256 matches the pinned value, exit 0 with a "already present" line.
* If the file exists but SHA mismatches (corrupt / wrong version), exit non-zero
  with the actual hash printed so the operator can decide (delete + re-run, or
  pin a new hash).
* If the file is missing, download from the pinned URL with a bounded retry
  ladder, atomic-rename after SHA verify, exit 0.

Usage:
    python scripts/ensure_upscale_models.py
    python scripts/ensure_upscale_models.py --print-sha  # print + exit; no download

First-run bootstrapping (when the SHA is empty in ``ASSETS`` below): the
downloader fetches, computes the SHA, prints it clearly, and exits 0. The
operator then pastes the SHA into ``ASSETS[...]["sha256"]`` in a follow-up
commit; subsequent runs verify against the pinned value.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# ComfyUI root is <comfy>/custom_nodes/ComfyUI-OldTimeRadio/, so parents[2] = <comfy>.
COMFY_ROOT = REPO_ROOT.parents[1]
MODELS_DIR = COMFY_ROOT / "models" / "upscale_models"


# Pinned checkpoint catalog. Empty sha256 = first-run bootstrap; the downloader
# fetches, computes the SHA, prints it, and exits 0. The operator pastes the
# printed SHA back into this dict in a follow-up commit.
ASSETS: "list[dict]" = [
    {
        "asset_id": "real-esrgan-x2plus",
        "filename": "RealESRGAN_x2plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.1/RealESRGAN_x2plus.pth",
        # PINNED 2026-08-08 from the local copy of the v0.2.1 release asset,
        # after loading it through spandrel and confirming ESRGAN / scale=2 /
        # 3->3 ch / tags ['64nf','23nb','unshuffle'] at 67,061,725 bytes.
        # Must match eng_spandrel_esrgan.SpandrelEsrgan._model_sha256.
        "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        "license": "BSD-3-Clause",
        "notes": "Real-ESRGAN x2plus super-resolution weights; used by the "
                 "shipped `spandrel_esrgan` upscale engine.",
    },
]


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download_atomic(url: str, dest: Path, *, retries: int = 3,
                     timeout: float = 60.0) -> None:
    """Download ``url`` to ``dest`` via a tempfile + atomic rename.

    Bounded retry ladder (retries * exponential backoff). On success ``dest``
    exists and is complete. On repeated failure, no partial file is left
    behind (tempfile is unlinked in the finally block).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: "Exception | None" = None
    for attempt in range(1, retries + 1):
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{dest.name}.", dir=str(dest.parent))
        os.close(tmp_fd)
        tmp = Path(tmp_path)
        try:
            print(f"[ensure_upscale_models] downloading (try {attempt}/{retries}): {url}",
                  flush=True)
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            # Atomic rename.
            os.replace(str(tmp), str(dest))
            print(f"[ensure_upscale_models] downloaded {dest}", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            print(f"[ensure_upscale_models] attempt {attempt} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        if attempt < retries:
            backoff = min(2 ** attempt, 30)
            print(f"[ensure_upscale_models] sleeping {backoff}s before retry", flush=True)
            time.sleep(backoff)
    raise RuntimeError(
        f"download failed after {retries} attempt(s): {url} :: "
        f"{type(last_err).__name__}: {last_err}")


def ensure_one(asset: dict, *, print_only: bool) -> int:
    """Ensure a single asset is present + SHA-valid.

    Returns 0 on success, non-zero on unrecoverable mismatch.
    """
    filename = asset["filename"]
    dest = MODELS_DIR / filename
    pinned_sha = str(asset.get("sha256") or "")

    if dest.is_file():
        actual = _sha256_file(dest)
        if not pinned_sha:
            # First-run bootstrap: print the hash so the operator can pin it.
            print(f"[ensure_upscale_models] BOOTSTRAP {asset['asset_id']}", flush=True)
            print(f"  filename : {filename}", flush=True)
            print(f"  path     : {dest}", flush=True)
            print(f"  sha256   : {actual}", flush=True)
            print(f"  license  : {asset.get('license', '<unknown>')}", flush=True)
            print(f"  notes    : {asset.get('notes', '')}", flush=True)
            print(f"[ensure_upscale_models] paste the sha256 into ASSETS[...] "
                  f"in this script (scripts/ensure_upscale_models.py) so "
                  f"subsequent runs verify against it.", flush=True)
            return 0
        if actual == pinned_sha:
            print(f"[ensure_upscale_models] OK {asset['asset_id']} present + "
                  f"sha matches ({actual})", flush=True)
            return 0
        print(f"[ensure_upscale_models] SHA MISMATCH {asset['asset_id']}", flush=True)
        print(f"  path     : {dest}", flush=True)
        print(f"  actual   : {actual}", flush=True)
        print(f"  pinned   : {pinned_sha}", flush=True)
        print(f"[ensure_upscale_models] file is corrupt or wrong version. "
              f"Delete it and re-run to re-download, OR pin a new sha256 in "
              f"ASSETS[...] if you deliberately swapped versions.", flush=True)
        return 2

    if print_only:
        print(f"[ensure_upscale_models] MISSING {asset['asset_id']} (would download "
              f"from {asset['url']!r} to {dest}) -- --print-sha given, not "
              f"downloading", flush=True)
        return 3

    # Download.
    _download_atomic(asset["url"], dest)
    actual = _sha256_file(dest)
    if pinned_sha and actual != pinned_sha:
        print(f"[ensure_upscale_models] DOWNLOAD SHA MISMATCH", flush=True)
        print(f"  actual   : {actual}", flush=True)
        print(f"  pinned   : {pinned_sha}", flush=True)
        # Unlink the bad file so a rerun doesn't skip verification.
        try:
            dest.unlink()
        except OSError:
            pass
        return 2
    # Success (matches pin OR first-run bootstrap).
    if not pinned_sha:
        print(f"[ensure_upscale_models] BOOTSTRAP {asset['asset_id']}", flush=True)
        print(f"  filename : {filename}", flush=True)
        print(f"  path     : {dest}", flush=True)
        print(f"  sha256   : {actual}", flush=True)
        print(f"[ensure_upscale_models] paste the sha256 into ASSETS[...] "
              f"in this script for pin-on-next-run.", flush=True)
    else:
        print(f"[ensure_upscale_models] OK {asset['asset_id']} downloaded + "
              f"sha matches ({actual})", flush=True)
    return 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-sha", action="store_true",
        help="print pinned catalog entries and what would be fetched; do not download")
    args = parser.parse_args(argv)
    rc = 0
    for asset in ASSETS:
        one_rc = ensure_one(asset, print_only=args.print_sha)
        if one_rc != 0:
            rc = one_rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
