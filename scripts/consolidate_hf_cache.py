"""consolidate_hf_cache.py -- Safe Hugging Face cache consolidation.

Recovers disk space by moving or deleting duplicate model copies that
some HF-using tools accidentally seed in non-canonical cache locations
(``~/.cache/huggingface``, ComfyUI Desktop's ``AppData`` shadow, etc.).

Why a custom utility instead of ``rmtree``:
    - Some "rogue" copies are actually the only good copy on the
      machine -- the canonical copy may be partial / corrupt from a
      previous failed download. We detect that via a 1 MB sample-hash
      of the first blob in each model's ``blobs/`` directory.
    - HF cache layout is content-addressed, so two valid copies of the
      same model SHOULD have the same first-blob hash. Hash mismatch
      means at least one is corrupt -- we skip and ask the user to
      resolve manually instead of silently picking a winner.
    - Size also matters: even with matching hashes, if the canonical
      copy is smaller than the rogue copy, the canonical may have lost
      blobs (partial download). Defensive comparison keeps the larger
      copy as canonical.

Usage:
    python scripts/consolidate_hf_cache.py             # dry run, prints actions
    python scripts/consolidate_hf_cache.py --execute   # actually move/delete

The canonical destination is HF_HOME (env var) if set, else falls back
to ComfyUI's ``models/huggingface`` convention. Rogue paths are the
known offenders on Windows: user ``.cache``, ComfyUI Desktop's bundled
``resources`` shadow, and any custom ``HF_HUB_CACHE`` if it differs from
HF_HOME.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path


def _canonical_cache_root() -> Path:
    """Return the canonical ``hub/`` cache root.

    Resolution order:
    1. HF_HOME env var (HF standard)
    2. comfy_models_dir() / huggingface (project convention)
    3. ~/Documents/ComfyUI/models/huggingface (Jeffrey-on-Windows default)
    """
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home) / "hub"

    try:
        # Best-effort import without making this script depend on
        # ComfyUI's folder_paths package.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
        from _otr_paths import comfy_models_dir  # type: ignore
        return comfy_models_dir() / "huggingface" / "hub"
    except Exception:  # noqa: BLE001
        pass

    return Path.home() / "Documents" / "ComfyUI" / "models" / "huggingface" / "hub"


CANONICAL = _canonical_cache_root()

ROGUE_PATHS = [
    Path.home() / ".cache" / "huggingface" / "hub",
    Path.home() / "AppData" / "Local" / "Programs" / "ComfyUI" / "resources" / "ComfyUI" / "models" / "huggingface" / "hub",
]


def dir_size(p: Path) -> int:
    """Sum file sizes under ``p``. Best-effort; permission errors skipped."""
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def file_count(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file())


def hash_first_blob(p: Path) -> str:
    """Sample-hash the first blob to detect partial / corrupt downloads.

    HF cache layout: ``models--{...}/blobs/<sha256-name>`` are the actual
    weight chunks; ``snapshots/`` are dirs of symlinks pointing at blobs.
    Hashing the first blob's first 1 MB is a fast smoke test -- two valid
    copies of the same model produce identical hashes; a partial copy
    produces a different hash because the truncated tail changes the
    first-MB SHA when the file is shorter than 1 MB.
    """
    blobs_dir = p / "blobs"
    if not blobs_dir.exists():
        return "no_blobs"
    blobs = list(blobs_dir.iterdir())
    if not blobs:
        return "empty_blobs"
    first = sorted(blobs)[0]
    h = hashlib.sha256()
    try:
        with open(first, "rb") as f:
            h.update(f.read(1024 * 1024))
    except OSError as exc:  # noqa: BLE001
        return f"read_error:{type(exc).__name__}"
    return h.hexdigest()[:16]


def fmt_gb(bytes_: int) -> str:
    if bytes_ < 1 << 20:
        return f"{bytes_ / (1 << 10):.1f} KB"
    if bytes_ < 1 << 30:
        return f"{bytes_ / (1 << 20):.1f} MB"
    return f"{bytes_ / (1 << 30):.2f} GB"


def inventory(dry_run: bool = True) -> int:
    """Walk rogue paths, print/execute the action plan. Returns exit code."""
    CANONICAL.mkdir(parents=True, exist_ok=True)
    print(f"Canonical: {CANONICAL}")
    print(f"Mode:      {'DRY RUN' if dry_run else 'EXECUTE'}")
    print()

    actions = []  # list of (action, src_dir, dst_dir_or_None)
    skipped = []  # list of (reason, model_name)
    bytes_recoverable = 0

    for rogue in ROGUE_PATHS:
        if not rogue.exists():
            print(f"[SKIP path] {rogue} (not present)")
            continue
        print(f"\n[SCAN] {rogue}")
        for model_dir in sorted(rogue.glob("models--*")):
            target = CANONICAL / model_dir.name
            r_size = dir_size(model_dir)
            r_count = file_count(model_dir)
            r_hash = hash_first_blob(model_dir)

            if r_count == 0 or r_hash in ("no_blobs", "empty_blobs"):
                # Stub directories with zero or near-zero content.
                # Safe to delete outright -- nothing to lose.
                print(f"  [WOULD DELETE STUB] {model_dir.name} ({fmt_gb(r_size)}, {r_count} files)")
                actions.append(("delete", model_dir, None))
                bytes_recoverable += r_size
                continue

            if not target.exists():
                # Rogue has data, canonical doesn't -- move it.
                print(f"  [WOULD MOVE]  {model_dir.name} ({fmt_gb(r_size)}, {r_count} files)")
                actions.append(("move", model_dir, target))
                continue

            t_size = dir_size(target)
            t_hash = hash_first_blob(target)

            if r_hash == t_hash and t_size >= r_size:
                print(f"  [WOULD DELETE ROGUE] {model_dir.name} ({fmt_gb(r_size)} duplicate; canonical {fmt_gb(t_size)})")
                actions.append(("delete", model_dir, None))
                bytes_recoverable += r_size
            else:
                reason = (
                    f"hash_mismatch ({r_hash} vs canonical {t_hash})"
                    if r_hash != t_hash
                    else f"canonical_smaller ({fmt_gb(t_size)} vs rogue {fmt_gb(r_size)})"
                )
                print(f"  [CONFLICT - SKIP] {model_dir.name}  reason={reason}")
                skipped.append((reason, model_dir.name))

    print(f"\n--- Summary ---")
    print(f"  Actions queued:   {len(actions)}")
    print(f"  Skipped conflicts: {len(skipped)}")
    print(f"  Bytes recoverable: {fmt_gb(bytes_recoverable)}")

    if skipped:
        print()
        print("Conflicts to resolve manually:")
        for reason, name in skipped:
            print(f"  - {name}  ({reason})")

    if dry_run:
        print()
        print("Re-run with --execute to perform.")
        return 0

    if not actions:
        print()
        print("Nothing to do.")
        return 0

    print()
    print("Executing...")
    for action, src, dst in actions:
        try:
            if action == "move":
                print(f"  MOVE   {src.name}")
                shutil.move(str(src), str(dst))
            elif action == "delete":
                print(f"  DELETE {src.name}")
                shutil.rmtree(src)
        except Exception as exc:  # noqa: BLE001 -- best effort per action
            print(f"  ERROR on {src.name}: {type(exc).__name__}: {exc}")
    print("Done.")
    return 0


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    sys.exit(inventory(dry_run=not execute))
