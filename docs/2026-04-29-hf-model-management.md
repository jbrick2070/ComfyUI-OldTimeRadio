# Hugging Face Model Management & Consolidation (OTR)

**Date:** 2026-04-29
**Objective:** Establish a unified, fail-safe architecture for Hugging Face model caching locally, and provide a robust model resolution path for users installing the OTR node package.

## 1. Local Environment Setup (Canonical Storage)

Do not rely on directory junctions. The canonical method to prevent model duplication across `~/.cache` and `AppData` is setting the `HF_HOME` environment variable at the OS level.

```cmd
setx HF_HOME "C:\Users\jeffr\Documents\ComfyUI\models\huggingface"
```

**Note:** Restart your shell and ComfyUI after setting. Models will automatically download to `<HF_HOME>/hub/`.

**Warning:** Ensure the target canonical directory is NOT subject to OneDrive sync.

## 2. Model Resolver for Shipped Nodes

When shipping OTR nodes, you cannot dictate the user's `HF_HOME`. Use the following resolver helper function to gracefully discover models. It honors explicit overrides, ComfyUI conventions, and defaults back to standard Hugging Face behavior.

```python
import os
from pathlib import Path


def resolve_hf_model_path(repo_id: str) -> str:
    """
    Resolve a HuggingFace repo_id to a local path.

    Resolution order:
    1. OTR_MODELS_DIR env var (user explicit override)
    2. HF_HOME env var + /hub/ (HF standard)
    3. ComfyUI's models/huggingface/hub (project convention)
    4. Default HF cache (~/.cache/huggingface/hub)
    5. Fall back to repo_id and let huggingface_hub download

    Always returns either a valid local path or the repo_id itself --
    callers pass the result to from_pretrained() which handles both.
    """
    cache_dirname = "models--" + repo_id.replace("/", "--")

    candidates = []
    if os.environ.get("OTR_MODELS_DIR"):
        candidates.append(
            Path(os.environ["OTR_MODELS_DIR"]) / "huggingface" / "hub" / cache_dirname
        )
    if os.environ.get("HF_HOME"):
        candidates.append(
            Path(os.environ["HF_HOME"]) / "hub" / cache_dirname
        )

    try:
        from ._otr_paths import comfy_models_dir
        candidates.append(comfy_models_dir() / "huggingface" / "hub" / cache_dirname)
    except ImportError:
        pass

    candidates.append(Path.home() / ".cache" / "huggingface" / "hub" / cache_dirname)

    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return str(c.parent.parent)

    return repo_id
```

## 3. Safe Cache Consolidation Utility

Use this script to clean up duplicate downloads across rogue cache directories. Unlike standard `rmtree` scripts, this verifies file sizes and performs a blob hash-check to ensure the canonical copy is not partial or corrupt before deleting the rogue duplicate.

```python
import hashlib
import os
import shutil
import sys
from pathlib import Path

CANONICAL = Path(os.path.expanduser(
    "~/Documents/ComfyUI/models/huggingface/hub"
))

ROGUE_PATHS = [
    Path(os.path.expanduser("~/.cache/huggingface/hub")),
    Path(os.path.expanduser(
        "~/AppData/Local/Programs/ComfyUI/resources/ComfyUI/models/huggingface/hub"
    )),
]


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def file_count(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file())


def hash_first_blob(p: Path) -> str:
    """Sample-hash one blob file to detect partial/corrupt downloads."""
    blobs_dir = p / "blobs"
    if not blobs_dir.exists():
        return "no_blobs"
    blobs = list(blobs_dir.iterdir())
    if not blobs:
        return "empty_blobs"
    first = sorted(blobs)[0]
    h = hashlib.sha256()
    with open(first, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def inventory(dry_run: bool = True):
    CANONICAL.mkdir(parents=True, exist_ok=True)
    print(f"Canonical: {CANONICAL}")
    print(f"Mode: {'DRY RUN (nothing will be moved or deleted)' if dry_run else 'EXECUTE'}\n")

    actions = []
    for rogue in ROGUE_PATHS:
        if not rogue.exists():
            print(f"[SKIP] {rogue} (not present)")
            continue
        print(f"\n[SCAN] {rogue}")
        for model_dir in rogue.glob("models--*"):
            target = CANONICAL / model_dir.name
            r_size = dir_size(model_dir)
            r_count = file_count(model_dir)
            r_hash = hash_first_blob(model_dir)

            if not target.exists():
                print(f"  [WOULD MOVE] {model_dir.name} ({r_size/1e9:.2f} GB, {r_count} files)")
                actions.append(("move", model_dir, target))
            else:
                t_size = dir_size(target)
                t_hash = hash_first_blob(target)

                if r_hash == t_hash and t_size >= r_size:
                    print(f"  [WOULD DELETE ROGUE] {model_dir.name}")
                    actions.append(("delete", model_dir, None))
                else:
                    print(f"  [CONFLICT - SKIP] {model_dir.name}")
                    print(f"    => canonical may be partial; resolve manually")

    print(f"\n--- Summary: {len(actions)} actions ---")
    if dry_run:
        print("Re-run with --execute to perform.")
        return

    for action, src, dst in actions:
        if action == "move":
            print(f"MOVE {src.name}")
            shutil.move(str(src), str(dst))
        elif action == "delete":
            print(f"DELETE {src.name}")
            shutil.rmtree(src)


if __name__ == "__main__":
    execute = "--execute" in sys.argv
    inventory(dry_run=not execute)
```

## Implementation chain

1. **Set HF_HOME at OS level** -- once via `setx`, then restart ComfyUI
2. **Add `resolve_hf_model_path()` to `nodes/_otr_paths.py`** -- becomes the canonical lookup helper for any OTR node loading from HF
3. **Add `scripts/consolidate_hf_cache.py`** -- runnable utility, dry-run by default, `--execute` to actually move/delete
4. **Run consolidation in dry-run mode first** -- inspect the actions list before committing
5. **Run with `--execute`** -- recovers ~50 GB of duplicates without risk to the canonical copy

## Why this beats junctions

- Junctions can break (someone deletes the link target, OneDrive moves the source folder, Windows update repairs default paths to real folders)
- `setx` writes to `HKCU\Environment` -- inherited by every new process at launch -- and survives reboots
- Hash-check before delete protects against the case where a "rogue" copy is actually the only good copy and the "canonical" copy is partial/corrupt
- Resolver function lets shipped nodes work for any user regardless of their HF_HOME / ComfyUI install layout
