"""
train_embedding.py — Textual Inversion embedding trainer (out-of-band).

Trains a Textual Inversion .safetensors for a character token.
Updates assets/embeddings/manifest.json with {token, path, sha256,
trained_on_model, trained_at}.

NEVER called during a run. This is a standalone utility.

Usage:
    python scripts/train_embedding.py --token="cpt_vega" --images=<dir> --steps=3000

Status: STUB — T13 placeholder for Gate C+ work.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_PATH = os.path.join(
    _REPO_ROOT, "assets", "embeddings", "manifest.json")


def _sha256(path):
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_manifest():
    """Load or initialize the embeddings manifest."""
    if os.path.isfile(_MANIFEST_PATH):
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": "1.0", "embeddings": []}


def _save_manifest(manifest):
    """Write the embeddings manifest."""
    os.makedirs(os.path.dirname(_MANIFEST_PATH), exist_ok=True)
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _update_manifest(token, embedding_path, model_name):
    """Add or update an embedding entry in the manifest."""
    manifest = _load_manifest()
    sha = _sha256(embedding_path)
    rel_path = os.path.relpath(embedding_path, _REPO_ROOT)

    # Remove existing entry for this token
    manifest["embeddings"] = [
        e for e in manifest["embeddings"]
        if e.get("token") != token
    ]

    # Add new entry
    manifest["embeddings"].append({
        "token": token,
        "path": rel_path,
        "sha256": sha,
        "trained_on_model": model_name,
        "trained_at": datetime.now().isoformat(),
    })

    _save_manifest(manifest)
    log.info("[TrainEmbedding] Manifest updated: %s -> %s", token, rel_path)


def train(token, images_dir, steps=3000, model_name="sd3.5_medium"):
    """Train a textual inversion embedding.

    Args:
        token: The token name (e.g., "cpt_vega").
        images_dir: Directory containing training images.
        steps: Number of training steps.
        model_name: Base model used for training.

    Returns:
        Path to the trained .safetensors file.

    Status: STUB — returns a placeholder path. Real training requires
    integration with ComfyUI's textual inversion training pipeline
    or a standalone trainer (kohya_ss, etc.).
    """
    # Validate inputs
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    log.info("[TrainEmbedding] Token: %s, Images: %d, Steps: %d, Model: %s",
             token, len(image_files), steps, model_name)

    # ── STUB: Real training goes here ─────────────────────────────
    # When implemented, this will:
    #   1. Load the base model
    #   2. Create a textual inversion embedding for the token
    #   3. Train for `steps` iterations on the provided images
    #   4. Save as .safetensors
    #   5. Update manifest
    #
    # For now, raise NotImplementedError to make it clear this is a stub.
    raise NotImplementedError(
        "Embedding training is a T13 stub. Actual training will be "
        "implemented post-Gate B when the animation pipeline is validated. "
        "Use ComfyUI's built-in textual inversion training or kohya_ss "
        "for now."
    )


def main():
    parser = argparse.ArgumentParser(
        description="OldTimeRadio Textual Inversion Embedding Trainer (STUB)")
    parser.add_argument("--token", required=True,
                        help="Token name for the embedding (e.g., cpt_vega)")
    parser.add_argument("--images", required=True,
                        help="Directory containing training images")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Training steps (default: 3000)")
    parser.add_argument("--model", default="sd3.5_medium",
                        help="Base model name (default: sd3.5_medium)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    try:
        result = train(args.token, args.images, args.steps, args.model)
        print(f"Embedding saved: {result}")
    except NotImplementedError as e:
        print(f"STUB: {e}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
