"""
MiniMax H3 Fetcher -- downloads the H3 stack into the ComfyUI models tree.

Puts each artifact where ComfyUI's loaders actually look, instead of the
HuggingFace blob cache. Because prestartup_script.py sets HF_HOME to
<ComfyUI>/models/huggingface, a bare `hf download Comfy-Org/MiniMax-H3`
writes ~42 GB into models/huggingface/hub/ as blobs plus symlinks -- on
disk, but invisible to the diffusion_models / text_encoders / vae
loaders. This script uses local_dir so the repo's path prefix is
preserved under the models root and every file lands in its real home.

What it does:
  1. Resolves the live models root (folder_paths if importable, else the
     prestartup-derived user-directory path, else --models-root)
  2. Refuses to start without enough free disk, with headroom
  3. Downloads one variant (FL2VA or Ref2VA) -- never both, they are the
     same size and only one is needed
  4. Skips any file already present at the correct size
  5. Prints a per-file plan with totals before touching the network

LICENSING -- read before running:
  The US is an Excluded Territory under the MiniMax H3 Community License
  (SS I.5), so running these weights requires the written authorization on
  file for Blueberrky Kale Yoga Books. See
  docs/licensing/MINIMAX_H3_AUTHORIZATION.md. That grant does not extend
  to anyone else who clones this repo.

VRAM -- this download does not make H3 runnable:
  The smallest published diffusion model is 15.6 GB and the smallest text
  encoder is 14.6 GB, both over the 14.5 GB ceiling. H3 needs sequential
  residency plus RAM streaming, which ROADMAP.md currently lists under
  discarded ideas. See docs/superpowers/specs/2026-08-08-minimax-h3-recipe-gate.md.

Usage:
    python fetch_minimax_h3.py --dry-run              # plan only, no network
    python fetch_minimax_h3.py                        # comfy profile, FL2VA
    python fetch_minimax_h3.py --profile gguf         # smallest VRAM path
    python fetch_minimax_h3.py --variant ref2va
    python fetch_minimax_h3.py --models-root "E:\\ComfyModels"
"""

import argparse
import os
import shutil
import sys

# Fallback root, derived the same way prestartup_script.py derives HF_HOME:
# three directories up from this repo, then models/.
DERIVED_ROOT = r"C:\Users\jeffr\Documents\ComfyUI\models"

COMFY_REPO = "Comfy-Org/MiniMax-H3"
GGUF_REPO = "Abiray/MiniMax-H3-GGUF"

# Sizes are the Hub's own byte counts, used for the disk check and to
# detect half-finished downloads. Update if upstream repacks.
SHARED_VAE = [
    ("vae/minimax_h3_video_vae_fp16.safetensors", 5_207_808_496),
    ("vae/minimax_h3_audio_vae_fp32.safetensors", 605_254_808),
]

PROFILES = {
    # ComfyUI-native safetensors. Repo layout already mirrors ComfyUI's
    # subfolder names, so local_dir at the models root is exact.
    "comfy": {
        "repo": COMFY_REPO,
        "variant_file": "diffusion_models/minimax_h3_{v}_pruned_fp8_scaled.safetensors",
        "variant_size": 20_958_205_608,
        "variant_case": str.lower,
        "extra": [
            ("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", 15_687_142_551),
        ],
        "vae_repo": COMFY_REPO,
    },
    # Smallest VRAM path. Note the encoder comes from the GGUF repo, but
    # that repo's nvfp4_awq safetensors is listed at 27.1 GB -- byte-for-byte
    # Comfy-Org's int8_convrot, so it looks mislabeled. Q4_K_M is the real
    # small encoder; never pull nvfp4 from here.
    "gguf": {
        "repo": GGUF_REPO,
        "variant_file": "unet/MiniMax-H3-{v}-Q3_K_M.gguf",
        "variant_size": 15_567_048_992,
        "variant_case": None,  # exact token, see VARIANT_TOKENS
        "extra": [
            ("text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf", 14_576_977_960),
        ],
        "vae_repo": GGUF_REPO,
    },
}

VARIANT_TOKENS = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}

HEADROOM_BYTES = 10 * 1000**3  # never fill the volume to the brim


def gb(n):
    return f"{n / 1000**3:.1f} GB"


def resolve_models_root(override):
    """Live root wins over the derived one -- the desktop app splits its
    install directory from its user directory and the two can disagree."""
    if override:
        return os.path.abspath(override), "--models-root"
    try:
        import folder_paths  # only importable inside ComfyUI's env
        return folder_paths.models_dir, "folder_paths.models_dir"
    except Exception:
        return DERIVED_ROOT, "derived fallback (folder_paths unavailable)"


def build_plan(profile, variant):
    spec = PROFILES[profile]
    token = VARIANT_TOKENS[variant]
    if spec["variant_case"]:
        token = spec["variant_case"](token)

    plan = [(spec["repo"], spec["variant_file"].format(v=token), spec["variant_size"])]
    plan += [(spec["repo"], f, s) for f, s in spec["extra"]]
    plan += [(spec["vae_repo"], f, s) for f, s in SHARED_VAE]
    return plan


def main():
    ap = argparse.ArgumentParser(description="Fetch MiniMax H3 into the ComfyUI models tree.")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="comfy")
    ap.add_argument("--variant", choices=sorted(VARIANT_TOKENS), default="fl2va")
    ap.add_argument("--models-root", default=None, help="override the destination root")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    root, how = resolve_models_root(args.models_root)
    plan = build_plan(args.profile, args.variant)

    print(f"\nMiniMax H3 fetch -- profile={args.profile} variant={args.variant}")
    print(f"Models root: {root}")
    print(f"  resolved via {how}\n")

    todo, have = [], []
    for repo, fname, size in plan:
        dest = os.path.join(root, fname.replace("/", os.sep))
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            have.append((fname, size))
        else:
            todo.append((repo, fname, size, dest))

    for fname, size in have:
        print(f"  [have] {fname}  ({gb(size)})")
    for repo, fname, size, _ in todo:
        print(f"  [get ] {fname}  ({gb(size)})  <- {repo}")

    need = sum(s for _, _, s, _ in todo)
    print(f"\nTo download: {gb(need)} across {len(todo)} file(s)")

    if not todo:
        print("Everything is already in place. Nothing to do.")
        return 0

    if not os.path.isdir(root):
        print(f"\nERROR: models root does not exist: {root}")
        print("Create it or pass --models-root. Refusing to guess.")
        return 2

    free = shutil.disk_usage(root).free
    print(f"Free on that volume: {gb(free)}")
    if free < need + HEADROOM_BYTES:
        print(f"\nERROR: need {gb(need + HEADROOM_BYTES)} including headroom, have {gb(free)}.")
        print("Free up space, or point --models-root at another volume and add an")
        print("extra_model_paths.yaml entry for it.")
        return 2

    if args.dry_run:
        print("\n--dry-run: stopping before any network access.")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("\nERROR: huggingface_hub not importable. Run this with ComfyUI's interpreter:")
        print(r'  C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\fetch_minimax_h3.py')
        return 2

    print()
    for i, (repo, fname, size, dest) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {fname}  ({gb(size)})")
        # local_dir keeps the repo's path prefix and writes real files --
        # this is what keeps the weights out of the blob cache.
        hf_hub_download(repo_id=repo, filename=fname, local_dir=root)
        actual = os.path.getsize(dest) if os.path.exists(dest) else 0
        if actual != size:
            print(f"  WARNING: expected {gb(size)}, got {gb(actual)}. Upstream may have repacked.")

    print(f"\nDone. {len(todo)} file(s) written under {root}")
    print("Next: gate step 0 in docs/superpowers/specs/2026-08-08-minimax-h3-recipe-gate.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
