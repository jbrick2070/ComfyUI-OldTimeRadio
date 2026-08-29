"""Preflight weight fetcher for the VIDEO lanes -- the one manual step left.

WHY THIS EXISTS. Every OTR video engine is fail-closed: a missing weight is a
NAMED refusal and nothing is fetched at render time. That rule is correct and
this script does not weaken it -- it fetches BEFORE a render, on purpose, when
the operator asks. The writer, image and audio models already auto-download
through the HF cache; the video lanes were the only manual step in the pack.

Kokoro VOICES are deliberately absent: `nodes/_otr_kokoro_voice_prefetch.py`
already fetches them at BOOT from `prestartup_script.py`, and duplicating that
here would write to a second models root the engine does not read.

Every source below is UNGATED (verified against the HF API 2026-08-29): no
account, no licence click, no token. The one gated video repo, Lightricks/
LTX-2.5, is deliberately NOT offered here -- it reports gated:"auto" and needs
the operator's own terms click, which a script must never paper over.

Usage:
    python scripts/otr_fetch_lane_weights.py --list
    python scripts/otr_fetch_lane_weights.py haunted
    python scripts/otr_fetch_lane_weights.py minimax_h3 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

#: lane -> list of (hf_repo, path_in_repo, models_subfolder)
#: Sizes in the comments are the real blob sizes read from the HF API.
LANES = {
    # ~3.8 GB total. No image model needed: the haunted lane is text_to_video
    # (accepts_still = False), so it also skips z_image entirely.
    #
    # THESE ARE THE v3 WEIGHTS, AND `mm-p_0.5.pth` IS NOT ONE OF THEM. The
    # first version of this bundle fetched mm-p_0.5 and no adapter, which
    # would have handed a fresh user 1.7 GB of the wrong file and left the
    # lane unable to start. The lane is `animatediff15_v3_haunted_video` ->
    # `GhostSignalV3HauntedEngine`, which extends the **V3** engine
    # (`motion_module_name = MM_V3_NAME = "v3_sd15_mm.ckpt"`) and is the one
    # sibling that also sets `lora_name = ADAPTER_V3_NAME`. Read off the
    # classes, not the docs. `mm-p_0.5.pth` belongs to the GOLDEN lane
    # (`GhostSignalEngine`), a different engine this bundle does not install.
    #
    # The adapter is small and easy to skip, and skipping it is not harmless:
    # the haunted lane's whole difference from the clean v3 lane is that LoRA
    # on the model path. Without it the render either fails the artifact check
    # or -- worse -- produces the CLEAN picture while stamping a haunted
    # receipt, which the engine's own docstring calls the one outcome the lane
    # may never produce.
    #
    # guoyww/animatediff is UNGATED (verified against the HF API, gated:False),
    # so both files download with no token and no licence click.
    "haunted": [
        ("Comfy-Org/stable-diffusion-v1-5-archive",
         "v1-5-pruned-emaonly-fp16.safetensors", "checkpoints"),        # 2.0 GB
        ("guoyww/animatediff",
         "v3_sd15_mm.ckpt", "animatediff_models"),                      # 1.67 GB
        ("guoyww/animatediff",
         "v3_sd15_adapter.ckpt", "loras"),                              # 0.10 GB
    ],
    # ~39.6 GB total, but it carries NATIVE AUDIO and is proven at 8 GB by
    # vram-recipe-lab/eightgb_bench (864x480, 90f, 24fps, requires_audio).
    # I2V: needs a first frame, so it also wants an image lane.
    "minimax_h3": [
        ("Comfy-Org/MiniMax-H3",
         "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         "diffusion_models"),                                            # 19.5 GB
        ("Comfy-Org/MiniMax-H3",
         "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
         "text_encoders"),                                               # 14.6 GB
        ("Comfy-Org/MiniMax-H3",
         "vae/minimax_h3_video_vae_fp16.safetensors", "vae"),            # 4.9 GB
        ("Comfy-Org/MiniMax-H3",
         "vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),            # 0.6 GB
    ],
    # ~9 GB. Ungated Comfy-Org repackages.
    "wan_ti2v": [
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
         "split_files/vae/wan_2.1_vae.safetensors", "vae"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
         "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
         "text_encoders"),
    ],
}

#: Convenience bundles: everything a named profile needs that is not already
#: auto-fetched by transformers (writer, musicgen) on first use.
BUNDLES = {
    "otr_nvidia_8gb_haunted": ["haunted"],
    "otr_nvidia_8gb_h3": ["minimax_h3"],
}


def models_root() -> str:
    """The one authority on where weights live -- never a hardcoded guess."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from nodes._otr_gguf_backend import _models_root
        return str(_models_root())
    except Exception:
        return os.environ.get("OTR_COMFYUI_MODELS_ROOT") or r"C:\ComfyUI-Models"


def human(n: float) -> str:
    return "%.2f GB" % (n / 1073741824.0)


def fetch(repo: str, path_in_repo: str, subfolder: str, root: str,
          dry_run: bool) -> bool:
    name = path_in_repo.split("/")[-1]
    dest_dir = os.path.join(root, subfolder)
    dest = os.path.join(dest_dir, name)
    url = "https://huggingface.co/%s/resolve/main/%s" % (repo, path_in_repo)

    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        print("  PRESENT  %-52s %s" % (name[:52], human(os.path.getsize(dest))))
        return True
    if dry_run:
        print("  WOULD GET %-51s <- %s" % (name[:51], repo))
        return True

    os.makedirs(dest_dir, exist_ok=True)
    tmp = dest + ".part"
    print("  FETCHING %-52s <- %s" % (name[:52], repo), flush=True)
    try:
        # Download to .part and rename only on success, so an interrupted
        # fetch never leaves a truncated file the engine would load.
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
        print("  OK       %-52s %s" % (name[:52], human(os.path.getsize(dest))))
        return True
    except Exception as exc:
        for leftover in (tmp,):
            try:
                os.remove(leftover)
            except OSError:
                pass
        print("  FAILED   %-52s %s" % (name[:52], str(exc)[:60]))
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lane", nargs="?", help="lane to fetch (see --list)")
    ap.add_argument("--list", action="store_true", help="list known lanes")
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be fetched, download nothing")
    args = ap.parse_args()

    if args.list or not args.lane:
        print("lanes:")
        for lane, files in LANES.items():
            print("  %-24s %d file(s)" % (lane, len(files)))
        print("")
        print("profile bundles (fetch everything a profile needs):")
        for b, lanes in BUNDLES.items():
            print("  %-24s = %s" % (b, " + ".join(lanes)))
        print("\nGATED, deliberately not offered: Lightricks/LTX-2.5 "
              "(accept the terms yourself, then set HF_TOKEN)")
        return 0

    if args.lane in BUNDLES:
        lanes = BUNDLES[args.lane]
    elif args.lane in LANES:
        lanes = [args.lane]
    else:
        print("unknown lane %r; try --list" % args.lane)
        return 2

    root = models_root()
    print("models root: %s" % root)
    print("target: %s -> %s" % (args.lane, ", ".join(lanes)))
    ok = True
    for lane in lanes:
        for repo, path_in_repo, subfolder in LANES[lane]:
            ok = fetch(repo, path_in_repo, subfolder, root, args.dry_run) and ok
    print("DONE" if ok else "INCOMPLETE -- see failures above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
