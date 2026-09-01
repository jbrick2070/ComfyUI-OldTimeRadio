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
import hashlib
import os
import sys
import urllib.parse
import urllib.request
from typing import NamedTuple


class WeightSpec(NamedTuple):
    """One fetchable artifact.

    Older lanes predate reproducible source receipts, so ``revision``,
    ``expected_bytes`` and ``expected_sha256`` remain optional for them. New
    promoted lanes must fill all three. Receipt-bearing lanes prove both remote
    identity and landed bytes before the final filename becomes visible to
    ComfyUI.
    """

    repo: str
    path_in_repo: str
    destination: str
    revision: str = "main"
    expected_bytes: int | None = None
    expected_sha256: str | None = None

#: lane -> list of (hf_repo, path_in_repo, models_subfolder)
#: Sizes in the comments are the real blob sizes read from the HF API.
#: THE PICK LIST. What each lane gives you and what it costs, so a person can
#: choose BEFORE spending the bandwidth instead of discovering the bill after.
#: Sizes are the sum of the per-file figures in LANES below; keep them in step
#: when a lane changes. A wrong number here is worse than no number, because it
#: is the one thing somebody uses to decide.
LANE_INFO = {
    "haunted": (3.65, "AnimateDiff video -- SD1.5 + motion module. The cheapest "
                      "complete video lane."),
    "z_image_blackwell": (12.00, "IMAGE model, nvfp4. Blackwell (sm_120) only."),
    "z_image_int8": (13.58, "IMAGE model, int8. Smallest universal precision."),
    "z_image": (19.26, "IMAGE model, bf16. Any NVIDIA; largest download."),
    "stable_audio_3": (3.46, "THE MUSIC MODEL. Sits on the shared path, so "
                             "without it EVERY profile fails at the music node."),
    "wan_ti2v_gguf": (9.37, "Wan 2.2 TI2V, quantised. Needs ComfyUI-GGUF."),
    "wan_ti2v": (0.0, "Wan 2.2 TI2V, safetensors. An ALTERNATIVE to the GGUF "
                      "set above -- fetch one or the other, never both."),
    "ltx_8gb": (16.13, "LTX 2b distilled + T5 encoder. Real video diffusion."),
    "humo": (26.74, "HuMo 14B talking-face lane. Locally episode-proven on "
                    "a 16 GB Blackwell 5080 at 13.06 GiB VRAM / 27.53 GiB "
                    "host RAM; use at least 32 GiB host RAM. Other NVIDIA "
                    "families remain lab candidates until a live receipt."),
    "minimax_h3": (59.084, "MiniMax H3 FL2VA + REF2VA operator-local lane. "
                           "The NVFP4 text encoder is not Blackwell-only; "
                           "physical 8 GB remains unqualified."),
}

#: The least you can install and still render an episode. Everything else is a
#: choice, not a prerequisite.
MINIMUM_HINT = ("haunted + one z_image precision + stable_audio_3 "
                "= about 20 GB and one complete episode")


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
    # 26.74 GiB. COMPLETE 14B HuMo recipe: every destination is read from
    # HuMoEngine._loader_names(), and the primary UNET is exactly
    # eng_humo._HUMO_DEFAULT_UNET. This is intentionally Kijai's
    # `...scaled_KJ` file, not Comfy-Org's similarly named humo_17B artifact.
    # That wrong download was a fresh-install breaker: it consumed ~16 GB and
    # the engine still reported not installed.
    #
    # Unlike the legacy lane rows below, every HuMo source is revision-pinned
    # and carries exact bytes + SHA-256. fetch() verifies the temporary file
    # before os.replace(), so neither a moved upstream ref nor an interrupted
    # transfer can become a loadable final checkpoint.
    "humo": [
        WeightSpec(
            "Kijai/WanVideo_comfy_fp8_scaled",
            "HuMo/Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors",
            "diffusion_models/Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors",
            "033a4e487f60220b3d6e469599a6aebc46e13cee",
            17_892_294_098,
            "a67ed82a7c008892f9192cdc5b23bbfe2e2a8e2f87d0b5b8dfb0226fafec022d",
        ),
        WeightSpec(
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "617a7633e636506f850e043bc4605f290a466a8e",
            6_735_906_897,
            "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
        ),
        WeightSpec(
            "Comfy-Org/HuMo_ComfyUI",
            "split_files/audio_encoders/whisper_large_v3_fp16.safetensors",
            "audio_encoders/whisper_large_v3_fp16.safetensors",
            "3a5e6947d865c3910cb2407cf2dac6a8df506b5a",
            3_087_130_976,
            "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
        ),
        WeightSpec(
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "split_files/vae/wan_2.1_vae.safetensors",
            "vae/wan_2.1_vae.safetensors",
            "c4f60d30c55a624e35427060fdd217579a6c1d77",
            253_815_318,
            "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        ),
        WeightSpec(
            "Kijai/WanVideo_comfy",
            "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "8260d429d19fd7a72304cad059160b95d843913f",
            738_005_744,
            "85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3",
        ),
    ],
    # 63,440,965,087 bytes (59.084 GiB). COMPLETE operator-local H3 recipe for
    # both OTR adapters: FL2VA and REF2VA DiTs, their shared NVFP4 encoder and
    # video VAE, plus the audio VAE REF2VA uses for conditioning. The NVFP4
    # encoder is explicitly documented by Comfy-Org as usable without
    # Blackwell. The lawful local receipts are 124 model / 129 canvas frames;
    # they do not qualify a physical 8 GB card. This lane is explicit only and
    # is never selected by a public profile or machine bundle.
    "minimax_h3": [
        WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "4cc1d817b6184899b41293954329f576cb5ae86b",
            20_970_379_616,
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        ),
        WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "4cc1d817b6184899b41293954329f576cb5ae86b",
            20_970_379_616,
            "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
        ),
        WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "4cc1d817b6184899b41293954329f576cb5ae86b",
            15_687_142_551,
            "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
        ),
        WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "4cc1d817b6184899b41293954329f576cb5ae86b",
            5_207_808_496,
            "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        ),
        WeightSpec(
            "Comfy-Org/MiniMax-H3",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            "4cc1d817b6184899b41293954329f576cb5ae86b",
            605_254_808,
            "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        ),
    ],
    # ~9 GB. Ungated Comfy-Org repackages.
    "wan_ti2v": [
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
         "split_files/vae/wan_2.1_vae.safetensors", "vae"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
         "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
         "text_encoders"),
    ],
    # ~16.1 GB. COMPLETE: `eng_ltx_8gb` names exactly these two files, and the
    # destinations are read off its own resolver -- the checkpoint from
    # ("checkpoints",) and the encoder from ("text_encoders", "clip").
    #
    # This lane, and the GGUF one below, were added 2026-08-30 after the asset
    # index made the real gap visible: OTR's engines name FILES but almost never
    # name SOURCES. Only three lanes had a recorded provenance, so every other
    # weight on the reference machine had been placed there by hand and could
    # not be obtained by anyone else from anything in the repo. Sizes below were
    # read from the Hub, not estimated.
    "ltx_8gb": [
        ("Lightricks/LTX-Video",
         "ltxv-2b-0.9.8-distilled.safetensors", "checkpoints"),         # 6.34 GB
        ("comfyanonymous/flux_text_encoders",
         "t5xxl_fp16.safetensors", "text_encoders"),                    # 9.79 GB
    ],
    # ~9.4 GB. The GGUF route for Wan 2.2 TI2V, which `eng_wan_ti2v` also
    # accepts -- it names both the safetensors set (the `wan_ti2v` lane above)
    # and these quantized files. They are ALTERNATIVES: fetch one set or the
    # other, not both.
    # ~3.4 GB. THE MUSIC MODEL, and it blocks far more than a "music lane":
    # OTR_StableAudioTheme runs on the shared path, so a machine without this
    # fails EVERY profile that reaches the music node -- eight consecutive lanes
    # died here on 2026-08-31, about twelve minutes into each, after the script,
    # cast and voices were already done.
    #
    # USE THE COMFY-ORG REPACKAGE, NOT STABILITY'S RAW REPO -- and note the
    # engine's own error message says exactly this ("fetch Comfy-Org/
    # stable-audio-3"). Ignoring it cost a full soak round.
    #
    # The raw repo LOOKS right and fails at render time. Its checkpoint is
    # byte-identical (2,270,384,940 either way), but its text encoder is not:
    #
    #   stabilityai .../t5gemma-b-b-ul2/model.safetensors   1,183,022,944
    #   Comfy-Org   .../t5gemma_b_b_ul2.safetensors         1,187,264,003
    #                                             difference     4,241,003
    #
    # which is exactly the size of `tokenizer.model` in the raw repo. ComfyUI's
    # SA3 CLIP wants the SentencePiece model EMBEDDED in the state dict as
    # `spiece_model` (see comfy/text_encoders/sa3.py and spiece_tokenizer.py);
    # the raw safetensors has no such key, so the loader reaches the file-path
    # branch, finds nothing, and raises a bare `ValueError: invalid tokenizer`
    # from deep inside sd1_clip -- a message that names neither SA3 nor the
    # missing tokenizer. Both repos are ungated, so there is no reason to prefer
    # the raw one.
    # ~19.3 GB. THE DEFAULT IMAGE ENGINE, and until 2026-08-31 it had no lane
    # at all -- so every machine that followed the docs fetched video weights
    # and then died at OTR_ImageGenDispatcher with 'z_image_turbo diffusion
    # model not found'. It is the `image` value for EVERY class in the machine
    # matrix, so a missing image model fails far more lanes than a missing
    # video model does. bf16 is the universal choice: the adapter RANKS
    # installed z_image_turbo*.safetensors nvfp4 > fp8 > bf16, so a Blackwell
    # box that also fetches `z_image_blackwell` gets the faster file
    # automatically and this one simply stops being chosen.
    "z_image": [
        ("Comfy-Org/z_image_turbo",
         "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
         "diffusion_models"),                                       # 11.46 GB
        ("Comfy-Org/z_image_turbo",
         "split_files/text_encoders/qwen_3_4b.safetensors",
         "text_encoders"),                                          # 7.49 GB
        ("Comfy-Org/z_image_turbo",
         "split_files/vae/ae.safetensors", "vae"),                  # 0.31 GB
    ],
    # ~12.0 GB. Blackwell (sm_120) only -- nvfp4 needs hardware fp4 and an
    # 8 GB 4060 (sm_89) cannot execute it. Fetch this INSTEAD of `z_image` on
    # such a card, never as well: the ranking prefers nvfp4 whenever it is
    # present, so installing it on a card that cannot run it makes the engine
    # choose the one file that fails. CLIP and VAE are not ranked -- they
    # resolve by exact filename -- so they are the same two files either way.
    # ~13.6 GB. The smallest UNIVERSAL precision -- smaller download and less
    # offload pressure than bf16, on any NVIDIA card.
    #
    # It is NOT a fits/does-not-fit choice. z_image_turbo is the low-VRAM lane
    # and is proven at 8 GB (the 4060, nine published episodes); the adapter
    # offloads the text encoder before the diffusion peak. A reading of
    # 19.3 GB memory.used on a 20 GB card was briefly mistaken here for a
    # requirement -- it was ComfyUI expanding into free memory, and the 8 GB
    # proof already contradicted it.
    #
    # The TEXT ENCODER is the full qwen_3_4b, not the fp8 variant, and that is
    # deliberate: the CLIP is NOT ranked -- it resolves by the exact filename
    # `qwen_3_4b.safetensors` unless OTR_ZIMAGE_CLIP is set -- so shipping
    # qwen_3_4b_fp8_mixed alone would resolve to nothing on a machine whose
    # owner never set that variable.
    #
    # Fetch this INSTEAD of `z_image`, never alongside it: the DiT ranking is
    # nvfp4 > fp8 > bf16 > other, and `z_image_turbo_int8_convrot` matches
    # none of the first three, so it sorts LAST. With bf16 also present the
    # ranking picks bf16 and the saving is lost.
    "z_image_int8": [
        ("Comfy-Org/z_image_turbo",
         "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
         "diffusion_models"),                                       # 5.78 GB
        ("Comfy-Org/z_image_turbo",
         "split_files/text_encoders/qwen_3_4b.safetensors",
         "text_encoders"),                                          # 7.49 GB
        ("Comfy-Org/z_image_turbo",
         "split_files/vae/ae.safetensors", "vae"),                  # 0.31 GB
    ],
    "z_image_blackwell": [
        ("Comfy-Org/z_image_turbo",
         "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
         "diffusion_models"),                                       # 4.20 GB
        ("Comfy-Org/z_image_turbo",
         "split_files/text_encoders/qwen_3_4b.safetensors",
         "text_encoders"),                                          # 7.49 GB
        ("Comfy-Org/z_image_turbo",
         "split_files/vae/ae.safetensors", "vae"),                  # 0.31 GB
    ],
    "stable_audio_3": [
        ("Comfy-Org/stable-audio-3",
         "checkpoints/stable_audio_3_small_music.safetensors",
         "checkpoints"),                                                 # 2.27 GB
        ("Comfy-Org/stable-audio-3",
         "text_encoders/t5gemma_b_b_ul2.safetensors",
         "text_encoders"),                                               # 1.19 GB
    ],
    "wan_ti2v_gguf": [
        ("QuantStack/Wan2.2-TI2V-5B-GGUF",
         "Wan2.2-TI2V-5B-Q5_K_M.gguf", "diffusion_models"),             # 3.81 GB
        ("city96/umt5-xxl-encoder-gguf",
         "umt5-xxl-encoder-Q5_K_M.gguf", "text_encoders"),              # 4.15 GB
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
         "split_files/vae/wan2.2_vae.safetensors", "vae"),              # 1.41 GB
    ],
}

#: Convenience bundles: everything a named profile needs that is not already
#: auto-fetched by transformers (writer, musicgen) on first use.
BUNDLES = {
    "otr_nvidia_8gb_haunted": ["haunted"],
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


#: Extensions that mark the third manifest element as a full destination PATH
#: rather than a bare folder.
_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".gguf", ".pth", ".bin", ".onnx")


def weight_spec(entry) -> WeightSpec:
    """Normalize a legacy three-tuple or a receipt-bearing WeightSpec."""
    if isinstance(entry, WeightSpec):
        return entry
    if len(entry) != 3:
        raise ValueError("lane artifact rows must be WeightSpec or legacy 3-tuples")
    return WeightSpec(*entry)


def destination_path(root: str, entry) -> str:
    """Return the exact final path this manifest row lands on."""
    spec = weight_spec(entry)
    if spec.destination.endswith(_WEIGHT_SUFFIXES):
        return os.path.join(root, *spec.destination.replace("\\", "/").split("/"))
    return os.path.join(root, spec.destination,
                        spec.path_in_repo.rsplit("/", 1)[-1])


def destination_name(entry) -> str:
    """The exact basename ComfyUI will resolve after a successful fetch."""
    spec = weight_spec(entry)
    if spec.destination.endswith(_WEIGHT_SUFFIXES):
        return os.path.basename(spec.destination)
    return spec.path_in_repo.rsplit("/", 1)[-1]


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validate_file(path: str, entry, *, verify_hash: bool) -> tuple[bool, str]:
    spec = weight_spec(entry)
    try:
        actual_bytes = os.path.getsize(path)
    except OSError:
        return False, "missing"
    if spec.expected_bytes is not None and actual_bytes != spec.expected_bytes:
        return False, "wrong bytes %d != %d" % (actual_bytes, spec.expected_bytes)
    if spec.expected_bytes is None and actual_bytes <= 1_000_000:
        return False, "file is too small (%d bytes)" % actual_bytes
    if verify_hash and spec.expected_sha256:
        actual_sha = _sha256_file(path)
        if actual_sha.lower() != spec.expected_sha256.lower():
            return False, "SHA-256 %s != %s" % (actual_sha, spec.expected_sha256)
    return True, "%d bytes%s" % (
        actual_bytes, ", SHA-256 verified" if verify_hash and spec.expected_sha256 else "")


def _already_present(root, entry, *, verify_hash=False):
    """Is this exact file on disk already?

    So the pick list can say what you ALREADY have. Someone choosing what to
    download needs to see that half a lane is present, not be told a size and
    left to guess whether they would be paying it again. Mirrors the naming
    rule in fetch(): a destination ending in a weight suffix names the file.
    """
    ok, _detail = _validate_file(
        destination_path(root, entry), entry, verify_hash=verify_hash)
    return ok


def fetch(entry, root: str, dry_run: bool) -> bool:
    """Fetch one manifest row into its exact destination.

    Some engines look for names upstream does not use -- Stable Audio 3 ships
    ``model.safetensors`` and OTR asks for ``stable_audio_3_small_music.safetensors``
    -- so a row can name the final file directly. A bare directory such as
    ``"checkpoints"`` keeps the legacy behaviour of taking the upstream
    basename, which is what every pre-existing lane row relies on.
    """
    spec = weight_spec(entry)
    dest = destination_path(root, spec)
    dest_dir = os.path.dirname(dest)
    name = os.path.basename(dest)
    url = "https://huggingface.co/%s/resolve/%s/%s" % (
        spec.repo,
        urllib.parse.quote(spec.revision, safe=""),
        urllib.parse.quote(spec.path_in_repo, safe="/"),
    )

    present, present_detail = _validate_file(dest, spec, verify_hash=True)
    if present:
        print("  PRESENT  %-52s %s" % (name[:52], present_detail))
        return True
    if os.path.isfile(dest):
        print("  MISMATCH %-52s %s" % (name[:52], present_detail))
    if dry_run:
        print("  WOULD GET %-51s <- %s@%s" %
              (name[:51], spec.repo, spec.revision))
        return True

    os.makedirs(dest_dir, exist_ok=True)
    tmp = dest + ".part"
    print("  FETCHING %-52s <- %s@%s" %
          (name[:52], spec.repo, spec.revision), flush=True)
    try:
        # Download to .part and rename only on success, so an interrupted
        # fetch never leaves a truncated file the engine would load.
        urllib.request.urlretrieve(url, tmp)
        valid, detail = _validate_file(tmp, spec, verify_hash=True)
        if not valid:
            raise ValueError("download integrity check failed: %s" % detail)
        os.replace(tmp, dest)
        print("  OK       %-52s %s" % (name[:52], detail))
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
        root = models_root()
        print("PICK WHAT YOU NEED. Nothing here is fetched unless you name it.")
        print("  %-20s %8s %7s  %s"
              % ("lane", "size", "on disk", "what it gives you"))
        print("  " + "-" * 96)
        for lane, files in LANES.items():
            gb, what = LANE_INFO.get(lane, (0.0, ""))
            # Listing is intentionally cheap: exact byte count when supplied,
            # never a multi-gigabyte SHA pass. Naming a lane performs the full
            # hash verification before it reports PRESENT.
            have = sum(1 for entry in files if _already_present(root, entry))
            mark = "all" if have == len(files) else "%d/%d" % (have, len(files))
            size = ("%.1f GB" % gb) if gb else "varies"
            print("  %-20s %8s %7s  %s" % (lane, size, mark, what[:52]))
            if len(what) > 52:
                print("  %-37s  %s" % ("", what[52:110]))
        print("")
        print("  minimum for one episode: %s" % MINIMUM_HINT)
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
        for entry in LANES[lane]:
            ok = fetch(entry, root, args.dry_run) and ok
    print("DONE" if ok else "INCOMPLETE -- see failures above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
