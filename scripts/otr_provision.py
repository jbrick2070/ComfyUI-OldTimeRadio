#!/usr/bin/env python
"""Install everything a machine needs to render, unattended. THE DOER.

    python scripts/otr_provision.py --profile otr_nvidia_8gb_haunted
    python scripts/otr_provision.py --profile otr_5080_haunted_12b_overnight --with-indextts2
    python scripts/otr_provision.py --list

WHY THIS EXISTS, AND WHY IT IS NOT A CHECKER. Operator, 2026-08-31: *"I need a
practical way to get up and running on these machines quickly, not a checker, a
DOER or an instruction manual. If it can't be automatic that's fine."*

The failure it is built against: on a rented pod, two missing assets sat behind
each other, and each cost a full round to discover because a lane runs seven to
thirteen minutes before it reaches the node that needs them. Sixteen legs died
that way in one session. **The answer is not detecting the gap sooner -- that
moves the discovery earlier without removing any of the work. The answer is
installing everything up front so there is nothing left to discover.**

So this installs first and prints an auditable receipt. A missing required
dependency, incompatible pinned node pack, failed automatic download, or
unverified manual tier is an honest nonzero result -- never a false-ready pod.

MACHINE-AGNOSTIC ON PURPOSE. The five things it does -- locate the tree ComfyUI
scans, resolve the model roots, install node packs, fetch lane weights, build
the isolated TTS environment -- are none of them pod-specific, and the pod-shaped
bash version could not serve the Windows boxes at all. What genuinely cannot be
automated stays in the manual: SSH keys, template environment, GPU and volume
selection. Those happen BEFORE the machine exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

GGUF_PACK_NAME = "ComfyUI-GGUF"
GGUF_URL = "https://github.com/city96/ComfyUI-GGUF"
GGUF_PIN = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
GGUF_CLEAN_SHA256 = "b66b5f39a656b1ada80cc452e18cf1e71323cd52b1a61b6852cf90dbf4842345"
GGUF_PATCHED_SHA256 = "63f8146be990b557728e5e806547fe6f904b87318ff6c4c87dde3c73f17bdf85"
GGUF_PATCH_SHA256 = "d9185b7a8129f85b59b4df527488aa396da7c99217d336a3580a4c3d0fd4fa04"
GGUF_PATCH_PATH = os.path.join(_REPO, "patches", "ComfyUI-GGUF-ltx25-gemma4.patch")

LTXVIDEO_PACK_NAME = "ComfyUI-LTXVideo"
LTXVIDEO_URL = "https://github.com/Lightricks/ComfyUI-LTXVideo"
LTXVIDEO_PIN = "3b9c5cde4700917074823d45e25401d81049f8fc"
LTXVIDEO_CLEAN_SHA256 = "08d2b18cfd325a3610683abc574e058fd209ddc7453c19b47cc108a8882a7dc1"
LTXVIDEO_PATCHED_SHA256 = "19ac341bad75f8ea03988aef664924896fc24960accd2a79f415536c2833997e"
LTXVIDEO_PATCH_SHA256 = "109fbe2927b9c07d95d431470f7449942094fc6047dcbc9ad4a519a57ac0c993"
LTXVIDEO_PATCH_PATH = os.path.join(
    _REPO, "patches", "ComfyUI-LTXVideo-kornia-pad.patch"
)

ANIMATEDIFF_PACK_NAME = "ComfyUI-AnimateDiff-Evolved"
ANIMATEDIFF_URL = "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved"
# Release 1.6.0 (2026-07-28): the checkout on the RTX 5080 behind every published
# AnimateDiff receipt. Until 2026-09-02 this pack was the one node pack cloned at
# whatever HEAD was current that day, so two provisioned machines could differ.
ANIMATEDIFF_PIN = "9257651221002dcba0a12f9cff37e1944e58fb60"

INDEXTTS2_URL = "https://github.com/index-tts/index-tts.git"
INDEXTTS2_PIN = "830f6f8f94a51fea23ab1d639027a86200075a4e"
INDEXTTS2_PYTHON = "3.10"

# Exact manual tiers for sources that cannot yet be fetched unattended. Every
# manual file has one reproducible identity and destination. Automatic sources
# belong only to ``otr_fetch_lane_weights.py``; in particular, HuMo 14B is the
# receipt-bearing ``humo`` lane there and is not duplicated here. Manual
# downloads use `<destination>.part`, verification, then rename; see
# docs/RUNPOD_INSTALL.md.
MANUAL_TIERS = {
    "ltx25": [
        {
            "role": "Q3 DiT",
            "repo": "realrebelai/LTX-2.5_GGUFs",
            "revision": "112436f97aaf99ce13ecb7b7eca7e2f6c128d3ec",
            "path": "LTX-2.5-Distilled-Q3_K_M.gguf",
            "destination": "diffusion_models/LTX-2.5-Distilled-Q3_K_M.gguf",
            "bytes": 11_525_623_808,
            "sha256": "4286f8de1074c0c4fddfb92f38bd7df9161782b53c1717ebd69f1189c7933265",
            "gated": False,
        },
        {
            "role": "Q5 encoder",
            "repo": "elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF",
            "revision": "085ceddbbac3c0370de7f59ebec8bef4763f04b5",
            "path": "gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf",
            "destination": "text_encoders/gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf",
            "bytes": 9_514_920_864,
            "sha256": "1d35d4fbfa34cca1513d8e9fdd77c0573778b21ffdcbe4ca9c906f37a8c502f9",
            "gated": True,
        },
        {
            "role": "video VAE",
            "repo": "Lightricks/LTX-2.5",
            "revision": "5e6e71018ee1756ed329b697a7b4aedc934dfce9",
            "path": "vae/ltx-2.5-video-vae-bf16.safetensors",
            "destination": "vae/ltx-2.5-video-vae-bf16.safetensors",
            "bytes": 1_472_223_346,
            "sha256": "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
            "gated": True,
        },
        {
            "role": "audio VAE",
            "repo": "Lightricks/LTX-2.5",
            "revision": "5e6e71018ee1756ed329b697a7b4aedc934dfce9",
            "path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
            "destination": "vae/ltx-2.5-audio-vae-bf16.safetensors",
            "bytes": 364_866_540,
            "sha256": "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
            "gated": True,
        },
        {
            "role": "spatial upscaler",
            "repo": "Lightricks/LTX-2.5",
            "revision": "5e6e71018ee1756ed329b697a7b4aedc934dfce9",
            "path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            "destination": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            "bytes": 995_778_752,
            "sha256": "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
            "gated": True,
        },
    ],
    "humo_1_7b": [
        {
            "role": "1.7B DiT",
            "repo": "Comfy-Org/HuMo_ComfyUI",
            "revision": "3a5e6947d865c3910cb2407cf2dac6a8df506b5a",
            "path": "split_files/diffusion_models/humo_1.7B_fp16.safetensors",
            "destination": "diffusion_models/humo_1.7B_fp16.safetensors",
            "bytes": 3_483_511_088,
            "sha256": "3f8c08e7db17e807397b9a9ed9d9b28a6e42c8083029395674e95544191b1b15",
            "gated": False,
        },
        {
            "role": "UMT5",
            "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "revision": "617a7633e636506f850e043bc4605f290a466a8e",
            "path": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "destination": "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "bytes": 6_735_906_897,
            "sha256": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
            "gated": False,
        },
        {
            "role": "Whisper",
            "repo": "Comfy-Org/HuMo_ComfyUI",
            "revision": "3a5e6947d865c3910cb2407cf2dac6a8df506b5a",
            "path": "split_files/audio_encoders/whisper_large_v3_fp16.safetensors",
            "destination": "audio_encoders/whisper_large_v3_fp16.safetensors",
            "bytes": 3_087_130_976,
            "sha256": "a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95",
            "gated": False,
        },
        {
            "role": "Wan VAE",
            "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "revision": "c4f60d30c55a624e35427060fdd217579a6c1d77",
            "path": "split_files/vae/wan_2.1_vae.safetensors",
            "destination": "vae/wan_2.1_vae.safetensors",
            "bytes": 253_815_318,
            "sha256": "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
            "gated": False,
        },
    ],
    "flux2_klein": [
        {
            "role": "4B Q4 DiT",
            "repo": "Latentiq/FLUX.2-klein-4B-GGUF",
            "revision": "4dc94114f28d56e7b63e7bb624a1c1f20353245b",
            "path": "flux-2-klein-4b-Q4_K_M.gguf",
            "destination": "diffusion_models/flux-2-klein-4b-Q4_K_M.gguf",
            "bytes": 2_604_311_104,
            "sha256": "0b25d143c8469b342bc5af3bce92b783bf6b0636d285f7b2f75e38af63af9a15",
            "gated": False,
        },
        {
            "role": "Qwen3-4B encoder",
            "repo": "Comfy-Org/flux2-klein",
            "revision": "5f526678002e43af5551dadb73ce2e8c91b43afe",
            "path": "split_files/text_encoders/qwen_3_4b.safetensors",
            "destination": "text_encoders/qwen_3_4b.safetensors",
            "bytes": 8_044_982_048,
            "sha256": "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
            "gated": False,
        },
        {
            "role": "Flux2 VAE",
            "repo": "Comfy-Org/flux2-dev",
            "revision": "ab9055628ea245000e610f2aa2c96f4746093546",
            "path": "split_files/vae/flux2-vae.safetensors",
            "destination": "vae/flux2-vae.safetensors",
            "bytes": 336_213_556,
            "sha256": "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
            "gated": False,
        },
    ],
}

OPERATOR_ONLY_TIERS = {
    "h3_operator_only": (
        "H3 is operator-local/offline and is never auto-selected by public "
        "provisioning. After reviewing the H3 operating contract, fetch the "
        "complete pinned lane explicitly with: python "
        "scripts/otr_fetch_lane_weights.py minimax_h3"
    ),
}

# Operator-only means "never selected or downloaded automatically", not
# "permanently unverifiable". The exact artifacts stay owned by the fetcher;
# this map lets the provisioner verify a completed explicit fetch without
# copying its manifest or weakening the opt-in boundary.
OPERATOR_ONLY_FETCH_LANES = {
    "h3_operator_only": "minimax_h3",
}

#: Every step appends here; the receipt is printed from it at the end.
_LOG: list = []


class ProvisionFailure(RuntimeError):
    """A required provisioning step could not be proved complete."""


def say(state: str, what: str, detail: str = "") -> None:
    _LOG.append((state, what, detail))
    print("  %-9s %-42s %s" % (state, what[:42], detail[:60]), flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# --------------------------------------------------------------------------- #
# Where things go.
# --------------------------------------------------------------------------- #
def comfy_root() -> str:
    """The tree ComfyUI actually scans.

    NOT a guess from the repo's own location: a pod image put ComfyUI at
    /workspace/runpod-slim/ComfyUI, and assuming /workspace/ComfyUI cost a round
    trip. Ask the library where it lives.
    """
    override = os.environ.get("OTR_COMFY_ROOT", "").strip()
    if override:
        override = os.path.abspath(os.path.expanduser(override))
        if not os.path.isfile(os.path.join(override, "folder_paths.py")):
            raise ProvisionFailure(
                "OTR_COMFY_ROOT does not name a ComfyUI tree (folder_paths.py missing): %s"
                % override)
        return override
    r = run([sys.executable, "-c",
             "import folder_paths,os;print(os.path.dirname(folder_paths.__file__))"])
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    # Two levels up from custom_nodes/<pack>/scripts is the usual layout.
    guess = os.path.dirname(os.path.dirname(_REPO))
    return guess


def models_root(comfy: str) -> str:
    """The models root, resolved by the PACK'S OWN authority.

    DO NOT REIMPLEMENT THIS. The first version of this function checked the env
    vars, then fell back to `<comfy>/models` if that directory existed -- and on
    the reference machine it does exist and holds some things, so the guess won
    and returned the wrong root. The real one is `C:\\ComfyUI-Models`. That is
    precisely the trap the project rules call out: a plausible tree that is not
    the tree, which makes a wrong answer look verified.

    `nodes/_otr_gguf_backend.py::_models_root()` is the single owner of this
    question. Ask it, and only fall back when it cannot be imported at all.
    """
    # BOTH paths go in: the repo so the pack imports, and ComfyUI's root so
    # `folder_paths` does. Without the second, the pack's friendly default
    # cannot reach its folder_paths step -- it would try to import from a cwd
    # of <comfy>/custom_nodes, fail, and fall through to the Windows literal.
    # On a fresh Linux box that literal is a RELATIVE directory name, and this
    # function would accept it as a successful answer.
    code = ("import sys;sys.path.insert(0,%r);sys.path.insert(0,%r)" + chr(10) +
            "from nodes._otr_gguf_backend import _models_root" + chr(10) +
            "print(_models_root())" + chr(10)) % (_REPO, comfy)
    r = run([sys.executable, "-c", code], cwd=os.path.dirname(_REPO))
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    for var in ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    say("WARN", "models root not resolvable from the pack",
        "falling back to <comfy>/models")
    return os.path.join(comfy, "models")


def ensure_hf_home(root: str) -> str:
    """Pin HF_HOME to the project convention, <models_root>/huggingface.

    THE DOER OWNS THIS, and that is the point. Left to a human, a second cache
    root gets created beside the first and the same weights download twice: 84 GB
    at an invented path plus 71 GB at the convention, discovered only as a disk
    quota error on a volume that was half empty. A step that cannot be forgotten
    cannot cause that.
    """
    want = os.path.join(root, "huggingface")
    os.makedirs(want, exist_ok=True)
    cur = os.environ.get("HF_HOME", "").strip()
    if os.path.normpath(cur or "") != os.path.normpath(want):
        os.environ["HF_HOME"] = want
        say("SET", "HF_HOME", want)
    else:
        say("OK", "HF_HOME already correct", want)
    return want


# --------------------------------------------------------------------------- #
# Steps.
# --------------------------------------------------------------------------- #
def _normalized_sha256(path: str) -> str:
    """Hash text after CRLF and lone-CR normalization to LF."""
    with open(path, "rb") as fh:
        data = fh.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git(dest: str, *args: str) -> str:
    r = run(["git", "-C", dest] + list(args))
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "git command failed").strip()
        raise ProvisionFailure("git %s failed in %s: %s" %
                               (" ".join(args), dest, detail[:300]))
    return (r.stdout or "").strip()


def _git_head(dest: str) -> str:
    return _git(dest, "rev-parse", "HEAD").strip().lower()


def _git_changed_paths(dest: str) -> list[str]:
    out = _git(dest, "diff", "--name-only", "HEAD")
    return sorted(line.strip().replace("\\", "/") for line in out.splitlines()
                  if line.strip())


def _git_untracked_paths(dest: str) -> list[str]:
    out = _git(dest, "ls-files", "--others", "--exclude-standard")
    return sorted(line.strip().replace("\\", "/") for line in out.splitlines()
                  if line.strip())


def _fetch_exact_repo(url: str, pin: str, dest: str) -> None:
    """Create a detached, shallow checkout of one exact commit."""
    if os.path.exists(dest) and not os.path.isdir(dest):
        raise ProvisionFailure("node-pack destination is not a directory: %s" % dest)
    os.makedirs(dest, exist_ok=True)
    if os.listdir(dest):
        raise ProvisionFailure("refusing exact checkout into non-empty directory: %s" % dest)
    _git(dest, "init", "-q")
    _git(dest, "remote", "add", "origin", url)
    _git(dest, "fetch", "-q", "--depth", "1", "origin", pin)
    _git(dest, "checkout", "-q", "--detach", "FETCH_HEAD")
    if _git_head(dest) != pin.lower():
        raise ProvisionFailure("checkout verification failed for %s at %s" % (dest, pin))


def _apply_gguf_patch(dest: str) -> None:
    loader = os.path.join(dest, "loader.py")
    if not os.path.isfile(GGUF_PATCH_PATH):
        raise ProvisionFailure("required GGUF patch is missing: %s" % GGUF_PATCH_PATH)
    if _normalized_sha256(GGUF_PATCH_PATH) != GGUF_PATCH_SHA256:
        raise ProvisionFailure("GGUF patch identity does not match the pinned SHA-256")
    if _normalized_sha256(loader) != GGUF_CLEAN_SHA256:
        raise ProvisionFailure("GGUF loader preimage is not the pinned clean file")
    r = run(["git", "-C", dest, "apply", "--ignore-space-change",
             "--ignore-whitespace", os.path.abspath(GGUF_PATCH_PATH)])
    if r.returncode != 0:
        raise ProvisionFailure("GGUF LTX 2.5 patch failed: %s" %
                               ((r.stderr or r.stdout or "unknown error").strip()[:300]))


def _verify_patched_gguf_git(dest: str) -> None:
    loader = os.path.join(dest, "loader.py")
    if _normalized_sha256(loader) != GGUF_PATCHED_SHA256:
        raise ProvisionFailure("GGUF patched loader postimage does not match the pinned SHA-256")
    changed = _git_changed_paths(dest)
    untracked = _git_untracked_paths(dest)
    if changed != ["loader.py"] or untracked:
        raise ProvisionFailure(
            "GGUF checkout drift: expected only loader.py changed; changed=%s untracked=%s"
            % (changed, untracked))


def _apply_ltxvideo_patch(dest: str) -> None:
    target = os.path.join(dest, "pyramid_blending.py")
    if not os.path.isfile(LTXVIDEO_PATCH_PATH):
        raise ProvisionFailure(
            "required LTXVideo patch is missing: %s" % LTXVIDEO_PATCH_PATH
        )
    if _normalized_sha256(LTXVIDEO_PATCH_PATH) != LTXVIDEO_PATCH_SHA256:
        raise ProvisionFailure(
            "LTXVideo patch identity does not match the pinned SHA-256"
        )
    if _normalized_sha256(target) != LTXVIDEO_CLEAN_SHA256:
        raise ProvisionFailure(
            "LTXVideo pyramid_blending.py preimage is not the pinned clean file"
        )
    r = run([
        "git", "-C", dest, "apply", "--ignore-space-change",
        "--ignore-whitespace", os.path.abspath(LTXVIDEO_PATCH_PATH),
    ])
    if r.returncode != 0:
        raise ProvisionFailure(
            "LTXVideo Kornia pad patch failed: %s"
            % ((r.stderr or r.stdout or "unknown error").strip()[:300])
        )


def _verify_patched_ltxvideo_git(dest: str) -> None:
    target = os.path.join(dest, "pyramid_blending.py")
    if _normalized_sha256(target) != LTXVIDEO_PATCHED_SHA256:
        raise ProvisionFailure(
            "LTXVideo patched pyramid_blending.py does not match the pinned SHA-256"
        )
    changed = _git_changed_paths(dest)
    untracked = _git_untracked_paths(dest)
    if changed != ["pyramid_blending.py"] or untracked:
        raise ProvisionFailure(
            "LTXVideo checkout drift: expected only pyramid_blending.py changed; "
            "changed=%s untracked=%s" % (changed, untracked)
        )


def ensure_gguf_pack(comfy: str) -> None:
    """Install or verify the one supported GGUF base plus the LTX 2.5 patch."""
    dest = os.path.join(comfy, "custom_nodes", GGUF_PACK_NAME)
    fresh = not os.path.isdir(dest) or not os.listdir(dest)
    if fresh:
        _fetch_exact_repo(GGUF_URL, GGUF_PIN, dest)

    loader = os.path.join(dest, "loader.py")
    if not os.path.isfile(loader):
        raise ProvisionFailure("%s is missing loader.py" % GGUF_PACK_NAME)

    git_dir = os.path.isdir(os.path.join(dest, ".git"))
    loader_sha = _normalized_sha256(loader)
    if not git_dir:
        if loader_sha == GGUF_CLEAN_SHA256:
            raise ProvisionFailure(
                "Manager-installed ComfyUI-GGUF is the clean base, not the required patched build; "
                "move it aside and rerun --packs-only")
        if loader_sha != GGUF_PATCHED_SHA256:
            raise ProvisionFailure(
                "unverifiable non-git ComfyUI-GGUF loader; move the pack aside and rerun --packs-only")
        install_pack_requirements(GGUF_PACK_NAME, dest, required=True)
        say("PRESENT", GGUF_PACK_NAME, "verified patched Manager install")
        return

    head = _git_head(dest)
    if head != GGUF_PIN:
        raise ProvisionFailure(
            "%s is at %s, required %s; move it aside and rerun --packs-only"
            % (GGUF_PACK_NAME, head, GGUF_PIN))
    untracked = _git_untracked_paths(dest)
    changed = _git_changed_paths(dest)
    if loader_sha == GGUF_CLEAN_SHA256:
        if changed or untracked:
            raise ProvisionFailure(
                "GGUF clean loader sits in a dirty checkout; refusing to overwrite drift")
        _apply_gguf_patch(dest)
        _verify_patched_gguf_git(dest)
        state = "PATCHED"
    elif loader_sha == GGUF_PATCHED_SHA256:
        _verify_patched_gguf_git(dest)
        state = "PRESENT"
    else:
        raise ProvisionFailure(
            "GGUF loader is neither the pinned clean nor pinned patched file; refusing partial drift")
    install_pack_requirements(GGUF_PACK_NAME, dest, required=True)
    say(state, GGUF_PACK_NAME, "%s + LTX 2.5 patch" % GGUF_PIN[:12])


def ensure_ltxvideo_pack(comfy: str) -> None:
    """Install the exact LTXVideo commit plus its Kornia 0.8.3 API repair."""
    dest = os.path.join(comfy, "custom_nodes", LTXVIDEO_PACK_NAME)
    fresh = not os.path.isdir(dest) or not os.listdir(dest)
    if fresh:
        _fetch_exact_repo(LTXVIDEO_URL, LTXVIDEO_PIN, dest)
    if not os.path.isdir(os.path.join(dest, ".git")):
        raise ProvisionFailure(
            "ComfyUI-LTXVideo is not a verifiable git checkout; move it aside and rerun --packs-only")
    head = _git_head(dest)
    if head != LTXVIDEO_PIN:
        raise ProvisionFailure(
            "ComfyUI-LTXVideo is at %s, required %s; move it aside and rerun --packs-only"
            % (head, LTXVIDEO_PIN)
        )
    target = os.path.join(dest, "pyramid_blending.py")
    if not os.path.isfile(target):
        raise ProvisionFailure(
            "ComfyUI-LTXVideo is missing pyramid_blending.py"
        )
    target_sha = _normalized_sha256(target)
    changed = _git_changed_paths(dest)
    untracked = _git_untracked_paths(dest)
    if target_sha == LTXVIDEO_CLEAN_SHA256:
        if changed or untracked:
            raise ProvisionFailure(
                "LTXVideo clean pyramid_blending.py sits in a dirty checkout; "
                "refusing to overwrite drift"
            )
        _apply_ltxvideo_patch(dest)
        _verify_patched_ltxvideo_git(dest)
        state = "PATCHED"
    elif target_sha == LTXVIDEO_PATCHED_SHA256:
        _verify_patched_ltxvideo_git(dest)
        state = "PRESENT"
    else:
        raise ProvisionFailure(
            "LTXVideo pyramid_blending.py is neither the pinned clean nor "
            "pinned patched file; refusing partial drift"
        )
    install_pack_requirements(LTXVIDEO_PACK_NAME, dest, required=True)
    say(
        state,
        LTXVIDEO_PACK_NAME,
        "%s + Kornia 0.8.3 pad patch" % LTXVIDEO_PIN[:12],
    )


def ensure_animatediff_pack(comfy: str) -> None:
    """Install or verify the AnimateDiff pack at its pinned commit.

    Same wrong-commit discipline as the GGUF and LTXVideo packs: a fresh install
    is an exact detached checkout of ANIMATEDIFF_PIN, and an existing git checkout
    must be AT that commit (a different commit is named and refused, never reset).
    A Manager install has no .git and cannot be verified, so it is accepted as
    PRESENT and said so (GGUF accepts one only when its patched loader hashes
    match; LTXVideo refuses any). This pack carries no patch to hash, and calling
    a working Manager install absent used to make the provisioner clone into a
    non-empty directory and report FAILED for it.
    """
    dest = os.path.join(comfy, "custom_nodes", ANIMATEDIFF_PACK_NAME)
    fresh = not os.path.isdir(dest) or not os.listdir(dest)
    if fresh:
        _fetch_exact_repo(ANIMATEDIFF_URL, ANIMATEDIFF_PIN, dest)
        install_pack_requirements(ANIMATEDIFF_PACK_NAME, dest)
        say("OK", ANIMATEDIFF_PACK_NAME, ANIMATEDIFF_PIN[:12])
        return
    if not os.path.isdir(os.path.join(dest, ".git")):
        install_pack_requirements(ANIMATEDIFF_PACK_NAME, dest)
        say("PRESENT", ANIMATEDIFF_PACK_NAME, "non-git install, commit unverifiable")
        return
    head = _git_head(dest)
    if head != ANIMATEDIFF_PIN:
        raise ProvisionFailure(
            "%s is at %s, required %s; move it aside and rerun --packs-only"
            % (ANIMATEDIFF_PACK_NAME, head, ANIMATEDIFF_PIN))
    install_pack_requirements(ANIMATEDIFF_PACK_NAME, dest)
    say("PRESENT", ANIMATEDIFF_PACK_NAME, ANIMATEDIFF_PIN[:12])


def install_node_packs(comfy: str) -> None:
    cn = os.path.join(comfy, "custom_nodes")
    os.makedirs(cn, exist_ok=True)
    # WITHOUT THESE THE MEATY VIDEO LANES DO NOT RUN. Only AnimateDiff was
    # listed here, so a provisioned machine could render the AnimateDiff lane
    # and nothing else: wan_ti2v died 17 minutes in with WrapperNodeMissing,
    # after writing, casting, voices and stills had all completed. The engines
    # resolve these node CLASSES by name at render time, which is why the
    # failure arrives late and looks nothing like a missing install.
    ensure_gguf_pack(comfy)
    ensure_ltxvideo_pack(comfy)
    ensure_animatediff_pack(comfy)


def install_pack_requirements(name: str, dest: str, required: bool = False) -> None:
    """A cloned pack whose own dependencies are missing does not load.

    Cloning was treated as installing, and it is not: ComfyUI-GGUF needs the
    `gguf` wheel, and without it the pack registers nothing -- so
    UnetLoaderGGUF is absent and wan_ti2v and ltx25 both fail at render time
    with WrapperNodeMissing, exactly as if the pack had never been cloned.

    Into sys.executable deliberately: this script is run BY the interpreter
    ComfyUI uses, and installing a node pack anywhere else is the same
    mistake this file exists to prevent.
    """
    req = os.path.join(dest, "requirements.txt")
    if not os.path.isfile(req):
        if required:
            raise ProvisionFailure("%s is missing required requirements.txt" % name)
        say("SKIP", "%s deps" % name, "requirements.txt not present")
        return
    r = run([sys.executable, "-m", "pip", "install", "-q", "-r", req])
    if r.returncode != 0:
        raise ProvisionFailure("dependency install failed for %s: %s" %
                               (name, (r.stderr or r.stdout or "").strip()[:300]))
    say("OK", "%s deps" % name)


def install_requirements() -> None:
    req = os.path.join(_REPO, "requirements.txt")
    if not os.path.isfile(req):
        raise ProvisionFailure("OTR requirements.txt is missing")
    r = run([sys.executable, "-m", "pip", "install", "-q", "-r", req])
    if r.returncode != 0:
        raise ProvisionFailure("OTR dependency install failed: %s" %
                               (r.stderr or r.stdout or "").strip()[:300])
    say("OK", "pip install requirements")


def fetch_lane_weights(lanes) -> None:
    """Delegate to the fetcher, which is the ONE place sources are recorded."""
    fetcher = os.path.join(_HERE, "otr_fetch_lane_weights.py")
    for lane in lanes:
        r = run([sys.executable, fetcher, lane])
        tail = [ln for ln in (r.stdout or "").splitlines() if ln.strip()][-1:] or [""]
        say("OK" if r.returncode == 0 else "FAILED", "weights: %s" % lane,
            tail[0].strip()[:60])


WRITER_TRANSFORMERS_BACKENDS = frozenset({
    "transformers_safetensors",
    "transformers_multimodal_text_only",
    "transformers_gptq_int4",
})


def _load_writer_catalog():
    """Load the pure catalog without importing the ComfyUI node package."""
    path = os.path.join(_REPO, "nodes", "_otr_model_catalog.py")
    name = "otr_provision_writer_catalog"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProvisionFailure("cannot load writer model catalog: %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def warm_profile_writer_models(profile: dict, _snapshot_download=None) -> None:
    """Download the selected local Transformers writer rows before render.

    The creative and technical slots commonly select the same model; preserve
    order and fetch it once. Remote providers have no local payload, and the
    GGUF-native lane has its own explicitly managed artifact, so both receive
    an honest receipt rather than an accidental Hub download.
    """
    llm = profile.get("llm") or {}
    selected = list(dict.fromkeys(
        str(llm.get(key) or "").strip()
        for key in ("creative_model", "technical_model")
        if str(llm.get(key) or "").strip()
    ))
    if not selected:
        say("SKIP", "writer model warm", "profile selects no writer models")
        return
    catalog = _load_writer_catalog()
    rows = {row.repo_id: row for row in catalog.CURATED_LLM_MODELS}
    cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
    os.makedirs(cache_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    if _snapshot_download is None:
        from huggingface_hub import snapshot_download as _snapshot_download

    for model_id in selected:
        row = rows.get(model_id)
        if row is None:
            lowered = model_id.lower()
            if "gguf" in lowered:
                say("SKIP", "writer: %s" % model_id,
                    "GGUF-native artifact is managed by its explicit lane")
            else:
                say("SKIP", "writer: %s" % model_id,
                    "not a static local Transformers catalog row")
            continue
        provider = getattr(row, "provider", "local")
        if provider == "gguf_native" or row.loader_backend == "gguf_native":
            say("SKIP", "writer: %s" % model_id,
                "GGUF-native artifact is managed by its explicit lane")
            continue
        if provider != "local":
            say("SKIP", "writer: %s" % model_id,
                "provider %s has no local Hub payload" % provider)
            continue
        if row.loader_backend not in WRITER_TRANSFORMERS_BACKENDS:
            say("SKIP", "writer: %s" % model_id,
                "backend %s is not a local Transformers lane" %
                row.loader_backend)
            continue
        try:
            local_path = _snapshot_download(
                repo_id=model_id,
                allow_patterns=list(catalog.ALLOW_PATTERNS),
                cache_dir=cache_dir,
                token=token,
            )
        except Exception as exc:  # noqa: BLE001 - receipt owns provider detail
            say("FAILED", "writer: %s" % model_id,
                "%s: %s" % (type(exc).__name__, str(exc)[:180]))
            continue
        say("OK", "writer: %s" % model_id, str(local_path))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manual_artifact_path(root: str, artifact: dict) -> str:
    return os.path.join(root, *str(artifact["destination"]).split("/"))


def _load_fetcher_manifest():
    path = os.path.join(_HERE, "otr_fetch_lane_weights.py")
    spec = importlib.util.spec_from_file_location(
        "otr_fetch_lane_weights_manifest", path)
    if spec is None or spec.loader is None:
        raise ProvisionFailure("cannot load automatic lane manifest: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lanes = getattr(module, "LANES", None)
    if not isinstance(lanes, dict):
        raise ProvisionFailure("automatic lane manifest has no LANES mapping")
    return module


def _verify_operator_fetch_tier(root: str, tier_id: str) -> bool:
    """Verify a lane that only the operator may explicitly fetch.

    The fetcher remains the sole artifact authority. This path never downloads
    anything; it only proves that the exact receipt-bearing final files exist.
    """
    lane = OPERATOR_ONLY_FETCH_LANES[tier_id]
    fetcher = _load_fetcher_manifest()
    entries = fetcher.LANES.get(lane)
    if not entries:
        raise ProvisionFailure(
            "operator tier %r maps to missing fetch lane %r" % (tier_id, lane))

    complete = True
    for entry in entries:
        artifact = fetcher.weight_spec(entry)
        if artifact.expected_bytes is None or not artifact.expected_sha256:
            raise ProvisionFailure(
                "operator lane %r has an unpinned artifact %r" %
                (lane, artifact.path_in_repo))
        path = fetcher.destination_path(root, artifact)
        label = "%s: %s" % (tier_id, os.path.basename(path))
        if not os.path.isfile(path):
            part_note = " (.part exists but is not a completed file)" \
                if os.path.isfile(path + ".part") else ""
            say("MISSING", label, artifact.destination + part_note + "; " +
                OPERATOR_ONLY_TIERS[tier_id])
            complete = False
            continue
        actual_size = os.path.getsize(path)
        if actual_size != artifact.expected_bytes:
            say("MISSING", label, "wrong bytes %d != %d" %
                (actual_size, artifact.expected_bytes))
            complete = False
            continue
        actual_sha = _sha256_file(path)
        if actual_sha.lower() != artifact.expected_sha256.lower():
            say("MISSING", label, "SHA-256 %s != %s" %
                (actual_sha, artifact.expected_sha256))
            complete = False
            continue
        say("PRESENT", label, "%d bytes, SHA-256 verified" % actual_size)
    return complete


def verify_manual_tier(root: str, tier_id: str) -> bool:
    """Verify every final manual artifact; `.part` files never count."""
    if tier_id in OPERATOR_ONLY_FETCH_LANES:
        return _verify_operator_fetch_tier(root, tier_id)
    if tier_id in OPERATOR_ONLY_TIERS:
        say("MISSING", "manual tier: %s" % tier_id, OPERATOR_ONLY_TIERS[tier_id])
        return False
    artifacts = MANUAL_TIERS.get(tier_id)
    if artifacts is None:
        raise ProvisionFailure("unknown manual tier %r" % tier_id)
    complete = True
    for artifact in artifacts:
        path = manual_artifact_path(root, artifact)
        label = "%s: %s" % (tier_id, os.path.basename(path))
        if not os.path.isfile(path):
            part_note = " (.part exists but is not a completed file)" \
                if os.path.isfile(path + ".part") else ""
            say("MISSING", label, artifact["destination"] + part_note)
            complete = False
            continue
        actual_size = os.path.getsize(path)
        if actual_size != artifact["bytes"]:
            say("MISSING", label, "wrong bytes %d != %d" %
                (actual_size, artifact["bytes"]))
            complete = False
            continue
        actual_sha = _sha256_file(path)
        if actual_sha.lower() != artifact["sha256"]:
            say("MISSING", label, "SHA-256 %s != %s" %
                (actual_sha, artifact["sha256"]))
            complete = False
            continue
        say("PRESENT", label, "%d bytes, SHA-256 verified" % actual_size)
    return complete


def _fetcher_lane_names() -> set[str]:
    return set(_load_fetcher_manifest().LANES)


def _indextts2_source_problems(path: str) -> list[str]:
    if not os.path.isdir(os.path.join(path, ".git")):
        return ["not an exact git checkout"]
    try:
        head = _git_head(path)
        changed = _git_changed_paths(path)
        untracked = _git_untracked_paths(path)
    except ProvisionFailure as exc:
        return [str(exc)]
    problems = []
    if head != INDEXTTS2_PIN:
        problems.append("HEAD %s != %s" % (head, INDEXTTS2_PIN))

    def allowed(item: str) -> bool:
        norm = item.replace("\\", "/")
        return (norm.startswith(".venv/") or
                norm.startswith(".uv-python/") or
                norm.startswith("checkpoints/"))

    drift = [item for item in changed + untracked if not allowed(item)]
    if drift:
        problems.append("source drift outside .venv/.uv-python/checkpoints: %s" %
                        ", ".join(sorted(set(drift))))
    return problems


def _indextts2_source_root(comfy: str) -> str:
    override = os.environ.get("OTR_INDEXTTS2_ROOT", "").strip()
    comfy = os.path.abspath(comfy)
    default = (os.path.join(comfy, "index-tts") if os.name == "nt" else
               os.path.join(os.path.dirname(comfy), "index-tts"))
    return os.path.abspath(os.path.expanduser(override or default))


def _exclude_and_link_indextts2_root(comfy: str, source: str) -> None:
    """Keep the pinned ComfyUI checkout clean without editing qualified bytes.

    The shipped IndexTTS2 voice route fingerprints both its adapter and worker;
    changing either operational default would honestly require a new human
    audition. Linux therefore installs the persistent checkout beside ComfyUI
    and exposes the adapter's historical ``ComfyUI/index-tts`` path as a local
    symlink. The exact local alias is excluded through Git's own info/exclude so
    a second pod-provision pass still proves the tracked core is clean.

    Windows keeps its historical in-tree default (and explicit overrides still
    win), but receives the same local exclusion when ComfyUI is a Git checkout.
    """
    comfy = os.path.abspath(comfy)
    source = os.path.abspath(source)
    legacy = os.path.join(comfy, "index-tts")

    git_path = run(["git", "-C", comfy, "rev-parse", "--git-path",
                    "info/exclude"])
    if git_path.returncode == 0 and (git_path.stdout or "").strip():
        exclude = (git_path.stdout or "").strip()
        if not os.path.isabs(exclude):
            exclude = os.path.join(comfy, exclude)
        os.makedirs(os.path.dirname(exclude), exist_ok=True)
        existing = ""
        try:
            with open(exclude, "r", encoding="utf-8") as handle:
                existing = handle.read()
        except FileNotFoundError:
            pass
        patterns = {line.strip() for line in existing.splitlines()}
        if "/index-tts" not in patterns and "/index-tts/" not in patterns:
            prefix = "" if not existing or existing.endswith(("\n", "\r")) else "\n"
            with open(exclude, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(prefix + "/index-tts\n")

    if os.path.normcase(source) == os.path.normcase(os.path.abspath(legacy)):
        return
    if os.name == "nt":
        return  # an explicit Windows override remains explicit at runtime
    if os.path.lexists(legacy):
        if os.path.islink(legacy) and os.path.realpath(legacy) == os.path.realpath(source):
            return
        raise ProvisionFailure(
            "ComfyUI/index-tts exists but does not resolve to the managed "
            "IndexTTS2 source %s" % source)
    os.symlink(source, legacy, target_is_directory=True)


def ensure_indextts2_source(comfy: str) -> str:
    """Install or verify the exact IndexTTS2 source checkout."""
    path = _indextts2_source_root(comfy)
    if not os.path.exists(path) or (os.path.isdir(path) and not os.listdir(path)):
        _fetch_exact_repo(INDEXTTS2_URL, INDEXTTS2_PIN, path)
    problems = _indextts2_source_problems(path)
    if problems:
        raise ProvisionFailure(
            "IndexTTS2 source is not the pinned clean checkout (%s); move %s "
            "aside and rerun" % ("; ".join(problems), path))
    _exclude_and_link_indextts2_root(comfy, path)
    say("PRESENT", "index-tts source", INDEXTTS2_PIN[:12])
    return path


def _indextts2_venv_python(source: str) -> str:
    override = os.environ.get("OTR_INDEXTTS2_VENV", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if os.name == "nt":
        return os.path.join(source, ".venv", "Scripts", "python.exe")
    return os.path.join(source, ".venv", "bin", "python")


def _indextts2_model_dir(source: str) -> str:
    override = os.environ.get("OTR_INDEXTTS2_DIR", "").strip()
    path = os.path.abspath(os.path.expanduser(
        override or os.path.join(source, "checkpoints")))
    if os.path.basename(os.path.normpath(path)) != "checkpoints":
        raise ProvisionFailure(
            "OTR_INDEXTTS2_DIR must end in literal lower-case 'checkpoints'; "
            "pinned vendor code resolves ./checkpoints/hf_cache relative to "
            "its launch directory")
    return path


def _load_idx_weights_module():
    path = os.path.join(_HERE, "_otr_idx_download_weights.py")
    spec = importlib.util.spec_from_file_location("otr_idx_weights_manifest", path)
    if spec is None or spec.loader is None:
        raise ProvisionFailure("cannot load IndexTTS2 weight manifest: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_voice_bank_module():
    """Load the dependency-free bank authority without importing all nodes."""
    path = os.path.join(_REPO, "nodes", "_otr_voice_bank.py")
    name = "otr_voice_bank_provision"
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ProvisionFailure("cannot load voice bank authority: %s" % path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module through sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - name the standalone load failure
        raise ProvisionFailure("cannot import voice bank authority: %s" % exc)
    return module


def _wav_problem(path: str) -> str:
    module = getattr(_wav_problem, "_contract", None)
    if module is None:
        contract_path = os.path.join(_HERE, "otr_pcm_reference.py")
        spec = importlib.util.spec_from_file_location(
            "_otr_provision_pcm_reference", contract_path)
        if spec is None or spec.loader is None:
            return "cannot load PCM reference contract at %s" % contract_path
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - admission returns named issue
            return "cannot load PCM reference contract: %s" % exc
        _wav_problem._contract = module
    return module.wav_problem(path)


def verify_registered_indextts2_refs(root: str, bank_path: str | None = None):
    """Return ``(usable_ids, problems, registered_count)`` for local char refs.

    A random WAV in the directory proves nothing: casting selects registered
    bank rows. Every selectable IndexTTS2 char row must resolve under the
    canonical models root and match its exact bank SHA-256.
    """
    bank_path = bank_path or os.environ.get(
        "OTR_VOICE_REFERENCE_BANK", "").strip() or os.path.join(
            _REPO, "config", "voice_reference_bank.json")
    try:
        authority = _load_voice_bank_module()
        rows, _bank_sha = authority.load_voice_bank(bank_path)
        reserved = authority.reserved_voice_ref_ids()
    except Exception as exc:  # noqa: BLE001 - schema authority owns details
        return [], ["voice bank invalid: %s" % exc], 0

    root_abs = os.path.abspath(root)
    usable_records = []
    problems = []
    registered = 0
    for row in rows:
        if getattr(row, "engine", None) != "indextts2":
            continue
        if "char_voice" not in (getattr(row, "roles", ()) or ()):
            continue
        if getattr(row, "quality_tier", "") == "reject":
            continue
        if getattr(row, "voice_ref_id", "") in reserved:
            continue
        registered += 1
        voice_id = str(getattr(row, "voice_ref_id", "") or "<unnamed>")
        gender = str(getattr(row, "gender", "") or "")
        if gender not in {"male", "female", "other"}:
            problems.append(
                "%s has unsupported gender %r; runtime spelling is exact"
                % (voice_id, gender))
            continue
        ref_path = str(getattr(row, "ref_path", "") or "").replace("\\", "/")
        prefix = "models/TTS/refs/indextts2/"
        if not ref_path.startswith(prefix) or not ref_path.lower().endswith(".wav"):
            problems.append("%s has invalid IndexTTS2 ref_path %r" %
                            (voice_id, ref_path))
            continue
        relative = ref_path[len("models/"):]
        resolved = os.path.abspath(os.path.join(root_abs, *relative.split("/")))
        try:
            inside = os.path.commonpath([root_abs, resolved]) == root_abs
        except ValueError:
            inside = False
        if not inside:
            problems.append("%s escapes the models root" % voice_id)
            continue
        expected = str(getattr(row, "ref_sha256", "") or "").lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            problems.append("%s has no exact ref_sha256" % voice_id)
            continue
        if not os.path.isfile(resolved) or os.path.getsize(resolved) <= 0:
            problems.append("%s is missing: %s" % (voice_id, resolved))
            continue
        wav_problem = _wav_problem(resolved)
        if wav_problem:
            problems.append("%s %s" % (voice_id, wav_problem))
            continue
        actual = _sha256_file(resolved)
        if actual.lower() != expected:
            problems.append("%s SHA-256 %s != %s" %
                            (voice_id, actual.lower(), expected))
            continue
        usable_records.append((voice_id, gender, actual.lower()))

    digest_genders = {}
    for _voice_id, gender, digest in usable_records:
        digest_genders.setdefault(digest, set()).add(gender)
    cross_gender = {
        digest: genders for digest, genders in digest_genders.items()
        if len(genders) > 1
    }
    for digest, genders in sorted(cross_gender.items()):
        ids = sorted(voice_id for voice_id, _gender, row_digest
                     in usable_records if row_digest == digest)
        problems.append(
            "IndexTTS2 reference SHA-256 %s is registered across genders %s "
            "(%s); male/female references must be distinct" %
            (digest, ", ".join(sorted(genders)), ", ".join(ids)))
    usable_records = [row for row in usable_records
                      if row[2] not in cross_gender]
    usable = [voice_id for voice_id, _gender, _digest in usable_records]
    usable_genders = {gender for _voice_id, gender, _digest in usable_records}
    if registered == 0:
        problems.append("no registered IndexTTS2 char_voice rows")
    missing_genders = sorted({"male", "female"} - usable_genders)
    if missing_genders:
        problems.append("IndexTTS2 char bank lacks %s coverage" %
                        ", ".join(missing_genders))
    return usable, problems, registered


def _runtime_cache_revision_path(cache: str, repo_id: str, revision: str) -> str:
    repo_dir = "models--" + repo_id.replace("/", "--")
    return os.path.join(cache, repo_dir, "snapshots", revision)


def _runtime_cache_problems(
        cache: str, repo_id: str, revision: str,
        expected: dict[str, int] | None = None) -> list[str]:
    snapshot = _runtime_cache_revision_path(cache, repo_id, revision)
    problems = []
    if not os.path.isdir(snapshot):
        problems.append("snapshot missing")
    else:
        expected = expected or {}
        if not expected and not any(
                os.path.isfile(os.path.join(parent, name))
                for parent, _dirs, names in os.walk(snapshot)
                for name in names):
            problems.append("snapshot empty")
        for relative, expected_bytes in expected.items():
            artifact = os.path.join(snapshot, *relative.split("/"))
            if not os.path.isfile(artifact):
                problems.append("missing %s" % relative)
            elif os.path.getsize(artifact) != expected_bytes:
                problems.append("wrong bytes for %s" % relative)
    ref = os.path.join(
        cache, "models--" + repo_id.replace("/", "--"), "refs", "main")
    try:
        with open(ref, "rb") as handle:
            resolved = handle.read()
    except OSError:
        resolved = b""
    if resolved != revision.encode("ascii"):
        problems.append("refs/main is not the exact 40-byte pinned commit")
    return problems


def _probe_indextts2_worker(source: str) -> tuple[bool, str]:
    """Boot the real worker once and require its first protocol line to be ready."""
    venv_py = _indextts2_venv_python(source)
    worker = os.path.abspath(os.path.expanduser(
        os.environ.get("OTR_INDEXTTS2_WORKER", "").strip()
        or os.path.join(_HERE, "_otr_indextts2_worker.py")))
    checkpoints = _indextts2_model_dir(source)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [venv_py, worker, "--model-dir", checkpoints]
    if os.environ.get("OTR_INDEXTTS2_FP16", "0") == "1":
        command.append("--fp16")
    try:
        result = run(
            command,
            cwd=os.path.dirname(checkpoints),
            env=env,
            input='{"stop": true}\n',
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return False, "worker readiness timed out after 600 seconds"
    if result.returncode != 0:
        return False, "worker exited %d: %s" % (
            result.returncode, (result.stderr or result.stdout or "")[-300:])
    first = (result.stdout or "").splitlines()
    try:
        receipt = json.loads(first[0]) if first else {}
    except ValueError:
        receipt = {}
    if receipt.get("ready") is not True:
        return False, "worker did not report ready: %s" % (
            first[0][:300] if first else "no protocol line")
    return True, "real worker loaded pinned model and stopped cleanly"


def verify_indextts2_install(comfy: str, root: str) -> bool:
    """Read-only completion gate for source, venv, weights, caches, and refs."""
    complete = True
    source = _indextts2_source_root(comfy)
    source_problems = _indextts2_source_problems(source)
    if source_problems:
        say("MISSING", "index-tts pinned source", "; ".join(source_problems))
        complete = False
    else:
        say("PRESENT", "index-tts pinned source", INDEXTTS2_PIN[:12])

    venv_py = _indextts2_venv_python(source)
    if not os.path.isfile(venv_py):
        say("MISSING", "index-tts isolated Python", venv_py)
        complete = False
    else:
        say("PRESENT", "index-tts isolated Python", venv_py)

    manifest = _load_idx_weights_module()
    checkpoints = _indextts2_model_dir(source)
    weight_problems = manifest.validate_model_dir(checkpoints)
    if weight_problems:
        say("MISSING", "index-tts pinned weights", "; ".join(weight_problems[:3]))
        complete = False
    else:
        say("PRESENT", "index-tts pinned weights", manifest._REPO_REVISION[:12])

    cache = os.path.join(checkpoints, "hf_cache")
    missing_runtime = [
        "%s@%s (%s)" % (repo_id, revision, "; ".join(problems))
        for repo_id, revision in manifest._RUNTIME_REPOS
        for problems in [_runtime_cache_problems(
            cache, repo_id, revision,
            manifest._RUNTIME_EXPECTED.get(repo_id, {}))]
        if problems
    ]
    if missing_runtime:
        say("MISSING", "index-tts runtime cache", ", ".join(missing_runtime[:2]))
        complete = False
    else:
        say("PRESENT", "index-tts runtime cache", "4 pinned repos")

    usable, ref_problems, registered = verify_registered_indextts2_refs(root)
    if ref_problems or not usable or len(usable) != registered:
        say("MISSING", "registered IndexTTS2 reference WAVs",
            "%d/%d usable; %s" %
            (len(usable), registered, "; ".join(ref_problems)))
        complete = False
    else:
        say("PRESENT", "registered IndexTTS2 reference WAVs",
            "%d/%d SHA-256 verified" % (len(usable), registered))

    if complete:
        ready, detail = _probe_indextts2_worker(source)
        say("PRESENT" if ready else "MISSING",
            "index-tts real worker readiness", detail)
        complete = complete and ready
    return complete


def _ensure_uv() -> str:
    candidates = [
        shutil.which("uv"),
        os.path.expanduser("~/.local/bin/uv"),
        os.path.expanduser("~/.cargo/bin/uv"),
        os.path.expanduser("~/.local/bin/uv.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    if os.name == "nt":
        raise ProvisionFailure(
            "uv is required for the pinned IndexTTS2 lock; install uv or run "
            "scripts/_otr_indextts2_install.ps1")
    r = run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    if r.returncode != 0:
        raise ProvisionFailure("uv bootstrap failed: %s" %
                               (r.stderr or r.stdout or "unknown error")[:300])
    for candidate in candidates[1:3]:
        if os.path.isfile(candidate):
            return candidate
    raise ProvisionFailure("uv bootstrap completed but the uv executable was not found")


def _uv_python_env(engine_root: str) -> dict[str, str]:
    """Environment whose managed interpreter survives a pod recreation.

    A uv venv contains a symlink to uv's managed Python. The default managed
    location is under the container user's home, which RunPod discards on a
    recreate even when the venv itself lives on a network volume. Put that
    small interpreter under the same engine root as the venv by default. An
    explicit UV_PYTHON_INSTALL_DIR still wins for operators with a shared
    persistent interpreter store.
    """
    env = dict(os.environ)
    if not env.get("UV_PYTHON_INSTALL_DIR", "").strip():
        env["UV_PYTHON_INSTALL_DIR"] = os.path.join(
            os.path.abspath(engine_root), ".uv-python")
    # A suitable system Python would otherwise be allowed to win discovery.
    # Force the venv to link to the managed interpreter we place above.
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    return env


def install_indextts2(comfy: str, root: str) -> None:
    """Build and fully verify the pinned isolated IndexTTS2 environment."""
    source = ensure_indextts2_source(comfy)
    uv = _ensure_uv()
    uv_env = _uv_python_env(source)
    r = run(
        [uv, "python", "install", INDEXTTS2_PYTHON],
        cwd=source,
        env=uv_env,
    )
    if r.returncode != 0:
        raise ProvisionFailure("IndexTTS2 managed Python install failed: %s" %
                               (r.stderr or r.stdout or "unknown error")[:300])
    r = run(
        [uv, "sync", "--frozen", "--python", INDEXTTS2_PYTHON],
        cwd=source,
        env=uv_env,
    )
    if r.returncode != 0:
        raise ProvisionFailure("IndexTTS2 uv sync failed: %s" %
                               (r.stderr or r.stdout or "unknown error")[:300])
    link_indextts2_runtime_python(source)
    venv_py = _indextts2_venv_python(source)
    if not os.path.isfile(venv_py):
        raise ProvisionFailure("IndexTTS2 uv sync did not create %s" % venv_py)
    r = run([venv_py, "-c", "import torch,huggingface_hub;print(torch.__version__)"])
    if r.returncode != 0:
        raise ProvisionFailure("IndexTTS2 isolated import failed: %s" %
                               (r.stderr or r.stdout or "unknown error")[:300])
    say("OK", "index-tts locked dependencies", (r.stdout or "").strip()[:60])

    downloader = os.path.join(_HERE, "_otr_idx_download_weights.py")
    checkpoints = _indextts2_model_dir(source)
    env = dict(os.environ, OTR_INDEXTTS2_DIR=checkpoints)
    r = run([venv_py, downloader], cwd=_REPO, env=env)
    if r.returncode != 0:
        raise ProvisionFailure("IndexTTS2 weights/runtime warm failed: %s" %
                               (r.stderr or r.stdout or "unknown error")[-500:])
    say("OK", "index-tts weights/runtime warm", "pinned downloader complete")
    if not verify_indextts2_install(comfy, root):
        raise ProvisionFailure(
            "IndexTTS2 is incomplete; repair the named source/weight/cache/reference step")


def link_windows_shaped_python(root: str) -> str:
    """Make `.venv/Scripts/python.exe` resolve on Linux, by symlink.

    Every isolated adapter falls back to `<engine>/.venv/Scripts/python.exe`
    -- a Windows shape -- when OTR_<ENGINE>_VENV is unset. Branching on
    os.name inside the adapter is the forbidden fix: qualified voice routes
    are pinned to an adapter FINGERPRINT, and editing one un-qualifies every
    route audited against it while the episode still renders.

    A symlink moves no fingerprint and needs no environment. That matters
    more than it sounds: env vars are set by whoever launches ComfyUI, and a
    pod restart boots it from /start.sh with none of them, so an env-only
    install silently loses all three cloners on every restart.
    """
    if os.name == "nt":
        return os.path.join(root, ".venv", "Scripts", "python.exe")
    real = os.path.join(root, ".venv", "bin", "python")
    scripts_dir = os.path.join(root, ".venv", "Scripts")
    windows_shaped = os.path.join(scripts_dir, "python.exe")
    if os.path.exists(real):
        os.makedirs(scripts_dir, exist_ok=True)
        if os.path.islink(windows_shaped) or os.path.exists(windows_shaped):
            os.remove(windows_shaped)
        os.symlink(os.path.join("..", "bin", "python"), windows_shaped)
    return real


def link_indextts2_runtime_python(root: str) -> str:
    """Publish IndexTTS2's Linux compatibility executable without a re-audition.

    The qualified adapter and worker bytes are immutable until a human approves
    a new audition. On Linux the adapter's historical Windows-shaped executable
    is therefore an external launcher: it establishes the vendor cwd and forces
    the already-pinned Hugging Face cache offline, then execs uv's real Python.
    Windows keeps its qualified native executable unchanged.
    """
    if os.name == "nt":
        return link_windows_shaped_python(root)
    real = os.path.join(root, ".venv", "bin", "python")
    scripts_dir = os.path.join(root, ".venv", "Scripts")
    launcher = os.path.join(scripts_dir, "python.exe")
    if os.path.exists(real):
        os.makedirs(scripts_dir, exist_ok=True)
        if os.path.lexists(launcher):
            os.remove(launcher)
        body = """#!/bin/sh
set -eu
launcher_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
engine_root=$(CDPATH= cd -- "$launcher_dir/../.." && pwd)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$engine_root"
exec "$engine_root/.venv/bin/python" "$@"
"""
        with open(launcher, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
        os.chmod(launcher, 0o755)
    return launcher


def cuda_wheel_tag():
    """The CUDA build ComfyUI itself runs, as a PyTorch wheel tag ("cu128").

    NOT a compute-capability table. ComfyUI's own torch is already proven to
    have kernels for this GPU -- it is what renders every episode -- so
    matching it is correct and self-maintaining, where a hand-kept table of
    architectures needs an edit every time a new one ships.

    This exists because pip's default resolution is blind to the GPU. On a
    Blackwell pod it installed torch 2.6.0+cu124 for chatterbox and
    2.6.0+cu126 for dia; both imported fine and both died at the first kernel
    launch with 'no kernel image is available for execution on the device',
    while index-tts happened to get cu128 and worked.
    """
    r = run([sys.executable, "-c", "import torch;print(torch.version.cuda or '')"])
    ver = (r.stdout or "").strip()
    if r.returncode != 0 or not ver:
        return None
    return "cu" + ver.replace(".", "")


def install_isolated_voice(comfy: str, name: str, pip_args: list) -> None:
    """Build one isolated voice engine's venv and report its env var.

    THE ENV VAR IS THE POINT. Each adapter resolves a Windows-shaped default
    (`.venv/Scripts/python.exe`), and the obvious fix -- branching on os.name
    inside the adapter -- is the wrong one: qualified voice routes are pinned to
    an adapter FINGERPRINT, so editing that file un-qualifies every route
    audited against it and the cast silently drops to an ordinary draw. That
    cost the shipped Lemmy route once already. Setting OTR_<ENGINE>_VENV moves
    no fingerprint.

    chatterbox (MIT) and dia (Apache) are the two commercial-clean cloners and
    the ONLY cloning engines the announcer accepts -- indextts2 is excluded
    there by its non-commercial licence. So these are not spare options; they
    are the only route to a cloned announcer.
    """
    if os.name == "nt":
        installer = os.path.join(_HERE, "_otr_%s_install.ps1" % name)
        if not os.path.isfile(installer):
            raise ProvisionFailure(
                "%s Windows installer is missing: %s" % (name, installer))
        shell = shutil.which("powershell.exe") or shutil.which("powershell")
        if not shell:
            raise ProvisionFailure(
                "%s requires PowerShell on Windows" % name)
        result = run([
            shell, "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", installer,
        ])
        if result.returncode != 0:
            raise ProvisionFailure(
                "%s Windows installer failed: %s" %
                (name, (result.stderr or result.stdout or "unknown error")[-500:]))
        say("OK", name, "Windows isolated installer completed")
        return
    root = os.path.join(comfy, name)   # see install_indextts2: the adapters
    #  resolve <comfy>/<engine>, so installing to <comfy>/../<engine> put the
    #  venv somewhere nothing looks and made the env var load-bearing.
    venv_py = os.path.join(root, ".venv", "bin", "python")
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv):
        run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
        uv = os.path.expanduser("~/.local/bin/uv")
    uv_env = _uv_python_env(root)
    if not os.path.exists(venv_py):
        os.makedirs(root, exist_ok=True)
        r = run(
            [uv, "python", "install", "3.11"],
            env=uv_env,
        )
        if r.returncode != 0:
            say("FAILED", "%s managed Python" % name,
                (r.stderr or r.stdout or "unknown error").strip()[:60])
            return
        r = run(
            [uv, "venv", "--python", "3.11", os.path.join(root, ".venv")],
            env=uv_env,
        )
        if r.returncode != 0:
            say("FAILED", "%s venv" % name, r.stderr.strip()[:60])
            return
    env = dict(uv_env, VIRTUAL_ENV=os.path.join(root, ".venv"))
    # Torch FIRST, from the index matching ComfyUI's own CUDA build. Left to
    # pip's default resolution this lands whatever wheel is newest-compatible
    # with the pinned package, which is blind to the GPU -- see cuda_wheel_tag.
    tag = cuda_wheel_tag()
    if tag:
        say("PIN", "%s torch" % name, "%s (matching ComfyUI)" % tag)
        run([uv, "pip", "install", "-q", "--index-url",
             "https://download.pytorch.org/whl/" + tag,
             "torch", "torchaudio"], env=env)
    run([uv, "pip", "install", "-q"] + pip_args, env=env)
    # VERIFY WITH A REAL KERNEL LAUNCH, not an import. `import torch` succeeds
    # on a build with no kernels for this card, so the old check reported OK
    # for two venvs that then failed at render time. Launching one tiny kernel
    # turns that silent time-bomb into an install-time failure naming itself.
    probe = (
        "import torch; "
        "cap = torch.cuda.is_available(); "
        "val = float(torch.zeros(8, device='cuda').sum()) if cap else 0.0; "
        "print(torch.__version__, 'no-cuda' if not cap else "
        "('kernel-ok' if val == 0.0 else 'kernel-bad'))"
    )
    r = run([venv_py, "-c", probe])
    ok = r.returncode == 0 and "kernel-bad" not in (r.stdout or "")
    say("OK" if ok else "FAILED", "%s venv" % name,
        ((r.stdout or "").strip() or (r.stderr or "").strip())[:56])
    if ok:
        link_windows_shaped_python(root)
        say("SET", "OTR_%s_VENV" % name.upper(), venv_py)
        os.environ["OTR_%s_VENV" % name.upper()] = venv_py


#: The isolated voice engines and what each venv needs.
ISOLATED_VOICES = {
    "chatterbox": ["chatterbox-tts", "soundfile", "torch", "torchaudio"],
    "dia": ["git+https://github.com/nari-labs/dia.git", "soundfile",
            "torch", "torchaudio"],
}


_PUBLIC_VIDEO_IDS = {
    "wan22_high_video": "wan_ti2v",
    "ltx25_high_video": "ltx25_video",
    "ltx25_high_foley_plus": "ltx25_foley_plus",
    "ltx25_high_mime": "ltx25_mime",
    "humo14_high_audio_in_portrait": "humo",
    "humo14_high_audio_in_wide": "humo_14B_169",
    "humo17_high_audio_in_portrait": "humo_1.7B",
    "humo17_high_audio_in_wide": "humo_1.7B_169",
    "h3_low_video": "minimax_h3_video",
    "h3_low_audio_in": "minimax_h3_audio_in",
    "ltx098_low_video": "ltx_8gb",
}
_LTX25_ENGINES = {"ltx25_video", "ltx25_foley_plus", "ltx25_mime"}
_HUMO14_ENGINES = {"humo", "humo_14B_169"}
_HUMO17_ENGINES = {"humo_1.7B", "humo_1.7B_169"}
_H3_ENGINES = {"minimax_h3_video", "minimax_h3_audio_in", "minimax_h3_music"}
_ANIMATEDIFF_ENGINES = {
    "animatediff15_v3_haunted_video",
    "animatediff15_v3_video",
    "animatediff15_v2_video",
    "animatediff15_video",
    "ghost_signal_official",
}
# Registered OTR-native video routes that own no downloadable video weights.
# They still consume the separately routed image/music lanes below. Keeping
# this explicit avoids importing ComfyUI's registry before dependencies are
# installed while ensuring an exact no-weight selection is not mistaken for an
# unknown engine or silently replaced with another lane.
_NO_WEIGHT_VIDEO_ENGINES = {
    "still_flat",
    "still_motion",
    "still_pan",
    "still_word",
    "viz_camera",
    "viz_green",
    "viz_mxc_cpu",
    "viz_mxc_mandala",
}
# These engines explicitly consume no init still. Keep the capability beside
# the lightweight install-routing table: importing their ComfyUI adapters
# before packs and weights exist would make provisioning depend on the thing it
# is meant to install. AnimateDiff Ghost Signal owns its subject from text
# (`required_inputs=("text_prompt",)`, `accepts_still=False`), so its proven
# 4060 path must not be gated on an ~11 GB Klein bundle it cannot invoke.
# Still/pan/motion routes remain image consumers despite owning no video
# weights.
_NO_STILL_VIDEO_ENGINES = _ANIMATEDIFF_ENGINES | {
    "viz_camera",
    "viz_green",
    "viz_mxc_cpu",
    "viz_mxc_mandala",
}
# Provider-side routes own no local video weights. Keep them distinct from the
# OTR-native set: provisioning can truthfully skip a download, while the
# selected adapter still fails loud at invocation when its provider key is
# absent. Only routes exercised by a shipping profile belong here.
_REMOTE_NO_WEIGHT_VIDEO_ENGINES = {
    "cloud_wan_i2v",
    "cloud_wan_i2v_audio",
    "google_omni_video",
    "google_veo_video",
    "word_razzle",
}
_REMOTE_NO_WEIGHT_IMAGE_ENGINES = {
    "cloud_flux_pro",
    "cloud_nano_banana_2",
    "cloud_seedream_2",
    "cloud_krea_2_turbo",
    "cloud_luma_photon_flash",
    "ideo",
    "google_image",
}
_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_profile(profile_id: str) -> dict:
    profile_id = str(profile_id or "")
    if not _PROFILE_ID_RE.fullmatch(profile_id):
        raise ProvisionFailure("invalid profile id: %r" % profile_id)
    path = os.path.join(_REPO, "config", "profiles", profile_id + ".json")
    if not os.path.isfile(path):
        raise ProvisionFailure(
            "profile %r does not exist; use an exact config/profiles id" % profile_id)
    try:
        profile = json.load(io.open(path, encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProvisionFailure("cannot load profile %r: %s" % (profile_id, exc))
    if not isinstance(profile, dict):
        raise ProvisionFailure("profile %r is not a JSON object" % profile_id)
    if str(profile.get("id") or "") != profile_id:
        raise ProvisionFailure(
            "profile filename/id drift: requested %r, document says %r"
            % (profile_id, profile.get("id")))
    return profile


def load_machine_profile(machine_key: str) -> dict:
    path = os.path.join(_HERE, "otr_machine_profile.py")
    spec = importlib.util.spec_from_file_location("otr_machine_profile_provision", path)
    if spec is None or spec.loader is None:
        raise ProvisionFailure("cannot load machine matrix helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        matrix = module.load_matrix()
        row = module.resolve(machine_key, matrix)
        return module.build_profile(row, matrix)
    except SystemExit as exc:
        raise ProvisionFailure(str(exc))


def profile_lanes(profile) -> dict:
    """Resolve one exact profile dict into automatic lanes and manual tiers."""
    if isinstance(profile, str):
        profile = load_profile(profile)
    if not isinstance(profile, dict):
        raise ProvisionFailure("profile router requires a profile dict or exact id")
    pid = str(profile.get("id") or "").strip()
    roles = profile.get("role_overrides", {}) or {}
    slots = profile.get("slot_overrides", {}) or {}
    slot_video = str(slots.get("video_render_engine") or "").strip()
    role_pairs = (
        ("announcer_visual", "announcer_image"),
        ("music_visual", "music_image"),
        ("character_visual", "character_image"),
    )
    selected_role_videos = {}
    for video_key, image_key in role_pairs:
        raw = str(roles.get(video_key) or slot_video).strip()
        selected_role_videos[image_key] = _PUBLIC_VIDEO_IDS.get(raw, raw)
    selected_videos = {
        selected for selected in selected_role_videos.values() if selected
    }
    if slot_video:
        selected_videos.add(_PUBLIC_VIDEO_IDS.get(slot_video, slot_video))
    images = {
        str(roles.get(image_key) or "").strip()
        for _video_key, image_key in role_pairs
        if (str(roles.get(image_key) or "").strip()
            and selected_role_videos[image_key]
            not in _NO_STILL_VIDEO_ENGINES)
    }
    music = str(slots.get("music_engine") or "").strip()

    automatic = []
    manual = []
    for video in sorted(selected_videos):
        if video in _H3_ENGINES:
            manual.append("h3_operator_only")
        elif video in _LTX25_ENGINES:
            manual.append("ltx25")
        elif video in _HUMO14_ENGINES:
            automatic.append("humo")
        elif video in _HUMO17_ENGINES:
            manual.append("humo_1_7b")
        elif video == "wan_ti2v":
            automatic.append("wan_ti2v_gguf")
        elif video == "ltx_8gb":
            automatic.append("ltx_8gb")
        elif video in _ANIMATEDIFF_ENGINES:
            automatic.append("haunted")
        elif video in (_NO_WEIGHT_VIDEO_ENGINES |
                       _REMOTE_NO_WEIGHT_VIDEO_ENGINES):
            pass
        else:
            raise ProvisionFailure(
                "profile %r selects unrecognized video engine %r; "
                "no fallback was chosen" % (pid, video))

    try:
        low_vram = float(
            (profile.get("llm", {}) or {}).get("vram_ceiling_gb")) <= 8.0
    except (TypeError, ValueError):
        low_vram = False
    for image in sorted(images):
        if image == "flux2_klein":
            manual.append("flux2_klein")
        elif image == "z_image_turbo":
            automatic.append("z_image_int8" if low_vram else "z_image")
        elif image in _REMOTE_NO_WEIGHT_IMAGE_ENGINES:
            pass
        else:
            raise ProvisionFailure(
                "profile %r selects unrecognized image engine %r; "
                "no fallback was chosen" % (pid, image))

    if music == "stable_audio_3":
        automatic.append("stable_audio_3")

    def dedupe(values):
        return list(dict.fromkeys(values))

    automatic = dedupe(automatic)
    manual = dedupe(manual)
    unknown = [lane for lane in automatic if lane not in _fetcher_lane_names()]
    if unknown:
        raise ProvisionFailure("profile router selected unknown automatic lane(s): %s"
                               % ", ".join(unknown))
    return {"automatic": automatic, "manual": manual}


def profile_requires_indextts2(profile: dict) -> bool:
    """Whether this exact profile can route character/announcer speech to IndexTTS2."""
    slots = profile.get("slot_overrides", {}) or {}
    return any(
        str(slots.get(name) or "").strip() == "indextts2"
        for name in ("char_voice_engine", "announcer_voice_engine")
    )


def profile_python_issue(profile: dict, version_info=None) -> str:
    """Return the deterministic interpreter/voice incompatibility, if any."""
    version_info = sys.version_info if version_info is None else version_info
    slots = profile.get("slot_overrides", {}) or {}
    voices = {
        str(slots.get(name) or "").strip()
        for name in ("char_voice_engine", "announcer_voice_engine")
    }
    if "kokoro" in voices and tuple(version_info[:2]) >= (3, 13):
        return (
            "selected Kokoro voice cannot be installed on Python 3.13; "
            "use Python 3.12 or earlier; on NVIDIA, the Bark-based "
            "otr_4060_floor profile is the supported Python 3.13 floor"
        )
    return ""


def _print_receipt(success: str = "everything installed.") -> int:
    print("\nreceipt")
    bad = [r for r in _LOG if r[0] in ("FAILED", "MISSING")]
    for state, what, detail in _LOG:
        print("  %-9s %s%s" % (state, what, (" -- " + detail) if detail else ""))
    if bad:
        print("\n  INCOMPLETE: %d required step(s) did not complete." % len(bad))
        print("  Repair the named step and rerun this same command.")
        return 1
    print("\n  %s" % success)
    return 0


def main(argv=None) -> int:
    _LOG.clear()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    selector = ap.add_mutually_exclusive_group()
    selector.add_argument("--profile",
                          help="exact id from config/profiles (default: otr_nvidia_8gb_haunted)")
    selector.add_argument("--machine",
                          help="exact config/machine_classes.json key")
    ap.add_argument("--with-all-voices", action="store_true",
                    help="build EVERY isolated voice engine: indextts2, "
                         "chatterbox, dia. Large; each gets its own venv.")
    ap.add_argument("--with-indextts2", action="store_true",
                    help="build the isolated voice-cloning environment (large)")
    plan_mode = ap.add_mutually_exclusive_group()
    plan_mode.add_argument("--list", action="store_true",
                           help="show what would be installed, install nothing")
    plan_mode.add_argument(
        "--check-plan", action="store_true",
        help="validate exact route ownership and interpreter compatibility; "
             "install nothing and return nonzero when the plan cannot run",
    )
    ap.add_argument("--packs-only", action="store_true",
                    help="install/verify node packs and OTR dependencies only")
    args = ap.parse_args(argv)

    profile = None
    routes = None
    if not args.packs_only:
        try:
            if args.machine:
                profile = load_machine_profile(args.machine)
            else:
                profile = load_profile(args.profile or "otr_nvidia_8gb_haunted")
            routes = profile_lanes(profile)
        except ProvisionFailure as exc:
            say("FAILED", "profile selection", str(exc))
            return _print_receipt()

        if args.check_plan:
            issue = profile_python_issue(profile)
            if issue:
                say("MISSING", "selected voice runtime", issue)
                return _print_receipt()
            print("READY: complete provision plan for %s" % profile.get("id", "?"))
            return 0

        if args.list:
            print("OTR provision plan (nothing installed or hashed)")
            print("  selector     : %s" % profile.get("id", "?"))
            print("  automatic    : %s" %
                  (", ".join(routes["automatic"]) or "none"))
            print("  manual tiers : %s" % (", ".join(routes["manual"]) or "none"))
            required_keys = list(
                (profile.get("preflight") or {}).get("required_keys") or [])
            print("  required keys: %s" %
                  (", ".join(required_keys) or "none"))
            print("  IndexTTS2    : %s" %
                  ("required by profile" if profile_requires_indextts2(profile)
                   else "not selected by profile"))
            issue = profile_python_issue(profile)
            print("  Python       : %s" % (issue or "compatible"))
            for tier_id in routes["manual"]:
                if tier_id in OPERATOR_ONLY_TIERS:
                    print("\n  %s: %s" % (tier_id, OPERATOR_ONLY_TIERS[tier_id]))
                    continue
                artifacts = MANUAL_TIERS[tier_id]
                print("\n  %s (%d bytes total):" %
                      (tier_id, sum(item["bytes"] for item in artifacts)))
                for item in artifacts:
                    print("    %s" % item["destination"])
                    print("      %s@%s/%s" %
                          (item["repo"], item["revision"], item["path"]))
                    print("      %d bytes  sha256=%s%s" %
                          (item["bytes"], item["sha256"],
                           "  TERMS/TOKEN" if item.get("gated") else ""))
            return 0

        issue = profile_python_issue(profile)
        if issue:
            say("MISSING", "selected voice runtime", issue)
            return _print_receipt()

    try:
        comfy = comfy_root()
    except ProvisionFailure as exc:
        say("FAILED", "ComfyUI root", str(exc))
        return _print_receipt()

    if args.packs_only:
        print("OTR provision")
        print("  comfy root  : %s" % comfy)
        print("  mode        : packs only")
        print("")
        try:
            install_node_packs(comfy)
            install_requirements()
        except ProvisionFailure as exc:
            say("FAILED", "required pack/dependency", str(exc))
        return _print_receipt("all required node packs and dependencies verified.")

    root = models_root(comfy)

    print("OTR provision")
    print("  comfy root  : %s" % comfy)
    print("  models root : %s" % root)
    print("  selector    : %s" % profile.get("id", "?"))
    print("  automatic   : %s" % (", ".join(routes["automatic"]) or "none"))
    print("  manual tiers: %s" % (", ".join(routes["manual"]) or "none"))
    print("")

    try:
        os.environ.setdefault("OTR_COMFYUI_MODELS_ROOT", root)
        ensure_hf_home(root)
        install_node_packs(comfy)
        install_requirements()
        fetch_lane_weights(routes["automatic"])
        warm_profile_writer_models(profile)
        manual_results = [
            verify_manual_tier(root, tier_id) for tier_id in routes["manual"]
        ]
        manual_complete = all(manual_results)
        if routes["manual"] and not manual_complete:
            say("MISSING", "MANUAL WEIGHTS REMAIN",
                "read the named tier receipt above; final .part files never count")
        want_indextts2 = args.with_all_voices or args.with_indextts2
        if want_indextts2:
            install_indextts2(comfy, root)
        elif profile_requires_indextts2(profile):
            if not verify_indextts2_install(comfy, root):
                say("MISSING", "selected profile requires IndexTTS2",
                    "rerun with --with-indextts2 after providing the registered reference WAVs")
        else:
            say("SKIP", "index-tts", "not selected by this profile")

        if args.with_all_voices:
            for name, pip_args in ISOLATED_VOICES.items():
                install_isolated_voice(comfy, name, pip_args)
    except ProvisionFailure as exc:
        say("FAILED", "required provisioning step", str(exc))
    return _print_receipt()


if __name__ == "__main__":
    sys.exit(main())
