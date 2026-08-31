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

So this never gates a render and never returns a verdict. It installs, then
prints a receipt of what it did -- a doer confirming its own work.

MACHINE-AGNOSTIC ON PURPOSE. The five things it does -- locate the tree ComfyUI
scans, resolve the model roots, install node packs, fetch lane weights, build
the isolated TTS environment -- are none of them pod-specific, and the pod-shaped
bash version could not serve the Windows boxes at all. What genuinely cannot be
automated stays in the manual: SSH keys, template environment, GPU and volume
selection. Those happen BEFORE the machine exists.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

#: Every step appends here; the receipt is printed from it at the end.
_LOG: list = []


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
    code = ("import sys;sys.path.insert(0,%r)\n"
            "from nodes._otr_gguf_backend import _models_root\n"
            "print(_models_root())\n" % _REPO)
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
def install_node_packs(comfy: str) -> None:
    cn = os.path.join(comfy, "custom_nodes")
    os.makedirs(cn, exist_ok=True)
    packs = [
        ("ComfyUI-AnimateDiff-Evolved",
         "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved", None),
    ]
    for name, url, branch in packs:
        dest = os.path.join(cn, name)
        if os.path.isdir(os.path.join(dest, ".git")):
            say("PRESENT", name)
            continue
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["-b", branch]
        r = run(cmd + [url, dest])
        say("OK" if r.returncode == 0 else "FAILED", name,
            "" if r.returncode == 0 else r.stderr.strip()[:60])


def install_requirements() -> None:
    req = os.path.join(_REPO, "requirements.txt")
    if not os.path.isfile(req):
        say("SKIP", "requirements.txt", "not found")
        return
    r = run([sys.executable, "-m", "pip", "install", "-q", "-r", req])
    say("OK" if r.returncode == 0 else "FAILED", "pip install requirements",
        "" if r.returncode == 0 else r.stderr.strip()[:60])


def fetch_lane_weights(lanes) -> None:
    """Delegate to the fetcher, which is the ONE place sources are recorded."""
    fetcher = os.path.join(_HERE, "otr_fetch_lane_weights.py")
    for lane in lanes:
        r = run([sys.executable, fetcher, lane])
        tail = [ln for ln in (r.stdout or "").splitlines() if ln.strip()][-1:] or [""]
        say("OK" if r.returncode == 0 else "FAILED", "weights: %s" % lane,
            tail[0].strip()[:60])


def install_indextts2(comfy: str, root: str) -> None:
    """The isolated voice-cloning environment.

    Four things, none guessable, all learned the expensive way:
      * the project pins >=3.10,<3.12 and a host may only have 3.12, so the venv
        needs a DIFFERENT INTERPRETER -- `uv` fetches one;
      * it ships pyproject.toml and NO requirements.txt, so a guard on the
        latter installs nothing and leaves a venv without torch, reporting
        success;
      * snapshot_download gives ~5.6 GB of an 11 GB model -- the rest is four
        more repos it pulls at RENDER time, inside a node with a timeout;
      * the engine resolves <comfy_root>/index-tts, so the install is symlinked
        into place rather than duplicated.
    """
    it = os.path.join(os.path.dirname(comfy), "index-tts")
    if os.name == "nt":
        say("SKIP", "index-tts", "Windows: use scripts/_otr_indextts2_install.ps1")
        return
    if not os.path.isdir(os.path.join(it, ".git")):
        r = run(["git", "clone", "--depth", "1",
                 "https://github.com/index-tts/index-tts.git", it])
        if r.returncode != 0:
            say("FAILED", "index-tts clone", r.stderr.strip()[:60])
            return
    say("OK", "index-tts source", it)

    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv):
        run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
        uv = os.path.expanduser("~/.local/bin/uv")
    venv_py = os.path.join(it, ".venv", "bin", "python")
    if not os.path.exists(venv_py):
        r = run([uv, "venv", "--python", "3.11", os.path.join(it, ".venv")])
        if r.returncode != 0:
            say("FAILED", "index-tts venv (3.11)", r.stderr.strip()[:60])
            return
    env = dict(os.environ, VIRTUAL_ENV=os.path.join(it, ".venv"))
    run([uv, "pip", "install", "-q", "."], cwd=it, env=env)
    r = run([venv_py, "-c", "import torch;print(torch.__version__)"])
    say("OK" if r.returncode == 0 else "FAILED", "index-tts deps",
        (r.stdout or r.stderr).strip()[:60])

    link = os.path.join(comfy, "index-tts")
    if not os.path.exists(link):
        try:
            os.symlink(it, link)
            say("OK", "index-tts symlink", link)
        except OSError as exc:
            say("FAILED", "index-tts symlink", str(exc)[:60])

    # The four repos it pulls at render time, warmed here where a slow download
    # is free rather than inside a node with a wall-clock budget.
    cache = os.path.join(it, "checkpoints", "hf_cache")
    os.makedirs(cache, exist_ok=True)
    code = (
        "import os;os.environ['HF_HOME']=%r\n"
        "from huggingface_hub import snapshot_download\n"
        "for r in ('facebook/w2v-bert-2.0','amphion/MaskGCT','funasr/campplus',"
        "'nvidia/bigvgan_v2_22khz_80band_256x'):\n"
        "    snapshot_download(r)\n" % cache)
    r = run([venv_py, "-c", code])
    say("OK" if r.returncode == 0 else "FAILED", "index-tts runtime repos",
        "4 repos" if r.returncode == 0 else r.stderr.strip()[:60])

    refs = os.path.join(root, "TTS", "refs", "indextts2")
    n = len(glob.glob(os.path.join(refs, "*.wav")))
    say("OK" if n else "MISSING", "voice reference WAVs",
        "%d in %s" % (n, refs) if n
        else "none -- indextts2 refuses without them; copy them here")


def profile_lanes(profile_id: str) -> list:
    """Which weight bundles a profile needs, read from the profile."""
    path = os.path.join(_REPO, "config", "profiles", profile_id + ".json")
    if not os.path.isfile(path):
        return ["haunted"]
    d = json.load(io.open(path, encoding="utf-8"))
    ro, so = d.get("role_overrides", {}) or {}, d.get("slot_overrides", {}) or {}
    eng = str(ro.get("character_visual") or so.get("video_render_engine") or "")
    lanes = []
    if "animatediff" in eng or "ghost" in eng:
        lanes.append("haunted")
    if "minimax" in eng or "h3" in eng:
        lanes.append("minimax_h3")
    if "wan_ti2v" in eng:
        lanes.append("wan_ti2v")
    if "ltx_8gb" in eng:
        lanes.append("ltx_8gb")
    # Stable Audio 3 sits on the SHARED path: OTR_StableAudioTheme runs for every
    # profile that reaches the music node, so a machine without it fails every
    # lane, not just a "music lane". Always fetched.
    lanes.append("stable_audio_3")
    return lanes or ["haunted", "stable_audio_3"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="otr_nvidia_8gb_haunted")
    ap.add_argument("--with-indextts2", action="store_true",
                    help="build the isolated voice-cloning environment (large)")
    ap.add_argument("--list", action="store_true",
                    help="show what would be installed, install nothing")
    args = ap.parse_args(argv)

    comfy = comfy_root()
    root = models_root(comfy)
    lanes = profile_lanes(args.profile)

    print("OTR provision")
    print("  comfy root  : %s" % comfy)
    print("  models root : %s" % root)
    print("  profile     : %s" % args.profile)
    print("  weight lanes: %s" % ", ".join(lanes))
    if args.list:
        print("  (--list: nothing installed)")
        return 0
    print("")

    os.environ.setdefault("OTR_COMFYUI_MODELS_ROOT", root)
    ensure_hf_home(root)
    install_node_packs(comfy)
    install_requirements()
    fetch_lane_weights(lanes)
    if args.with_indextts2:
        install_indextts2(comfy, root)
    else:
        say("SKIP", "index-tts", "pass --with-indextts2 to build it")

    # THE RECEIPT. Not a gate -- a doer saying what it did.
    print("\nreceipt")
    bad = [r for r in _LOG if r[0] in ("FAILED", "MISSING")]
    for state, what, detail in _LOG:
        print("  %-9s %s%s" % (state, what, (" -- " + detail) if detail else ""))
    if bad:
        print("\n  %d step(s) did not complete. This does NOT block a render --"
              % len(bad))
        print("  the engines refuse by name if something they need is absent.")
        print("  See docs/RUNPOD_INSTALL.md section 5 for the symptom index.")
    else:
        print("\n  everything installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
