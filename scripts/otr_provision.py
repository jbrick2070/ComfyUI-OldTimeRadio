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
import glob
import hashlib
import io
import json
import os
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
    """Install or verify the exact LTXVideo node-pack commit."""
    dest = os.path.join(comfy, "custom_nodes", LTXVIDEO_PACK_NAME)
    fresh = not os.path.isdir(dest) or not os.listdir(dest)
    if fresh:
        _fetch_exact_repo(LTXVIDEO_URL, LTXVIDEO_PIN, dest)
    if not os.path.isdir(os.path.join(dest, ".git")):
        raise ProvisionFailure(
            "ComfyUI-LTXVideo is not a verifiable git checkout; move it aside and rerun --packs-only")
    head = _git_head(dest)
    changed = _git_changed_paths(dest)
    untracked = _git_untracked_paths(dest)
    if head != LTXVIDEO_PIN or changed or untracked:
        raise ProvisionFailure(
            "ComfyUI-LTXVideo must be clean at %s; found head=%s changed=%s untracked=%s"
            % (LTXVIDEO_PIN, head, changed, untracked))
    install_pack_requirements(LTXVIDEO_PACK_NAME, dest, required=True)
    say("OK" if fresh else "PRESENT", LTXVIDEO_PACK_NAME, LTXVIDEO_PIN[:12])


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
    packs = [
        # AnimateDiff lane.
        ("ComfyUI-AnimateDiff-Evolved",
         "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved", None),
    ]
    for name, url, branch in packs:
        dest = os.path.join(cn, name)
        # PRESENT means the directory is there with something in it, NOT that
        # it is a git clone. ComfyUI-Manager installs packs without a .git
        # directory -- the reference 5080 carries ComfyUI-GGUF exactly that
        # way -- and the old .git test called such a pack absent, then tried
        # to clone into a non-empty directory, failed, and reported FAILED for
        # a pack that was installed and working.
        if os.path.isdir(dest) and os.listdir(dest):
            say("PRESENT", name)
            install_pack_requirements(name, dest)
            continue
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["-b", branch]
        r = run(cmd + [url, dest])
        if r.returncode != 0:
            raise ProvisionFailure("clone failed for %s: %s" %
                                   (name, (r.stderr or "").strip()[:300]))
        say("OK", name)
        install_pack_requirements(name, dest)


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
    it = os.path.join(comfy, "index-tts")   # where eng_indextts2._default looks:
    #  _COMFY_ROOT is the ComfyUI dir ITSELF, not its parent. Installing a
    #  level up made OTR_INDEXTTS2_VENV mandatory rather than optional.
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

    # The symlink that used to live here is gone: it linked <comfy>/index-tts
    # to a clone one directory ABOVE <comfy>, because that is where the clone
    # went. The clone now goes straight to <comfy>/index-tts -- the path the
    # adapter actually resolves -- so the indirection has nothing left to do.
    # chatterbox and dia never had that symlink, which is the whole reason
    # they could not be found without OTR_<ENGINE>_VENV set by hand.
    link_windows_shaped_python(it)

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
        say("SKIP", name, "Windows: use scripts/_otr_%s_install.ps1" % name)
        return
    root = os.path.join(comfy, name)   # see install_indextts2: the adapters
    #  resolve <comfy>/<engine>, so installing to <comfy>/../<engine> put the
    #  venv somewhere nothing looks and made the env var load-bearing.
    venv_py = os.path.join(root, ".venv", "bin", "python")
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv):
        run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
        uv = os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(venv_py):
        os.makedirs(root, exist_ok=True)
        r = run([uv, "venv", "--python", "3.11", os.path.join(root, ".venv")])
        if r.returncode != 0:
            say("FAILED", "%s venv" % name, r.stderr.strip()[:60])
            return
    env = dict(os.environ, VIRTUAL_ENV=os.path.join(root, ".venv"))
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
    ap.add_argument("--profile", default="otr_nvidia_8gb_haunted")
    ap.add_argument("--with-all-voices", action="store_true",
                    help="build EVERY isolated voice engine: indextts2, "
                         "chatterbox, dia. Large; each gets its own venv.")
    ap.add_argument("--with-indextts2", action="store_true",
                    help="build the isolated voice-cloning environment (large)")
    ap.add_argument("--list", action="store_true",
                    help="show what would be installed, install nothing")
    ap.add_argument("--packs-only", action="store_true",
                    help="install/verify node packs and OTR dependencies only")
    args = ap.parse_args(argv)

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

    try:
        os.environ.setdefault("OTR_COMFYUI_MODELS_ROOT", root)
        ensure_hf_home(root)
        install_node_packs(comfy)
        install_requirements()
        fetch_lane_weights(lanes)
        if args.with_all_voices:
            install_indextts2(comfy, root)
            for name, pip_args in ISOLATED_VOICES.items():
                install_isolated_voice(comfy, name, pip_args)
        elif args.with_indextts2:
            install_indextts2(comfy, root)
        else:
            say("SKIP", "index-tts", "pass --with-indextts2 to build it")
    except ProvisionFailure as exc:
        say("FAILED", "required provisioning step", str(exc))
    return _print_receipt()


if __name__ == "__main__":
    sys.exit(main())
