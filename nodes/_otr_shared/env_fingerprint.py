"""What this process is running on: ComfyUI, every custom node pack, Python,
torch, CUDA, the GPU, its driver and the OS. (PLAN row 2.1,
planning/creatoros-concepts/PLAN.md.)

WHY THIS EXISTS. A ledger records only OTR's own commit. When an episode breaks
after a ComfyUI or node-pack update, nothing on disk said what moved, and with
two boxes there was no quick way to see how they differ. ``Ledger.save`` stamps
``fingerprint()`` into ``meta.env`` (with ``meta.env_version``), and
``scripts/otr_env_drift.py`` diffs two of them.

WHAT IT RECORDS, AND WHAT IT NEVER DOES.
  * Versions and commits only. No path, user name or machine name: the stamp
    travels in ledgers and replay bundles, and a pack is named by its folder
    name alone.
  * Read-only metadata. Nothing branches on it, so no render on any machine
    changes because of it.
  * No fallbacks: a value it cannot read is ``"unknown: <reason>"``, never a
    guess.
  * Computed once per process (it is a boot constant) and bounded in time: git
    state is read straight from ``.git`` files, not by a git subprocess per
    pack, and the one ``nvidia-smi`` call has a 5 s timeout.
  * It never imports torch. Inside ComfyUI torch is already loaded; anywhere
    else the torch fields say so.
"""
from __future__ import annotations

import os
import platform
import re
import sys
import threading
from typing import Any, Dict, Iterable, List, Optional

try:
    from . import proc as otr_proc
except ImportError:  # pragma: no cover -- flat load
    import proc as otr_proc  # type: ignore

#: Bump when a field changes meaning, so a reader can tell two shapes apart.
ENV_VERSION = 1

_NVIDIA_SMI_TIMEOUT_S = 5
_SHA = re.compile(r"^[0-9a-f]{40}$")
_LOCK = threading.Lock()
_CACHE: Optional[Dict[str, Any]] = None


def _unknown(reason: str) -> str:
    return "unknown: %s" % reason


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _git_dir(repo: str) -> Optional[str]:
    """The git dir of ``repo``: ``.git`` itself, or where a ``.git`` FILE
    (a worktree or a submodule) points."""
    dot_git = os.path.join(repo, ".git")
    if os.path.isdir(dot_git):
        return dot_git
    text = _read_text(dot_git) if os.path.isfile(dot_git) else None
    if text and text.startswith("gitdir:"):
        target = text[len("gitdir:"):].strip()
        if not os.path.isabs(target):
            target = os.path.normpath(os.path.join(repo, target))
        return target if os.path.isdir(target) else None
    return None


def _common_dir(git_dir: str) -> str:
    """A worktree keeps HEAD in its own git dir and the refs in the common one."""
    text = _read_text(os.path.join(git_dir, "commondir"))
    if text and text.strip():
        common = text.strip()
        if not os.path.isabs(common):
            common = os.path.normpath(os.path.join(git_dir, common))
        if os.path.isdir(common):
            return common
    return git_dir


def _resolve_ref(git_dir: str, ref: str) -> Optional[str]:
    for base in dict.fromkeys((git_dir, _common_dir(git_dir))):
        loose = _read_text(os.path.join(base, *ref.split("/")))
        if loose and _SHA.match(loose.strip()):
            return loose.strip()
        packed = _read_text(os.path.join(base, "packed-refs"))
        for line in (packed or "").splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2 and parts[1] == ref and _SHA.match(parts[0]):
                return parts[0]
    return None


def git_commit(repo: str) -> Optional[str]:
    """The commit ``repo`` has checked out (12 hex characters), or None when
    it is not a git checkout or its HEAD cannot be read."""
    git_dir = _git_dir(repo)
    if git_dir is None:
        return None
    head = (_read_text(os.path.join(git_dir, "HEAD")) or "").strip()
    if _SHA.match(head):
        return head[:12]
    if head.startswith("ref:"):
        sha = _resolve_ref(git_dir, head[len("ref:"):].strip())
        return sha[:12] if sha else None
    return None


def _pyproject_version(folder: str) -> Optional[str]:
    text = _read_text(os.path.join(folder, "pyproject.toml"))
    if not text:
        return None
    match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return match.group(1).strip() if match else None


def pack_state(folder: str) -> str:
    """One custom node folder: ``git <sha>``, else ``version <v>`` from its
    pyproject.toml, else ``unknown: <reason>``."""
    sha = git_commit(folder)
    if sha:
        return "git %s" % sha
    if _git_dir(folder) is not None:
        return _unknown("a git checkout whose HEAD cannot be read")
    version = _pyproject_version(folder)
    if version:
        return "version %s" % version
    return _unknown("no .git and no pyproject.toml version")


def _comfyui_root() -> Optional[str]:
    """The ComfyUI core folder of THIS process, or None outside ComfyUI."""
    module = sys.modules.get("folder_paths")
    if module is None:
        try:
            import folder_paths as module  # type: ignore
        except Exception:  # noqa: BLE001 -- outside ComfyUI
            return None
    path = getattr(module, "__file__", None)
    return os.path.dirname(os.path.abspath(path)) if path else None


def _custom_node_dirs() -> List[str]:
    module = sys.modules.get("folder_paths")
    try:
        dirs = module.get_folder_paths("custom_nodes") if module else []
    except Exception:  # noqa: BLE001
        dirs = []
    return [d for d in dirs if isinstance(d, str) and os.path.isdir(d)]


def comfyui_state(root: Optional[str]) -> Dict[str, str]:
    if root is None:
        reason = "not running inside ComfyUI (folder_paths not importable)"
        return {"version": _unknown(reason), "commit": _unknown(reason)}
    text = _read_text(os.path.join(root, "comfyui_version.py")) or ""
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    version = match.group(1) if match else _unknown("no __version__ in comfyui_version.py")
    commit = git_commit(root) or _unknown("the core folder is not a git checkout")
    return {"version": version, "commit": commit}


def packs_state(dirs: Iterable[str]) -> Dict[str, str]:
    """Every pack folder under ``dirs``, by folder name. The first directory
    wins a name collision, as ComfyUI's own load order does."""
    packs: Dict[str, str] = {}
    for base in dirs:
        try:
            names = sorted(os.listdir(base))
        except OSError:
            continue
        for name in names:
            folder = os.path.join(base, name)
            if name.startswith(".") or name == "__pycache__" or name in packs:
                continue
            if not os.path.isdir(folder) or name.endswith(".disabled"):
                continue
            packs[name] = pack_state(folder)
    return packs


def torch_state() -> Dict[str, str]:
    torch = sys.modules.get("torch")
    if torch is None:
        reason = "torch is not loaded in this process"
        return {"torch": _unknown(reason), "cuda": _unknown(reason), "gpu": _unknown(reason)}
    out = {"torch": str(getattr(torch, "__version__", "") or _unknown("no __version__"))}
    version = getattr(torch, "version", None)
    cuda = getattr(version, "cuda", None)
    hip = getattr(version, "hip", None)
    out["cuda"] = (str(cuda) if cuda else ("hip %s" % hip if hip
                   else _unknown("torch was built without CUDA")))
    try:
        if torch.cuda.is_available():
            out["gpu"] = str(torch.cuda.get_device_name(torch.cuda.current_device()))
        elif getattr(getattr(torch, "backends", None), "mps", None) is not None \
                and torch.backends.mps.is_available():
            out["gpu"] = "Apple MPS"
        else:
            out["gpu"] = _unknown("torch sees no GPU")
    except Exception as exc:  # noqa: BLE001
        out["gpu"] = _unknown("torch could not name the GPU: %s" % type(exc).__name__)
    return out


def driver_state() -> str:
    """The NVIDIA driver version, from one bounded ``nvidia-smi`` call."""
    try:
        done = otr_proc.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S, check=False)
    except FileNotFoundError:
        return _unknown("no nvidia-smi")
    except otr_proc.TimeoutExpired:
        return _unknown("nvidia-smi gave no answer within %d s" % _NVIDIA_SMI_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        return _unknown("nvidia-smi failed: %s" % type(exc).__name__)
    lines = [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]
    if done.returncode != 0 or not lines:
        return _unknown("nvidia-smi exited %s with no driver version" % done.returncode)
    return lines[0]


def os_state() -> str:
    try:
        parts = [platform.system(), platform.release(), platform.version()]
    except Exception:  # noqa: BLE001
        return _unknown("platform could not be read")
    text = " ".join(p for p in parts if p)
    return text or _unknown("platform reported nothing")


def compute() -> Dict[str, Any]:
    """The fingerprint, freshly read. Never raises."""
    env: Dict[str, Any] = {}
    root = _comfyui_root()
    for key, read in (
        ("comfyui", lambda: comfyui_state(root)),
        ("packs", lambda: packs_state(_custom_node_dirs())),
        ("python", lambda: platform.python_version()),
        ("os", os_state),
        ("driver", driver_state),
    ):
        try:
            env[key] = read()
        except Exception as exc:  # noqa: BLE001 -- reported, never raised
            # The type alone: an exception's text can carry a local path.
            env[key] = _unknown("%s while reading it" % type(exc).__name__)
    try:
        env.update(torch_state())
    except Exception as exc:  # noqa: BLE001
        for key in ("torch", "cuda", "gpu"):
            env[key] = _unknown("%s while reading it" % type(exc).__name__)
    return env


def fingerprint() -> Dict[str, Any]:
    """The fingerprint of this process, computed once and then reused. A copy
    is returned, so a caller that edits it never edits the cache."""
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            _CACHE = compute()
        return _copy(_CACHE)


def _copy(env: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in env.items()}


def stamp(meta: Dict[str, Any]) -> bool:
    """Put this process's fingerprint in ``meta`` unless it already holds one.
    Returns True when it stamped."""
    if isinstance(meta.get("env"), dict) and meta.get("env"):
        return False
    meta["env"] = fingerprint()
    meta["env_version"] = ENV_VERSION
    return True


def _reset_cache_for_tests() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = None


__all__ = ["ENV_VERSION", "fingerprint", "compute", "stamp", "git_commit", "pack_state"]
