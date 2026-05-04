"""
_otr_hf_env.py  --  HF_HOME + canonical snapshot resolver
==========================================================

Reads HF_HOME from the Windows user environment (HKCU\\Environment) because
the running ComfyUI Desktop process often does not inherit the User-scope
env vars (Electron processes started from non-Explorer parents miss them).

Provides:

    ensure_hf_home() -> str
        Resolve HF_HOME (registry > os.environ > default), export it back
        into os.environ so transformers / huggingface_hub pick it up
        without explicit ``cache_dir=`` arguments. Returns the resolved
        path. Idempotent.

    resolve_snapshot_dir(model_id, hf_home=None) -> str | None
        Return the absolute path to the model's snapshot directory in
        the canonical cache, or None if the model is not cached. Useful
        for passing directly to ``model_loader()`` as a path instead
        of a Hub ID, bypassing transformers' Hub-resolution logic that
        sometimes mis-handles Windows symlinks under local_files_only.

Why this exists (BUG-LOCAL-085):
    Last night's full re-render OOM'd at Mistral-Nemo prefill with 24 GiB
    allocated on a 16 GiB GPU. Diagnosis: NF4 quantization config WAS
    built and passed to model_loader, but transformers fell back to
    full bf16 because cache resolution failed on the Mistral-Nemo
    sharded-safetensors layout under local_files_only. Tested in
    isolation: passing the snapshot directory path directly to
    model_loader loads cleanly at 7.79 GiB with all 280 Linear
    modules quantized to 4-bit NF4 -- proves the bug is the cache
    resolution path, not the quantization config.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("OTR._otr_hf_env")

_REG_KEY = "Environment"
_REG_HF_HOME = "HF_HOME"
_DEFAULT_HF_HOME = r"C:\ComfyUI-Models\huggingface"

_CACHE: dict[str, str | None] = {"hf_home": None, "resolved": False}


def _read_hf_home_from_winreg() -> str | None:
    """Look up HF_HOME in HKEY_CURRENT_USER\\Environment.

    Returns the string value or None when the key is absent, the
    platform is not Windows, or the lookup raises for any reason.
    """
    try:
        import winreg  # Windows-only stdlib module
    except ImportError:
        return None

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _REG_KEY,
            0,
            winreg.KEY_READ,
        ) as key:
            value, _type = winreg.QueryValueEx(key, _REG_HF_HOME)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        log.debug("_otr_hf_env: winreg HF_HOME lookup failed: %s", exc)
        return None
    return None


def ensure_hf_home() -> str:
    """Resolve HF_HOME and export it into os.environ.

    Order of precedence:
        1. os.environ['HF_HOME']  (already in process env)
        2. HKCU\\Environment\\HF_HOME  (Windows User-scope)
        3. _DEFAULT_HF_HOME  ('C:\\ComfyUI-Models\\huggingface')

    Result is cached for the lifetime of the process. Idempotent --
    safe to call from multiple module init paths.

    Returns the absolute HF_HOME path. Also writes:
        os.environ['HF_HOME']      = resolved value
        os.environ['HF_HUB_CACHE'] = resolved value (belt + suspenders)
    """
    if _CACHE["resolved"] and _CACHE["hf_home"]:
        return _CACHE["hf_home"]

    # 1. Already in process env?
    env_val = (os.environ.get("HF_HOME") or "").strip()
    if env_val:
        resolved = env_val
        source = "os.environ"
    else:
        # 2. Windows registry?
        reg_val = _read_hf_home_from_winreg()
        if reg_val:
            resolved = reg_val
            source = "HKCU\\Environment"
        else:
            # 3. Default
            resolved = _DEFAULT_HF_HOME
            source = "default"

    # Export so downstream HF tooling picks it up automatically.
    os.environ["HF_HOME"] = resolved
    os.environ["HF_HUB_CACHE"] = resolved
    _CACHE["hf_home"] = resolved
    _CACHE["resolved"] = True
    log.info(
        "[OTR_HF_ENV] HF_HOME=%r (source=%s); exported to os.environ",
        resolved, source,
    )
    return resolved


def _model_id_to_cache_dirname(model_id: str) -> str:
    """Convert 'mistralai/Mistral-Nemo-Instruct-2407' to
    'models--mistralai--Mistral-Nemo-Instruct-2407'.
    """
    return "models--" + model_id.replace("/", "--")


def resolve_snapshot_dir(model_id: str, hf_home: str | None = None) -> str | None:
    """Return the absolute path to the model's snapshot directory under
    the canonical HF cache, or None if the model is not cached.

    Layout expected:
        <hf_home>/hub/models--<org>--<name>/snapshots/<commit_sha>/

    Picks the most-recently-modified snapshot if multiple exist (HF
    keeps prior snapshots until cleanup). Returns None when:
        - hf_home directory missing
        - models--<...> directory missing
        - snapshots/ directory missing or empty

    The returned path is suitable for passing directly to
    ``AutoModelForCausalLM.model_loader(snapshot_dir, ...)`` --
    bypasses transformers' Hub-resolution layer entirely.
    """
    home = hf_home or ensure_hf_home()
    cache_dirname = _model_id_to_cache_dirname(model_id)
    snapshots_dir = Path(home) / "hub" / cache_dirname / "snapshots"
    if not snapshots_dir.is_dir():
        log.debug(
            "[OTR_HF_ENV] no snapshot dir for %s under %s",
            model_id, snapshots_dir,
        )
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Most recent wins (HF cleanup keeps prior snapshots until pruned).
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    log.info(
        "[OTR_HF_ENV] snapshot resolved %s -> %s",
        model_id, chosen,
    )
    return str(chosen)


__all__ = ["ensure_hf_home", "resolve_snapshot_dir"]
