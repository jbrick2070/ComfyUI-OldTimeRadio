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
        Return the newest MATERIALIZED snapshot (one containing a real
        weight blob), or None if no complete snapshot is cached. Useful for
        passing directly to ``model_loader()`` as a path instead of a Hub ID.

    resolve_snapshot_file(model_id, filename, hf_home=None) -> str | None
        Return the newest materialized copy of one metadata file across all
        cached snapshots. This supports HF's split-snapshot layout where a
        newer metadata revision (for example ``chat_template.jinja``) sits
        beside an older, complete weight revision.

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
from pathlib import Path

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR._otr_hf_env")

_REG_KEY = "Environment"
_REG_HF_HOME = "HF_HOME"
_DEFAULT_HF_HOME = r"C:\ComfyUI-Models\huggingface"
_WEIGHT_SUFFIXES = (".safetensors", ".bin")

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
        os.environ['HF_HUB_CACHE'] = <resolved value> / 'hub'
    """
    if _CACHE["resolved"] and _CACHE["hf_home"]:
        return _CACHE["hf_home"]

    # 1. Already in process env?
    env_val = (otr_env.get("HF_HOME") or "").strip()
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
    otr_env.pin("HF_HOME", resolved)
    # HF_HOME is the cache *root*; HF_HUB_CACHE is the hub subdirectory.
    # Pointing both variables at the same path makes huggingface_hub scan
    # ``...\huggingface\models--*`` while OTR stores the weights under
    # ``...\huggingface\hub\models--*``. That mismatch can turn a complete
    # offline cache into a false miss and trigger an unwanted download.
    hub_cache = str(Path(resolved) / "hub")
    otr_env.pin("HF_HUB_CACHE", hub_cache)
    _CACHE["hf_home"] = resolved
    _CACHE["resolved"] = True
    log.info(
        "[OTR_HF_ENV] HF_HOME=%r HF_HUB_CACHE=%r (source=%s); "
        "exported to os.environ",
        resolved, hub_cache, source,
    )
    return resolved


def _model_id_to_cache_dirname(model_id: str) -> str:
    """Convert 'mistralai/Mistral-Nemo-Instruct-2407' to
    'models--mistralai--Mistral-Nemo-Instruct-2407'.
    """
    return "models--" + model_id.replace("/", "--")


def _snapshot_has_weights(snapshot_path: Path) -> bool:
    """Return True only for a snapshot with a materialized weight blob."""
    try:
        for child in snapshot_path.iterdir():
            if child.suffix.lower() not in _WEIGHT_SUFFIXES:
                continue
            try:
                if child.stat().st_size > 0:  # follows HF snapshot symlinks
                    return True
            except OSError:
                continue
    except OSError:
        return False
    return False


def _snapshot_candidates(model_id: str, hf_home: str) -> list[Path]:
    snapshots_dir = (
        Path(hf_home)
        / "hub"
        / _model_id_to_cache_dirname(model_id)
        / "snapshots"
    )
    if not snapshots_dir.is_dir():
        return []
    try:
        candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    except OSError:
        return []
    candidates.sort(
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    return candidates


def resolve_snapshot_dir(model_id: str, hf_home: str | None = None) -> str | None:
    """Return the absolute path to the model's snapshot directory under
    the canonical HF cache, or None if the model is not cached.

    Layout expected:
        <hf_home>/hub/models--<org>--<name>/snapshots/<commit_sha>/

    Picks the most-recently-modified snapshot that contains at least one
    materialized ``.safetensors`` or ``.bin`` weight file. Metadata-only
    snapshots are intentionally skipped. Returns None when:
        - hf_home directory missing
        - models--<...> directory missing
        - snapshots/ directory missing or empty

    The returned path is suitable for passing directly to
    ``AutoModelForCausalLM.model_loader(snapshot_dir, ...)`` --
    bypasses transformers' Hub-resolution layer entirely.
    """
    home = hf_home or ensure_hf_home()
    candidates = _snapshot_candidates(model_id, home)
    if not candidates:
        log.debug(
            "[OTR_HF_ENV] no snapshot dir for %s under %s",
            model_id, Path(home) / "hub",
        )
        return None
    weighted = [p for p in candidates if _snapshot_has_weights(p)]
    if not weighted:
        log.warning(
            "[OTR_HF_ENV] cached snapshots for %s contain no materialized "
            "weight blob under %s",
            model_id, Path(home) / "hub",
        )
        return None
    chosen = weighted[0]
    log.info(
        "[OTR_HF_ENV] snapshot resolved %s -> %s",
        model_id, chosen,
    )
    return str(chosen)


def resolve_snapshot_file(
    model_id: str,
    filename: str,
    hf_home: str | None = None,
) -> str | None:
    """Return the newest cached copy of ``filename`` across snapshots.

    Only a simple basename is accepted; callers cannot escape the snapshot
    directory. Files must resolve to a materialized, non-empty target.
    """
    name = str(filename or "").strip()
    if not name or Path(name).name != name:
        raise ValueError("filename must be one non-empty basename")
    home = hf_home or ensure_hf_home()
    for snapshot in _snapshot_candidates(model_id, home):
        candidate = snapshot / name
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                log.info(
                    "[OTR_HF_ENV] snapshot metadata resolved %s:%s -> %s",
                    model_id, name, candidate,
                )
                return str(candidate)
        except OSError:
            continue
    return None


__all__ = [
    "ensure_hf_home",
    "resolve_snapshot_dir",
    "resolve_snapshot_file",
]
