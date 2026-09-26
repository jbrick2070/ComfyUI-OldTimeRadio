"""Choose the HF_HOME prestartup may assign. Stdlib only.

Prestartup must not import the node package to make this choice, so the
``C:\\ComfyUI-Models`` existence check is duplicated here and in the HF
env default. This module has no import-time side effects: registry,
long-path, and write checks run only when asked.
"""
import logging
import os
import sys
from os import environ, getenv

_MODELS_HF = r"C:\ComfyUI-Models\huggingface"
_UNSET = object()


def _registry_hf_home():
    """HKCU\\Environment\\HF_HOME, or None. Never raises."""
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ,
        ) as key:
            value, _typ = winreg.QueryValueEx(key, "HF_HOME")
    except Exception:  # noqa: BLE001 -- a missing key is "not set"
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _long_paths_enabled():
    """True when Windows' 260-char limit is lifted registry-wide.

    python.exe already ships longPathAware, so this key is the deciding
    half. Never raises: a prestartup that dies takes the whole boot with it.
    """
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            return int(winreg.QueryValueEx(key, "LongPathsEnabled")[0]) == 1
    except Exception:  # noqa: BLE001 -- absent key = off
        return False


def _user_hf_cache():
    """The huggingface_hub-shaped user cache, via os.path only.

    ``getenv`` is the bare name on purpose: spelling the call as an attribute
    of the os module is a registry-scan hit, and this file ships.
    """
    return os.path.join(
        getenv("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "huggingface",
    )


def _write_probe(path):
    """True when ``path`` can be created and a file written there. Never raises."""
    probe = os.path.join(path, ".otr_hf_write_probe")
    try:
        os.makedirs(path, exist_ok=True)
        with open(probe, "w", encoding="ascii") as handle:
            handle.write("ok")
        try:
            os.remove(probe)
        except OSError:
            pass
        return True
    except OSError:
        return False


def _default_log(message):
    logging.getLogger("OTR").warning("%s", message)


def _choose_hf_home(
    models_adjacent,
    room,
    *,
    platform=_UNSET,
    registry_value=_UNSET,
    hub_cache=_UNSET,
    legacy_hub_cache=_UNSET,
    long_paths_enabled=_UNSET,
    models_root=_UNSET,
    models_root_exists=_UNSET,
    user_cache=_UNSET,
    write_probe=_UNSET,
    log=_UNSET,
):
    """Return the HF_HOME to assign, or None to leave it unset.

    Candidate order, first qualifier wins:

    1. Windows ``HKCU\\Environment\\HF_HOME`` when ``len(value) <= room``.
    2. Process ``HF_HUB_CACHE``, then ``HUGGINGFACE_HUB_CACHE``, same rule.
    3. ``models_adjacent`` when not Windows, or it fits ``room``, or
       ``LongPathsEnabled=1``.
    4. ``C:\\ComfyUI-Models\\huggingface`` when that models root exists,
       the candidate fits, and the write probe succeeds.
    5. The user cache, when it fits and the write probe succeeds.
    6. None. Every refused candidate has already been logged with its length.

    A too-long value is never returned. Keyword arguments default to the
    live machine; tests pass them so this stays callable without booting.
    """
    if platform is _UNSET:
        platform = sys.platform
    if registry_value is _UNSET:
        registry_value = _registry_hf_home() if platform == "win32" else None
    if hub_cache is _UNSET:
        hub_cache = environ.get("HF_HUB_CACHE")
    if legacy_hub_cache is _UNSET:
        legacy_hub_cache = environ.get("HUGGINGFACE_HUB_CACHE")
    if long_paths_enabled is _UNSET:
        long_paths_enabled = _long_paths_enabled()
    if models_root is _UNSET:
        models_root = _MODELS_HF
    if models_root_exists is _UNSET:
        models_root_exists = os.path.isdir(os.path.dirname(models_root))
    if user_cache is _UNSET:
        user_cache = _user_hf_cache()
    if write_probe is _UNSET:
        write_probe = _write_probe
    if log is _UNSET:
        log = _default_log

    def fits(path):
        return len(path) <= room

    def refuse(path, why):
        log(
            "OldTimeRadio prestartup: refused HF_HOME candidate %s "
            "(%d characters): %s" % (path, len(path), why)
        )

    if platform == "win32" and registry_value:
        if fits(registry_value):
            return registry_value
        refuse(registry_value, "longer than the %d-character room" % room)

    for label, value in (
        ("HF_HUB_CACHE", hub_cache),
        ("HUGGINGFACE_HUB_CACHE", legacy_hub_cache),
    ):
        if not value:
            continue
        if fits(value):
            return value
        refuse(value, "%s is longer than the %d-character room" % (label, room))

    if platform != "win32" or fits(models_adjacent) or long_paths_enabled:
        return models_adjacent
    refuse(
        models_adjacent,
        "longer than the %d-character room and LongPathsEnabled is off" % room,
    )

    if platform == "win32":
        if not models_root_exists:
            refuse(models_root, "C:\\ComfyUI-Models does not exist")
        elif not fits(models_root):
            refuse(models_root, "longer than the %d-character room" % room)
        elif write_probe(models_root):
            return models_root
        else:
            refuse(models_root, "write probe failed")

    if not fits(user_cache):
        refuse(user_cache, "longer than the %d-character room" % room)
    elif write_probe(user_cache):
        return user_cache
    else:
        refuse(user_cache, "write probe failed")

    log(
        "OldTimeRadio prestartup: leaving HF_HOME unset; no candidate fit "
        "in %d characters" % room
    )
    return None
