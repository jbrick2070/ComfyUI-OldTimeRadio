"""Two-file API keys for Google, OpenRouter, and Comfy Cloud.

The README heading is the user-facing copy. This module is the only reader
of those files. Environment variables still win when they are set.

Two ways to enter a key, same recipe for every lane:

1. Put the key on the first real line of ``<lane>.secret`` in the pack folder.
2. Or keep the key wherever you want, and put that file's full path on the
   first real line of ``<lane>_api_key.location`` in the pack folder.

``OTR_TEST_MODE`` skips the files so a leftover secret on the box cannot
satisfy a "no key" test. A test that wants the files sets
``OTR_ALLOW_KEY_FILES=1`` and, if needed, ``OTR_API_KEY_PACK_ROOT``.
"""
from __future__ import annotations

from pathlib import Path

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    import env as otr_env  # type: ignore

PACK_ROOT = Path(__file__).resolve().parents[2]

LANES = {
    "google": {
        "label": "Google",
        "secret": "google.secret",
        "location": "google_api_key.location",
        "env": ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
    "openrouter": {
        "label": "OpenRouter",
        "secret": "openrouter.secret",
        "location": "openrouter_api_key.location",
        "env": ("OPENROUTER_API_KEY",),
    },
    # No "comfy" lane (rip 2026-09-19): the Comfy credential is ONLY the
    # api_key_comfy_org hidden input ComfyUI injects -- app sign-in, or a
    # headless submitter's extra_data. Never an env var or a pack file.
}


class KeyFileError(RuntimeError):
    """A location file named a path that is missing or empty."""


def pack_root() -> Path:
    override = otr_env.get("OTR_API_KEY_PACK_ROOT")
    if isinstance(override, str) and override.strip():
        return Path(override.strip())
    return PACK_ROOT


def files_enabled() -> bool:
    if str(otr_env.get("OTR_ALLOW_KEY_FILES") or "") == "1":
        return True
    if otr_env.get("OTR_TEST_MODE"):
        return False
    return True


def _first_real_line(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        return line
    return None


def env_key(lane: str) -> str | None:
    spec = LANES[lane]
    for name in spec["env"]:
        value = otr_env.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def file_key(lane: str) -> str | None:
    """Read the two pack files. Raises :class:`KeyFileError` on a broken pointer."""
    if not files_enabled():
        return None
    spec = LANES[lane]
    root = pack_root()
    direct = _first_real_line(root / spec["secret"])
    if direct:
        return direct
    pointer = _first_real_line(root / spec["location"])
    if not pointer:
        return None
    target = Path(pointer).expanduser()
    if not target.is_file():
        raise KeyFileError(
            "%s key location %s names %s, which is not a file. No request "
            "was sent." % (spec["label"], spec["location"], target)
        )
    key = _first_real_line(target)
    if not key:
        raise KeyFileError(
            "%s key file %s is empty. No request was sent."
            % (spec["label"], target)
        )
    return key


def resolve_lane_key(lane: str) -> str | None:
    """Environment first, then the two pack files. None if nothing is set."""
    return env_key(lane) or file_key(lane)


def missing_key_hint(lane: str) -> str:
    spec = LANES[lane]
    env = ", ".join(spec["env"])
    return (
        "No %s API key. Two ways to enter it, both in this pack folder: "
        "put the key on the first line of %s, or put the path to your own "
        "key file on the first line of %s. Environment %s also works. "
        "No request was sent."
        % (spec["label"], spec["secret"], spec["location"], env)
    )


__all__ = [
    "KeyFileError",
    "LANES",
    "PACK_ROOT",
    "env_key",
    "file_key",
    "files_enabled",
    "missing_key_hint",
    "pack_root",
    "resolve_lane_key",
]
