"""OTR_SaveToEpisodeWorkspace -- per-episode-aware image save sink.

Replaces stock ``SaveImage`` nodes in the OTR workflow that previously
hardcoded ``filename_prefix="otr/stills/full_env"``. With the per-episode
workspace reorg (Phase B/E, 2026-05-02 EVENING), every episode artifact
should land under ``output/otr/episodes/<episode_id>/`` so VideoComposite
and downstream consumers can discover the per-episode subdir without
tree-walking. This node reads the in-flight Ledger singleton at runtime,
derives the episode_id, and writes images to the canonical per-episode
``stills/`` or ``portraits/`` directory.

Falls back to the legacy flat dirs (``_legacy_stills/`` /
``_legacy_portraits/``) when no singleton is available -- preserves
backward compatibility for headless / standalone test invocations.
Never raises: a save failure is logged and silently skipped so a broken
save doesn't crash the whole workflow.

BUG-LOCAL-028 fix (2026-05-03 EVENING).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore
from PIL import Image  # type: ignore

# Late imports: support both ``python -m`` package context (relative)
# and direct script context (used by some test harnesses).
try:
    from . import _otr_paths as _OTRP  # type: ignore
    from . import _otr_ledger as _OTRL  # type: ignore
except ImportError:  # pragma: no cover -- direct-script fallback
    import sys as _sys
    _NODES_DIR = Path(__file__).resolve().parent
    if str(_NODES_DIR) not in _sys.path:
        _sys.path.insert(0, str(_NODES_DIR))
    import _otr_paths as _OTRP  # type: ignore
    import _otr_ledger as _OTRL  # type: ignore

log = logging.getLogger(__name__)

_ROLE_KINDS = ("stills", "portraits")


def _resolve_episode_id() -> Optional[str]:
    """Return the in-flight episode_id, or None if no singleton is set.

    Uses ``_otr_ledger.in_flight_ledger_path()`` (Phase G discovery,
    BUG-LOCAL-021) which itself falls back to the mtime walker when no
    singleton is active. We add a final ``None`` filter so callers can
    distinguish "active episode resolved" vs "fall back to legacy".
    """
    try:
        ledger_p = _OTRL.in_flight_ledger_path()
        if ledger_p is None:
            return None
        led = _OTRL.load_ledger_safe(ledger_p)
        if led is None:
            return None
        ep = (led.get("episode_id") or "").strip()
        return ep or None
    except Exception as exc:  # noqa: BLE001
        log.warning("[SaveToEpisodeWorkspace] episode_id lookup failed: %s", exc)
        return None


def _resolve_target_dir(role_kind: str, episode_id: Optional[str]) -> Path:
    """Pick the right per-episode (or legacy fallback) directory.

    Unknown ``role_kind`` defaults to ``stills`` to preserve existing
    behavior for legacy workflows that pass an empty string. The
    underlying ``otr_stills_dir`` / ``otr_portraits_dir`` helpers
    handle the ``episode_id == ""`` legacy fallback themselves.
    """
    if role_kind == "portraits":
        return _OTRP.otr_portraits_dir(episode_id or "")
    # Default: stills for any unknown role_kind.
    return _OTRP.otr_stills_dir(episode_id or "")


def _next_index(target_dir: Path, pattern: str) -> int:
    """Find the next free counter for ``<pattern>_NNNNN_.png`` files.

    Per-episode counter starts at 1 -- we don't share a counter across
    episodes the way stock SaveImage's flat-dir behavior did.
    """
    if not target_dir.exists():
        return 1
    max_seen = 0
    for p in target_dir.iterdir():
        if not p.is_file():
            continue
        name = p.stem
        if not name.startswith(pattern + "_"):
            continue
        tail = name[len(pattern) + 1:].rstrip("_")
        try:
            n = int(tail)
        except ValueError:
            continue
        if n > max_seen:
            max_seen = n
    return max_seen + 1


def _tensor_to_pil(img_tensor) -> Image.Image:
    """Convert a ComfyUI image tensor (HxWxC float 0..1) to PIL."""
    if hasattr(img_tensor, "detach"):
        arr = img_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(img_tensor)
    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(arr)


class SaveToEpisodeWorkspace:
    """Per-episode-aware image save sink.

    Replaces stock ``SaveImage`` nodes whose hardcoded ``filename_prefix``
    can't track the in-flight episode. Reads the Ledger singleton at
    runtime via ``in_flight_ledger_path()`` and routes images to the
    canonical per-episode workspace.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "role_kind": (list(_ROLE_KINDS), {"default": "stills"}),
                "filename_pattern": (
                    "STRING",
                    {"default": "full_env", "multiline": False},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "OldTimeRadio/save"

    def save(
        self,
        images,
        role_kind: str = "stills",
        filename_pattern: str = "full_env",
        prompt=None,
        extra_pnginfo=None,
    ):
        episode_id = _resolve_episode_id()
        target_dir = _resolve_target_dir(role_kind, episode_id)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[SaveToEpisodeWorkspace] mkdir failed for %s: %s",
                target_dir,
                exc,
            )
            return {}

        if episode_id:
            log.info(
                "[SaveToEpisodeWorkspace] saving to per-episode dir: %s "
                "(role_kind=%s, ep=%s, pattern=%s)",
                target_dir,
                role_kind,
                episode_id,
                filename_pattern,
            )
        else:
            log.warning(
                "[SaveToEpisodeWorkspace] no in-flight ledger singleton -- "
                "falling back to legacy dir: %s (role_kind=%s, pattern=%s)",
                target_dir,
                role_kind,
                filename_pattern,
            )

        next_idx = _next_index(target_dir, filename_pattern)
        ui_results = []

        # ComfyUI passes ``images`` as a 4D tensor (B, H, W, C). Iterate
        # the batch dim explicitly so a single-image render works the
        # same as a multi-image batch.
        try:
            iterator = list(images)
        except TypeError:
            iterator = [images]

        for img in iterator:
            try:
                pil_img = _tensor_to_pil(img)
                fname = f"{filename_pattern}_{next_idx:05d}_.png"
                out_path = target_dir / fname
                pil_img.save(out_path)
                log.info("[SaveToEpisodeWorkspace] saved: %s", out_path)
                # Build a UI entry so ComfyUI shows the thumbnail. The
                # ``subfolder`` field must be relative to the ComfyUI
                # output dir (``comfy_output_dir()``) for the preview
                # to resolve correctly in the web UI.
                comfy_out = _OTRP.comfy_output_dir()
                try:
                    rel = target_dir.resolve().relative_to(comfy_out.resolve())
                    subfolder = str(rel).replace("\\", "/")
                except ValueError:
                    subfolder = ""
                ui_results.append(
                    {
                        "filename": fname,
                        "subfolder": subfolder,
                        "type": "output",
                    }
                )
                next_idx += 1
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[SaveToEpisodeWorkspace] save failed for image %d: %s",
                    next_idx,
                    exc,
                )

        return {"ui": {"images": ui_results}}


__all__ = ["SaveToEpisodeWorkspace"]
