"""OTR_SaveToEpisodeWorkspace -- per-episode-aware image save sink.

Replaces stock ``SaveImage`` nodes in the OTR workflow that previously
hardcoded ``filename_prefix="otr/stills/full_env"``. With the per-episode
workspace reorg (Phase B/E, 2026-05-02 EVENING), every episode artifact
should land under ``output/otr/episodes/<episode_id>/`` so VideoComposite
and downstream consumers can discover the per-episode subdir without
tree-walking. This node reads the in-flight Ledger singleton at runtime,
derives the episode_id, and writes images to the canonical per-episode
``stills/`` or ``portraits/`` directory.

OH-1 (output-tree contract, 2026-06-11): the legacy flat-dir fallback
(``_legacy_stills/`` / ``_legacy_portraits/``) is REMOVED. A save with
no resolvable in-flight episode now fails LOUD (RuntimeError) instead of
silently littering the otr/ top level -- the same loud-failure posture
as the BAD_IMAGE_SAVE gate below.

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
    """Pick the right per-episode directory (OH-1: episode_id REQUIRED).

    Unknown ``role_kind`` defaults to ``stills`` to preserve existing
    behavior for legacy workflows that pass an empty string. The
    underlying ``otr_stills_dir`` / ``otr_portraits_dir`` helpers RAISE
    LOUD (``OtrPathContractError``) on an empty episode_id -- the
    legacy flat-dir fallback is gone.
    """
    if not episode_id:
        raise RuntimeError(
            "SaveToEpisodeWorkspace: no in-flight episode could be "
            "resolved (ledger singleton missing) -- refusing to save "
            "outside the per-episode workspace (output-tree contract, "
            "2026-06-11). Run the writer first so a ledger exists.")
    if role_kind == "portraits":
        return _OTRP.otr_portraits_dir(episode_id)
    # Default: stills for any unknown role_kind.
    return _OTRP.otr_stills_dir(episode_id)


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
        extra_pnginfo=None,  # kept: ComfyUI hidden input contract -- runtime always passes it, body does not need to consume
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

        log.info(
            "[SaveToEpisodeWorkspace] saving to per-episode dir: %s "
            "(role_kind=%s, ep=%s, pattern=%s)",
            target_dir,
            role_kind,
            episode_id,
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
                # Round-robin Step 3 follow-up (post 13:07 clean run, 2026-05-08):
                # Recognize the benign all-zero placeholder tensor that
                # ``OTR_BatchFluxRender`` emits when ``skip_env_stills=True``
                # (BUG-LOCAL-078 follow-up). Saving it produces the
                # 74-byte PNG that the gate below would otherwise raise on,
                # but in this case the empty image is INTENTIONAL --
                # nothing to save. Skip silently with a debug log so the
                # runtime trace stays useful, but don't pollute disk
                # with empty PNGs and don't false-fire the gate.
                # Sentinel predicate: any tensor 64x64 or smaller with
                # all zeros is treated as an upstream skip-stub. Real
                # FLUX/LTX renders are >=512x512; a tiny zero tensor
                # only appears as a deliberate empty marker (e.g.
                # BatchFluxRender's (16,16,3) placeholder when
                # skip_env_stills=True). This avoids accidentally
                # swallowing a legitimately-black large render while
                # still catching the known sentinel.
                _is_zero_marker = False
                try:
                    import torch  # type: ignore
                    if isinstance(img, torch.Tensor) and img.numel() > 0:
                        # Last two dims for (..., H, W, C) or (..., H, W).
                        shp = tuple(img.shape)
                        if len(shp) >= 2:
                            _h, _w = shp[-3], shp[-2] if len(shp) >= 3 else (shp[-2], shp[-1])
                        else:
                            _h = _w = max(shp) if shp else 0
                        if _h <= 64 and _w <= 64:
                            _tmin = float(img.min().item())
                            _tmax = float(img.max().item())
                            if _tmin == 0.0 and _tmax == 0.0:
                                _is_zero_marker = True
                except Exception:  # noqa: BLE001
                    pass
                if _is_zero_marker:
                    log.info(
                        "[SaveToEpisodeWorkspace] skipping all-zero "
                        "marker tensor (shape=%s); upstream node "
                        "produced an intentional empty placeholder, "
                        "no PNG written",
                        tuple(getattr(img, "shape", ())),
                    )
                    next_idx += 1
                    continue

                pil_img = _tensor_to_pil(img)
                fname = f"{filename_pattern}_{next_idx:05d}_.png"
                out_path = target_dir / fname
                pil_img.save(out_path)
                # Round-robin Step 3 (post-Stage 1, 2026-05-08): catch
                # the empty-tensor failure mode early. The 2026-05-08
                # morning Stage 1 run wrote a 74-byte PNG (header
                # chunks only, no IDAT) for stills/full_env_00001_.png
                # because the upstream FLUX node yielded an empty
                # IMAGE tensor. Without this gate the empty save was
                # invisible until VideoComposite tried to use the
                # still and the 74-byte file silently produced a
                # black/garbage frame. The 4096-byte threshold catches
                # any plausibly-empty PNG (a real 1024x1024 FLUX still
                # is hundreds of KB to MB) while still letting tiny
                # legitimate test fixtures through.
                _MIN_VALID_PNG_BYTES = 4096
                try:
                    saved_bytes = int(out_path.stat().st_size)
                except OSError:
                    saved_bytes = 0
                if saved_bytes < _MIN_VALID_PNG_BYTES:
                    # Build a diagnostic about the input tensor so the
                    # caller knows whether the upstream produced empty
                    # data or this save logic mishandled it. shape /
                    # dtype / min / max in one line, no second run
                    # needed.
                    try:
                        import torch  # type: ignore
                        if isinstance(img, torch.Tensor):
                            tshape = tuple(img.shape)
                            tdtype = str(img.dtype)
                            tmin = float(img.min().item()) if img.numel() else float("nan")
                            tmax = float(img.max().item()) if img.numel() else float("nan")
                        else:
                            tshape = getattr(img, "shape", "n/a")
                            tdtype = str(getattr(img, "dtype", type(img)))
                            tmin = tmax = float("nan")
                    except Exception:  # noqa: BLE001
                        tshape = "n/a"
                        tdtype = "n/a"
                        tmin = tmax = float("nan")
                    # Unlink the bad PNG so a stale 74-byte file
                    # doesn't pollute the episode dir or get picked up
                    # by downstream consumers. RuntimeError surfaces
                    # the diagnostic so the workflow halts loudly
                    # instead of silently producing garbage video.
                    try:
                        out_path.unlink()
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"BAD_IMAGE_SAVE: {fname} wrote {saved_bytes} "
                        f"bytes (under {_MIN_VALID_PNG_BYTES}-byte gate); "
                        f"input tensor shape={tshape} dtype={tdtype} "
                        f"min={tmin:.4f} max={tmax:.4f}. Likely upstream "
                        f"produced empty IMAGE -- check the previous "
                        f"node's output tensor before re-queueing."
                    )
                log.info("[SaveToEpisodeWorkspace] saved: %s (%d bytes)",
                         out_path, saved_bytes)
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
