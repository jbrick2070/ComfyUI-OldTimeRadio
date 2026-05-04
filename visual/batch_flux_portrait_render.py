"""
batch_flux_portrait_render.py  --  per-cast portrait render via FLUX
=====================================================================

BUG-LOCAL-078 fix (2026-05-03 EVENING). Generates ONE clean head-and-
shoulders portrait per cast member using FLUX, saves to the per-episode
``portraits/`` directory, and stamps ``cast[i].portrait_path`` into the
ledger so downstream HuMo's ``_find_portrait`` picks them up at tier 1
(highest priority) instead of falling through to the env-still tier 4
stopgap.

Why this exists:
  Pre-fix, ``portraits/`` stayed empty in every episode workspace.
  HuMo's portrait resolver fell through to tier 4 (FLUX env stills
  ``full_env_NNNNN_.png`` indexed by cast position). Env stills are
  SCENE shots — they happen to contain characters in scene context,
  but lighting / outfit / angle vary widely and HuMo's lipsync
  reference quality suffers. Each character maps to a random env still
  per BUG-078 stopgap comment in batch_humo_render.py:322-327.

After this fix:
  Tier 1 hits for every cast member with a populated portrait_path.
  Portraits are clean centered headshots at 1024x1024 with neutral
  lighting and a frontal pose (FLUX-friendly composition that gives
  HuMo a stable reference for facial geometry across all of that
  character's lines).

Cast traits (appearance, voice, etc.) come from the ledger's ``cast[]``
section populated by LLMDirector. Each portrait prompt is built as:

    "<period style anchor>, head and shoulders portrait of {speaker},
     {appearance}, neutral expression, centered composition, soft
     studio lighting, 35mm film grain, no other characters in frame"

Output:
  * ``output/otr/episodes/<ep>/portraits/<char_id>_portrait.png`` (1024x1024)
  * ``ledger.cast[i].portrait_path`` = absolute path string above

Production:
  ~10-15s per portrait on RTX 5080. With 2-4 cast members per episode,
  total cost is ~30-60s — small compared to HuMo's per-line cost.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore

# Make the repo root importable so we can pull `nodes/_otr_paths` etc.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from nodes import _otr_paths as _OTRP  # type: ignore
from nodes import _otr_ledger as _OTRL  # type: ignore

log = logging.getLogger("OTR")

# Default render dims: FLUX-native square portrait. 1024x1024 is the
# canonical FLUX training resolution; produces the cleanest headshots
# for downstream HuMo reference.
DEFAULT_PORTRAIT_W = 1024
DEFAULT_PORTRAIT_H = 1024


def _slugify_char_id(s: str) -> str:
    """Convert a char_id like 'c01' or a freeform speaker name into a
    filesystem-safe slug for the portrait filename."""
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def _build_portrait_prompt(speaker: str, appearance: str, style_anchor: str) -> str:
    """Compose the FLUX prompt for one cast member's portrait.

    Period style anchor + character description + composition guidance.
    Designed to give HuMo a stable, frontal, well-lit reference frame
    that's NOT a scene shot.
    """
    speaker = (speaker or "Unnamed character").strip()
    appearance = (appearance or "").strip()
    style_anchor = (style_anchor or "1940s noir radio drama style").strip()
    parts = [
        style_anchor,
        f"head and shoulders portrait of {speaker}",
    ]
    if appearance:
        parts.append(appearance)
    parts.extend([
        "neutral expression",
        "centered composition",
        "frontal pose facing camera",
        "soft studio lighting",
        "35mm film grain",
        "no other characters in frame",
        "no background props",
    ])
    return ", ".join(parts)


class BatchFluxPortraitRender:
    """Generate one clean FLUX portrait per cast member.

    Reads cast[] from the ledger, renders each portrait sequentially
    (low VRAM cost since FLUX is already loaded for env stills), and
    stamps cast[i].portrait_path into the ledger so HuMo's tier 1
    portrait lookup hits.
    """

    CATEGORY = "OTR/v2/Visual"
    OUTPUT_NODE = True
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("portrait_batch", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "FLUX MODEL output (CheckpointLoaderSimple)",
                }),
                "clip": ("CLIP", {
                    "tooltip": "FLUX CLIP output (CheckpointLoaderSimple)",
                }),
                "vae": ("VAE", {
                    "tooltip": "FLUX VAE output (CheckpointLoaderSimple)",
                }),
                "ledger_json": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Ledger source. Empty -> auto-pick most recent "
                        "from disk via in_flight_ledger_path. String "
                        "starting with '{' -> parsed as JSON. Filesystem "
                        "path -> loaded directly."
                    ),
                }),
            },
            "optional": {
                "style_anchor": ("STRING", {
                    "multiline": False,
                    "default": "1940s noir radio drama style, cinematic",
                    "tooltip": (
                        "Period style anchor prepended to each portrait "
                        "prompt. Keep all cast members visually consistent "
                        "by sharing the same style across the cast."
                    ),
                }),
                "width": ("INT", {
                    "default": DEFAULT_PORTRAIT_W,
                    "min": 256, "max": 2048, "step": 64,
                }),
                "height": ("INT", {
                    "default": DEFAULT_PORTRAIT_H,
                    "min": 256, "max": 2048, "step": 64,
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 100, "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1,
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "FluxGuidance strength (3.5 is the canonical FLUX default)",
                }),
                "sampler_name": (["euler"], {"default": "euler"}),
                "scheduler": (["simple"], {"default": "simple"}),
                "seed": ("INT", {
                    "default": 100, "min": 0, "max": 2**32 - 1,
                    "tooltip": (
                        "Base seed; each cast member's portrait gets "
                        "seed+i so re-renders are reproducible."
                    ),
                }),
                "skip_announcer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Skip rendering portraits for cast members whose "
                        "char_id maps to an announcer role (LTX renders "
                        "the radio scene for them; HuMo never asks for "
                        "their portrait)."
                    ),
                }),
                "flux_done_gate": ("IMAGE", {
                    "tooltip": (
                        "Optional ordering edge. Wire the IMAGE output "
                        "from OTR_BatchFluxRender here so this portrait "
                        "node executes AFTER FLUX env stills + radio "
                        "bookend (and therefore BEFORE OTR_UnloadAll "
                        "unloads the FLUX checkpoint). Value is "
                        "ignored; only the dependency edge matters. "
                        "Mirrors the BUG-LOCAL-086 pattern in "
                        "OTR_BatchHumoRender."
                    ),
                }),
            },
        }

    def execute(
        self,
        model,
        clip,
        vae,
        ledger_json: str = "",
        style_anchor: str = "1940s noir radio drama style, cinematic",
        width: int = DEFAULT_PORTRAIT_W,
        height: int = DEFAULT_PORTRAIT_H,
        steps: int = 20,
        cfg: float = 1.0,
        guidance: float = 3.5,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        seed: int = 100,
        skip_announcer: bool = True,
        flux_done_gate=None,
    ):
        # Ordering edge only; consume to silence "unused variable" lints.
        del flux_done_gate
        from PIL import Image  # type: ignore
        try:
            from nodes import (  # type: ignore  # ComfyUI nodes namespace
                CLIPTextEncode, EmptyLatentImage, KSampler, VAEDecode,
            )
        except Exception as exc:
            raise RuntimeError(
                f"OTR_BatchFluxPortraitRender: cannot import ComfyUI core "
                f"sampler nodes: {exc}"
            )
        try:
            from comfy_extras.nodes_flux import FluxGuidance  # type: ignore
            guidance_node = FluxGuidance()
        except Exception:
            guidance_node = None
            log.warning(
                "[OTR_BatchFluxPortraitRender] FluxGuidance not "
                "available; portraits may render with default guidance"
            )

        text_enc = CLIPTextEncode()
        empty_latent_cls = EmptyLatentImage()
        sampler = KSampler()
        decoder = VAEDecode()
        report_lines: list[str] = []
        report_lines.append(
            f"OTR_BatchFluxPortraitRender start | dims={width}x{height} "
            f"steps={steps} cfg={cfg} guidance={guidance} "
            f"sampler={sampler_name}/{scheduler} seed_base={seed}"
        )

        # ---- Load ledger ----
        led, led_path = self._load_ledger(ledger_json)
        if led is None:
            raise RuntimeError(
                "OTR_BatchFluxPortraitRender: cannot load ledger from "
                f"{ledger_json!r}"
            )
        episode_id = str(led.get("episode_id") or "episode")
        cast = led.get("cast") or []
        report_lines.append(
            f"OTR_BatchFluxPortraitRender: loaded ledger "
            f"{led_path.name if led_path else '<inline>'} "
            f"episode_id={episode_id} cast={len(cast)}"
        )
        if not cast:
            report_lines.append(
                "  no cast members in ledger; nothing to render"
            )
            empty = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (empty, "\n".join(report_lines))

        # ---- Resolve output dir ----
        portraits_dir = _OTRP.otr_portraits_dir(episode_id)
        portraits_dir.mkdir(parents=True, exist_ok=True)

        # ---- Build negative cond (shared) ----
        negative = text_enc.encode(
            clip,
            "blurry, low quality, distorted face, multiple people, "
            "extra limbs, deformed, watermark, text",
        )[0]

        # ---- Render each cast portrait ----
        rendered_imgs: list[torch.Tensor] = []
        for i, c in enumerate(cast):
            char_id = _slugify_char_id(
                c.get("char_id") or c.get("name") or f"c{i+1:02d}"
            )
            speaker = (c.get("name") or c.get("speaker") or char_id).strip()
            role = (c.get("speaker_role") or c.get("role") or "").lower()
            if skip_announcer and role == "announcer":
                report_lines.append(
                    f"  cast[{i}] {speaker} ({char_id}) role=announcer; "
                    f"skip per skip_announcer=True"
                )
                continue
            appearance = (
                c.get("appearance")
                or c.get("description")
                or c.get("traits")
                or ""
            )
            prompt = _build_portrait_prompt(speaker, appearance, style_anchor)
            t0 = time.time()
            try:
                positive = text_enc.encode(clip, prompt)[0]
                if guidance_node is not None:
                    positive = guidance_node.append(positive, guidance)[0]
                latent = empty_latent_cls.generate(width, height, 1)[0]
                samples = sampler.sample(
                    model, seed + i, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, 1.0,
                )[0]
                img = decoder.decode(vae, samples)[0]  # [1, H, W, C] in 0..1
                arr = (
                    img[0].detach().cpu().numpy()
                    if hasattr(img, "detach")
                    else np.asarray(img[0])
                )
                arr_u8 = np.clip(arr * 255.0, 0, 255).astype("uint8")
                pil = Image.fromarray(arr_u8)
                out_path = portraits_dir / f"{char_id}_portrait.png"
                pil.save(out_path)
                # Stamp into ledger immediately so a crash on a later
                # portrait doesn't lose the earlier ones.
                c["portrait_path"] = str(out_path)
                rendered_imgs.append(img)
                elapsed = time.time() - t0
                report_lines.append(
                    f"  cast[{i}] {speaker} ({char_id}) "
                    f"-> {out_path.name} ({elapsed:.1f}s)"
                )
                log.info(
                    "[OTR_BatchFluxPortraitRender] %s -> %s (%.1fs)",
                    char_id, out_path.name, elapsed,
                )
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"  cast[{i}] {speaker} ({char_id}) FAILED: {exc}; "
                    f"HuMo will fall back to env-still tier"
                )
                log.warning("[OTR_BatchFluxPortraitRender] %s", msg)
                report_lines.append(msg)

        # ---- Persist ledger ----
        if led_path is not None:
            try:
                _OTRL.save_ledger_safe(led_path, led)
                report_lines.append(
                    f"OTR_BatchFluxPortraitRender: ledger updated "
                    f"with {sum(1 for c in cast if c.get('portrait_path'))} "
                    f"portrait_path entries -> {led_path.name}"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_BatchFluxPortraitRender] ledger save failed: %s",
                    exc,
                )

        # ---- Build IMAGE output (stack rendered) ----
        if rendered_imgs:
            try:
                out_batch = torch.cat(
                    [im if im.dim() == 4 else im.unsqueeze(0)
                     for im in rendered_imgs],
                    dim=0,
                )
            except Exception:
                out_batch = rendered_imgs[0]
        else:
            out_batch = torch.zeros((1, height, width, 3), dtype=torch.float32)

        report_lines.append(
            f"OTR_BatchFluxPortraitRender done | rendered={len(rendered_imgs)}/"
            f"{len(cast)} | out_dir={portraits_dir}"
        )
        return (out_batch, "\n".join(report_lines))

    # ----- helpers -----

    def _load_ledger(self, ledger_json: str) -> tuple[dict | None, Path | None]:
        """Load ledger from inline JSON OR filesystem path OR auto-pick."""
        s = (ledger_json or "").strip()
        if not s:
            try:
                p = _OTRL.in_flight_ledger_path()
                if p is not None and p.exists():
                    return json.loads(p.read_text(encoding="utf-8")), p
            except Exception:
                pass
            return None, None
        if s.startswith("{"):
            try:
                return json.loads(s), None
            except Exception:
                return None, None
        p = Path(s)
        if p.exists() and p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8")), p
            except Exception:
                return None, None
        return None, None


NODE_CLASS_MAPPINGS = {"OTR_BatchFluxPortraitRender": BatchFluxPortraitRender}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_BatchFluxPortraitRender": "[POR] Batch FLUX Portrait Render (per-cast)",
}
