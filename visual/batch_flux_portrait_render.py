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
section. Historically populated by the legacy LLMDirector; post-
voice-path-cleanbreak populated by OTR_LedgerScriptWriter directly.
Each portrait prompt is built as:

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

# folder_paths is the ComfyUI canonical path resolver. The portrait
# write site uses _OTRP.otr_portraits_dir(episode_id) which composes
# folder_paths.get_output_directory() under the hood, so this top-level
# import documents the Bug Bible BUG-01.02 contract that every
# OUTPUT_NODE module references the canonical resolver.
import folder_paths  # noqa: F401

log = logging.getLogger("OTR")


def _lazy_otr_imports():
    """Mirror the lazy-import pattern used by the sibling
    ``batch_flux_render.py``. ComfyUI loads custom node modules from
    inside ``custom_nodes/<pkg>/`` with a different sys.path setup
    than a standalone Python invocation, so top-level ``from nodes
    import ...`` fails at module-load time. Doing it lazily inside
    a helper -- and inserting ``nodes/`` directly onto sys.path so
    we ``import _otr_paths`` (no ``nodes.`` prefix) -- matches what
    the working sibling does (see batch_flux_render.py:786-792).
    """
    _NODES_DIR = Path(__file__).resolve().parents[1] / "nodes"
    if str(_NODES_DIR) not in sys.path:
        sys.path.insert(0, str(_NODES_DIR))
    import _otr_paths as _OTRP  # type: ignore
    import _otr_ledger as _OTRL  # type: ignore
    return _OTRP, _OTRL

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


def _build_portrait_prompt(
    speaker: str,
    appearance: str,
    style_anchor: str,
    lighting: str = "",
) -> str:
    """Compose the FLUX prompt for one cast member's portrait.

    Period style anchor + character description + (optional) brief-
    derived lighting + atmosphere + composition guidance. Designed to
    give HuMo a stable, frontal, well-lit reference frame that's NOT
    a scene shot.

    `lighting` is the output of `get_story_brief_lighting(meta)` --
    a comma-joined string of lighting + atmosphere terms (refinement
    section 6.2). Setting terms are deliberately excluded since
    portraits do not want env / prop noise pulling composition toward
    the scene. Empty string when the brief is absent or failed; the
    portrait falls through to the legacy composition guidance.
    """
    speaker = (speaker or "Unnamed character").strip()
    appearance = (appearance or "").strip()
    style_anchor = (style_anchor or "head-and-shoulders studio portrait, neutral lighting").strip()
    lighting = (lighting or "").strip()
    parts = [
        style_anchor,
        f"head and shoulders portrait of {speaker}",
    ]
    if appearance:
        parts.append(appearance)
    # Sprint C C5d (2026-05-15): brief-derived lighting + atmosphere
    # inserted after the character appearance, before the fixed
    # composition guidance. Per refinement section 6.2 the lighting
    # helper returns lighting + atmosphere terms only (no setting).
    if lighting:
        parts.append(lighting)
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
    # Sprint D D0d (2026-05-16): added third output `portraits_dir` so
    # downstream HuMo can wire its face-reference input to the actual
    # write directory instead of falling through to `comfy_output_dir()`.
    # Sprint C shipped HuMo's `portraits_dir` input unlinked; this
    # output is the wiring partner so the JSON link can land cleanly.
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("portrait_batch", "report", "portraits_dir")

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
                    "default": "head-and-shoulders studio portrait, neutral lighting, cinematic",
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
        style_anchor: str = "head-and-shoulders studio portrait, neutral lighting, cinematic",
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
        # Sprint D D0d: resolve portraits_dir up here so both the
        # no-cast early-return path AND the normal return path can
        # surface a real directory string in the third output socket.
        # Was previously computed at the per-character render site only.
        _OTRP_early, _ = _lazy_otr_imports()
        portraits_dir = _OTRP_early.otr_portraits_dir(episode_id)
        portraits_dir.mkdir(parents=True, exist_ok=True)
        # Sprint C C5d (2026-05-15): resolve brief lighting + status once
        # per ledger; passed into every per-character portrait prompt.
        # Lighting helper returns lighting + atmosphere terms only
        # (refinement section 6.2). Status surfaces in the report for
        # E-07 observability.
        from ..nodes._otr_story_brief_helpers import (
            get_story_brief_lighting,
            get_story_brief_status,
        )
        _meta = led.get("meta") if isinstance(led.get("meta"), dict) else {}
        _brief_lighting = get_story_brief_lighting(_meta)
        _brief_status = get_story_brief_status(_meta)
        report_lines.append(
            f"OTR_BatchFluxPortraitRender: loaded ledger "
            f"{led_path.name if led_path else '<inline>'} "
            f"episode_id={episode_id} cast={len(cast)} "
            f"story_brief_status={_brief_status}"
        )
        if not cast:
            report_lines.append(
                "  no cast members in ledger; nothing to render"
            )
            empty = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (empty, "\n".join(report_lines), str(portraits_dir))

        # BUG-LOCAL-094 (2026-05-04 EVENING): build a per-char_id flag
        # of "has at least one character-role line" by walking
        # ledger.lines[]. Cast entries themselves don't carry
        # speaker_role -- it lives per-line -- so the pre-094 check
        # (``c.get("speaker_role") or c.get("role")``) always returned
        # an empty string and the skip_announcer guard never fired.
        # Result: ANNOUNCER cast members got a wasted ~30s FLUX
        # portrait that HuMo never used (announcer beats route to LTX
        # via BUG-129b). Same for any future cast member whose lines
        # are entirely music/sfx.
        #
        # Detection: walk lines[], group by char_id, check if any line
        # has speaker_role == "character". If a cast member has at
        # least one character line, render the portrait. If not, skip.
        # Falls back to "render anyway" when lines block is missing
        # (degraded ledger; safer to render than to skip silently).
        try:
            from nodes._otr_speaker_role import (  # type: ignore
                resolve_speaker_role,
                SPEAKER_ROLE_CHARACTER,
            )
        except ImportError:
            try:
                from _otr_speaker_role import (  # type: ignore
                    resolve_speaker_role,
                    SPEAKER_ROLE_CHARACTER,
                )
            except ImportError:
                resolve_speaker_role = None
                SPEAKER_ROLE_CHARACTER = "character"  # fallback
        char_id_has_character_line: dict[str, bool] = {}
        ledger_lines = led.get("lines") or []
        if resolve_speaker_role is not None:
            for ln in ledger_lines:
                cid = (ln.get("char_id") or "").strip()
                if not cid:
                    continue
                role = resolve_speaker_role(ln)
                if role == SPEAKER_ROLE_CHARACTER:
                    char_id_has_character_line[cid] = True
                else:
                    # Default to False only if not already seen as
                    # character; preserve True if any prior line had it.
                    char_id_has_character_line.setdefault(cid, False)
        if char_id_has_character_line:
            report_lines.append(
                f"  BUG-094 cast filter: {sum(char_id_has_character_line.values())}"
                f"/{len(char_id_has_character_line)} cast member(s) have "
                f">=1 character-role line"
            )

        # ---- Output dir already resolved early in execute() per D0d ----
        _OTRP, _OTRL = _lazy_otr_imports()
        # portraits_dir computed near line 308 so the no-cast early
        # return at line 331 can surface it in the third output socket.

        # ---- Build negative cond (shared) ----
        negative = text_enc.encode(
            clip,
            "blurry, low quality, distorted face, multiple people, "
            "extra limbs, deformed, watermark, text",
        )[0]

        # ---- BUG-LOCAL-231 fix (2026-05-18) ----
        # Two-pass refactor: pre-encode ALL non-skipped cast positives
        # BEFORE pinning MODEL. Pre-fix the encode happened inside the
        # render loop at each iteration, AFTER the previous iteration's
        # sampler.sample() had loaded MODEL onto GPU. ComfyUI's
        # model_management then had to evict MODEL each time CLIP
        # needed to encode the next prompt, causing per-portrait
        # MODEL-unload + MODEL-reload cycles. Pre-encoding all
        # positives upfront lets CLIP get evicted ONCE before MODEL
        # is pinned for the whole sampling pass.

        # ---- PRE-PASS: skip checks + per-prompt encode ----
        prepared = []  # list of (i, c, char_id, speaker, positive)
        for i, c in enumerate(cast):
            char_id = _slugify_char_id(
                c.get("char_id") or c.get("name") or f"c{i+1:02d}"
            )
            speaker = (c.get("name") or c.get("speaker") or char_id).strip()
            # BUG-LOCAL-094 (2026-05-04 EVENING): two-tier skip.
            # Tier 1 (line-driven): line-block visibility AND
            # skip_announcer AND char_id has zero character-role lines.
            # Tier 2 (legacy name-match): no lines block AND
            # skip_announcer AND name == "ANNOUNCER".
            cast_char_id_for_filter = (c.get("char_id") or "").strip()
            line_visible = bool(char_id_has_character_line)
            if (
                skip_announcer
                and line_visible
                and cast_char_id_for_filter
                and not char_id_has_character_line.get(
                    cast_char_id_for_filter, True
                )
            ):
                report_lines.append(
                    f"  cast[{i}] {speaker} ({char_id}) "
                    f"all lines non-character; skip per skip_announcer=True "
                    f"(BUG-LOCAL-094)"
                )
                continue
            if (
                skip_announcer
                and not line_visible
                and speaker.upper().strip() == "ANNOUNCER"
            ):
                report_lines.append(
                    f"  cast[{i}] {speaker} ({char_id}) "
                    f"name=ANNOUNCER (legacy fallback, no lines block); "
                    f"skip per skip_announcer=True"
                )
                continue
            appearance = (
                c.get("appearance")
                or c.get("character_description")
                or c.get("description")
                or c.get("traits")
                or ""
            )
            prompt = _build_portrait_prompt(
                speaker, appearance, style_anchor,
                lighting=_brief_lighting,
            )
            try:
                positive = text_enc.encode(clip, prompt)[0]
                if guidance_node is not None:
                    positive = guidance_node.append(positive, guidance)[0]
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"  cast[{i}] {speaker} ({char_id}) PRE-ENCODE FAILED: "
                    f"{exc}; HuMo will fall back to env-still tier"
                )
                log.warning("[OTR_BatchFluxPortraitRender] %s", msg)
                report_lines.append(msg)
                continue
            prepared.append((i, c, char_id, speaker, positive))
        log.info(
            "[OTR_BatchFluxPortraitRender] pre-encoded %d portrait "
            "positive(s) from %d cast member(s)",
            len(prepared), len(cast),
        )

        # ---- BUG-LOCAL-231 fix: nuclear eviction + pin ----
        # Original fix used mm.free_memory(11.5 GB, [model.load_device]).
        # 2026-05-18 post-fix smoke (HEAD 36bcfc0) showed mm.free_memory
        # was NOT sufficient. Jeffrey's pre-authorized escalation: swap
        # to Option B mm.unload_all_models() per the acceptance gate
        # rule "if LHM shows D3D Shared > 200 MB during sampler the
        # eviction didn't work and the fix needs to escalate to option
        # B (unload_all_models)."
        try:
            import comfy.model_management as mm  # type: ignore
            import gc as _gc
            import torch as _torch  # type: ignore
            try:
                mm.unload_all_models()
                # BUG-07.03 invariant (Bug Bible regression): every
                # unload_all_models() call must be paired with
                # gc.collect() + torch.cuda.empty_cache() to actually
                # release VRAM (the model registry can be cleared
                # without freeing the caching allocator's reserved
                # blocks).
                _gc.collect()
                _torch.cuda.empty_cache()
                log.info(
                    "[OTR_BatchFluxPortraitRender] unload_all_models() "
                    "+ gc.collect() + empty_cache() complete (nuclear "
                    "eviction before MODEL pin, BUG-LOCAL-231 Option "
                    "B escalation)"
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_BatchFluxPortraitRender] mm.unload_all_models() "
                    "raised %r; proceeding to load_models_gpu without "
                    "explicit eviction (sampler may thrash)", exc,
                )
            mm.load_models_gpu([model])
            log.info(
                "[OTR_BatchFluxPortraitRender] pinned MODEL via "
                "load_models_gpu"
            )
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "[OTR_BatchFluxPortraitRender] pin skipped: %s", exc
            )

        # ---- RENDER PASS: sampler + decoder + save per prepared portrait ----
        rendered_imgs: list[torch.Tensor] = []
        for i, c, char_id, speaker, positive in prepared:
            t0 = time.time()
            try:
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
        return (out_batch, "\n".join(report_lines), str(portraits_dir))

    # ----- helpers -----

    def _load_ledger(self, ledger_json: str) -> tuple[dict | None, Path | None]:
        """Load ledger from inline JSON OR filesystem path OR auto-pick."""
        _OTRP, _OTRL = _lazy_otr_imports()
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
