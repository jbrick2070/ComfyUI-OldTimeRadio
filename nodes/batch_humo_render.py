"""
batch_humo_render.py  --  OTR_BatchHumoRender ComfyUI node
==========================================================

Render N HuMo lip-sync clips in lockstep from a Production Ledger,
in a single graph run, sharing one HuMo + Lora + CLIP + VAE +
Whisper load.

This is the in-graph counterpart to ``scripts/render_humo_batch.py``
(the subprocess orchestrator). The CLI script remains for ad-hoc
smoke tests; production runs use this node so progress is visible
in ComfyUI's UI and the FULL workflow is a single self-contained
JSON with no hidden subprocess.

Pattern mirrors OTR_BatchFluxRender:
  - take pre-loaded MODEL/CLIP/VAE/AUDIO_ENCODER as INPUT_TYPES so
    Comfy's model-management owns the lifecycle
  - load core ComfyUI nodes lazily at execute() time
  - loop N ledger lines, calling node objects directly (NOT via
    /prompt HTTP API)
  - each line: slice audio tensor, load portrait from disk, encode
    text + audio + ref_image, run WanHuMoImageToVideo + KSampler +
    VAEDecode + CreateVideo + SaveVideo, write ``<line_id>.mp4``
  - return clips_dir + count + report

Per CLAUDE.md C5: HuMo runs at fp8_e4m3fn weights staged via
ComfyUI's dynamic VRAM loading. UnloadAll wires after this node to
free HuMo before VideoComposite reads the proc gen + clips.

Ported logic from render_humo_batch.py:
  - HUMO_FPS / HUMO_MIN_FRAMES / HUMO_MAX_FRAMES constants
  - humo_length_for_dur(): 4n+1 frame snap with floor + ceiling
  - find_portrait_for_speaker(): cast-position fallback
  - find_composite_for_shot_speaker(): per-(shot,speaker) pass3
    composite picker
  - cid_to_name speaker resolution (BUG-LOCAL-074)
  - Save filename = ``<line_id>.mp4`` so OTR_VideoComposite (or
    render_episode_concat.py) finds clips by ledger line_id
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("OTR.batch_humo_render")


# ---------------------------------------------------------------------------
# HuMo constants (ported from render_humo_batch.py)
# ---------------------------------------------------------------------------

HUMO_FPS = 25
HUMO_MIN_FRAMES = 33   # smaller frame counts have hung this hardware
HUMO_MAX_FRAMES = 177  # last empirically verified value on RTX 5080 Laptop 16GB

# ByteDance Chinese negative prompt -- empirically the best HuMo neg
# on the official template.
_CHINESE_NEGATIVE = (
    "色调艳丽，过曝，静态，"
    "细节模糊不清，字幕，风格"
    "，作品，画作，画面，静止"
    "，整体发灰，最差质量，低"
    "质量，JPEG压缩残留，丑陋的"
    "，残缺的，多余的手指，画"
    "得不好的手部，画得不好的"
    "脸部，畸形的，毁容的，形"
    "态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背"
    "景，三条腿，背景人很多，"
    "倒着走"
)

_DEFAULT_POS_SUFFIX = (
    "dimly lit interior, ambient cinematic lighting, "
    "35mm film grain, shallow depth of field"
)

# Slug rule must stay in lockstep with render_flux_batch.slugify so
# pass3 composite filenames written by FLUX get found by HuMo.
_SLUG_RE_CONSUME = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def humo_length_for_dur(dur_s: float, *, fps: int = HUMO_FPS) -> int:
    """Pick the smallest valid HuMo `length` >= round(dur_s * fps).

    Wan 2.1 VAE temporal compression requires length = 4n + 1.
    Floored at HUMO_MIN_FRAMES. Capped at HUMO_MAX_FRAMES.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 3) // 4
    frames = 4 * n + 1
    if frames < HUMO_MIN_FRAMES:
        frames = HUMO_MIN_FRAMES
    if frames > HUMO_MAX_FRAMES:
        frames = HUMO_MAX_FRAMES
    return frames


def _composite_slug(s: str, limit: int = 40) -> str:
    s = (s or "").strip().lower()
    s = _SLUG_RE_CONSUME.sub("_", s).strip("_")
    return (s[:limit] or "unnamed")


def _find_portrait(
    speaker: str,
    cast: list[dict],
    portraits_dir: Path,
) -> Path | None:
    """Find a portrait image for a speaker.

    Strategy (extends render_humo_batch.find_portrait_for_speaker
    with a final FLUX env-still fallback so the workflow runs even
    when no per-cast PASS1 portraits have been pre-rendered):

      1. cast[].portrait_path if populated and exists
      2. otr_humo_pass1_portrait_*.png by index in cast list
      3. any otr_humo_pass1_portrait_*.png
      4. **NEW**: full_env_*.png by index in cast list (uses FLUX
         environment stills as face stand-ins — visually wrong per
         line but the workflow runs end-to-end without a separate
         portrait-render pass; BUG-LOCAL-078 stopgap)
      5. any full_env_*.png
      6. None
    """
    speaker_norm = (speaker or "").upper().strip()

    # 1. Direct path from ledger cast entry
    for c in cast:
        if (c.get("name") or "").upper().strip() == speaker_norm:
            p = c.get("portrait_path")
            if p and Path(p).exists():
                return Path(p)

    cast_names = [(c.get("name") or "").upper().strip() for c in cast]

    # 2. PASS1 humo portraits indexed by cast position (canonical naming)
    if speaker_norm in cast_names:
        idx = cast_names.index(speaker_norm)
        candidates = sorted(
            list(portraits_dir.glob("otr/stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_humo_pass1_portrait_*.png"))
        )
        if candidates:
            return candidates[idx % len(candidates)]

    # 3. Any HuMo portrait (no cast match)
    candidates = sorted(portraits_dir.glob("otr_humo_pass1_portrait_*.png"))
    if candidates:
        return candidates[0]

    # 4. FLUX env-stills indexed by cast position (BUG-078 stopgap).
    #    These are typically full_env_NNNNN_.png from BatchFluxRender's
    #    environment-token output. Visually wrong per-line (each cast
    #    member maps to a random env still) but produces a runnable
    #    end-to-end pipeline. Replace with proper PASS1 portraits when
    #    a portrait render path is wired into the workflow.
    if speaker_norm in cast_names:
        idx = cast_names.index(speaker_norm)
        candidates = sorted(
            list(portraits_dir.glob("otr/stills/full_env_*.png"))
            + list(portraits_dir.glob("otr_stills/full_env_*.png"))
        )
        if candidates:
            return candidates[idx % len(candidates)]

    # 5. Any FLUX env still (last resort)
    candidates = sorted(
        list(portraits_dir.glob("otr/stills/full_env_*.png"))
        + list(portraits_dir.glob("otr_stills/full_env_*.png"))
    )
    if candidates:
        return candidates[0]

    return None


def _find_composite(
    shot_id: str | None,
    speaker: str | None,
    portraits_dir: Path,
) -> Path | None:
    """Per-(shot, speaker) FLUX pass3 composite picker."""
    if not shot_id or not speaker:
        return None
    shot_slug = _composite_slug(shot_id, limit=24)
    speaker_slug = _composite_slug(speaker, limit=40)
    new_pat = f"otr/stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    mid_pat = f"otr_stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    legacy_pat = f"otr_humo_pass3_{shot_slug}_{speaker_slug}_*.png"
    candidates = sorted(
        list(portraits_dir.glob(new_pat))
        + list(portraits_dir.glob(mid_pat))
        + list(portraits_dir.glob(legacy_pat)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _png_to_tensor(png_path: Path):
    """Load a PNG file as a ComfyUI IMAGE tensor [1, H, W, C] in
    [0, 1] float32. Bypasses ComfyUI's LoadImage which requires the
    file to be in input/ -- we read directly from the otr/ tree."""
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W, C]
    return tensor


def _slice_audio_tensor(audio_dict: dict, start_s: float, dur_s: float) -> dict:
    """Slice an AUDIO dict by start_s and dur_s. The AUDIO type in
    ComfyUI is ``{"waveform": Tensor[B, C, samples], "sample_rate":
    int}``. Returns a new dict with the sliced waveform; sample_rate
    unchanged."""
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    start_sample = max(0, int(start_s * sr))
    end_sample = max(start_sample + 1, int((start_s + dur_s) * sr))
    if waveform.dim() == 2:
        # Some callers pass [C, samples] instead of [B, C, samples]
        sliced = waveform[..., start_sample:end_sample]
    else:
        sliced = waveform[..., start_sample:end_sample]
    return {"waveform": sliced.contiguous(), "sample_rate": sr}


def _build_pos_prompt(speaker: str, ln: dict, cast: list[dict]) -> str:
    """Build a HuMo positive prompt from the ledger line + cast desc."""
    speaker_desc = ""
    for c in cast:
        if (c.get("name") or "").upper().strip() == (speaker or "").upper().strip():
            speaker_desc = (c.get("description") or "").strip()
            break
    if not speaker_desc:
        speaker_desc = (
            f"A {speaker.lower()} character speaks calmly with subtle facial "
            f"expressions"
        ) if speaker else "A character speaks calmly with subtle facial expressions"
    return f"{speaker_desc}, {_DEFAULT_POS_SUFFIX}"


def _lazy_humo_nodes() -> dict[str, Any]:
    """Resolve the ComfyUI / extension node classes BatchHumoRender
    needs. Lazy import so the module loads even if Comfy isn't fully
    initialized (e.g., during static analysis or pytest)."""
    refs: dict[str, Any] = {}
    try:
        import nodes  # type: ignore
        refs["CLIPTextEncode"] = getattr(nodes, "CLIPTextEncode", None)
        refs["KSampler"] = getattr(nodes, "KSampler", None)
        refs["VAEDecode"] = getattr(nodes, "VAEDecode", None)
        node_mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    except Exception:
        node_mappings = {}

    # Wan 2.1 / HuMo extension nodes (community / Kijai pack)
    refs["WanHuMoImageToVideo"] = node_mappings.get("WanHuMoImageToVideo")
    refs["AudioEncoderEncode"] = node_mappings.get("AudioEncoderEncode")
    refs["CreateVideo"] = node_mappings.get("CreateVideo")
    refs["SaveVideo"] = node_mappings.get("SaveVideo")
    return refs


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class BatchHumoRender:
    """Render N HuMo lip-sync clips in one graph execution.

    OUTPUT_NODE = True (BUG-LOCAL-077 lesson): this node has
    side-effects (writes per-line .mp4 files to output/otr/videos/
    <ep_id>/) AND its output STRINGs are diagnostic. Without
    OUTPUT_NODE the executor would prune it when downstream consumers
    don't fully chain.
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("clips_dir", "clip_count", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers  # type: ignore
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["uni_pc"]
            schedulers = ["simple"]
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "HuMo model with lightx2v Lora applied (post ModelSamplingSD3).",
                }),
                "clip": ("CLIP", {
                    "tooltip": "umt5_xxl text encoder loaded via CLIPLoader.",
                }),
                "vae": ("VAE", {
                    "tooltip": "wan_2.1 VAE loaded via VAELoader.",
                }),
                "audio_encoder": ("AUDIO_ENCODER", {
                    "tooltip": "Whisper Large v3 fp16 loaded via AudioEncoderLoader.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Full episode audio from EpisodeAssembler.",
                }),
                "ledger_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Production Ledger JSON. Either the JSON string "
                        "itself or a path to *_ledger.json on disk."
                    ),
                }),
                "portraits_dir": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Directory holding PASS1 character portraits "
                        "(otr_humo_pass1_portrait_*.png) and optional "
                        "PASS3 (shot,speaker) composites. Empty -> "
                        "auto-resolve to output/otr/portraits/<ep_id>/."
                    ),
                }),
                "clip_length": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.32,
                    "max": 7.08,
                    "step": 0.04,
                    "tooltip": (
                        "Per-clip duration in seconds. Default 7.0 -> "
                        "175 frames -> 177 (Wan 2.1 4n+1 = 7.08s)."
                    ),
                }),
                "max_clips": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "0 = render every line; smoke-test caps with positive int.",
                }),
                "seed": ("INT", {"default": 7, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "uni_pc"}),
                "scheduler": (schedulers, {"default": "simple"}),
                "width": ("INT", {"default": 480, "min": 256, "max": 1280, "step": 8}),
                "height": ("INT", {"default": 832, "min": 256, "max": 1280, "step": 8}),
            },
        }

    def execute(
        self,
        model, clip, vae, audio_encoder, audio,
        ledger_json: str,
        portraits_dir: str,
        clip_length: float,
        max_clips: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        width: int,
        height: int,
    ):
        t_start = time.time()

        # ---- 1. Parse ledger ----
        ledger = self._load_ledger(ledger_json)
        cast = ledger.get("cast") or []
        lines = ledger.get("lines") or []
        episode_id = ledger.get("episode_id", "episode")
        cid_to_name = {
            (c.get("char_id") or ""): (c.get("name") or "")
            for c in cast if c.get("char_id")
        }
        log.info("[BatchHumoRender] episode_id=%s cast=%d lines=%d",
                 episode_id, len(cast), len(lines))

        # ---- 2. Resolve portraits dir ----
        # Default = ComfyUI output root so _find_portrait's globs hit
        # both `otr/portraits/<ep_id>/otr_humo_pass1_portrait_*.png`
        # AND `otr/stills/full_env_*.png` AND `otr/stills/pass1_*.png`.
        # User-supplied input wins if non-empty.
        comfy_output = Path(r"C:\Users\jeffr\Documents\ComfyUI\output")
        if not portraits_dir or not portraits_dir.strip():
            portraits_dir_path = comfy_output
        else:
            portraits_dir_path = Path(portraits_dir)
        log.info("[BatchHumoRender] portraits_dir=%s", portraits_dir_path)

        # ---- 3. Resolve clips output_dir (canonical OTR tree) ----
        # output_dir = ComfyUI/output/otr/videos/<episode_id>/
        clips_dir_path = comfy_output / "otr" / "videos" / episode_id
        clips_dir_path.mkdir(parents=True, exist_ok=True)

        # ---- 4. Lazy-load Comfy nodes ----
        refs = _lazy_humo_nodes()
        missing = [k for k in (
            "CLIPTextEncode", "KSampler", "VAEDecode",
            "WanHuMoImageToVideo", "AudioEncoderEncode",
            "CreateVideo", "SaveVideo",
        ) if refs.get(k) is None]
        if missing:
            raise RuntimeError(
                f"BatchHumoRender: missing required ComfyUI node classes: {missing}"
            )

        text_enc = refs["CLIPTextEncode"]()
        sampler = refs["KSampler"]()
        vae_decoder = refs["VAEDecode"]()
        humo_node = refs["WanHuMoImageToVideo"]()
        audio_enc_node = refs["AudioEncoderEncode"]()
        create_video = refs["CreateVideo"]()
        save_video = refs["SaveVideo"]()

        # ---- 5. Pin model on GPU ----
        try:
            import comfy.model_management as mm  # type: ignore
            try:
                mm.load_models_gpu([model], force_full_load=True)
            except TypeError:
                mm.load_models_gpu([model])
            log.info("[BatchHumoRender] pinned MODEL via load_models_gpu")
        except Exception as exc:
            log.debug("[BatchHumoRender] pre-pin skipped: %s", exc)

        # ---- 6. Encode negative once (shared across clips) ----
        try:
            negative = text_enc.encode(clip, _CHINESE_NEGATIVE)[0]
        except Exception as exc:
            raise RuntimeError(f"BatchHumoRender: negative encode failed: {exc}")

        # ---- 7. Loop lines ----
        report_lines: list[str] = [
            f"BatchHumoRender: episode={episode_id} target_lines={len(lines)}",
        ]
        if max_clips and max_clips > 0:
            lines = lines[:max_clips]
            report_lines.append(f"  capped to {max_clips} clips")

        rendered = 0
        for idx, ln in enumerate(lines):
            line_id = str(ln.get("line_id") or f"l{idx + 1:03d}")
            speaker = (
                ln.get("speaker")
                or cid_to_name.get(ln.get("char_id") or "", "")
                or ""
            ).strip()

            # Per-line timing: prefer ledger.start_s/dur_s, fall back
            # to auto-slice (idx * clip_length).
            start_s = ln.get("start_s")
            dur_s = ln.get("dur_s")
            if start_s is None or start_s == "":
                start_s = idx * clip_length
            if dur_s is None or dur_s == "" or dur_s <= 0:
                dur_s = clip_length
            start_s = float(start_s)
            dur_s = float(dur_s)

            # Cap to clip_length
            dur_s = min(dur_s, float(clip_length))
            humo_length = humo_length_for_dur(dur_s)

            # Resolve portrait
            shot_id = ln.get("shot_id")
            ref_png = _find_composite(shot_id, speaker, portraits_dir_path)
            if not ref_png:
                ref_png = _find_portrait(speaker, cast, portraits_dir_path)
            if not ref_png:
                report_lines.append(f"  l{idx+1:03d}: SKIP no portrait")
                log.warning("[BatchHumoRender] line %s speaker=%r: no portrait", line_id, speaker)
                continue

            # Build prompts
            pos_text = _build_pos_prompt(speaker, ln, cast)

            # Slice audio
            line_audio = _slice_audio_tensor(audio, start_s, dur_s)

            # Load portrait as IMAGE tensor
            try:
                ref_image = _png_to_tensor(ref_png)
            except Exception as exc:
                report_lines.append(f"  {line_id}: SKIP portrait load failed: {exc}")
                continue

            # Run pipeline for this clip
            shot_t0 = time.time()
            shot_seed = (seed + idx * 1009) & 0x7FFFFFFFFFFFFFFF
            try:
                positive = text_enc.encode(clip, pos_text)[0]
                audio_emb = audio_enc_node.encode(audio_encoder, line_audio)[0]
                humo_pos, humo_neg, humo_latent = humo_node.encode(
                    width, height, humo_length, 1,
                    positive, negative, vae,
                    audio_emb,
                    ref_image,
                )[:3]
                samples = sampler.sample(
                    model, shot_seed, steps, cfg, sampler_name, scheduler,
                    humo_pos, humo_neg, humo_latent, 1.0,
                )[0]
                images_out = vae_decoder.decode(vae, samples)[0]
                video_obj = create_video.create_video(images_out, line_audio, 25.0)[0]
                save_video.save_video(
                    video_obj,
                    filename_prefix=f"otr/videos/{episode_id}/humo_{line_id}",
                    format="auto",
                    codec="auto",
                )

                # Rename SaveVideo's "humo_<line_id>_NNNNN_.mp4" to
                # canonical "<line_id>.mp4" so downstream concat
                # lookup is straightforward (BUG-074 convention).
                self._rename_savevideo_output(
                    clips_dir_path, line_id, report_lines,
                )

                shot_ms = int((time.time() - shot_t0) * 1000)
                report_lines.append(
                    f"  {line_id} ({speaker}): {shot_ms} ms "
                    f"(length={humo_length} ref={ref_png.name})"
                )
                log.info("[BatchHumoRender] %s done in %d ms", line_id, shot_ms)
                rendered += 1
            except Exception as exc:
                log.exception("[BatchHumoRender] line %s failed: %s", line_id, exc)
                report_lines.append(f"  {line_id}: FAILED ({exc})")

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"Total: {rendered}/{len(lines)} clip(s) in {total_ms} ms"
        )
        log.info("[BatchHumoRender] complete: %d/%d clips in %d ms",
                 rendered, len(lines), total_ms)

        return (str(clips_dir_path), rendered, "\n".join(report_lines))

    @staticmethod
    def _load_ledger(ledger_arg: str) -> dict:
        """Accept either:
          - inline JSON string (starts with '{')
          - path to *_ledger.json
          - path to *.mp4 (audio episode); ledger inferred via
            suffix swap (.mp4 -> _ledger.json), since that's the
            convention OTR_SignalLostVideo / EpisodeAssembler write.
            Lets us wire BatchHumoRender's ledger_json input directly
            from SignalLostVideo.video_path -- no separate ledger
            output node required.
          - empty -> auto-pick newest non-pending in the canonical
            audio dirs (BUG-LOCAL-076 fallback chain).
        """
        s = (ledger_arg or "").strip()

        # Auto-pick fallback when input empty
        if not s:
            audio_dirs = [
                Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\audio"),
                Path(r"C:\Users\jeffr\Documents\ComfyUI\output\old_time_radio"),
            ]
            cands = []
            for d in audio_dirs:
                if d.exists():
                    cands.extend(
                        p for p in d.glob("*_ledger.json")
                        if not p.name.startswith("pending_")
                    )
            if not cands:
                raise RuntimeError("BatchHumoRender: ledger_json empty and auto-pick found no ledger")
            p = max(cands, key=lambda x: x.stat().st_mtime)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        if s.startswith("{"):
            return json.loads(s)

        p = Path(s)
        # .mp4 path -> swap suffix to _ledger.json (SignalLostVideo
        # convention)
        if p.suffix.lower() == ".mp4":
            ledger_p = p.with_suffix("").parent / f"{p.stem}_ledger.json"
            if ledger_p.exists():
                with open(ledger_p, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise RuntimeError(
                f"BatchHumoRender: derived ledger from .mp4 not found: {ledger_p}"
            )

        # Plain ledger.json path
        if not p.exists():
            raise RuntimeError(f"BatchHumoRender: ledger path not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _rename_savevideo_output(
        clips_dir: Path,
        line_id: str,
        report_lines: list[str],
    ) -> None:
        """SaveVideo writes ``humo_<line_id>_NNNNN_.mp4``. Rename
        the newest match to ``<line_id>.mp4`` so concat finds it
        without globbing for raw SaveVideo prefixes."""
        import shutil as _shutil
        candidates = sorted(
            clips_dir.glob(f"humo_{line_id}_*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            report_lines.append(f"  {line_id}: WARN SaveVideo output not found in {clips_dir}")
            return
        src = candidates[0]
        dst = clips_dir / f"{line_id}.mp4"
        try:
            _shutil.move(str(src), str(dst))
        except Exception as exc:
            report_lines.append(f"  {line_id}: WARN rename failed: {exc}")


__all__ = ["BatchHumoRender"]
