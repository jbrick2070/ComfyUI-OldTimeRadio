"""Stable Audio 3 music adapter -- ComfyUI-NATIVE (no stable_audio_tools).

Drives ComfyUI's own audio node classes (CheckpointLoaderSimple + optional
t5gemma CLIPLoader + CLIPTextEncode + ConditioningStableAudio + EmptyLatentAudio
+ KSampler + VAEDecodeAudio) so SA3 uses ComfyUI's `comfy.model_management` --
no PyPI dependency, no torch/numpy conflict on the Blackwell stack. Weights:
`Comfy-Org/stable-audio-3` (ungated). `interface == "clip"`.

Fail-closed: absent ComfyUI runtime or absent checkpoint raise a clear named
error (the 6-class taxonomy's MISSING_MODEL), never a silent fetch or crash.
SA3 community license = commercial-OK -> commercial_clean = True. Opt-in behind
OTR_ENABLE_STABLE_AUDIO_3 until F validates render-twice determinism on sm_120,
then promotion flips default_roles + clears the flag.
"""
from __future__ import annotations

import hashlib
import logging
import os

from .registry import EngineUnusable, EngineUsabilityReason, register

log = logging.getLogger("OTR")

# Default model files in ComfyUI's models/ tree (Comfy-Org/stable-audio-3).
_CKPT = os.environ.get("OTR_SA3_CKPT", "stable_audio_3_small_music.safetensors")
_TENC = os.environ.get("OTR_SA3_TEXT_ENCODER", "t5gemma_b_b_ul2.safetensors")
_CLIP_TYPE = os.environ.get("OTR_SA3_CLIP_TYPE", "stable_audio")

# BUG-408: SA3 is a different model than the old MusicGen default and does NOT
# respond to MusicGen-shaped abstract-mood prompts the same way -- it wants a
# genre + instrumentation + production anchor and a real negative prompt, and it
# needs a multi-second STRUCTURAL context (seconds_total) to sound like music
# rather than an unstructured 4-12s texture. The fixes below are SA3-only and
# keep the shared compose_music_prompt() contract unchanged.

# Era -> genre/instrumentation anchor (deterministic; keyed on period/setting
# keywords the brief already put in the prompt). NO fabricated musical key.
_SA3_PERIOD_GENRE = (
    (("1920", "1930", "roaring"),
     "vintage big-band orchestral, warm brass, upright bass, analog tape warmth"),
    (("1940", "1950", "atomic", "pulp", "radio", "noir"),
     "vintage orchestral sci-fi score, theremin, eerie strings, brass, timpani, "
     "analog tape warmth"),
    (("1960", "1970", "retro", "space age"),
     "retro-futurist analog synth score, mellotron, electric piano, tape echo, "
     "warm strings"),
)
_SA3_DEFAULT_GENRE = (
    "cinematic instrumental sci-fi underscore, small orchestra, low strings, "
    "soft brass, light percussion, analog tape warmth")

_SA3_NEG_DEFAULT = (
    "vocals, singing, speech, spoken words, lyrics, voiceover, crowd noise, "
    "harsh clipping, digital distortion, muddy mix, out of tune, low quality")


def _sa3_augment_prompt(prompt: str) -> str:
    """Prepend an SA3-shaped genre + instrumentation + production anchor to the
    brief-derived prompt (era-aware via keywords). Keeps the existing mood /
    setting / cue-arc text and the instrumental-only tail. Length-capped: one
    genre clause + the existing prompt."""
    low = (prompt or "").lower()
    genre = _SA3_DEFAULT_GENRE
    for keys, g in _SA3_PERIOD_GENRE:
        if any(k in low for k in keys):
            genre = g
            break
    base = (prompt or "").strip()
    return f"{genre}, {base}" if base else genre


def _sa3_clip_window(prompt: str, dur: float, context_s: float):
    """Place the ``dur``-second clip within a ``context_s`` structural window per
    cue so SA3 renders a real slice of a longer piece: an ``outro`` sits at the
    TAIL (resolving), an ``intro`` at the HEAD (build), anything else in the
    MIDDLE (unresolved bridge). Returns ``(seconds_start, seconds_total)``. The
    latent length stays ``dur`` -- only the CONDITIONING window changes, so clip
    length + seed determinism are unchanged."""
    dur = float(dur)
    ctx = max(float(context_s), dur)
    low = (prompt or "").lower()
    if "outro" in low:
        start = max(0.0, ctx - dur)
    elif "intro" in low:
        start = 0.0
    else:
        start = max(0.0, (ctx - dur) / 2.0)
    return start, ctx


@register
class StableAudio3Engine:
    name = "stable_audio_3"
    roles = ("music",)
    default_roles = ("music",)  # PROMOTED 2026-06-03: shipped music default
    commercial_clean = True     # SA3 community license: commercial use OK
    requires_flag = None        # default engine -> always usable; weights checked in load()
    native = True               # drives ComfyUI's own nodes -> no external dep pilot
    interface = "clip"
    sample_rate = 44100

    def __init__(self):
        self._bundle = None     # (model, clip, vae)

    # -- ComfyUI handles for the native pipeline (lazy; only at execute time) --
    def _native(self):
        try:
            import nodes as comfy_nodes
            import comfy_extras.nodes_audio as audio_nodes
        except Exception as exc:  # noqa: BLE001 -- absent ComfyUI runtime
            raise EngineUnusable(
                self.name, "music", EngineUsabilityReason.MALFORMED_CONFIG,
                "ComfyUI runtime not importable -- stable_audio_3 drives native "
                "nodes and only runs inside ComfyUI",
            ) from exc
        return comfy_nodes, audio_nodes

    def _ckpt_present(self):
        try:
            import folder_paths
            return folder_paths.get_full_path("checkpoints", _CKPT)
        except Exception:
            return None

    def load(self):
        if self._bundle is not None:
            return
        if self._ckpt_present() is None:
            raise EngineUnusable(
                self.name, "music", EngineUsabilityReason.MISSING_MODEL,
                "SA3 checkpoint %r not found in ComfyUI/models/checkpoints -- "
                "fetch Comfy-Org/stable-audio-3 (ungated) first" % _CKPT,
            )
        comfy_nodes, _ = self._native()
        model, clip, vae = comfy_nodes.CheckpointLoaderSimple().load_checkpoint(_CKPT)
        # SA3 ships t5gemma separately; if the checkpoint did not carry a usable
        # conditioner, load it explicitly. Defensive: prefer the bundled clip.
        if clip is None:
            clip = comfy_nodes.CLIPLoader().load_clip(_TENC, _CLIP_TYPE)[0]
        self._bundle = (model, clip, vae)

    def unload(self):
        import gc
        self._bundle = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def generate_clip(self, prompt, duration_s, seed):
        """Text prompt -> stereo AUDIO clip ``{"waveform", "sample_rate"}`` via
        ComfyUI's native SA3 graph. Determinism: KSampler takes the int seed and
        builds its generator internally (a bound torch.Generator cannot cross a
        node boundary), so the seed-int is the determinism carrier."""
        self.load()
        comfy_nodes, audio_nodes = self._native()
        model, clip, vae = self._bundle
        seed = int(seed)
        dur = float(duration_s)

        # BUG-408: SA3-shaped prompt + real negative + a structural seconds_total
        # context with a per-cue seconds_start. The LATENT stays exactly dur, so
        # clip length + seed determinism are unchanged; only the conditioning
        # window + prompt change. All knobs env-overridable for A/B tuning.
        context_s = float(os.environ.get("OTR_SA3_CONTEXT_S", "30.0"))
        seconds_start, seconds_total = _sa3_clip_window(prompt, dur, context_s)
        pos_text = _sa3_augment_prompt(prompt)
        neg_text = os.environ.get("OTR_SA3_NEG_PROMPT", _SA3_NEG_DEFAULT)
        steps = int(os.environ.get("OTR_SA3_STEPS", "100"))
        cfg = float(os.environ.get("OTR_SA3_CFG", "6.0"))
        sampler = os.environ.get("OTR_SA3_SAMPLER", "dpmpp_3m_sde_gpu")
        scheduler = os.environ.get("OTR_SA3_SCHEDULER", "exponential")

        pos = comfy_nodes.CLIPTextEncode().encode(clip, pos_text)[0]
        neg = comfy_nodes.CLIPTextEncode().encode(clip, neg_text)[0]
        pos, neg = audio_nodes.ConditioningStableAudio().append(
            pos, neg, seconds_start, seconds_total)
        latent = audio_nodes.EmptyLatentAudio().generate(dur, 1)[0]
        sampled = comfy_nodes.KSampler().sample(
            model, seed, steps, cfg, sampler, scheduler,
            pos, neg, latent, 1.0)[0]
        audio = audio_nodes.VAEDecodeAudio().decode(vae, sampled)[0]
        # traceability for A/B listens (no determinism impact)
        _phash = hashlib.blake2s((pos_text + "||" + neg_text).encode()).hexdigest()[:8]
        log.info("[OTR.sa3] cue_window start=%.1fs total=%.1fs dur=%.1fs seed=%d "
                 "steps=%d cfg=%.1f sampler=%s/%s prompt_hash=%s",
                 seconds_start, seconds_total, dur, seed, steps, cfg,
                 sampler, scheduler, _phash)
        # native AUDIO dict already carries {"waveform","sample_rate"}.
        sr = int(audio.get("sample_rate", self.sample_rate))
        return {"waveform": audio["waveform"], "sample_rate": sr}
