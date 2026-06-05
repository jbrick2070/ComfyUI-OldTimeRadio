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

import os

from .registry import EngineUnusable, EngineUsabilityReason, register

# Default model files in ComfyUI's models/ tree (Comfy-Org/stable-audio-3).
_CKPT = os.environ.get("OTR_SA3_CKPT", "stable_audio_3_small_music.safetensors")
_TENC = os.environ.get("OTR_SA3_TEXT_ENCODER", "t5gemma_b_b_ul2.safetensors")
_CLIP_TYPE = os.environ.get("OTR_SA3_CLIP_TYPE", "stable_audio")


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

        pos = comfy_nodes.CLIPTextEncode().encode(clip, prompt)[0]
        neg = comfy_nodes.CLIPTextEncode().encode(clip, "")[0]
        pos, neg = audio_nodes.ConditioningStableAudio().append(
            pos, neg, 0.0, dur)
        latent = audio_nodes.EmptyLatentAudio().generate(dur, 1)[0]
        sampled = comfy_nodes.KSampler().sample(
            model, seed, 100, 6.0, "dpmpp_3m_sde_gpu", "exponential",
            pos, neg, latent, 1.0)[0]
        audio = audio_nodes.VAEDecodeAudio().decode(vae, sampled)[0]
        # native AUDIO dict already carries {"waveform","sample_rate"}.
        sr = int(audio.get("sample_rate", self.sample_rate))
        return {"waveform": audio["waveform"], "sample_rate": sr}
