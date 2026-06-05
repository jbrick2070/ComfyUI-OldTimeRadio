"""MusicGen theme adapter -- self-contained clip engine (clean-break 1c).

interface == "clip": the theme node (OTR_StableAudioTheme) calls generate_clip
per cue; no delegation to a batch node. MusicGen inference (transformers
MusicgenForConditionalGeneration) is lazy-loaded inside load() / generate_clip so
importing the registry package stays light (C-5). guidance_scale is the
music_musicgen_v1 profile default (pinned by a drift guard). Cue prompts come
from the Meta brief protocol (nodes/_otr_music_prompt) via the theme node, so a
given brief yields the same music the legacy node produced.

UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

from .registry import register

_MUSICGEN_MODEL_ID = "facebook/musicgen-medium"
_MUSICGEN_TOKENS_PER_SEC = 50  # ~50 audio tokens per second at 32 kHz


@register
class MusicGenEngine:
    name = "musicgen"
    roles = ("music",)
    default_roles = ()                   # DEMOTED 2026-06-03: SA3 is now the music default; musicgen stays selectable
    commercial_clean = False             # Meta MusicGen CC-BY-NC-4.0 (non-commercial)
    requires_flag = None
    interface = "clip"
    sample_rate = 32000
    supports_external_generator = False  # MusicGen.generate binds no external Generator
    model_id = _MUSICGEN_MODEL_ID
    guidance_scale = 3.0                 # == music_musicgen_v1 profile default (pinned)

    def __init__(self):
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "transformers MusicGen is not available -- install transformers "
                "before using the musicgen engine"
            ) from exc
        import os

        import torch

        cache_dir = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=cache_dir,
        )
        model = MusicgenForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype, cache_dir=cache_dir,
        ).to(device)
        model.eval()
        self._model = model

    def unload(self):
        model, self._model = self._model, None
        self._processor = None
        try:
            import gc

            import torch

            if model is not None:
                model.cpu()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- teardown must never raise
            pass

    def generate_clip(self, prompt, duration_s, seed):
        """One music cue -> mono AUDIO {"waveform":[1,1,T], "sample_rate"}.

        Runs inside the caller's deterministic_inference wrap; seeds torch from
        the per-cue seed. Peak-normalized to ~-1 dBFS to match the legacy bus.
        """
        import numpy as np
        import torch

        self.load()
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        max_new_tokens = int(float(duration_s) * _MUSICGEN_TOKENS_PER_SEC) + 8
        inputs = self._processor(
            text=[prompt], padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            audio_values = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                guidance_scale=self.guidance_scale,
            )
        av = audio_values
        if hasattr(av, "audio_values"):
            av = av.audio_values
        av = av.detach().cpu().float()
        if av.dim() == 3:
            arr = av[0, 0].numpy()
        elif av.dim() == 2:
            arr = av[0].numpy()
        elif av.dim() == 1:
            arr = av.numpy()
        else:
            arr = av.flatten().numpy()
        peak = float(np.max(np.abs(arr))) or 1.0
        arr = (arr / peak * 0.89).astype(np.float32)
        wav = torch.from_numpy(arr).reshape(1, 1, -1)
        return {"waveform": wav, "sample_rate": self.sample_rate}
