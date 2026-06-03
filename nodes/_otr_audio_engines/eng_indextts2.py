"""IndexTTS2 voice adapter -- expressive zero-shot cloning, opt-in.

NON-commercial: IndexTTS2 weights carry the Bilibili license (written auth
required for commercial use), so this engine is opt-in behind
OTR_ENABLE_INDEXTTS2 and never a default. ``interface == "per_line"``. The
library import is deferred to ``load``; the actual inference call is wired and
verified in the GPU dependency pilot (it must not drag xformers onto cu130).
"""
from __future__ import annotations

from .registry import register


@register
class IndexTTS2Engine:
    name = "indextts2"
    roles = ("char_voice",)
    default_roles = ()
    commercial_clean = False  # Bilibili license -- non-commercial
    requires_flag = "OTR_ENABLE_INDEXTTS2"
    interface = "per_line"
    sample_rate = 22050

    def __init__(self):
        self._model = None

    def load(self):
        if self._model is not None:
            return
        try:
            from indextts.infer import IndexTTS2
        except ImportError as exc:
            raise RuntimeError(
                "indextts is not installed -- run the isolated dependency "
                "pilot and install it before enabling OTR_ENABLE_INDEXTTS2"
            ) from exc
        import os

        # GPU-VALIDATE (F): confirm the IndexTTS2(cfg, model_dir) constructor +
        # the weights-dir convention on the box. OTR_INDEXTTS2_DIR points at the
        # downloaded weights; cfg is the config.yaml beside them.
        model_dir = os.environ.get("OTR_INDEXTTS2_DIR", "")
        cfg = os.path.join(model_dir, "config.yaml") if model_dir else ""
        self._model = IndexTTS2(cfg, model_dir)

    def unload(self):
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def emo_list(self, delivery_vector):
        """8-dim delivery vector -> IndexTTS2 Emo-Vector order list."""
        from .._otr_delivery_vector import EMOTIONS

        dv = delivery_vector or {}
        return [round(float(dv.get(e, 0.0)), 3) for e in EMOTIONS]

    def prepare_text(self, text, delivery_vector=None):
        """Engine-neutral clean spoken text; the audio direction is routed
        into the emo-vector (see emo_list), not the words."""
        from .._otr_script_prep import clean_spoken_text

        return clean_spoken_text(text)

    def generate_voice(self, text, ref_clip_path, delivery_vector, seed):
        """One dialogue line -> mono AUDIO ``{"waveform", "sample_rate"}``.

        Implemented to the documented assumed_call:
            IndexTTS2(cfg, model_dir).infer(spk_audio_prompt=ref_clip_path,
                text=text, emo_vector=<self.emo_list(delivery_vector)>,
                seed=seed[, generator=<bound torch.Generator>])
        Runs inside the call site's ``deterministic_inference`` wrap. GPU-VALIDATE
        (F): scripts/otr_audio_dep_pilot pins the constructor, the ``infer``
        kwargs + return type, and external-generator support on the box, then
        flips ``supports_external_generator`` True. Flag-gated + default-off;
        ``supported_kwargs`` keeps an unverified kwarg from crashing.
        """
        import torch

        from .base import supported_kwargs

        self.load()
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        infer = getattr(self._model, "infer", None) or getattr(
            self._model, "generate", None
        )
        if infer is None:
            raise RuntimeError(
                "IndexTTS2 handle exposes no infer()/generate() -- the GPU "
                "pilot (F) wires the real inference entry point"
            )
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = supported_kwargs(
            infer,
            spk_audio_prompt=ref_clip_path,
            text=text,
            emo_vector=self.emo_list(delivery_vector),
            seed=seed,
            generator=torch.Generator(device=dev).manual_seed(seed),
        )
        return {
            "waveform": self._as_waveform(infer(**kwargs)),
            "sample_rate": self.sample_rate,
        }

    @staticmethod
    def _as_waveform(out):
        """Normalize an engine return (tensor / ``(sr, wav)`` / dict / WAV path)
        to a waveform tensor. GPU-VALIDATE (F): pin IndexTTS2.infer's real return
        type and drop this shim if it already returns a plain tensor.
        """
        import torch

        if isinstance(out, tuple) and out:
            out = out[-1]
        if isinstance(out, dict):
            out = out.get("waveform", out.get("wav"))
        if isinstance(out, str):
            import torchaudio

            wav, _sr = torchaudio.load(out)
            return wav
        return torch.as_tensor(out)
