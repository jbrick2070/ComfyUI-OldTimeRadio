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
            import indextts  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "indextts is not installed -- run the isolated dependency "
                "pilot and install it before enabling OTR_ENABLE_INDEXTTS2"
            ) from exc
        # VERIFY in the pilot: construct IndexTTS2 from its config + model dir
        # and cache the inference handle on self._model.
        self._model = indextts

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
        """One dialogue line -> mono AUDIO. Inference wired in the GPU pilot.

        TODO-for-F: the assumed library call is
            IndexTTS2(cfg, model_dir).infer(spk_audio_prompt=ref_clip_path,
                text=text, emo_vector=<self.emo_list(delivery_vector)>,
                seed=seed, generator=<bound torch.Generator>)
        scripts/otr_audio_dep_pilot.py verifies the constructor + infer kwargs +
        external-generator support on GPU before supports_external_generator is
        flipped True and this body is filled. Until then indextts2 is a
        flag-gated, default-off stub -- never run blind in production.
        """
        self.load()
        emo = self.emo_list(delivery_vector)
        raise RuntimeError(
            "IndexTTS2 inference is wired and verified in the GPU pilot; "
            f"engine registered and selectable (emo dims={len(emo)})"
        )
