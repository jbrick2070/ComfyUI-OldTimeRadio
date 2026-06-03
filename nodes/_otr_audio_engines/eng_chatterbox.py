"""Chatterbox voice adapter -- MIT-licensed, commercial-clean opt-in.

Serves both char_voice and announcer_voice (independent slots, shared engine
pool). Opt-in behind OTR_ENABLE_CHATTERBOX. ``interface == "per_line"``: the
generic voice node calls ``generate_voice`` per dialogue line. The library
import is deferred to ``load`` and MUST NOT pull xformers onto the cu130
Blackwell stack -- verify in the isolated dependency pilot before enabling.
"""
from __future__ import annotations

from .registry import register


@register
class ChatterboxEngine:
    name = "chatterbox"
    roles = ("char_voice", "announcer_voice")
    default_roles = ()
    commercial_clean = True  # MIT
    requires_flag = "OTR_ENABLE_CHATTERBOX"
    interface = "per_line"
    sample_rate = 24000

    def __init__(self):
        self._model = None

    def load(self):
        if self._model is not None:
            return
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError as exc:
            raise RuntimeError(
                "chatterbox is not installed -- run the isolated dependency "
                "pilot and install it before enabling OTR_ENABLE_CHATTERBOX"
            ) from exc
        self._model = ChatterboxTTS.from_pretrained(device="cuda")

    def unload(self):
        self._model = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def prepare_text(self, text, delivery_vector=None):
        """Engine-neutral clean spoken text. Audio direction lives in the
        delivery vector / exaggeration, not in the words."""
        from .._otr_script_prep import clean_spoken_text

        return clean_spoken_text(text)

    def generate_voice(self, text, ref_clip_path, delivery_vector, seed):
        """One dialogue line -> mono AUDIO. Inference wired in the GPU pilot.

        TODO-for-F: the assumed library call is
            ChatterboxTTS.generate(text, audio_prompt_path=ref_clip_path,
                exaggeration=<self._project(delivery_vector)>, cfg=0.5,
                temperature=0.6, generator=<bound torch.Generator>)
        scripts/otr_audio_dep_pilot.py verifies on GPU that ``generate`` binds an
        external ``torch.Generator`` (so render-twice is byte-identical) before
        ``supports_external_generator`` is flipped True and this body is filled.
        Until then chatterbox is a flag-gated, default-off stub -- never run
        blind in production.
        """
        self.load()
        exaggeration = self._project(delivery_vector)
        raise RuntimeError(
            "Chatterbox inference is wired and verified in the GPU pilot; "
            f"engine registered and selectable (exaggeration={exaggeration})"
        )

    @staticmethod
    def _project(delivery_vector):
        """Collapse the 8-dim delivery vector to Chatterbox's exaggeration."""
        if not delivery_vector:
            return 0.5
        calm = float(delivery_vector.get("calm", 0.5))
        return round(min(1.0, 0.3 + 0.7 * max(0.0, 1.0 - calm)), 3)
