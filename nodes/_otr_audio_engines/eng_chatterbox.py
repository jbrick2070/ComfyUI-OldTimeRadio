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
        """One dialogue line -> mono AUDIO ``{"waveform", "sample_rate"}``.

        Implemented to the documented assumed_call:
            ChatterboxTTS.generate(text, audio_prompt_path=ref_clip_path,
                exaggeration=<self._project(delivery_vector)>, cfg_weight=0.5,
                temperature=0.6[, generator=<bound torch.Generator>])
        Runs inside the call site's ``deterministic_inference`` wrap (global RNG
        already seeded); we seed locally too and bind a per-line Generator when
        ``generate`` accepts one. GPU-VALIDATE (F): scripts/otr_audio_dep_pilot
        confirms the kwargs + that ``generate`` binds an external
        ``torch.Generator`` on the box, then flips
        ``supports_external_generator`` True. Flag-gated + default-off until
        then; ``supported_kwargs`` keeps an unverified kwarg from crashing.
        """
        import torch

        from .base import supported_kwargs

        self.load()
        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = supported_kwargs(
            self._model.generate,
            audio_prompt_path=ref_clip_path,
            exaggeration=self._project(delivery_vector),
            cfg=0.5,
            cfg_weight=0.5,
            temperature=0.6,
            generator=torch.Generator(device=dev).manual_seed(seed),
        )
        wav = self._model.generate(text, **kwargs)
        return {"waveform": wav, "sample_rate": self.sample_rate}

    @staticmethod
    def _project(delivery_vector):
        """Collapse the 8-dim delivery vector to Chatterbox's exaggeration."""
        if not delivery_vector:
            return 0.5
        calm = float(delivery_vector.get("calm", 0.5))
        return round(min(1.0, 0.3 + 0.7 * max(0.0, 1.0 - calm)), 3)
