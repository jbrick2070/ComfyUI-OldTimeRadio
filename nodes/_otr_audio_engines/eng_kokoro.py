"""Kokoro announcer-voice adapter -- self-contained per_line (clean-break 1b).

interface == "per_line": the announcer node calls generate_voice per announcer
line; no delegation to a batch node. The announcer voice is ONE per episode,
chosen by a per-episode seeded pick from the curated British pool.

2026-08-05: that pick now DELEGATES to the voice bank rather than running its
own formula. This engine derived the seed as
``Random(f"{episode_seed}_kokoro_announcer")`` while the bank derived it as
``Random(sha1("kokoro_announcer_pick:<seed>"))`` -- one pool, two draws, so the
ledger could name one announcer and the render open another. **This DID change
what listeners hear**: measured over five sampled seeds the two formulas
disagree on three of them. That re-baseline is the point, not a side effect --
one episode, one announcer, whichever side is asked. The engine-local formula
survives only for a bank that cannot be served at all.

C-7: the voice .pt is NEVER fetched during execute. begin_episode picks the
voice and verifies its file on local disk, raising a NAMED EngineUnusable with
an out-of-band fetch command if missing -- it never networks inside a render.

Library + model imports are lazy (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import logging
import os
import random

from .registry import register

log = logging.getLogger("OTR")

# Curated British announcer pool (2 male + 2 female: BBC authoritative +
# documentary relaxed) -> natural 50/50 per-episode gender split. Canonical home
# for the pool; config/cast_pools.py ANNOUNCER_PRESETS mirrors it.
ANNOUNCER_VOICE_POOL = ["bm_george", "bm_fable", "bf_emma", "bf_lily"]
KOKORO_SAMPLE_RATE = 24000
_KOKORO_MODEL_SUBDIR = os.path.join("TTS", "KokoroTTS")


# S4 platform-portability (2026-07-10): the cuda->mps->cpu auto-waterfall is
# DELETED. The device is EXPLICIT: the voice node threads the CastLock ledger
# stamp (meta.voice_device) onto the adapter as ``requested_device``; a
# device the host cannot provide fails LOUD in KPipeline -- never a silent
# downgrade to a 10x slower backend.


def _kokoro_model_dir() -> str:
    """Absolute path to ComfyUI models/TTS/KokoroTTS (no network)."""
    try:
        import folder_paths

        return os.path.join(folder_paths.models_dir, _KOKORO_MODEL_SUBDIR)
    except Exception:  # noqa: BLE001 -- non-Comfy contexts (tests / CLI)
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(
            os.path.join(here, "..", "..", "..", "..", "models", _KOKORO_MODEL_SUBDIR)
        )


def _kokoro_voice_path(voice_id: str) -> str:
    """Local path to the voice .pt. LOCAL-DISK only -- never downloads (C-7)."""
    return os.path.join(_kokoro_model_dir(), "voices", f"{voice_id}.pt")


def _pick_announcer_voice(episode_seed, voice_override="random") -> str:
    """One voice per episode, drawn the SAME way the voice bank draws it.

    This used to run its own formula -- Random(f"{seed}_kokoro_announcer") --
    while the bank drew with Random(sha1("kokoro_announcer_pick:<seed>")). Two
    formulas over one pool means the ledger can name one announcer while the
    render opens another, which is precisely the defect the shared
    gender-agnostic selector was extracted to end on the character side.

    Delegates to the bank so there is ONE draw. The local formula survives only
    as the fallback for a bank that cannot be loaded at all -- an engine must
    still be able to speak.
    """
    if voice_override and voice_override != "random":
        return voice_override
    from .._otr_voice_bank import (
        VoiceBankError, VoiceCastingError, announcer_voice_ref,
    )
    try:
        return announcer_voice_ref("kokoro", episode_seed=episode_seed).voice_ref_id
    except (VoiceBankError, VoiceCastingError):
        # ONLY the bank's own typed contracts fall back -- a genuinely absent or
        # unservable bank, where an engine must still be able to speak. A bare
        # `except Exception` here would recreate the very divergence this
        # delegation removes: an import error, a schema change or a coding
        # mistake would silently switch back to the second formula and the
        # ledger would once again name a different announcer than the render.
        # Loud, because a seedless-style silent fallback is how the last one hid.
        log.warning(
            "[OTR.kokoro] voice bank could not serve an announcer; falling back "
            "to the engine-local pool. The ledger's announcer and this render "
            "may now disagree.", exc_info=True)
        return random.Random(f"{episode_seed}_kokoro_announcer").choice(
            ANNOUNCER_VOICE_POOL)


@register
class KokoroEngine:
    name = "kokoro"
    # char_voice added 2026-06-19 (opt-in): kokoro is a fast in-process per-line
    # engine whose generate_voice already takes a per-character voice_ref, so it
    # can serve character lines too. default_roles stays announcer-only -> the
    # shipped char_voice default is UNCHANGED (byte-identical); kokoro is used for
    # char only when explicitly selected (e.g. a fast soak voice lane).
    roles = ("announcer_voice", "char_voice")
    default_roles = ("announcer_voice",)   # internal default until promotion (I)
    commercial_clean = True                # Apache-2.0
    requires_flag = None
    interface = "per_line"
    sample_rate = KOKORO_SAMPLE_RATE
    supports_external_generator = False    # KPipeline binds no external Generator
    voice_ref_field = "voice_ref_id"       # bank-assigned voice (sprint 4); else seeded pick
    speed = 0.95                           # == announcer_kokoro_v1 profile default (pinned)

    def __init__(self):
        self._pipeline = None
        self._episode_voice = None

    def begin_episode(self, meta):
        """Pick ONE announcer voice for the whole episode (seeded) and verify its
        file is on local disk. C-7: a missing file is a NAMED error, never a
        download. Runs once, before the per-line loop."""
        from .registry import EngineUnusable, EngineUsabilityReason

        episode_seed = "" if not meta else str(meta.get("episode_seed") or "")
        voice_id = _pick_announcer_voice(episode_seed)
        self._episode_voice = voice_id
        path = _kokoro_voice_path(voice_id)
        if not os.path.exists(path):
            raise EngineUnusable(
                self.name, "announcer_voice", EngineUsabilityReason.MISSING_MODEL,
                f"kokoro announcer voice {voice_id!r} not found at {path}. Fetch "
                f"it OFFLINE (never during a render), e.g.: huggingface-cli "
                f"download 1038lab/KokoroTTS voices/{voice_id}.pt --local-dir "
                f"{_kokoro_model_dir()}",
            )

    def load(self):
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError(
                "kokoro is not installed -- run the dependency pilot / "
                "pip install kokoro before using the kokoro announcer engine"
            ) from exc
        # lang_code 'b' = British English; pin repo_id to suppress migration noise.
        # S4: EXPLICIT device from the CastLock ledger stamp (threaded by the
        # voice node as requested_device; default cuda = nv50 baseline).
        self._pipeline = KPipeline(
            lang_code="b",
            device=(getattr(self, "requested_device", None) or "cuda"),
            repo_id="hexgrad/Kokoro-82M",
        )

    def unload(self):
        pipe, self._pipeline = self._pipeline, None
        self._episode_voice = None
        try:
            if pipe is not None and hasattr(pipe, "model"):
                pipe.model.to("cpu")
            del pipe
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- teardown must never raise
            pass

    def prepare_text(self, text, delivery_vector=None):
        # Legacy parity: the Kokoro announcer spoke the raw (stripped) ledger line
        # -- no bracket/asterisk cleaning -- so keep this identity to preserve
        # exactly what listeners heard before.
        return text

    def generate_voice(self, text, voice_ref, delivery_vector, seed):
        """One announcer line -> mono AUDIO {"waveform":[1,1,T], "sample_rate"}.

        A bank-assigned voice_ref (voice_ref_id, sprint-4 path) takes precedence;
        otherwise the per-episode seeded pick from begin_episode is used. Runs
        inside the caller's deterministic_inference wrap; peak-normalized to
        ~-1 dBFS to match the legacy announcer bus.
        """
        import numpy as np
        import torch

        from .registry import EngineUnusable, EngineUsabilityReason

        voice_id = voice_ref or self._episode_voice
        if not voice_id:
            raise EngineUnusable(
                self.name, "announcer_voice",
                EngineUsabilityReason.MALFORMED_CONFIG,
                "no announcer voice resolved (begin_episode did not run and the "
                "cast carries no voice_ref_id)",
            )
        # LOCAL-ONLY voice guard (capstone soak catch 2026-06-09): a cast row
        # reached kokoro carrying a NON-kokoro voice ref (an indextts2 bank id,
        # e.g. 'vz_donor_*'); KPipeline would then try to DOWNLOAD voices/<id>.pt
        # from HF mid-render and 404-abort the whole episode. NO-FALLBACK (operator
        # 2026-07-03): FAIL LOUD instead of swapping to the seeded episode voice --
        # a non-kokoro id in the kokoro slot is a casting defect the operator must
        # fix, never a silent voice swap (and still never a mid-render hub fetch).
        if not os.path.exists(_kokoro_voice_path(voice_id)):
            raise EngineUnusable(
                self.name, "announcer_voice", EngineUsabilityReason.MISSING_MODEL,
                "kokoro voice %r has no local voice file at %s (a non-kokoro bank "
                "id leaked into the kokoro slot?). NO voice-swap fallback "
                "(no-fallback rip) -- cast a real kokoro voice; never downloads "
                "mid-render (V-9)." % (voice_id, _kokoro_voice_path(voice_id)),
            )
        self.load()
        segments = []
        for _, _, audio_data in self._pipeline(
            text, voice=voice_id, speed=self.speed, split_pattern=r"\n+",
        ):
            if torch.is_tensor(audio_data):
                arr = audio_data.detach().cpu().numpy()
            else:
                arr = np.asarray(audio_data, dtype=np.float32)
            segments.append(arr.astype(np.float32).squeeze())
        if not segments:
            raise RuntimeError("kokoro pipeline produced no audio")
        clip = np.concatenate(segments) if len(segments) > 1 else segments[0]
        peak = float(np.max(np.abs(clip))) or 1.0
        clip = clip / peak * 0.9  # peak-normalize to ~-1 dBFS (legacy parity)
        wav = torch.from_numpy(np.asarray(clip, dtype=np.float32)).reshape(1, 1, -1)
        return {"waveform": wav, "sample_rate": KOKORO_SAMPLE_RATE}
