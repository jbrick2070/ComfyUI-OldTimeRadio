"""Kokoro voice adapter -- self-contained per_line (clean-break 1b), two backends.

interface == "per_line": the voice nodes call generate_voice per line; no
delegation to a batch node. The announcer voice is ONE per episode, chosen by a
per-episode seeded pick from the curated British pool; character voices arrive as
bank-assigned ``voice_ref_id`` values from the same 28-voice English pool.

2026-09-02 (queue item 2, operator ruling 2026-09-01 "kokoro onnx is our new
go-to"): the synthesis call lives in ``_kokoro_backends`` and is selected per
``load()`` -- the torch ``kokoro`` package when it imports (the RTX 5080's 3.12
venv; its bytes are unchanged and proven by sha256), else ``kokoro-onnx`` on CPU
(Python 3.13 portables and Desktop, where torch kokoro cannot install --
PBUG-20260901-04). ``OTR_KOKORO_BACKEND=auto|torch|onnx`` forces one; a forced
backend that cannot import fails loud by name. Same engine name, same voice ids,
same ledger contract on every machine.

2026-08-05: the announcer pick DELEGATES to the voice bank rather than running its
own formula. This engine derived the seed as
``Random(f"{episode_seed}_kokoro_announcer")`` while the bank derived it as
``Random(sha1("kokoro_announcer_pick:<seed>"))`` -- one pool, two draws, so the
ledger could name one announcer and the render open another. **This DID change
what listeners hear**: measured over five sampled seeds the two formulas
disagree on three of them. That re-baseline is the point, not a side effect --
one episode, one announcer, whichever side is asked. The engine-local formula
survives only for a bank that cannot be served at all.

C-7: the voice .pt is NEVER fetched during execute, and neither is the ONNX
model. begin_episode picks the voice and verifies its file on local disk, raising
a NAMED EngineUnusable with an out-of-band fetch command if missing -- it never
networks inside a render; never downloads. The fetches live in
``_otr_kokoro_voice_prefetch`` at prestartup.

Library + model imports are lazy (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import logging
import os
import random

from .registry import register

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR")

# Curated British announcer pool (2 male + 2 female: BBC authoritative +
# documentary relaxed) -> natural 50/50 per-episode gender split. Canonical home
# for the pool; config/cast_pools.py ANNOUNCER_PRESETS mirrors it.
ANNOUNCER_VOICE_POOL = ["bm_george", "bm_fable", "bf_emma", "bf_lily"]
KOKORO_SAMPLE_RATE = 24000
_KOKORO_MODEL_SUBDIR = os.path.join("TTS", "KokoroTTS")
#: The ONNX model, relative to the model dir. Duplicated in
#: `_otr_kokoro_voice_prefetch.py` ON PURPOSE (boot must not import this module);
#: `tests/test_kokoro_voice_prefetch.py` pins the two equal.
_KOKORO_ONNX_REL_PATH = os.path.join("onnx", "model.onnx")
KOKORO_ONNX_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"


# S4 platform-portability (2026-07-10): the cuda->mps->cpu auto-waterfall is
# DELETED. The device is EXPLICIT: the voice node threads the CastLock ledger
# stamp (meta.voice_device) onto the adapter as ``requested_device``; a
# device the host cannot provide fails LOUD in KPipeline -- never a silent
# downgrade to a 10x slower backend. The ONNX backend is CPU by design and says
# so once at load (it is not a downgrade from anything: there is no ONNX-CUDA
# path in the pack, and an 82M model runs ~6x realtime on CPU).


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


def _kokoro_voices_dir() -> str:
    return os.path.join(_kokoro_model_dir(), "voices")


def _kokoro_onnx_model_path() -> str:
    """Local path to the ONNX model. LOCAL-DISK only -- never downloads (C-7)."""
    return os.path.join(_kokoro_model_dir(), _KOKORO_ONNX_REL_PATH)


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


def _onnx_model_fetch_hint() -> str:
    return (
        "Fetch it OFFLINE (never during a render), e.g.: huggingface-cli download %s "
        "onnx/model.onnx --local-dir %s" % (KOKORO_ONNX_REPO_ID, _kokoro_model_dir()))


@register
class KokoroEngine:
    name = "kokoro"
    # char_voice added 2026-06-19 (opt-in): kokoro is a fast in-process per-line
    # engine whose generate_voice already takes a per-character voice_ref, so it
    # can serve character lines too. default_roles stays announcer-only -> the
    # INPUT_TYPES default combo is UNCHANGED (byte-identical); the SHIPPED default
    # for characters is the canonical workflow's saved dropdown value (kokoro
    # since 2026-09-02), which is what a template loads.
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
        self._backend = None
        self._backend_name = None
        self._episode_voice = None

    # ------------------------------------------------------------------ #
    def _role(self) -> str:
        # The voice node sets ``adapter.role`` per call; a direct caller (tests,
        # CLI) may not, and an error message must never AttributeError.
        return getattr(self, "role", None) or "announcer_voice"

    def _unusable(self, reason, message):
        from .registry import EngineUnusable

        return EngineUnusable(self.name, self._role(), reason, message)

    def begin_episode(self, meta):
        """Pick ONE announcer voice for the whole episode (seeded) and verify its
        file is on local disk. C-7: a missing file is a NAMED error, never a
        download. Runs once, before the per-line loop."""
        from .registry import EngineUsabilityReason

        episode_seed = "" if not meta else str(meta.get("episode_seed") or "")
        voice_id = _pick_announcer_voice(episode_seed)
        self._episode_voice = voice_id
        path = _kokoro_voice_path(voice_id)
        if not os.path.exists(path):
            raise self._unusable(
                EngineUsabilityReason.MISSING_MODEL,
                f"kokoro announcer voice {voice_id!r} not found at {path}. Fetch "
                f"it OFFLINE (never during a render), e.g.: huggingface-cli "
                f"download hexgrad/Kokoro-82M voices/{voice_id}.pt --local-dir "
                f"{_kokoro_model_dir()}",
            )

    # ------------------------------------------------------------------ #
    def render_time_params(self) -> dict:
        """What the voice node folds into IS_CHANGED (`_otr_voice_node_common`).

        EMPTY under the torch backend, so every shipping torch render keeps its
        exact in-graph caching behaviour ("static"). Under ONNX the backend name and
        the model file's size + mtime bust the cache when the model is swapped --
        never a 326 MB hash per queue. Uses spec probes, not imports, so a queue
        never pays an import here; a probe that raises reads as "not present".
        """
        if _spec_present("kokoro") and otr_env.get("OTR_KOKORO_BACKEND", "auto").lower() != "onnx":
            return {}
        if not _spec_present("kokoro_onnx"):
            return {}
        path = _kokoro_onnx_model_path()
        try:
            st = os.stat(path)
            stamp = "%d:%d" % (st.st_size, st.st_mtime_ns)
        except OSError:
            stamp = "missing"
        return {"backend": "onnx", "onnx_model": stamp}

    # ------------------------------------------------------------------ #
    def load(self):
        """Select and load the backend (re-evaluated whenever nothing is loaded)."""
        if self._backend is not None:
            return
        from . import _kokoro_backends as _kb
        from .registry import EngineUsabilityReason

        try:
            name = _kb.select_backend_name(otr_env.get("OTR_KOKORO_BACKEND"))
        except _kb.BackendUnavailable as exc:
            # A bad OTR_KOKORO_BACKEND value is a config error; a package that is
            # not installed is the missing-model class (the fix is a pip line).
            reason = (EngineUsabilityReason.MALFORMED_CONFIG
                      if "must be auto, torch or onnx" in str(exc)
                      else EngineUsabilityReason.MISSING_MODEL)
            raise self._unusable(reason, str(exc)) from exc

        device = getattr(self, "requested_device", None) or "cuda"
        if name == "torch":
            backend = _kb.TorchKokoroBackend(device)
            backend.load()          # KPipeline errors stay as they were (S4: loud)
            log.info("[OTR.kokoro] backend=torch device=%s", device)
        else:
            model_path = _kokoro_onnx_model_path()
            if not os.path.exists(model_path):
                raise self._unusable(
                    EngineUsabilityReason.MISSING_MODEL,
                    "kokoro ONNX model not found at %s. %s" % (model_path, _onnx_model_fetch_hint()))
            try:
                voices_npz = _kb.ensure_voices_npz(_kokoro_voices_dir())
                providers = _kb.parse_onnx_providers(otr_env.get("OTR_KOKORO_ONNX_PROVIDERS"))
                backend = _kb.OnnxKokoroBackend(model_path, voices_npz, providers)
                backend.load()
            except _kb.BackendUnavailable as exc:
                raise self._unusable(EngineUsabilityReason.MALFORMED_CONFIG, str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 -- onnxruntime / espeak / phonemizer
                raise self._unusable(
                    EngineUsabilityReason.MALFORMED_CONFIG,
                    "kokoro ONNX backend failed to load (%s: %s)" % (type(exc).__name__, exc),
                ) from exc
            log.info(
                "[OTR.kokoro] backend=onnx provider=%s threads=%d model=%s; the ledger's "
                "voice_device=%r is not used by this backend (CPU by design)",
                ",".join(backend.providers_active), backend.threads, model_path, device)
        self._backend = backend
        self._backend_name = name

    def unload(self):
        backend, self._backend = self._backend, None
        self._backend_name = None
        self._episode_voice = None
        if backend is not None:
            backend.close()         # never raises

    def prepare_text(self, text, delivery_vector=None):
        # Legacy parity: the Kokoro announcer spoke the raw (stripped) ledger line
        # -- no bracket/asterisk cleaning -- so keep this identity to preserve
        # exactly what listeners heard before.
        return text

    def generate_voice(self, text, voice_ref, delivery_vector, seed):
        """One line -> mono AUDIO {"waveform":[1,1,T], "sample_rate"}.

        A bank-assigned voice_ref (voice_ref_id, sprint-4 path) takes precedence;
        otherwise the per-episode seeded pick from begin_episode is used. Runs
        inside the caller's deterministic_inference wrap; peak-normalized to
        ~-1 dBFS to match the legacy announcer bus.
        """
        import numpy as np
        import torch

        from .registry import EngineUsabilityReason

        voice_id = voice_ref or self._episode_voice
        if not voice_id:
            raise self._unusable(
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
            raise self._unusable(
                EngineUsabilityReason.MISSING_MODEL,
                "kokoro voice %r has no local voice file at %s (a non-kokoro bank "
                "id leaked into the kokoro slot?). NO voice-swap fallback "
                "(no-fallback rip) -- cast a real kokoro voice; never downloads "
                "mid-render (V-9)." % (voice_id, _kokoro_voice_path(voice_id)),
            )
        self.load()
        if self._backend_name == "onnx":
            from . import _kokoro_backends as _kb

            try:
                clip = self._backend.synthesize(text, voice_id, self.speed)
            except _kb.BackendUnavailable as exc:
                raise self._unusable(EngineUsabilityReason.MISSING_MODEL, str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 -- phonemizer / espeak / onnxruntime
                raise self._unusable(
                    EngineUsabilityReason.MALFORMED_CONFIG,
                    "kokoro ONNX synthesis failed for voice %r (%s: %s)"
                    % (voice_id, type(exc).__name__, exc),
                ) from exc
        else:
            clip = self._backend.synthesize(text, voice_id, self.speed)
        clip = np.asarray(clip, dtype=np.float32)
        peak = float(np.max(np.abs(clip))) or 1.0
        clip = clip / peak * 0.9  # peak-normalize to ~-1 dBFS (legacy parity)
        wav = torch.from_numpy(np.asarray(clip, dtype=np.float32)).reshape(1, 1, -1)
        return {"waveform": wav, "sample_rate": KOKORO_SAMPLE_RATE}


def _spec_present(name: str) -> bool:
    """`importlib.util.find_spec` that never raises (a `sys.modules` fake without
    `__spec__` raises ValueError) and never imports anything."""
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False
