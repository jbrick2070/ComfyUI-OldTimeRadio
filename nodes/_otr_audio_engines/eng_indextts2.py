"""IndexTTS2 voice adapter -- Path B isolated subprocess worker (opt-in).

IndexTTS2 hard-pins torch 2.8 / numpy 1.26 / transformers 4.52, which would brick
the Blackwell (torch 2.10 / cu130) ComfyUI venv. So it runs in its OWN isolated
venv (python 3.10 / torch 2.8 / cu128) as a supervised subprocess worker; this
adapter, in ComfyUI's venv, drives it over line-delimited JSON and reads back the
rendered WAV -- ZERO shared torch. PROMOTED 2026-06-04 to the shipped char_voice
default: its weights carry the non-commercial bilibili Model Use License, so it
emits a non-blocking commercial-use warning (I-8) and -- with no permanent legacy
fallback -- a render fails closed with a NAMED error until the Path B worker +
weights are installed (C-7). ``interface == "per_line"``.

Config (env, with box defaults under ``ComfyUI/index-tts``):
  ``OTR_INDEXTTS2_VENV``   isolated venv python (``.venv/Scripts/python.exe``)
  ``OTR_INDEXTTS2_DIR``    weights dir (``checkpoints``, holds ``config.yaml``)
  ``OTR_INDEXTTS2_WORKER`` worker script (``scripts/_otr_indextts2_worker.py``)
  ``OTR_INDEXTTS2_FP16``   ``1`` to load fp16 (default fp32)

Fail-closed: a missing venv / worker / weights raises a NAMED RuntimeError telling
the operator to run the Path B install -- never a silent swap, never an in-process
import of the conflicting library. Import-time is side-effect-free (C-5): the
worker is spawned lazily in ``load`` / ``generate_voice``, never at import.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile

from .registry import register

# realpath (NOT abspath): under the Desktop-v2 install the OldTimeRadio custom
# node is reached through a directory junction, so abspath(__file__) keeps the
# install-root path and _COMFY_ROOT resolves to the WRONG tree -- index-tts is
# installed in the REAL ComfyUI tree the junction targets (the documented box
# default "ComfyUI/index-tts"). realpath resolves the junction to that real tree;
# it is a no-op on a non-junction checkout and env overrides still win. (Without
# this, a headless render fails closed: "IndexTTS2 Path B not installed: isolated
# venv python missing at <install-root>\index-tts\.venv\...".)
_THIS = os.path.realpath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))   # ...\ComfyUI-OldTimeRadio
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))              # ...\ComfyUI


def _default(*parts):
    return os.path.join(_COMFY_ROOT, "index-tts", *parts)


# --------------------------------------------------------------------------- #
# VOICE IDENTITY (2026-08-18, PBUG-20260817-09) -- the two emotion constants.
#
# WHAT THE OPERATOR HEARD: "Nag 1 sounded good, Nag beat 2 was another voice."
# One character, two of his own lines, the same reference WAV -- and a different
# voice each time. Two causes, and they compound.
#
# HOW THE VENDOR ACTUALLY SPENDS THIS VECTOR (read from the installed
# `indextts/infer_v2.py`, not assumed). The emotion vector is not a tone knob
# laid over the speaker; its SUM is a budget spent AGAINST him:
#
#     emovec = emovec_mat + (1 - sum(weight_vector)) * emovec
#
# `emovec_mat` is a blend of generic emotion prototypes and the right-hand term
# is the speaker's OWN emotional embedding. So a vector summing to 1.0 leaves
# `1 - 1.0 == 0.0` of him in the result -- the identity the reference WAV was
# chosen for is fully displaced, and what survives is whatever the sampler drew
# under that line's seed. A neutral line stamped `calm=1.0` is the WORST case,
# not the safest one, which is exactly why the quiet beat is the one that
# stopped sounding like him.
#
# CARRY THE OPERATOR'S CAVEAT VERBATIM: that is the EMOTION-LATENT BLEND, not
# "26% of his vocal tract".
# --------------------------------------------------------------------------- #

#: Emotion-blend strength when ``OTR_INDEXTTS2_EMO_ALPHA`` is unset or unusable.
#:
#: PINNED AT 1.0 AND NO LONGER A TUNING KNOB (2026-08-18). It went 1.0 -> 0.4 on
#: the voice-identity fix, when two knobs shared one job; the ceiling below now
#: owns that job alone, so alpha is a pass-through on the default path -- at
#: exactly 1.0 :meth:`IndexTTS2Engine._apply_vendor_alpha` short-circuits and
#: the vendor's pre-scaling does nothing.
#:
#: IT IS KEPT, NOT DELETED, and the distinction is deliberate. It remains a
#: compatibility and diagnostic override: ``OTR_INDEXTTS2_EMO_ALPHA`` still
#: resolves per render, still keys the cache, and still reaches the worker, so a
#: control arm can reproduce a pre-fix blend without a code change. Deleting it
#: would touch the cache key, the per-line receipt, the profile schema, the
#: worker payload and the acceptance checker to buy nothing.
EMO_ALPHA_DEFAULT = 1.0

#: Ceiling on the EFFECTIVE emotion mass -- the sum of the weights the vendor
#: actually spends, AFTER alpha. With alpha pinned at 1.0 this is THE knob: it
#: alone decides how much of the generic emotion prototype is laid over the
#: speaker, and ``1 - cap`` is the share of his own embedding that survives.
#:
#: 0.560 IS THE OPERATOR'S OWN NUMBER, chosen by ear on
#: ``otr/episodes/lemmy_emotion_ladder_logodds_2026-08-18/`` -- a ladder that
#: pinned alpha at 1.0 and varied only this value, spaced evenly in
#: ``log(mass / (1 - mass))`` because that ratio is what the ear tracks. His
#: verdict: *"IF I WERE A KID I'D LIKE MORE BUT AS AN ADULT ARM0P560 IS
#: PERFECT."* Of the uncapped arm: *"its not a real emtion ist coimputer
#: emoption"* -- at mass 1.0 the vendor residual is 0, none of the speaker's own
#: emotional embedding survives, and what is left is the generic prototype.
#:
#: IT ALSO STILL COVERS VECTORS ALPHA CANNOT. A multiplier can promise nothing
#: about a vector it did not derive -- a hand-edited or pre-stamped ledger
#: summing to 8.0 spends all 8.0 at alpha 1.0. The ceiling is measured on what
#: the vendor actually receives, so that line lands here like any other.
EFFECTIVE_EMOTION_MASS_CAP = 0.56

#: The highest ceiling worth expressing: eight dimensions each clamped to 1.0,
#: so a cap at or above this can never bind. ``OTR_INDEXTTS2_EMO_MASS_CAP=8``
#: is therefore "no ceiling", which is what the pre-fix control arm needs.
EMOTION_MASS_CAP_DISABLED = 8.0


def sanitize_delivery_vector(delivery_vector) -> dict:
    """Any object -> a complete 8-key ``{emotion: 0.0..1.0}`` dict. Never raises.

    THE ONE SAFE VECTOR PREPARATION HELPER (QA-4). A delivery vector is
    HAND-EDITABLE -- it is stamped onto the ledger as ordinary JSON -- so a
    value can legitimately arrive as a string, a ``None``, a NaN or a number
    outside 0..1. Everything that reads a vector on this lane reads it through
    here: the outbound worker payload, the cap metrics, and the voice
    dispatch's per-line observability line, which used to call ``float(...)``
    on the RAW stamped values and would raise ``ValueError`` on a ledger
    carrying ``{"happy": "very"}``. THE LAW is that a render degrades, never
    raises, so there is exactly one sanitizer and every reader shares it.

    A non-dict yields the flat all-zero vector; one bad value zeroes or clamps
    THAT emotion only, leaving the rest intact. Values are rounded to three
    decimals -- the resolution the cache-key quantizer keeps -- so what is
    measured is what is sent.

    The emotion ORDER is imported from the delivery module, which owns that
    mapping and is not modified from here.
    """
    from .._otr_delivery_vector import EMOTIONS

    dv = delivery_vector if isinstance(delivery_vector, dict) else {}
    out = {}
    for emotion in EMOTIONS:
        try:
            value = float(dv.get(emotion, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value != value:  # NaN
            value = 0.0
        out[emotion] = round(min(1.0, max(0.0, value)), 3)
    return out


@register
class IndexTTS2Engine:
    name = "indextts2"
    roles = ("char_voice",)
    default_roles = ("char_voice",)  # PROMOTED 2026-06-04: shipped char_voice default
    commercial_clean = False  # bilibili Model Use License -- non-commercial
    requires_flag = None             # default engine -> always usable; venv/weights checked in load()
    interface = "per_line"
    sample_rate = 22050
    # Model-agnostic dispatch metadata (replaces the old _OTR_CLONE_ENGINES
    # tuple): a clone engine needs a per-character reference WAV. NO-FALLBACK
    # (operator 2026-07-03): a char_voice line with no usable ref now FAILS LOUD
    # (named EngineUnusable in the dispatch) -- it never silently renders on bark.
    requires_voice_ref = True
    voice_ref_kind = "wav_path"
    missing_ref_fallback = None

    def __init__(self):
        self._proc = None

    # ---- config resolution (env override -> box default) ----
    def _venv_python(self):
        return os.environ.get("OTR_INDEXTTS2_VENV") or _default(".venv", "Scripts", "python.exe")

    def _model_dir(self):
        return os.environ.get("OTR_INDEXTTS2_DIR") or _default("checkpoints")

    def _worker_script(self):
        return os.environ.get("OTR_INDEXTTS2_WORKER") or os.path.join(
            _REPO_ROOT, "scripts", "_otr_indextts2_worker.py")

    def _use_fp16(self):
        return os.environ.get("OTR_INDEXTTS2_FP16", "0") == "1"

    # ---- worker lifecycle ----
    def load(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        py = self._venv_python()
        worker = self._worker_script()
        model_dir = self._model_dir()
        for label, path in (("isolated venv python", py),
                            ("worker script", worker),
                            ("weights dir", model_dir)):
            if not os.path.exists(path):
                raise RuntimeError(
                    "IndexTTS2 Path B not installed: %s missing at %s -- run "
                    "scripts\\_otr_indextts2_install.ps1 (isolated venv + weights) "
                    "before rendering with indextts2 (the default char voice)" % (label, path))
        cfg = os.path.join(model_dir, "config.yaml")
        if not os.path.exists(cfg):
            raise RuntimeError(
                "IndexTTS2 weights incomplete: %s missing -- re-run "
                "scripts\\_otr_idx_download_weights.py" % cfg)

        args = [py, worker, "--model-dir", model_dir]
        if self._use_fp16():
            args.append("--fp16")
        err_path = os.path.join(_REPO_ROOT, "_otr_indextts2_worker.err")
        self._stderr = open(err_path, "ab", buffering=0)
        proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, encoding="utf-8", bufsize=1, cwd=os.path.dirname(model_dir))
        line = proc.stdout.readline()
        try:
            ready = json.loads(line) if line.strip() else {"ready": False, "error": "no readiness line"}
        except ValueError:
            ready = {"ready": False, "error": "bad readiness line: %r" % line[:200]}
        if not ready.get("ready"):
            try:
                proc.kill()
            except OSError:
                pass
            raise RuntimeError(
                "IndexTTS2 worker failed to start: %s (see %s)"
                % (ready.get("error"), err_path))
        self._proc = proc

    def unload(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write(json.dumps({"stop": True}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 -- shutdown must never raise
            pass
        finally:
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            sd = getattr(self, "_stderr", None)
            if sd is not None:
                try:
                    sd.close()
                except OSError:
                    pass

    # ---- emotion + text (adapter-side; sent to the worker) ----
    def emo_list(self, delivery_vector):
        """8-dim delivery vector -> IndexTTS2 Emo-Vector order list. Robust to a
        malformed (hand-editable) stamped vector: a non-dict or non-numeric value
        -> 0.0, every value clamped to 0..1, so an out-of-contract ledger never
        crashes the render or sends bad values to the worker (PD1).

        This is the PRE-ALPHA, PRE-CAP projection. The list that actually
        reaches the worker comes from :meth:`emotion_payload`, which applies the
        effective-mass ceiling on top."""
        from .._otr_delivery_vector import EMOTIONS
        safe = sanitize_delivery_vector(delivery_vector)
        return [safe[e] for e in EMOTIONS]

    def prepare_text(self, text, delivery_vector=None):
        """Engine-neutral clean spoken text; audio direction rides the emo-vector,
        not the words."""
        from .._otr_script_prep import clean_spoken_text
        return clean_spoken_text(text)

    @staticmethod
    def current_emo_alpha() -> float:
        """The active emotion-blend strength (whiny-fix P1.2).

        ``OTR_INDEXTTS2_EMO_ALPHA`` env, clamped to 0..1, default
        :data:`EMO_ALPHA_DEFAULT`. Read per render so a long-running server
        picks up env changes.

        THE DEFAULT WENT 1.0 -> 0.4 -> 1.0, AND THAT IS NOT A ROUND TRIP TO
        WHERE IT STARTED. The voice-identity fix dropped it to 0.4 while alpha
        and the ceiling shared one job; the ceiling now owns that job alone at
        0.56, so alpha returns to 1.0 as a pass-through and the emotion budget
        has exactly one owner. The pre-fix build was alpha 1.0 with NO ceiling,
        which is a different thing entirely -- it spent the vector's whole sum
        and left nothing of the speaker.

        So this is a DIAGNOSTIC override now, not a taste control. Turning it
        down still works and still keys, which is what a control arm needs; it
        is simply not how the shipped blend is set.

        ROUNDED TO THREE DECIMALS, AND THE ROUNDING IS LOAD-BEARING [QA-2].
        ``quantize_params`` keys this value at three decimals
        (``round(v * 1000)``), so an env value of ``0.4001`` would key
        identically to ``0.4`` while RENDERING differently -- the next
        identical line would replay the wrong blend from cache. Clamp first,
        then round, so the value the forward uses and the value the key
        records are the same number by construction.
        """
        raw = os.getenv("OTR_INDEXTTS2_EMO_ALPHA", str(EMO_ALPHA_DEFAULT))
        try:
            a = float(raw)
        except (TypeError, ValueError):
            return EMO_ALPHA_DEFAULT
        if a != a:  # NaN
            return EMO_ALPHA_DEFAULT
        return round(min(1.0, max(0.0, a)), 3)

    @staticmethod
    def current_emo_mass_cap() -> float:
        """The active ceiling on effective emotion mass.

        ``OTR_INDEXTTS2_EMO_MASS_CAP`` env, default
        :data:`EFFECTIVE_EMOTION_MASS_CAP`, clamped to
        ``0 .. EMOTION_MASS_CAP_DISABLED`` and rounded to three decimals for the
        same cache-key reason alpha is.

        THIS IS THE ONE KNOB (2026-08-18). Alpha is pinned at 1.0, so the
        effective mass of every line is decided here and nowhere else. Raise it
        for more emotion and less of the speaker; lower it for the reverse.
        ``OTR_INDEXTTS2_EMO_MASS_CAP=8`` is the disabled sentinel and restores
        pre-fix intensity for a control arm.

        IT FLATTENS TOTAL INTENSITY ON PURPOSE, AND THE OPERATOR CHOSE THAT.
        Across 57 character lines sampled from the six most recent episode
        ledgers, every derived vector summed above 0.56, so in practice every
        line lands on this ceiling and the per-line variation in TOTAL emotion
        budget is gone. What still varies is the vector's SHAPE -- which
        emotions, in what proportion. The previous 0.4 ceiling already pinned
        81% of those lines, so this removes the remainder rather than
        introducing the behaviour. A vector that does sum below the ceiling
        passes through untouched; that is valid and tested, just not typical.

        The review gate's stated risk was that capping would read as flattened
        performance. His ear went the other way and called the UNCAPPED arm
        *"too emotional"* and not *"a real emtion"*, which is why the ceiling is
        the knob that survived.
        """
        raw = os.getenv("OTR_INDEXTTS2_EMO_MASS_CAP",
                        str(EFFECTIVE_EMOTION_MASS_CAP))
        try:
            cap = float(raw)
        except (TypeError, ValueError):
            return EFFECTIVE_EMOTION_MASS_CAP
        if cap != cap:  # NaN
            return EFFECTIVE_EMOTION_MASS_CAP
        return round(min(EMOTION_MASS_CAP_DISABLED, max(0.0, cap)), 3)

    @staticmethod
    def _apply_vendor_alpha(emo_vector, alpha):
        """Mirror of the vendor's OWN post-alpha transform. Read, not assumed.

        ``indextts/infer_v2.py`` pre-scales the emotion vector by alpha itself
        -- emotion vectors cannot be alpha-blended later in its pipeline -- and
        then, because OTR never supplies a separate emotion reference clip,
        forces ``emo_alpha = 1.0`` for the rest of the forward. So alpha's ONE
        effect is this scaling, and the weights below are literally what gets
        summed into ``1 - sum(weight_vector)``.

        Two details are copied deliberately: the scale is only applied when it
        is not exactly 1.0, and the result is TRUNCATED to four decimals
        (``int(x * scale * 10000) / 10000``), never rounded. Measuring an
        idealised ``alpha * sum`` instead of this would describe a blend the
        vendor does not use [QA-3].
        """
        scale = max(0.0, min(1.0, float(alpha)))
        if scale != 1.0:
            return [int(x * scale * 10000) / 10000 for x in emo_vector]
        return [float(x) for x in emo_vector]

    def emotion_payload(self, delivery_vector, alpha=None, mass_cap=None) -> dict:
        """The EXACT emotion arguments this adapter will hand the worker.

        Returns ``{"emo_vector", "emo_alpha", "effective_mass", "mass_capped",
        "vector_state"}``. Pure and deterministic in its inputs, so the voice
        dispatch can call it for the per-line receipt and get the same numbers
        the forward sends -- one resolution, not two that can drift [QA-2].

        THE CEILING IS APPLIED AFTER ALPHA, NEVER BEFORE [QA-3, QA-4 order].
        Capping the raw vector first and letting alpha scale the capped result
        would soften the delivery twice. The order is invisible at the shipped
        alpha of 1.0 -- scaling by 1.0 commutes with everything -- but it is
        still load-bearing for any arm that turns alpha down: at alpha 0.4 a
        stamped ``calm=1.0`` line lands at ``min(0.4, 0.56) == 0.4``, where
        cap-then-alpha would give ``0.56 * 0.4 == 0.224``. So the vendor
        transform runs first, the mass is measured on ITS output, and only an
        overweight result is scaled back.

        WHY A CEILING AND NOT A SMALLER ALPHA. Alpha is a multiplier, so it
        cannot promise anything about a vector it did not derive: a hand-edited
        or pre-stamped ledger summing to 3.0 spends all 3.0 at alpha 1.0 --
        three times the whole speaker. The ceiling is measured on what the
        vendor actually receives, so it holds regardless of where the vector
        came from. That is why it, and not alpha, is the knob that ships.

        THE RETURNED ``effective_mass`` IS THE AUTHORITY, not the arithmetic.
        ``_apply_vendor_alpha`` truncates to four decimals and the rescale
        FLOORS to three, so the result lands at or just under an idealised
        ``min(alpha * sum, cap)`` -- 0.5590 rather than 0.5600 on a real
        emotional line. Anything asserting a mass reads this field.
        """
        from .._otr_delivery_vector import EMOTIONS

        emo_alpha = self.current_emo_alpha() if alpha is None else round(
            min(1.0, max(0.0, float(alpha))), 3)
        cap = self.current_emo_mass_cap() if mass_cap is None else round(
            min(EMOTION_MASS_CAP_DISABLED, max(0.0, float(mass_cap))), 3)
        safe = sanitize_delivery_vector(delivery_vector)
        vector = [safe[e] for e in EMOTIONS]
        state = "omitted" if delivery_vector is None else (
            "nonzero" if any(v > 0.0 for v in vector) else "zero")

        applied = self._apply_vendor_alpha(vector, emo_alpha)
        mass = sum(applied)
        capped = False
        if mass > cap:
            capped = True
            factor = cap / mass
            # FLOOR, not round. Rounding a rescaled weight up is precisely how
            # an enforced 0.4 comes back as 0.401 [QA-3]; flooring can only
            # ever land at or under the ceiling.
            vector = [math.floor(v * factor * 1000) / 1000 for v in vector]

        # MEASURED AFTER SERIALIZATION, on the values the worker will actually
        # parse [QA-3]. The round-trip is what makes "the exact outbound list"
        # a fact rather than an intention; `generate_voice` then serializes
        # these same numbers.
        vector = json.loads(json.dumps(vector))
        applied = self._apply_vendor_alpha(vector, emo_alpha)
        mass = sum(applied)

        # Bounded, deterministic shave. Unreachable with the floor above -- it
        # is here because an unenforced ceiling is a comment, not a ceiling,
        # and because THE LAW forbids raising over it mid-render.
        for _ in range(len(vector)):
            if mass <= cap:
                break
            capped = True
            heaviest = max(range(len(vector)), key=lambda i: (vector[i], -i))
            vector[heaviest] = max(0.0, round(vector[heaviest] - 0.001, 3))
            vector = json.loads(json.dumps(vector))
            applied = self._apply_vendor_alpha(vector, emo_alpha)
            mass = sum(applied)

        return {
            "emo_vector": vector,
            "emo_alpha": emo_alpha,
            "emo_mass_cap": cap,
            "effective_mass": round(mass, 4),
            "mass_capped": capped,
            "vector_state": state,
        }

    def render_time_params(self) -> dict:
        """``emo_alpha`` MUST key -- it is resolved per render, from the env.

        THE DEFECT THIS CLOSES (Lemmy chunk A1). ``current_emo_alpha`` is read
        inside :meth:`generate_voice`, at GENERATE time, while the cache key
        captured ``profile.default_params`` at REQUEST-BUILD time. So exporting
        ``OTR_INDEXTTS2_EMO_ALPHA`` changed the RENDER and not the KEY: the next
        identical line replayed audio made under the previous alpha, and the
        receipt described a blend the clip does not have. ``IS_CHANGED`` carried
        no alpha term either, so an in-graph rerun did not save it.

        Resolved through :meth:`current_emo_alpha` -- the SAME function the
        forward calls -- so the key and the render cannot disagree about the
        value. Per-render env pickup is preserved exactly: a new request is
        built per line per render, so a changed env still takes effect on the
        next render. It now also takes effect on the KEY.
        """
        return {"emo_alpha": self.current_emo_alpha(),
                "emo_mass_cap": self.current_emo_mass_cap()}

    # ---- one dialogue line -> mono AUDIO {"waveform","sample_rate"} ----
    def _resolve_ref(self, ref):
        """Bank ref_paths are relative to the ComfyUI root (e.g.
        ``models/TTS/refs/...``). Resolve to an absolute path the isolated worker
        can open regardless of its own cwd.

        DELEGATES to the ONE shared resolver (Lemmy chunk B). This used to be a
        private copy that tried a single candidate, so it could miss a reference
        the voice node's own broader check had just confirmed exists."""
        from .base import resolve_voice_ref_path
        return resolve_voice_ref_path(ref)

    def generate_voice(self, text, ref_clip_path, delivery_vector, seed):
        self.load()
        ref_clip_path = self._resolve_ref(ref_clip_path)
        out_path = tempfile.mktemp(suffix=".wav", prefix="otr_idx2_")
        # ONE resolution of the emotion arguments, shared with the per-line
        # receipt the dispatch writes [QA-2]. The vector here is already capped
        # to EFFECTIVE_EMOTION_MASS_CAP measured AFTER alpha, so the speaker's
        # own embedding always keeps its share of the blend.
        emotion = self.emotion_payload(delivery_vector)
        req = {
            "text": text,
            "ref_clip": ref_clip_path,
            "emo_vector": emotion["emo_vector"],
            "emo_alpha": emotion["emo_alpha"],
            "seed": int(seed),
            "out_path": out_path,
            "verbose": False,
        }
        self._proc.stdin.write(json.dumps(req, ensure_ascii=True) + "\n")
        self._proc.stdin.flush()
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            self._proc = None
            raise RuntimeError(
                "IndexTTS2 worker closed unexpectedly (see _otr_indextts2_worker.err)")
        resp = json.loads(resp_line)
        if not resp.get("ok"):
            raise RuntimeError("IndexTTS2 render failed: %s" % resp.get("error"))
        return self._load_wav(resp["out_path"], resp.get("sample_rate", self.sample_rate))

    @staticmethod
    def _load_wav(path, sample_rate):
        """Load the worker's WAV into the main venv as an AUDIO dict. Uses
        soundfile, NOT torchaudio.load -- the Blackwell venv's torchaudio routes
        load() through torchcodec, which is not installed here."""
        import soundfile as sf
        import torch
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
        try:
            os.remove(path)
        except OSError:
            pass
        wav = torch.from_numpy(data.T).contiguous()  # [C, T]
        return {"waveform": wav, "sample_rate": int(sr or sample_rate)}
