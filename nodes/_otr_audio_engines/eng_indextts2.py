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
    # tuple): a clone engine needs a per-character reference WAV; a char_voice
    # line with no usable ref renders on bark so the episode always renders (PD1).
    requires_voice_ref = True
    voice_ref_kind = "wav_path"
    missing_ref_fallback = "bark"

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
        crashes the render or sends bad values to the worker (PD1)."""
        from .._otr_delivery_vector import EMOTIONS
        dv = delivery_vector if isinstance(delivery_vector, dict) else {}
        out = []
        for e in EMOTIONS:
            try:
                val = float(dv.get(e, 0.0))
            except (TypeError, ValueError):
                val = 0.0
            if val != val:  # NaN
                val = 0.0
            out.append(round(min(1.0, max(0.0, val)), 3))
        return out

    def prepare_text(self, text, delivery_vector=None):
        """Engine-neutral clean spoken text; audio direction rides the emo-vector,
        not the words."""
        from .._otr_script_prep import clean_spoken_text
        return clean_spoken_text(text)

    @staticmethod
    def current_emo_alpha() -> float:
        """The active emotion-blend strength (whiny-fix P1.2).

        ``OTR_INDEXTTS2_EMO_ALPHA`` env, clamped to 0..1, default 1.0 (today's
        shipped behavior). The worker already accepts a per-request value (G7)
        -- this is the cheapest dominant lever; the default is re-anchored from
        the operator's P0 audition matrix when it lands. Read per render so a
        long-running server picks up env changes."""
        raw = os.getenv("OTR_INDEXTTS2_EMO_ALPHA", "1.0")
        try:
            a = float(raw)
        except (TypeError, ValueError):
            return 1.0
        if a != a:  # NaN
            return 1.0
        return min(1.0, max(0.0, a))

    # ---- one dialogue line -> mono AUDIO {"waveform","sample_rate"} ----
    def _resolve_ref(self, ref):
        """Bank ref_paths are relative to the ComfyUI root (e.g.
        ``models/TTS/refs/...``). Resolve to an absolute path the isolated worker
        can open regardless of its own cwd."""
        if not ref or os.path.isabs(ref):
            return ref
        try:
            import folder_paths
            cand = os.path.join(os.path.dirname(folder_paths.models_dir), ref)
            if os.path.exists(cand):
                return cand
        except Exception:  # noqa: BLE001 -- non-Comfy contexts (tests / CLI)
            pass
        return os.path.abspath(ref)

    def generate_voice(self, text, ref_clip_path, delivery_vector, seed):
        self.load()
        ref_clip_path = self._resolve_ref(ref_clip_path)
        out_path = tempfile.mktemp(suffix=".wav", prefix="otr_idx2_")
        req = {
            "text": text,
            "ref_clip": ref_clip_path,
            "emo_vector": self.emo_list(delivery_vector),
            "emo_alpha": self.current_emo_alpha(),
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
