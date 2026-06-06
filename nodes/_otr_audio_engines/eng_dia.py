"""Dia voice adapter -- Path B isolated subprocess worker (opt-in, Apache-2.0).

Dia (Nari Labs, ``nari-labs/Dia-1.6B-0626``) is Apache-2.0 -> a COMMERCIAL-CLEAN
engine, which (unlike the bilibili-licensed IndexTTS2 default) is safe to ship in
Jeffrey's films. It is dialogue-native and zero-shot, so it reuses the existing CC0
reference bank. On Blackwell (RTX 5080, sm_120) it needs torch 2.8 nightly cu128
(Nari issue #26) which would conflict with the main torch-2.10/cu130 venv -- so it
runs in its OWN isolated venv as a supervised subprocess worker, driven over
line-delimited JSON; ZERO shared torch. Opt-in behind OTR_ENABLE_DIA; a render
fails CLOSED with a NAMED error until the Path B worker + venv are installed (C-7).
``interface == "per_line"``; char_voice only this pass (no Dia announcer yet).

Subprocess lifecycle (bounded read + idempotent teardown) lives in ``_otr_sidecar``.

Voice cloning: Dia's best clone prepends the TRANSCRIPT of the reference clip. The
CC0 refs ship without transcripts, so the official path here is audio_prompt-only.
If ``config/dia_ref_transcripts.json`` exists (keyed by reference WAV basename),
the adapter sends the matching transcript and the worker prepends it -- a drop-in
quality upgrade with no code change.

Config (env, with box defaults under ``ComfyUI/dia``):
  ``OTR_DIA_VENV``   isolated venv python (``.venv/Scripts/python.exe``)
  ``OTR_DIA_WORKER`` worker script (``scripts/_otr_dia_worker.py``)
  ``OTR_DIA_MODEL``  HF model id (default ``nari-labs/Dia-1.6B-0626``)

Import-time is side-effect-free (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from . import _otr_sidecar as _SC
from .registry import register

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))   # ...\ComfyUI-OldTimeRadio
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))              # ...\ComfyUI
_DEFAULT_MODEL = "nari-labs/Dia-1.6B-0626"


def _default(*parts):
    return os.path.join(_COMFY_ROOT, "dia", *parts)


@register
class DiaEngine:
    name = "dia"
    roles = ("char_voice",)
    default_roles = ()
    commercial_clean = True  # Apache-2.0
    requires_flag = "OTR_ENABLE_DIA"
    interface = "per_line"
    sample_rate = 44100      # Descript Audio Codec; must match char_dia_v1
    requires_voice_ref = True
    voice_ref_kind = "wav_path"
    missing_ref_fallback = "bark"

    def __init__(self):
        self._proc = None
        self._stderr = None
        self._transcripts = None  # lazily-loaded basename -> transcript map

    # ---- config resolution (env override -> box default) ----
    def _venv_python(self):
        return os.environ.get("OTR_DIA_VENV") or _default(".venv", "Scripts", "python.exe")

    def _worker_script(self):
        return os.environ.get("OTR_DIA_WORKER") or os.path.join(
            _REPO_ROOT, "scripts", "_otr_dia_worker.py")

    def _model_id(self):
        return os.environ.get("OTR_DIA_MODEL") or _DEFAULT_MODEL

    # ---- worker lifecycle ----
    def load(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._proc is not None:
            # a previously-spawned worker exited while idle -> reap it before
            # replacing, so a respawn never leaks the old pipes / stderr handle.
            _SC.close_worker(self._proc, self._stderr)
            self._proc = None
            self._stderr = None
        py = self._venv_python()
        worker = self._worker_script()
        for label, path in (("isolated venv python", py),
                            ("worker script", worker)):
            if not os.path.exists(path):
                raise RuntimeError(
                    "Dia Path B not installed: %s missing at %s -- run "
                    "scripts\\_otr_dia_install.ps1 (isolated venv + weights) and set "
                    "OTR_ENABLE_DIA=1 before rendering with dia" % (label, path))
        err_path = os.path.join(_REPO_ROOT, "_otr_dia_worker.err")
        stderr = open(err_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                [py, worker, "--model", self._model_id()], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=stderr, text=True, encoding="utf-8",
                bufsize=1)
        except Exception:
            stderr.close()
            raise
        try:
            line = _SC.read_protocol_line(proc, _SC.startup_timeout(), "Dia readiness")
            ready = json.loads(line) if line.strip() else {"ready": False, "error": "no readiness line"}
            if not isinstance(ready, dict):
                raise ValueError("readiness was not a JSON object: %r" % line[:200])
        except (TimeoutError, EOFError, ValueError) as exc:
            _SC.close_worker(proc, stderr)
            raise RuntimeError("Dia worker failed to start: %s (see %s)" % (exc, err_path))
        if ready.get("ready") is not True:
            _SC.close_worker(proc, stderr)
            raise RuntimeError(
                "Dia worker failed to start: %s (see %s)" % (ready.get("error"), err_path))
        self._proc = proc
        self._stderr = stderr

    def unload(self):
        proc, self._proc = self._proc, None
        stderr, self._stderr = self._stderr, None
        _SC.close_worker(proc, stderr)

    # ---- text + ref + (optional) clone transcript (adapter-side) ----
    def prepare_text(self, text, delivery_vector=None):
        """Engine-neutral clean spoken text. The worker adds Dia's [S1] speaker
        tag; non-verbals like (laughs) are left to the writer/script prep."""
        from .._otr_script_prep import clean_spoken_text
        return clean_spoken_text(text)

    def _resolve_ref(self, ref):
        if not ref or os.path.isabs(ref):
            return ref
        try:
            import folder_paths
            cand = os.path.join(os.path.dirname(folder_paths.models_dir), ref)
            if os.path.exists(cand):
                return cand
        except Exception:  # noqa: BLE001
            pass
        return os.path.abspath(ref)

    def _resolve_transcript(self, ref_clip_path):
        """Optional clone transcript for this reference WAV, or "".

        ``generate_voice`` receives only the ref PATH (not voice_ref_id), so the
        map ``config/dia_ref_transcripts.json`` is keyed by WAV BASENAME. Absent
        file / key -> "" (audio_prompt-only clone, the official path this pass)."""
        if self._transcripts is None:
            self._transcripts = {}
            path = os.path.join(_REPO_ROOT, "config", "dia_ref_transcripts.json")
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self._transcripts = {str(k): str(v) for k, v in data.items()}
            except Exception:  # noqa: BLE001 -- optional file; absent = no transcripts
                self._transcripts = {}
        if not ref_clip_path:
            return ""
        return self._transcripts.get(os.path.basename(ref_clip_path), "")

    # ---- one dialogue line -> mono AUDIO {"waveform","sample_rate"} ----
    def generate_voice(self, text, ref_clip_path, delivery_vector, seed):
        self.load()
        ref_clip_path = self._resolve_ref(ref_clip_path)
        out_path = tempfile.mktemp(suffix=".wav", prefix="otr_dia_")
        req = {
            "text": text,
            "ref_clip": ref_clip_path,
            "ref_transcript": self._resolve_transcript(ref_clip_path),
            "seed": int(seed),
            "out_path": out_path,
            "verbose": False,
        }
        try:
            self._proc.stdin.write(json.dumps(req, ensure_ascii=True) + "\n")
            self._proc.stdin.flush()
            resp_line = _SC.read_protocol_line(
                self._proc, _SC.request_timeout(), "Dia response")
            resp = json.loads(resp_line)
            if not isinstance(resp, dict):
                raise ValueError("response was not a JSON object: %r" % resp_line[:200])
            if resp.get("ok") is True and not resp.get("out_path"):
                raise ValueError("response missing out_path: %r" % resp_line[:200])
        except (TimeoutError, EOFError, ValueError, OSError) as exc:
            _SC.close_worker(self._proc, self._stderr)
            self._proc = None
            self._stderr = None
            _SC.remove_quietly(out_path)
            raise RuntimeError(
                "Dia worker request failed: %s (see _otr_dia_worker.err)" % exc)
        if not resp.get("ok"):
            _SC.remove_quietly(out_path)
            raise RuntimeError("Dia render failed: %s" % resp.get("error"))
        try:
            return self._load_wav(resp["out_path"], resp.get("sample_rate", self.sample_rate))
        except Exception as exc:  # noqa: BLE001 -- corrupt ok:true output -> named C-7 error
            raise RuntimeError("Dia worker output load failed: %s" % exc)

    @staticmethod
    def _load_wav(path, sample_rate):
        """Load the worker's WAV into the main venv as an AUDIO dict (soundfile,
        not torchaudio.load)."""
        import soundfile as sf
        import torch
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
        finally:
            _SC.remove_quietly(path)
        wav = torch.from_numpy(data.T).contiguous()  # [C, T]
        return {"waveform": wav, "sample_rate": int(sr or sample_rate)}
