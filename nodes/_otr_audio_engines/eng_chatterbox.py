"""Chatterbox voice adapter -- Path B isolated subprocess worker (opt-in, MIT).

Chatterbox (Resemble AI, MIT -> commercial-clean ENGINE) pins its own torch /
numpy that would brick the Blackwell (torch 2.10 / cu130) ComfyUI venv, so --
exactly like IndexTTS2 -- it runs in its OWN isolated venv as a supervised
subprocess worker. This adapter, in ComfyUI's venv, drives it over line-delimited
JSON and reads back the rendered WAV; ZERO shared torch. Opt-in behind
OTR_ENABLE_CHATTERBOX; a render fails CLOSED with a NAMED error until the Path B
worker + venv are installed (C-7). ``interface == "per_line"``.

Subprocess lifecycle (bounded read + idempotent teardown) lives in
``_otr_sidecar`` so a hung worker cannot block the render thread forever and a
failed start / mid-request crash never leaks the stderr handle or a zombie.

NOTE: every Chatterbox output carries Resemble AI's imperceptible PerTh watermark.

Config (env, with box defaults under ``ComfyUI/chatterbox``):
  ``OTR_CHATTERBOX_VENV``   isolated venv python (``.venv/Scripts/python.exe``)
  ``OTR_CHATTERBOX_WORKER`` worker script (``scripts/_otr_chatterbox_worker.py``)

Import-time is side-effect-free (C-5). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile

from . import _otr_sidecar as _SC
from .registry import register

# realpath (NOT abspath): the custom node is reached through a directory junction
# under the Desktop-v2 install, so abspath(__file__) would point _COMFY_ROOT at the
# install root instead of the REAL ComfyUI tree where the chatterbox box-default
# venv/weights live. realpath resolves the junction; no-op on a non-junction
# checkout; env overrides still win. (Same junction fix as eng_indextts2.)
_THIS = os.path.realpath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))   # ...\ComfyUI-OldTimeRadio
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))              # ...\ComfyUI


def _default(*parts):
    return os.path.join(_COMFY_ROOT, "chatterbox", *parts)


@register
class ChatterboxEngine:
    name = "chatterbox"
    roles = ("char_voice", "announcer_voice")
    default_roles = ()
    commercial_clean = True  # MIT
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    interface = "per_line"
    # FIXED 24000 (NOT a dynamic m.sr): _render_per_line packs at the profile
    # rate and pack_audio_batch raises on any clip whose rate differs.
    sample_rate = 24000
    # Model-agnostic dispatch metadata (clone engine -> needs a ref WAV). NO-
    # FALLBACK (operator 2026-07-03): a ref-less char_voice line FAILS LOUD (named
    # EngineUnusable in the dispatch) -- it never silently renders on bark.
    requires_voice_ref = True
    voice_ref_kind = "wav_path"
    missing_ref_fallback = None

    def __init__(self):
        self._proc = None
        self._stderr = None

    # ---- config resolution (env override -> box default) ----
    def _venv_python(self):
        return os.environ.get("OTR_CHATTERBOX_VENV") or _default(".venv", "Scripts", "python.exe")

    def _worker_script(self):
        return os.environ.get("OTR_CHATTERBOX_WORKER") or os.path.join(
            _REPO_ROOT, "scripts", "_otr_chatterbox_worker.py")

    # ---- worker lifecycle ----
    def load(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._proc is not None:
            # a previously-spawned worker exited while idle -> reap it (close its
            # pipes + stderr) before replacing, so a respawn never leaks handles.
            _SC.close_worker(self._proc, self._stderr)
            self._proc = None
            self._stderr = None
        py = self._venv_python()
        worker = self._worker_script()
        for label, path in (("isolated venv python", py),
                            ("worker script", worker)):
            if not os.path.exists(path):
                raise RuntimeError(
                    "Chatterbox Path B not installed: %s missing at %s -- run "
                    "scripts\\_otr_chatterbox_install.ps1 (isolated venv) and set "
                    "OTR_ENABLE_CHATTERBOX=1 before rendering with chatterbox" % (label, path))
        err_path = os.path.join(_REPO_ROOT, "_otr_chatterbox_worker.err")
        stderr = open(err_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                [py, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, text=True, encoding="utf-8", bufsize=1)
        except Exception:
            stderr.close()
            raise
        try:
            line = _SC.read_protocol_line(proc, _SC.startup_timeout(), "Chatterbox readiness")
            ready = json.loads(line) if line.strip() else {"ready": False, "error": "no readiness line"}
            if not isinstance(ready, dict):
                raise ValueError("readiness was not a JSON object: %r" % line[:200])
        except (TimeoutError, EOFError, ValueError) as exc:
            _SC.close_worker(proc, stderr)
            raise RuntimeError(
                "Chatterbox worker failed to start: %s (see %s)" % (exc, err_path))
        if ready.get("ready") is not True:
            _SC.close_worker(proc, stderr)
            raise RuntimeError(
                "Chatterbox worker failed to start: %s (see %s)"
                % (ready.get("error"), err_path))
        self._proc = proc
        self._stderr = stderr

    def unload(self):
        proc, self._proc = self._proc, None
        stderr, self._stderr = self._stderr, None
        _SC.close_worker(proc, stderr)  # always closes stderr, even if proc is None

    # ---- text + ref (adapter-side; sent to the worker) ----
    def prepare_text(self, text, delivery_vector=None):
        """Engine-neutral clean spoken text. Audio direction rides the
        exaggeration scalar, not the words."""
        from .._otr_script_prep import clean_spoken_text
        return clean_spoken_text(text)

    def _resolve_ref(self, ref):
        """Resolve a bank ref_path (relative to the ComfyUI root) to an absolute
        path the isolated worker can open regardless of its own cwd."""
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

    @staticmethod
    def _project(delivery_vector):
        """Collapse the 8-dim delivery vector to Chatterbox's exaggeration. Robust
        to a malformed stamped vector: a non-dict / non-numeric / missing 'calm'
        -> neutral 0.5; calm clamped to 0..1 (never crashes the render, PD1)."""
        if not isinstance(delivery_vector, dict) or not delivery_vector:
            return 0.5
        try:
            calm = float(delivery_vector.get("calm", 0.5))
        except (TypeError, ValueError):
            calm = 0.5
        if calm != calm:  # NaN
            calm = 0.5
        calm = min(1.0, max(0.0, calm))
        return round(min(1.0, 0.3 + 0.7 * max(0.0, 1.0 - calm)), 3)

    # ---- one dialogue line -> mono AUDIO {"waveform","sample_rate"} ----
    def generate_voice(self, text, ref_clip_path, delivery_vector, seed):
        self.load()
        ref_clip_path = self._resolve_ref(ref_clip_path)
        out_path = tempfile.mktemp(suffix=".wav", prefix="otr_cbx_")
        req = {
            "text": text,
            "ref_clip": ref_clip_path,
            "exaggeration": self._project(delivery_vector),
            "cfg_weight": 0.4,
            "temperature": 0.6,
            "seed": int(seed),
            "out_path": out_path,
            "verbose": False,
        }
        try:
            self._proc.stdin.write(json.dumps(req, ensure_ascii=True) + "\n")
            self._proc.stdin.flush()
            resp_line = _SC.read_protocol_line(
                self._proc, _SC.request_timeout(), "Chatterbox response")
            resp = json.loads(resp_line)
            if not isinstance(resp, dict):
                raise ValueError("response was not a JSON object: %r" % resp_line[:200])
            if resp.get("ok") is True and not resp.get("out_path"):
                raise ValueError("response missing out_path: %r" % resp_line[:200])
        except (TimeoutError, EOFError, ValueError, OSError) as exc:
            # Stream state is no longer trustworthy -> kill + reap + fail closed.
            _SC.close_worker(self._proc, self._stderr)
            self._proc = None
            self._stderr = None
            _SC.remove_quietly(out_path)
            raise RuntimeError(
                "Chatterbox worker request failed: %s "
                "(see _otr_chatterbox_worker.err)" % exc)
        if not resp.get("ok"):
            _SC.remove_quietly(out_path)
            raise RuntimeError("Chatterbox render failed: %s" % resp.get("error"))
        try:
            return self._load_wav(resp["out_path"], resp.get("sample_rate", self.sample_rate))
        except Exception as exc:  # noqa: BLE001 -- corrupt ok:true output -> named C-7 error
            raise RuntimeError("Chatterbox worker output load failed: %s" % exc)

    @staticmethod
    def _load_wav(path, sample_rate):
        """Load the worker's WAV into the main venv as an AUDIO dict. soundfile,
        NOT torchaudio.load -- the Blackwell venv routes load() through the
        uninstalled torchcodec backend."""
        import soundfile as sf
        import torch
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
        finally:
            _SC.remove_quietly(path)
        wav = torch.from_numpy(data.T).contiguous()  # [C, T]
        return {"waveform": wav, "sample_rate": int(sr or sample_rate)}
