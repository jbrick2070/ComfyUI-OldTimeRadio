"""LatentSync lipsync adapter -- Path-B isolated cu128 subprocess sidecar (A-S4).

LatentSync (ByteDance) is a lipsync-overlay engine: given a BASE CLIP (from a
provider engine -- a talking-head still/motion clip) plus an AUDIO reference, it
re-renders the mouth region so the lips match the speech. It is a CONSUMER in the
provider-before-consumer ordering: the platform must produce a base clip first,
then latentsync overlays onto it.

Isolation (V-12): LatentSync's UNet is fine-tuned from Stable Diffusion 1.5 and
pins its own diffusers / torch stack that would brick the Blackwell (torch 2.10 /
cu130) ComfyUI venv, so -- exactly like the Path-B audio sidecars -- it runs in
its OWN pinned py3.10 / cu128 venv as a supervised subprocess worker. This
adapter, in ComfyUI's venv, drives it over line-delimited JSON and reads back the
rendered MP4; ZERO shared torch. Default-OFF / dark: it is registered (so it
appears in the static per-role dropdown, V-6) but is NOT a default for any role
and is gated behind OTR_ENABLE_LATENTSYNC; a render fails CLOSED with a NAMED
error until the Path-B worker + venv are installed. The GPU smoke (operator) is
where the live load + the VRAM<=14.5 GB boundary are proven.

Single resident heavy engine: the adapter takes the SHARED cross-process GPU
residency lease (AS-3) before the worker loads weights and releases it on
teardown, so latentsync can never co-reside with another heavy engine.

License (verify-at-build): LatentSync CODE is Apache-2.0 but the WEIGHTS are
OpenRAIL++ (use-restricted, SD1.5-derived) -- treated here as NOT
commercial-clean until the operator confirms per the license matrix.

Config (env, with box defaults under ``ComfyUI/latentsync``):
  ``OTR_LATENTSYNC_VENV``   isolated venv python (``.venv/Scripts/python.exe``)
  ``OTR_LATENTSYNC_WORKER`` worker script (``scripts/_otr_latentsync_worker.py``)
  ``OTR_LATENTSYNC_REPO``   cloned bytedance/LatentSync repo (on the worker path)

Import-time is side-effect-free + cold-import clean (V-12): module scope imports
only the stdlib + the dep-free registry + shared lease/sidecar helpers. diffusers
/ torch / ffmpeg are never imported here -- the worker venv owns them. UTF-8, no
BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess

from .._otr_shared import gpu_residency as _GR
from .._otr_shared import sidecar as _SC
from .registry import EngineUnusable, EngineUsabilityReason, register

log = logging.getLogger("OTR.latentsync")

#: GATE A latentsync-100% fix (2026-06-11; 3D plan section 0). Today latentsync
#: is a ~1/5 face-lottery: only beats whose PROVIDER base clip carries a
#: detectable face land; the rest fall LOUD to the floor. With this env set to a
#: registered still-capable engine (canonically ``still_kenburns``), the adapter
#: synthesizes its OWN base clip from the request's clean cast portrait
#: (asset_refs init image) so EVERY beat has a face to track. Default unset =
#: shipped provider-base behavior, unchanged.
_BASE_ENGINE_ENV = "OTR_LSYNC_BASE_ENGINE"

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))   # ...\ComfyUI-OldTimeRadio
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))              # ...\ComfyUI

#: Machine-wide VRAM ceiling for the single resident heavy engine (A invariant).
_VRAM_CEILING_MB = 14_500


def _default(*parts):
    return os.path.join(_COMFY_ROOT, "latentsync", *parts)


@register
class LatentSyncEngine:
    """The latentsync lipsync-overlay adapter (default-OFF, sidecar-isolated)."""

    name = "latentsync"
    family = "lipsync_overlay"
    # Lipsync needs BOTH a base clip to overlay onto AND the speech audio; only
    # the roles that can supply both (announcer + character) offer it. role_compat
    # excludes it everywhere else (fail-closed) -- never a default (dark).
    roles = ("announcer_visual", "character_video")
    default_roles = ()
    required_inputs = ("base_clip_ref", "audio_ref")
    commercial_clean = False            # OpenRAIL++ weights -- verify-at-build
    requires_flag = "OTR_ENABLE_LATENTSYNC"
    engine_version = "1"
    #: LatentSync resamples audio to 16 kHz internally; the platform canvas fps
    #: is already 25 (its required input fps) so no fps re-time is needed.
    audio_sample_rate = 16_000
    target_fps = 25
    #: Family-degradation next hop -> the zero-VRAM radio floor (A-S6 chain
    #: humo -> latentsync -> still_kenburns; see nodes/_otr_shared/fallback.py).
    fallback_engine = "still_kenburns"

    def __init__(self):
        self._proc = None
        self._stderr = None

    # ---- config resolution (env override -> box default) ----
    def _venv_python(self):
        return os.environ.get("OTR_LATENTSYNC_VENV") or _default(
            ".venv", "Scripts", "python.exe")

    def _worker_script(self):
        return os.environ.get("OTR_LATENTSYNC_WORKER") or os.path.join(
            _REPO_ROOT, "scripts", "_otr_latentsync_worker.py")

    def _installed(self):
        """True iff BOTH the isolated venv python and the worker script exist on
        disk (no import -- cheap, headless-safe)."""
        return os.path.exists(self._venv_python()) and os.path.exists(
            self._worker_script())

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before the first forward. Raises EngineUnusable with a
        classified reason if the opt-in flag is unset (dark) or the Path-B venv /
        worker are not installed; returns the validated name otherwise. Imports
        nothing heavy -- this runs at lock/validate time on the CPU box."""
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "latentsync is opt-in; set %s=1 and install the Path-B venv"
                % self.requires_flag, kind="video")
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "latentsync Path-B not installed: run "
                "scripts\\_otr_latentsync_install.ps1 (isolated cu128 venv)",
                kind="video")
        return self.name

    # ---- worker lifecycle ----
    def load(self):
        """Spawn (or reuse) the isolated worker. Fails CLOSED with a NAMED error
        when the venv / worker are absent -- the dark default-OFF state."""
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._proc is not None:
            _SC.close_worker(self._proc, self._stderr)  # reap an idle-exited worker
            self._proc = None
            self._stderr = None
        py = self._venv_python()
        worker = self._worker_script()
        for label, path in (("isolated venv python", py), ("worker script", worker)):
            if not os.path.exists(path):
                raise RuntimeError(
                    "LatentSync Path B not installed: %s missing at %s -- run "
                    "scripts\\_otr_latentsync_install.ps1 (isolated cu128 venv) "
                    "and set OTR_ENABLE_LATENTSYNC=1 before rendering" % (label, path))
        env = dict(os.environ)
        repo = os.environ.get("OTR_LATENTSYNC_REPO") or _default("repo")
        env["OTR_LATENTSYNC_REPO"] = repo
        err_path = os.path.join(_REPO_ROOT, "_otr_latentsync_worker.err")
        stderr = open(err_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                [py, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=stderr, text=True, encoding="utf-8", bufsize=1, env=env)
        except Exception:
            stderr.close()
            raise
        try:
            line = _SC.read_protocol_line(proc, _SC.startup_timeout(), "LatentSync readiness")
            ready = json.loads(line) if line.strip() else {"ready": False, "error": "no readiness line"}
            if not isinstance(ready, dict):
                raise ValueError("readiness was not a JSON object: %r" % line[:200])
        except (TimeoutError, EOFError, ValueError) as exc:
            # BUG-09.02 belt-and-suspenders: close_worker reaps the process,
            # but if it ever fails partway we still guarantee the kill so a
            # wedged sidecar can never outlive a failed startup.
            try:
                _SC.close_worker(proc, stderr)
            finally:
                if proc.poll() is None:
                    proc.kill()
            raise RuntimeError(
                "LatentSync worker failed to start: %s (see %s)" % (exc, err_path))
        if ready.get("ready") is not True:
            _SC.close_worker(proc, stderr)
            raise RuntimeError(
                "LatentSync worker failed to start: %s (see %s)"
                % (ready.get("error"), err_path))
        self._proc = proc
        self._stderr = stderr

    def unload(self):
        proc, self._proc = self._proc, None
        stderr, self._stderr = self._stderr, None
        _SC.close_worker(proc, stderr)  # always closes stderr, even if proc is None

    # ---- render lifecycle (the per-clip-group contract; CPU-scaffolded) ----
    def prepare(self, host_caps, profile, session_ctx):
        """Take the SHARED single-heavy-engine lease (AS-3) BEFORE the worker
        loads weights, then start the worker. Returns the prepared context the
        render walk + teardown thread through. FAIL CLOSED: a held lease or a
        missing venv raises and no weights load."""
        lease = _GR.acquire(timeout_s=float(os.getenv("OTR_GPU_LEASE_TIMEOUT_S", "120")))
        try:
            self.load()
        except BaseException:
            _GR.release(lease)              # never strand the lease on a load failure
            raise
        return {"engine_id": self.name, "lease": lease}

    def render_clip(self, request, prepared):
        """Drive ONE lipsync overlay in the worker. Sends the base clip + audio
        paths + timing; reads back the worker's MP4 path. Returns the raw worker
        response (canonicalize normalizes it). Stream state is killed + reaped on
        any protocol error so a hung worker never wedges the render thread."""
        self.load()
        from ._tmp import otr_engine_tmp_mp4
        out_path = otr_engine_tmp_mp4("otr_lsync_")
        base_clip, base_source = self._resolve_base_clip(request)
        log.info(
            "LatentSync base clip resolved: source=%s path=%s", base_source, base_clip)
        req = self._build_worker_request(request, out_path, base_clip_override=base_clip)
        try:
            self._proc.stdin.write(json.dumps(req, ensure_ascii=True) + "\n")
            self._proc.stdin.flush()
            resp_line = _SC.read_protocol_line(
                self._proc, _SC.request_timeout(), "LatentSync response")
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
                "LatentSync worker request failed: %s "
                "(see _otr_latentsync_worker.err)" % exc)
        if not resp.get("ok"):
            _SC.remove_quietly(out_path)
            raise RuntimeError("LatentSync render failed: %s" % resp.get("error"))
        return resp

    def canonicalize(self, raw, request, profile):
        """Normalize the worker's MP4 into the platform's ALWAYS-SILENT clip
        contract. The worker already wrote a bt709 / yuv420p / silent MP4 at the
        requested frame count; this records that as the canonical descriptor
        (the ffprobe assert lives in OTR_SilentComposite at compose time)."""
        return self._clip_from_worker(raw, request)

    def teardown(self, prepared):
        """Kill the worker (frees its VRAM), then RELEASE the shared lease, then
        bounded-wait for machine-wide VRAM to fall below the ceiling (absorbs
        Windows teardown latency). Idempotent + never raises out of teardown.
        The boundary settle-wait only runs when a lease was actually held (a
        no-load teardown has nothing resident to wait on)."""
        self.unload()
        lease = (prepared or {}).get("lease")
        had_lease = lease is not None
        _GR.release(lease)
        if had_lease:
            # Boundary check (the GPU smoke proves VRAM<=ceiling; on a CPU box
            # NVML is absent -> wait_until_below_mb returns False, read by the gate).
            _GR.wait_until_below_mb(_VRAM_CEILING_MB, attempts=3, sleep_s=2.0)

    # ---- pure helpers (CPU-testable; no worker, no heavy import) ----
    @staticmethod
    def _ref_path(ref):
        """Pull a filesystem path out of a base_clip_ref / audio_ref that may be
        a bare string OR a mapping carrying a ``path`` key (the schema AudioRef
        shape). Returns "" when nothing path-like is present."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    @staticmethod
    def _portrait_path(request):
        """A clean portrait/still path from the request's asset_refs (the same
        key set the cheap families read: still / init_image / image), falling
        back to a top-level init_image. Returns "" when none present."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            for key in ("still", "init_image", "image"):
                v = assets.get(key)
                if isinstance(v, str) and v:
                    return v
                if isinstance(v, dict) and v.get("path"):
                    return v["path"]
        v = get("init_image")
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict) and v.get("path"):
            return v["path"]
        return ""

    def _resolve_base_clip(self, request):
        """Resolve the base clip latentsync overlays onto. Returns
        ``(path, source_tag)``.

        * ``OTR_LSYNC_BASE_ENGINE`` unset -> the provider base clip from
          ``base_clip_ref`` (shipped behavior, byte-identical).
        * env set (e.g. ``still_kenburns``) -> synthesize the base FROM the
          request's clean cast portrait via that registered engine, so every
          beat has a trackable face (the GATE A 100% fix). Missing portrait
          falls back LOUD to the provider base; missing BOTH raises (the
          driver's LOUD restamp-to-floor path -- never silent).
        """
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        provider_base = self._ref_path(get("base_clip_ref"))
        base_engine_id = (os.getenv(_BASE_ENGINE_ENV, "") or "").strip()
        if not base_engine_id:
            return provider_base, "provider"
        # The render_driver's _provide_lipsync_base seam may have ALREADY
        # rendered a base via this same env (driver-side provider). A present
        # base wins -- this adapter-side synthesis is the in-adapter backstop
        # for paths that bypass the driver seam, never a double render.
        if provider_base and os.path.exists(provider_base):
            return provider_base, "provider"

        portrait = self._portrait_path(request)
        if not portrait or not os.path.exists(portrait):
            if provider_base:
                log.warning(
                    "LatentSync: %s=%s but the request carries no portrait/init "
                    "image (asset_refs still/init_image/image); falling back "
                    "LOUD to the provider base clip %r.",
                    _BASE_ENGINE_ENV, base_engine_id, provider_base)
                return provider_base, "provider_fallback_no_portrait"
            raise RuntimeError(
                "LatentSync: %s=%s requires a portrait/init image on the "
                "request (asset_refs still/init_image/image) and none is "
                "present, and there is no provider base_clip_ref either -- "
                "failing closed (LOUD)." % (_BASE_ENGINE_ENV, base_engine_id))

        from .registry import get_engine  # registry import is dep-free
        try:
            base_eng = get_engine(base_engine_id)
        except KeyError as exc:
            raise RuntimeError(
                "LatentSync: %s=%r is not a registered video engine: %s"
                % (_BASE_ENGINE_ENV, base_engine_id, exc))
        if not callable(getattr(base_eng, "render_clip", None)):
            raise RuntimeError(
                "LatentSync: %s=%r cannot render a base clip (no render_clip)"
                % (_BASE_ENGINE_ENV, base_engine_id))

        base_request = {
            "shot_id": (get("shot_id") or "lsync_base"),
            "canvas": get("canvas") or {},
            "timing": get("timing") or {},
            "text_prompt": get("text_prompt") or "latentsync base portrait",
            "asset_refs": {"init_image": portrait},
        }
        clip = base_eng.render_clip(base_request, None)
        path = (clip or {}).get("path", "") if isinstance(clip, dict) else ""
        if not path or not os.path.exists(path):
            raise RuntimeError(
                "LatentSync: base engine %r did not produce a base clip on "
                "disk (got %r) -- failing closed (LOUD)." % (base_engine_id, path))
        return path, "base_engine:%s" % base_engine_id

    def _build_worker_request(self, request, out_path, base_clip_override=None):
        """Pure: build the line-JSON request the worker consumes from a platform
        VideoRequest-shaped object OR a plain dict. No IO, no heavy import.
        ``base_clip_override`` (from _resolve_base_clip) takes precedence over
        the request's base_clip_ref."""
        get = request.get if isinstance(request, dict) else (lambda k, d=None: getattr(request, k, d))
        base_clip = base_clip_override if base_clip_override is not None \
            else self._ref_path(get("base_clip_ref"))
        audio = self._ref_path(get("audio_ref"))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (lambda k, d=None: getattr(seeds, k, d))
        return {
            "base_clip": base_clip,
            "audio_path": audio,
            "out_path": out_path,
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "audio_sample_rate": int(self.audio_sample_rate),
            "seed": int(s_get("request_seed", 0) or 0),
        }

    def _clip_from_worker(self, resp, request):
        """Pure: shape the worker response into the CanonicalClip dict (silent,
        bt709/yuv420p). frame_count is the integer timing authority."""
        get = request.get if isinstance(request, dict) else (lambda k, d=None: getattr(request, k, d))
        return {
            "clip_id": get("shot_id") or get("request_id") or "latentsync_clip",
            "type": "video",
            "path": resp.get("out_path", ""),
            "container": "mp4",
            "codec": "h264",
            "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(resp.get("frame_count", 0) or 0),
            "has_audio": False,             # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709",
            "transfer": "bt709",
            "matrix": "bt709",
            "engine_id": self.name,
            "family": self.family,
        }


__all__ = ["LatentSyncEngine"]
