"""Kokoro synthesis backends -- torch (the `kokoro` package, KPipeline) and ONNX
(the `kokoro-onnx` package over onnxruntime).

WHY TWO BACKENDS BEHIND ONE ENGINE (operator ruling 2026-09-01, queue item 2 of
docs/GO_FORWARD_PLAN.md): the torch `kokoro` package cannot be pip-installed on
Python 3.13 (PBUG-20260901-04), which is the interpreter ComfyUI Desktop and the
portable ship, while the saved dropdown value is the one string "kokoro" on every
machine. So the ENGINE keeps its name, its voice ids, its ledger contract and its
per_line interface, and only the synthesis call differs:

* ``TorchKokoroBackend`` -- the code that ran before 2026-09-02, moved here
  VERBATIM (one ``KPipeline(text, voice=..., speed=..., split_pattern=r"\\n+")``
  call over the full line). The RTX 5080's 3.12 venv keeps selecting it and its
  output is byte-identical to the pre-change engine; that is proven by sha256,
  not asserted (docs/2026-09-02-kokoro-onnx/5080_torch_baseline_sha256.json).
* ``OnnxKokoroBackend`` -- CPU by design: an 82M model runs six times faster than
  realtime on CPU, the 8 GB tier's GPU is owed to the video engine, and
  onnxruntime-gpu would drag CUDA DLL matching into a voice. The CastLock ledger's
  ``voice_device`` stamp is logged as unused by this backend, once, at load.

ONE VOICE SOURCE. The ONNX backend reuses the ``voices/<id>.pt`` files the boot
prefetch already places (hexgrad/Kokoro-82M): they are float32 tensors of shape
(510, 1, 256), exactly the style table kokoro-onnx indexes by phoneme count. They
are converted once into a digest-named npz beside them (``ensure_voices_npz``);
the ``.pt`` stays the identity the bank and the cache fingerprint hash, and the
npz is derived state.

RULES THIS FILE KEEPS: C-5 -- nothing heavy (torch, numpy, onnxruntime, kokoro)
is imported at module top; the package imports ``eng_kokoro`` at init. C-7 --
nothing here ever networks; a missing model or voice is a NAMED error raised by
the adapter, and the fetch lives in ``_otr_kokoro_voice_prefetch`` at prestartup.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile

log = logging.getLogger("OTR")

SAMPLE_RATE = 24000

#: kokoro-onnx ``create()`` keyword arguments, PINNED so a library default change
#: cannot move the house cadence. ``lang="en-gb"`` mirrors the torch path's British
#: ``lang_code="b"``; ``trim=False`` mirrors the torch path, which never trims; the
#: two pauses only apply when kokoro-onnx has to split a chunk longer than 510
#: phonemes, which the torch path also splits internally.
ONNX_CREATE_KWARGS = {
    "lang": "en-gb",
    "trim": False,
    "sentence_pause": 0.25,
    "clause_pause": 0.1,
}

#: CPU by design (see the module docstring). ``OTR_KOKORO_ONNX_PROVIDERS`` is the
#: only override, for an operator who deliberately installed an accelerated
#: onnxruntime build and wants to use it.
DEFAULT_ONNX_PROVIDERS = ("CPUExecutionProvider",)

#: Intra-op thread cap so a 16-thread session does not fight the video encode
#: running beside it; RTF measured under this cap is ~0.15 on the dev box.
ONNX_THREAD_CAP = 4

TORCH_INSTALL_HINT = "pip install kokoro   (Python 3.12 or earlier)"
ONNX_INSTALL_HINT = "pip install kokoro-onnx   (Python 3.10 to 3.13; onnxruntime comes with it)"

_LINE_SPLIT = re.compile(r"\n+")
_NPZ_PREFIX = "_onnx_voices."
_NPZ_SUFFIX = ".npz"


class BackendUnavailable(RuntimeError):
    """A backend cannot be selected or loaded; the adapter maps this to a named
    ``EngineUnusable`` with the classified reason and the install / fetch line."""


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_backend_name(env_value=None) -> str:
    """``"torch"`` or ``"onnx"``, re-evaluated on every call (imports are cached, the
    choice is cheap, and tests swap ``sys.modules`` / the env between calls).

    ``env_value`` is ``OTR_KOKORO_BACKEND``: ``auto`` (default) prefers the torch
    package when it imports and falls to kokoro-onnx otherwise; ``torch`` / ``onnx``
    force one and fail LOUD by name when it cannot import -- a forced backend never
    falls through. Try-imports, not ``importlib.util.find_spec``: the adapter tests
    fake ``sys.modules["kokoro"]`` with a namespace that has no ``__spec__``, on
    which ``find_spec`` raises.
    """
    mode = str(env_value or "auto").strip().lower() or "auto"
    if mode not in ("auto", "torch", "onnx"):
        raise BackendUnavailable(
            "OTR_KOKORO_BACKEND must be auto, torch or onnx, got %r" % env_value)
    if mode in ("auto", "torch"):
        try:
            from kokoro import KPipeline  # noqa: F401 -- probe only
        except ImportError as exc:
            if mode == "torch":
                raise BackendUnavailable(
                    "OTR_KOKORO_BACKEND=torch but the kokoro package is not "
                    "installed: %s" % TORCH_INSTALL_HINT) from exc
        else:
            return "torch"
    try:
        import kokoro_onnx  # noqa: F401 -- probe only
    except ImportError as exc:
        if mode == "onnx":
            raise BackendUnavailable(
                "OTR_KOKORO_BACKEND=onnx but the kokoro-onnx package is not "
                "installed: %s" % ONNX_INSTALL_HINT) from exc
        raise BackendUnavailable(
            "neither kokoro backend is installed. Install one: %s  -- or --  %s"
            % (TORCH_INSTALL_HINT, ONNX_INSTALL_HINT)) from exc
    return "onnx"


def parse_onnx_providers(env_value=None) -> list:
    """``OTR_KOKORO_ONNX_PROVIDERS`` as a list, or the CPU default when unset.

    An empty or whitespace list RAISES rather than passing ``providers=[]`` to
    onnxruntime, which would mean "every available provider" and silently undo
    the CPU pin.
    """
    if env_value is None:
        return list(DEFAULT_ONNX_PROVIDERS)
    names = [p.strip() for p in str(env_value).split(",")]
    names = [p for p in names if p]
    if not names:
        raise BackendUnavailable(
            "OTR_KOKORO_ONNX_PROVIDERS is set but names no provider; unset it for "
            "the CPU default or name one, e.g. CPUExecutionProvider")
    return names


# --------------------------------------------------------------------------- #
# Voices: one npz derived from the .pt files, named by the digest of the set
# --------------------------------------------------------------------------- #
def _voice_files(voices_dir: str) -> list:
    try:
        names = sorted(n for n in os.listdir(voices_dir) if n.endswith(".pt"))
    except OSError:
        return []
    return [os.path.join(voices_dir, n) for n in names]


def voices_digest(voices_dir: str) -> str:
    """Short digest of the ``.pt`` set (names, sizes, mtimes). A changed set gives
    a new npz FILENAME, never a replace-in-place: ``np.load`` keeps a zip handle
    open for the session's life and Windows refuses ``os.replace`` onto it."""
    h = hashlib.sha1()
    for path in _voice_files(voices_dir):
        st = os.stat(path)
        h.update(("%s:%d:%d\n" % (os.path.basename(path), st.st_size, st.st_mtime_ns))
                 .encode("utf-8"))
    return h.hexdigest()[:16]


def npz_path_for(voices_dir: str, digest: str) -> str:
    return os.path.join(voices_dir, _NPZ_PREFIX + digest + _NPZ_SUFFIX)


def ensure_voices_npz(voices_dir: str) -> str:
    """Return the path of the npz holding every readable ``.pt`` voice, building it
    when the digest-named file does not exist yet.

    Disk-only (C-7). A single corrupt ``.pt`` is skipped and logged -- that voice
    then fails by name at its line, the others still speak. Writes beside the
    voices; when that directory is read-only, writes under the temp dir instead.
    Stale digests beside the voices are removed opportunistically (errors ignored:
    a live session may still hold one).
    """
    files = _voice_files(voices_dir)
    if not files:
        raise BackendUnavailable("no kokoro voice files (*.pt) under %s" % voices_dir)
    digest = voices_digest(voices_dir)
    target = npz_path_for(voices_dir, digest)
    if os.path.exists(target):
        return target
    fallback = npz_path_for(os.path.join(tempfile.gettempdir(), "otr_kokoro_voices"), digest)
    if os.path.exists(fallback):
        return fallback

    import numpy as np
    import torch

    arrays = {}
    for path in files:
        voice_id = os.path.splitext(os.path.basename(path))[0]
        try:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            arr = np.asarray(tensor.numpy() if hasattr(tensor, "numpy") else tensor,
                             dtype=np.float32)
            if arr.ndim != 3 or arr.shape[1:] != (1, 256):
                raise ValueError("unexpected voice shape %r (want (N, 1, 256))" % (arr.shape,))
            arrays[voice_id] = arr
        except Exception as exc:  # noqa: BLE001 -- one bad voice is not fatal
            log.warning("[OTR.kokoro] voice %s skipped for the ONNX table: %s", voice_id, exc)
    if not arrays:
        raise BackendUnavailable("no readable kokoro voice files under %s" % voices_dir)

    written = _write_npz(target, arrays)
    if written is None:
        os.makedirs(os.path.dirname(fallback), exist_ok=True)
        written = _write_npz(fallback, arrays)
        if written is None:
            raise BackendUnavailable(
                "cannot write the kokoro ONNX voice table beside %s or under %s"
                % (voices_dir, os.path.dirname(fallback)))
    else:
        _remove_stale_npz(voices_dir, keep=target)
    log.info("[OTR.kokoro] ONNX voice table built: %d voices -> %s", len(arrays), written)
    return written


def _write_npz(target: str, arrays: dict):
    """Write via a unique temp name then rename onto a path nothing holds yet.
    Returns the path, or None when the directory refuses the write."""
    import numpy as np

    tmp = "%s.%d.tmp" % (target, os.getpid())
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, target)
        return target
    except OSError as exc:
        log.info("[OTR.kokoro] cannot write %s (%s)", target, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None


def _remove_stale_npz(voices_dir: str, keep: str) -> None:
    try:
        for name in os.listdir(voices_dir):
            if name.startswith(_NPZ_PREFIX) and name.endswith(_NPZ_SUFFIX):
                path = os.path.join(voices_dir, name)
                if os.path.abspath(path) != os.path.abspath(keep):
                    try:
                        os.remove(path)
                    except OSError:
                        pass                     # a live session still holds it
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class TorchKokoroBackend:
    """The pre-2026-09-02 synthesis path, moved verbatim.

    ``load`` builds ``KPipeline`` exactly as the engine did: lang_code 'b'
    (British), the EXPLICIT device from the CastLock ledger stamp (S4 -- a device
    the host cannot provide fails LOUD in KPipeline, never a silent downgrade),
    ``repo_id`` only when this kokoro build accepts it (0.7.x does not).
    ``synthesize`` is ONE pipeline call over the full line with
    ``split_pattern=r"\\n+"`` -- pre-splitting would change the call shape and the
    bytes.
    """

    name = "torch"

    def __init__(self, device: str):
        self.device = device
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        import inspect

        from kokoro import KPipeline

        kwargs = {"lang_code": "b", "device": self.device}
        try:
            if "repo_id" in inspect.signature(KPipeline.__init__).parameters:
                kwargs["repo_id"] = "hexgrad/Kokoro-82M"
        except (TypeError, ValueError):
            kwargs["repo_id"] = "hexgrad/Kokoro-82M"
        self._pipeline = KPipeline(**kwargs)

    def synthesize(self, text: str, voice_id: str, speed: float):
        import numpy as np
        import torch

        segments = []
        for _, _, audio_data in self._pipeline(
            text, voice=voice_id, speed=speed, split_pattern=r"\n+",
        ):
            if torch.is_tensor(audio_data):
                arr = audio_data.detach().cpu().numpy()
            else:
                arr = np.asarray(audio_data, dtype=np.float32)
            segments.append(arr.astype(np.float32).squeeze())
        if not segments:
            raise RuntimeError("kokoro pipeline produced no audio")
        return np.concatenate(segments) if len(segments) > 1 else segments[0]

    def close(self) -> None:
        pipe, self._pipeline = self._pipeline, None
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


class OnnxKokoroBackend:
    """kokoro-onnx over an onnxruntime session the pack builds itself.

    The session is built HERE with an explicit provider list (CPU by default),
    never through kokoro-onnx's own "every available provider" resolution, so a
    box that happens to carry onnxruntime-gpu from another pack does not try
    unqualified CUDA DLLs for a voice. Voices are passed BY NAME from the npz
    ``ensure_voices_npz`` derived from the ``.pt`` files.
    """

    name = "onnx"

    def __init__(self, model_path: str, voices_npz: str, providers=None, threads=None):
        self.model_path = model_path
        self.voices_npz = voices_npz
        self.providers = list(providers or DEFAULT_ONNX_PROVIDERS)
        self.threads = int(threads or min(ONNX_THREAD_CAP, os.cpu_count() or 1))
        self.providers_active: list = []
        self._session = None
        self._kokoro = None

    def load(self) -> None:
        if self._kokoro is not None:
            return
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        available = list(ort.get_available_providers())
        unknown = [p for p in self.providers if p not in available]
        if unknown:
            raise BackendUnavailable(
                "onnxruntime provider(s) %s are not available in this build "
                "(available: %s); unset OTR_KOKORO_ONNX_PROVIDERS for the CPU default"
                % (unknown, available))
        # phonemizer logs "words count mismatch on 100.0% of the lines" at WARNING
        # for every line whose phoneme count differs from its word count -- which is
        # every line kokoro-onnx hands it, by design of its punctuation handling.
        # It is not a defect and it would drown the render log (clean-logs rule).
        logging.getLogger("phonemizer").setLevel(logging.ERROR)
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        self._session = ort.InferenceSession(
            self.model_path, sess_options=options, providers=self.providers)
        self._kokoro = Kokoro.from_session(self._session, self.voices_npz)
        self.providers_active = list(self._session.get_providers())

    def voice_ids(self) -> list:
        return list(self._kokoro.get_voices()) if self._kokoro is not None else []

    def synthesize(self, text: str, voice_id: str, speed: float):
        import numpy as np

        if self._kokoro is None:
            raise RuntimeError("ONNX backend not loaded")
        if voice_id not in self.voice_ids():
            raise BackendUnavailable(
                "voice %r is not in the ONNX voice table %s (its .pt file was missing "
                "or unreadable when the table was built)" % (voice_id, self.voices_npz))
        segments = []
        for chunk in _LINE_SPLIT.split(text or ""):
            chunk = chunk.strip()
            if not chunk:
                continue
            samples, rate = self._kokoro.create(
                chunk, voice=voice_id, speed=speed, **ONNX_CREATE_KWARGS)
            if int(rate) != SAMPLE_RATE:
                raise RuntimeError(
                    "kokoro-onnx returned %r Hz, expected %d" % (rate, SAMPLE_RATE))
            segments.append(np.asarray(samples, dtype=np.float32).squeeze())
        if not segments:
            raise RuntimeError("kokoro-onnx produced no audio")
        return np.concatenate(segments) if len(segments) > 1 else segments[0]

    def close(self) -> None:
        # onnxruntime's InferenceSession has no close(); dropping the references
        # is the unload. The npz handle kokoro-onnx holds goes with it.
        self._kokoro = None
        self._session = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001 -- teardown must never raise
            pass
