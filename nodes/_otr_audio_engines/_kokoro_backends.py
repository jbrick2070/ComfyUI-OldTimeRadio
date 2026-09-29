"""Kokoro synthesis backends -- torch (the `kokoro` package, KPipeline) and ONNX
(the `kokoro-onnx` package over onnxruntime).

WHY TWO BACKENDS BEHIND ONE ENGINE (operator ruling 2026-09-01, queue item 2 of
apple/GO_FORWARD_PLAN.md): the torch `kokoro` package cannot be pip-installed on
Python 3.13 (PBUG-20260901-04), which is the interpreter ComfyUI Desktop and the
portable ship, while the saved dropdown value is the one string "kokoro" on every
machine. So the ENGINE keeps its name, its voice ids, its ledger contract and its
per_line interface, and only the synthesis call differs:

* ``TorchKokoroBackend`` -- the code that ran before 2026-09-02, moved here
  VERBATIM (one ``KPipeline(text, voice=..., speed=..., split_pattern=r"\\n+")``
  call over the full line). The RTX 5080's 3.12 venv keeps selecting it and its
  output is byte-identical to the pre-change engine; that is proven by sha256,
  not asserted (a 5080 torch-baseline sha256 receipt from 2026-09-02).
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

THE ESPEAK LANGUAGES ON ONNX (2026-09-28). The torch package voices Spanish,
French, Hindi, Italian and Brazilian Portuguese through misaki's ``EspeakG2P``:
espeak-ng phonemes, tied affricates and diphthongs folded into Kokoro's own
symbols. misaki does not install on Python 3.13, but phonemizer and espeak-ng
come with kokoro-onnx, so ``EspeakPhonemizer`` repeats that G2P here and the
ONNX backend hands kokoro-onnx the phonemes. Measured the day it was written:
identical phoneme strings to misaki 0.9.4 on eleven sentences across the five
languages, and the same strings from the Desktop's Python 3.13.

MANDARIN ON ONNX (2026-09-28). misaki's Chinese G2P is Python over jieba,
pypinyin, cn2an and ordered-set, all of which install on 3.13; only misaki's own
package pin refuses. The pack carries a copy of it (``_misaki``, provenance
hashed) and ``MandarinPhonemizer`` feeds its phonemes to kokoro-onnx; the four
libraries are an opt-in install for the Mandarin row (``OWN_G2P_INSTALL_HINT``).

JAPANESE ON ONNX (the same night). ``JAG2P()`` defaults to misaki's Cutlet
route: fugashi (MeCab) over a UniDic dictionary, jaconv, a word list and
misaki's number-to-kana code. All of it installs on 3.13 except mojimoji, whose
two calls follow NFKC; measured over every code point, one is then a no-op and
the other a three-character table, which the pack's copy uses instead. The copy
also pins unidic-lite, so a `unidic` package without its dictionary (the
PBUG-20260918-06 trap) cannot take MeCab over. ``JapanesePhonemizer`` feeds its
phonemes to kokoro-onnx; fugashi, jaconv and unidic-lite are the opt-in install.

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
import unicodedata

log = logging.getLogger("OTR")

SAMPLE_RATE = 24000

#: kokoro-onnx ``create()`` keyword arguments, PINNED so a library default change
#: cannot move the house cadence. ``lang="en-gb"`` mirrors the torch path's British
#: ``lang_code="b"`` and applies to English lines only: an espeak row replaces it
#: with its own language and hands over phonemes. ``trim=False`` mirrors the torch
#: path, which never trims; the two pauses only apply when kokoro-onnx has to split
#: a chunk longer than 510 phonemes, which the torch path also splits internally.
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

#: Seconds of silence for a line with nothing the model can say: what the torch
#: pipeline gives the same line (measured by Sonnet QA of 503b2661).
UNSPEAKABLE_LINE_S = 0.25

#: Kokoro's lang_code -> the espeak-ng language its torch pipeline phonemizes
#: with (``kokoro.pipeline.LANG_CODES`` in kokoro 0.9.4). English has its own
#: G2P on each backend, and Japanese and Mandarin use misaki's, so none of the
#: three is here: this is exactly the set the ONNX backend voices through
#: ``EspeakPhonemizer``.
ESPEAK_LANGUAGES = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}

#: The rows the ONNX backend phonemizes with the pack's own copy of misaki
#: (``_misaki``), because misaki does not install on Python 3.13: lang_code ->
#: the modules that copy imports, and the pip line that installs them. Opt-in
#: per language, like the torch path's misaki extras; never an English tax.
OWN_G2P_MODULES = {"z": ("jieba", "pypinyin", "cn2an", "ordered_set"),
                   "j": ("fugashi", "jaconv", "unidic_lite")}
OWN_G2P_INSTALL_HINT = {"z": "pip install jieba pypinyin cn2an ordered-set",
                        "j": "pip install fugashi jaconv unidic-lite"}

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
    fallback = npz_path_for(_fallback_voices_dir(), digest)
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


def _otr_scratch_dir() -> str:
    """OTR's own scratch tier (``<output>/otr/episodes/_shared/tmp``), which the
    janitor sweeps -- never the ambient system TEMP, which nothing sweeps
    (tests/test_node_temp_hygiene.py)."""
    try:
        from .._otr_paths import otr_shared_tmp_dir
    except ImportError:                       # imported straight from nodes/
        from _otr_paths import otr_shared_tmp_dir  # type: ignore
    return str(otr_shared_tmp_dir())


def _fallback_voices_dir() -> str:
    """Where the voice table goes when the voices folder refuses the write."""
    return os.path.join(_otr_scratch_dir(), "kokoro_voices")


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
# The torch pipeline's espeak G2P, for the ONNX backend
# --------------------------------------------------------------------------- #
class EspeakPhonemizer:
    """misaki 0.9.4 ``EspeakG2P`` at its default version, repeated so the ONNX
    backend phonemizes a non-English line exactly as the torch pipeline does.

    The same espeak settings (stress marks, punctuation kept, ``^`` ties,
    language-switch flags removed), the same folds of tied pairs into Kokoro's
    single symbols, and the same bracket handling. espeak-ng is the copy
    kokoro-onnx loads (``espeakng_loader``), and ``Kokoro.from_session`` points
    phonemizer at it, so this is built after the session. The ``phonemizer``
    logger is passed in so the ERROR level ``OnnxKokoroBackend.load`` sets is
    kept; left to its default, phonemizer resets it to WARNING. Adapted from
    misaki (hexgrad, Apache-2.0).
    """

    #: Tied espeak pairs -> Kokoro's single symbols, in misaki's (sorted) order.
    FOLDS = tuple(sorted({
        "a^\u026a": "I", "a^\u028a": "W",
        "d^z": "\u02a3", "d^\u0292": "\u02a4",
        "e^\u026a": "A",
        "o^\u028a": "O", "\u0259^\u028a": "Q",
        "s^s": "S",
        "t^s": "\u02a6", "t^\u0283": "\u02a7",
        "\u0254^\u026a": "Y",
    }.items()))

    def __init__(self, language: str):
        import phonemizer

        self.language = language
        self._backend = phonemizer.backend.EspeakBackend(
            language=language, preserve_punctuation=True, with_stress=True,
            tie="^", language_switch="remove-flags",
            logger=logging.getLogger("phonemizer"))

    def __call__(self, text: str) -> str:
        # Angle quotes become curly ones, and brackets ride through espeak as
        # angle quotes, exactly as misaki does it.
        text = text.replace("\xab", "\u201c").replace("\xbb", "\u201d")
        text = text.replace("(", "\xab").replace(")", "\xbb")
        phonemes = self._backend.phonemize([text])
        if not phonemes:
            return ""
        ps = phonemes[0].strip()
        for old, new in self.FOLDS:
            ps = ps.replace(old, new)
        ps = ps.replace("^", "").replace("-", "")
        return ps.replace("\xab", "(").replace("\xbb", ")")


class MandarinPhonemizer:
    """misaki 0.9.4's ``ZHG2P`` on the legacy path hexgrad/Kokoro-82M uses
    (``version=None``), from the pack's copy in ``_misaki``: the phonemes the
    torch pipeline feeds its model for a Mandarin line. Needs the
    ``OWN_G2P_MODULES["z"]`` libraries; jieba's own console chatter (it logs
    every dictionary load at DEBUG to stderr) is turned down to warnings."""

    language = "zh"

    def __init__(self):
        import warnings

        with warnings.catch_warnings():
            # jieba 0.42's import of pkg_resources warns on every new setuptools.
            warnings.simplefilter("ignore", UserWarning)
            import jieba
            from ._misaki.zh import ZHG2P
        jieba.setLogLevel(logging.WARNING)
        try:
            # jieba caches its 9 MB dictionary index under the system TEMP by
            # default; keep it in OTR's own swept scratch tier instead.
            jieba.dt.tmp_dir = _otr_scratch_dir()
        except Exception:  # noqa: BLE001 -- no OTR output tree (a bare probe)
            pass
        self._g2p = ZHG2P()

    def __call__(self, text: str) -> str:
        phonemes, _tokens = self._g2p(text)
        return phonemes


#: Ten or more ASCII digits in a row, once NFKC has run.
_JA_LONG_DIGIT_RUN = re.compile(r"[0-9]{10,}")


def _nfkc_has_digit(ch: str) -> bool:
    return any("0" <= c <= "9" for c in unicodedata.normalize("NFKC", ch))


def japanese_long_numbers_spelled(text: str) -> str:
    """``text`` as misaki's Japanese reader can speak it, on both backends.

    misaki's number reader gives up past nine digits (it returns an error
    sentence, which Cutlet then asserts on) -- a phone number, a serial, the
    digits after a decimal point -- and it has no entry for a digit NFKC leaves
    foreign (Arabic-Indic, Devanagari, a dingbat digit): either took the line
    and the episode with it. So such a digit is written as its ASCII digit
    (``unicodedata.digit``, which covers the dingbat and Ethiopic digits
    ``decimal`` does not), and a run of characters that comes to ten or more
    digits once misaki's own NFKC has run -- counting full-width, circled and
    superscript digits -- is spaced out, which misaki reads digit by digit, as
    Japanese reads phone numbers and decimals. Everything else is left as
    written, because misaki applies some rules before its NFKC; a run of nine
    digits or fewer reads exactly as misaki reads it.
    """
    chars = []
    for ch in text:
        value = unicodedata.digit(ch, None)
        if value is not None and not unicodedata.normalize("NFKC", ch).isascii():
            ch = str(value)
        chars.append(ch)
    out: list = []
    run: list = []

    def flush() -> None:
        if run:
            joined = "".join(run)
            long_run = _JA_LONG_DIGIT_RUN.search(unicodedata.normalize("NFKC", joined))
            out.append(" ".join(run) if long_run else joined)
            run.clear()

    for ch in chars:
        if _nfkc_has_digit(ch):
            run.append(ch)
        else:
            flush()
            out.append(ch)
    flush()
    return "".join(out)


class JapanesePhonemizer:
    """misaki 0.9.4's Japanese phonemizer on the route ``JAG2P()`` takes by
    default (``version='cutlet'``), from the pack's copy in ``_misaki``: the
    phonemes the torch pipeline feeds its model for a Japanese line. Needs the
    ``OWN_G2P_MODULES["j"]`` libraries; the copy pins its MeCab dictionary to
    unidic-lite and replaces mojimoji, which has no Python 3.13 wheel, with the
    three-character table it measured to be after NFKC (see ``_misaki``)."""

    language = "ja"

    def __init__(self):
        from ._misaki.cutlet import Cutlet

        self._g2p = Cutlet()

    def __call__(self, text: str) -> str:
        phonemes, _tokens = self._g2p(japanese_long_numbers_spelled(text))
        return phonemes


#: lang_code -> the phonemizer the ONNX backend builds from the pack's copy, and
#: the copy's module (relative to this package) that a readiness check imports.
OWN_G2P = {"z": MandarinPhonemizer, "j": JapanesePhonemizer}
OWN_G2P_IMPORT = {"z": "._misaki.zh", "j": "._misaki.cutlet"}


#: lang_code -> the phonemizer builder that built here once. The readiness
#: check runs on every queued prompt whose language pool holds the row, and
#: fugashi's Tagger leaks about 0.7 MB per construction (Sonnet QA of
#: c7136448), so a success is remembered for the process rather than rebuilt.
#: Keyed to the builder object, so a swapped builder is built again; a failure
#: is never remembered, so a fixed install is seen on the next queue. The
#: trade: an install that breaks while ComfyUI is running still reads as ready
#: here, and the ONNX backend's load refuses it loudly instead.
_OWN_G2P_BUILT: dict = {}


def own_g2p_error(lang_code) -> "str | None":
    """Why the pack's copy of misaki for ``lang_code`` does not work here, or
    None when it does: it imports, and the phonemizer builds -- for Japanese that
    opens MeCab on its dictionary (38 ms measured on the 5080, most of it the
    first import; a dictionary that will not open fails in about 1 ms), so a
    dictionary that will not open refuses at the gate rather than at the voice
    node. An import alone is the half that passes. A build that succeeded once
    in this process is not repeated (``_OWN_G2P_BUILT``)."""
    import importlib
    import warnings

    code = str(lang_code or "").strip()
    module = OWN_G2P_IMPORT.get(code)
    if module is None or code not in OWN_G2P:
        return "no pack copy of a phonemizer for lang_code %r" % code
    builder = OWN_G2P[code]
    if _OWN_G2P_BUILT.get(code) is builder:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            importlib.import_module(module, __package__)
            builder()
    except Exception as exc:  # noqa: BLE001 -- any failure to build is "not ready"
        return _failure_reason(exc)
    _OWN_G2P_BUILT[code] = builder
    return None


def _failure_reason(exc: BaseException) -> str:
    """One line saying why: the exception's type, its first line, and its last
    line when there are more -- fugashi's MeCab error opens with "Failed
    initializing MeCab" and names the missing file on its last line. Never
    raises, whatever the exception's message does."""
    name = type(exc).__name__
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 -- a message that cannot render is still a failure
        return name
    lines = [line.strip() for line in text.splitlines() if any(ch.isalnum() for ch in line)]
    if not lines:
        return name
    detail = lines[0] if len(lines) == 1 else "%s ... %s" % (lines[0], lines[-1])
    return "%s: %s" % (name, detail[:400])


def own_g2p_ready(lang_code) -> bool:
    """True when ``own_g2p_error`` finds nothing wrong."""
    return own_g2p_error(lang_code) is None


def own_g2p_missing(lang_code) -> list:
    """The ``OWN_G2P_MODULES`` a row still needs on this box, by spec probe
    (nothing is imported, so a queue-time check pays no import). Empty for a
    row that needs none or has them all."""
    import importlib.util

    missing = []
    for name in OWN_G2P_MODULES.get(str(lang_code or "").strip(), ()):
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
#: The MeCab failure reads as a missing FILE, which sends the reader looking
#: for a corrupt install rather than a missing dictionary package.
_MECAB_SIGNATURES = ("mecabrc", "failed initializing mecab")

#: Rows whose G2P loads MeCab. No other language row touches it, so a MeCab
#: failure on any other row is something else and must not be reworded.
_MECAB_LANG_CODES = ("j",)


def _mecab_dictionary_error(error: Exception, lang_code: str) -> Exception:
    """Reword ONLY the empty-dictionary failure; return ``error`` otherwise.

    `misaki[ja]` declares `unidic`, whose wheel ships no dictionary at all --
    it downloads ~770 MB on a separate `python -m unidic download` step that
    nobody runs, so the Japanese row dies at the voice node with a path that
    does not exist. Worse, fugashi PREFERS `unidic` whenever it imports and
    never falls back, so installing `unidic-lite` beside it changes nothing
    and the error does not move. Both halves are named here because finding
    the second one cost a live leg (PBUG-20260918-06).

    Every other RuntimeError is returned untouched: KPipeline failures stay
    exactly as loud and as literal as they were.
    """
    text = str(error).lower()
    if str(lang_code or "").strip().lower() not in _MECAB_LANG_CODES:
        return error
    if not any(signature in text for signature in _MECAB_SIGNATURES):
        return error
    return RuntimeError(
        "Japanese voices need a MeCab dictionary and the installed one is "
        "empty. `misaki[ja]` pulls `unidic`, which ships a downloader rather "
        "than a dictionary. Fix it with BOTH of these, in this order:\n"
        "    pip uninstall -y unidic\n"
        "    pip install unidic-lite\n"
        "The uninstall is not optional -- fugashi prefers `unidic` whenever "
        "it is importable, so `unidic-lite` alone leaves this error "
        "unchanged. `unidic-lite` carries its dictionary inside the wheel, so "
        "nothing is downloaded at render time.\n"
        "Original error: %s" % error)


def _pause_only(text: str) -> bool:
    """True when ``text`` has something written but no letter or digit in it:
    pause marks ("...", Japanese middle dots, a dash) or a symbol (a star). The
    voice gate passes these as spoken lines, and English Kokoro voices "..." as
    a pause, so a phonemizer that finds no sound in one is right."""
    stripped = str(text or "").strip()
    return bool(stripped) and not any(ch.isalnum() for ch in stripped)


def _nothing_voiced(text: str, lang_code: str, backend: str):
    """What a backend gives a line it found no sound in.

    Pause marks only: a quarter second of silence, logged, and the episode
    goes on. A line with a letter or digit in it: loud, because its words were
    lost -- another language's script, letters the phonemizer does not read --
    and a silent clip would hide that (the no-fallback rule, operator
    2026-07-03, `_otr_voice_node_common`). A blank line is loud as it always
    was; it should never reach a voice.
    """
    import numpy as np

    line = str(text or "").strip()
    if _pause_only(line):
        # INFO, not WARNING: a pause line is expected content (Fable gate,
        # 2026-09-29); the unknown-phoneme pause stays a WARNING.
        log.info(
            "[%s] lang_code %r: %r has no sound to voice (pause marks or symbols "
            "only); voiced as a %.2f s pause", backend, lang_code, line[:80],
            UNSPEAKABLE_LINE_S)
        return np.zeros(int(SAMPLE_RATE * UNSPEAKABLE_LINE_S), dtype=np.float32)
    if not line:
        raise RuntimeError("%s produced no audio" % backend)
    raise RuntimeError(
        "%s produced no audio: lang_code %r found no sound in %r -- its letters "
        "are not ones this language's phonemizer reads (another language's "
        "script?). It is not voiced as silence; the line needs fixing."
        % (backend, lang_code, line[:120]))


class TorchKokoroBackend:
    """The pre-2026-09-02 synthesis path, moved verbatim.

    ``load`` builds ``KPipeline`` with the episode's lang_code (British ``b``
    on English; ``e``/``p``/``i``/``f``/``h``/``j``/``z`` on the other admitted
    rows). Cache identity is ``(lang_code, device)`` -- a language change
    rebuilds. The EXPLICIT device comes from the CastLock ledger stamp (S4 -- a
    device the host cannot provide fails LOUD in KPipeline, never a silent
    downgrade). ``repo_id`` only when this kokoro build accepts it (0.7.x does
    not).
    ``synthesize`` is ONE pipeline call over the full line with
    ``split_pattern=r"\\n+"`` -- pre-splitting would change the call shape and the
    bytes.
    """

    name = "torch"

    def __init__(self, device: str, lang_code: str = "b"):
        self.device = device
        self.lang_code = str(lang_code or "b").strip() or "b"
        self._pipeline = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        import inspect

        from kokoro import KPipeline

        kwargs = {"lang_code": self.lang_code, "device": self.device}
        try:
            if "repo_id" in inspect.signature(KPipeline.__init__).parameters:
                kwargs["repo_id"] = "hexgrad/Kokoro-82M"
        except (TypeError, ValueError):
            kwargs["repo_id"] = "hexgrad/Kokoro-82M"
        try:
            self._pipeline = KPipeline(**kwargs)
        except RuntimeError as error:
            reworded = _mecab_dictionary_error(error, self.lang_code)
            if reworded is error:
                # Not ours to explain. Re-raise the ORIGINAL untouched rather
                # than `raise error from error`, which would hang a
                # self-referential __cause__ on an exception this code has no
                # opinion about -- "errors stay exactly as loud" means the
                # metadata too, not only the message.
                raise
            raise reworded from error

    def synthesize(self, text: str, voice_id: str, speed: float):
        import numpy as np
        import torch

        if self.lang_code == "j":
            text = japanese_long_numbers_spelled(text)
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
            # The pipeline skips every chunk that phonemizes to nothing and
            # raises on nothing else (kokoro 0.9.4 pipeline.py), so no segment
            # at all means no chunk had a sound in it.
            return _nothing_voiced(text, self.lang_code, "kokoro pipeline")
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

    English lines go to kokoro-onnx as text (its own espeak ``en-gb``). A line
    in one of the ``ESPEAK_LANGUAGES`` goes as phonemes from
    ``EspeakPhonemizer``, and a Mandarin or Japanese line as phonemes from
    ``MandarinPhonemizer`` or ``JapanesePhonemizer`` -- in each case the torch
    pipeline's own G2P. The style vector is picked by kokoro-onnx from the
    count of phonemes the model knows, where the torch pipeline counts every
    character of the phoneme string, so a phoneme string with an unknown
    character (a Spanish inverted mark) can take the neighbouring style row:
    measured inaudible, waveforms 0.988-0.996 alike on the espeak rows. A line
    too long for one pass is cut where kokoro-onnx cuts it (at punctuation,
    within 510 phonemes), where the torch pipeline cuts the text into
    400-character pieces and truncates any that still run past 510 phonemes.
    """

    name = "onnx"

    def __init__(self, model_path: str, voices_npz: str, providers=None, threads=None,
                 lang_code: str = "b"):
        self.model_path = model_path
        self.voices_npz = voices_npz
        self.providers = list(providers or DEFAULT_ONNX_PROVIDERS)
        self.threads = int(threads or min(ONNX_THREAD_CAP, os.cpu_count() or 1))
        self.lang_code = str(lang_code or "b").strip() or "b"
        self.providers_active: list = []
        self._session = None
        self._kokoro = None
        self._g2p = None

    def load(self) -> None:
        if self._kokoro is not None:
            return
        if self.lang_code not in ("a", "b") and self.lang_code not in ESPEAK_LANGUAGES \
                and self.lang_code not in OWN_G2P:
            raise BackendUnavailable(
                "the ONNX kokoro has no phonemizer for lang_code %r; it voices "
                "English and lang_codes %s" % (
                    self.lang_code, ", ".join(sorted(set(ESPEAK_LANGUAGES) | set(OWN_G2P)))))
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
        session = ort.InferenceSession(
            self.model_path, sess_options=options, providers=self.providers)
        kokoro = Kokoro.from_session(session, self.voices_npz)
        # After the session (it points phonemizer at kokoro-onnx's espeak-ng),
        # and before anything is kept: a phonemizer that fails to build leaves
        # the backend unloaded, never loaded without it.
        if self.lang_code in ESPEAK_LANGUAGES:
            g2p = EspeakPhonemizer(ESPEAK_LANGUAGES[self.lang_code])
        elif self.lang_code in OWN_G2P:
            g2p = OWN_G2P[self.lang_code]()
        else:
            g2p = None
        self._session, self._kokoro, self._g2p = session, kokoro, g2p
        self.providers_active = list(session.get_providers())

    def voice_ids(self) -> list:
        return list(self._kokoro.get_voices()) if self._kokoro is not None else []

    def synthesize(self, text: str, voice_id: str, speed: float):
        import numpy as np

        if self._kokoro is None:
            raise RuntimeError("ONNX backend not loaded")
        if self._g2p is None and self.lang_code not in ("a", "b"):
            raise RuntimeError(
                "the ONNX backend for lang_code %r has no phonemizer; it will not "
                "read the line as English" % self.lang_code)
        if voice_id not in self.voice_ids():
            raise BackendUnavailable(
                "voice %r is not in the ONNX voice table %s (its .pt file was missing "
                "or unreadable when the table was built)" % (voice_id, self.voices_npz))
        segments = []
        unknown_only = []
        for chunk in _LINE_SPLIT.split(text or ""):
            chunk = chunk.strip()
            if not chunk:
                continue
            if self._g2p is None:
                samples, rate = self._kokoro.create(
                    chunk, voice=voice_id, speed=speed, **ONNX_CREATE_KWARGS)
            else:
                phonemes = self._g2p(chunk)
                if not phonemes:
                    continue            # the torch pipeline skips these too
                known = getattr(getattr(self._kokoro, "tokenizer", None), "known", None)
                if callable(known) and not known(phonemes):
                    # Nothing the model has a symbol for (a lone inverted
                    # question mark, a bracket, an emoji): kokoro-onnx refuses
                    # such a chunk outright, where the torch pipeline speaks a
                    # quarter second of nothing.
                    unknown_only.append(phonemes)
                    continue
                samples, rate = self._kokoro.create(
                    phonemes, voice=voice_id, speed=speed, is_phonemes=True,
                    **dict(ONNX_CREATE_KWARGS, lang=self._g2p.language))
            if int(rate) != SAMPLE_RATE:
                raise RuntimeError(
                    "kokoro-onnx returned %r Hz, expected %d" % (rate, SAMPLE_RATE))
            segments.append(np.asarray(samples, dtype=np.float32).squeeze())
        if not segments:
            if unknown_only:
                # The torch pipeline voices these phonemes as its own quarter
                # second of nothing, so the line is not the end of the episode
                # here either -- but the log says what was not voiced.
                log.warning(
                    "[kokoro-onnx] lang_code %r: %r has no phoneme the model knows "
                    "(%r); voiced as a %.2f s pause", self.lang_code,
                    str(text).strip()[:80], " ".join(unknown_only)[:80], UNSPEAKABLE_LINE_S)
                return np.zeros(int(SAMPLE_RATE * UNSPEAKABLE_LINE_S), dtype=np.float32)
            return _nothing_voiced(text, self.lang_code, "kokoro-onnx")
        return np.concatenate(segments) if len(segments) > 1 else segments[0]

    def close(self) -> None:
        # onnxruntime's InferenceSession has no close(); dropping the references
        # is the unload. The npz handle kokoro-onnx holds goes with it.
        self._kokoro = None
        self._session = None
        self._g2p = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001 -- teardown must never raise
            pass
