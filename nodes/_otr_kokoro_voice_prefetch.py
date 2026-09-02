"""Fetch Kokoro's English voice files ONCE, at boot, so a fresh install can
speak without the operator downloading anything by hand.

WHY THIS EXISTS (operator, 2026-08-24). Someone installs the pack from the
Comfy registry, clicks Run, and needs a voice. Only two local TTS engines
work with NO user-supplied audio: Bark, whose voices live inside its weights,
and Kokoro, whose voices are separate files. The other three local engines
(`indextts2`, `chatterbox`, `dia`) all declare `requires_voice_ref = True` --
they CLONE, so they need a reference WAV the user has to supply, and a fresh
install has none.

That left Bark as the only zero-setup option, and Bark is 4.2 GB on disk
against Kokoro's 327 MB -- a 13x tax on exactly the 8 GB tier that can least
afford it. The whole gap was a handful of 523 KB files.

WHAT IT DOES NOT DO, and this is the load-bearing part: it does NOT make the
render path network. `eng_kokoro` refuses to fetch a voice mid-render and
raises a named `EngineUnusable` instead, deliberately -- a mid-render hub
fetch once 404'd and aborted an entire finished episode (the V-9 / C-7 rule).
That rule is untouched. This runs from `prestartup_script.py`, BEFORE ComfyUI
loads a single node, so it is not inside any render at all.

NEVER FATAL. A prestartup failure is uniquely expensive: the banner lies, and
anything below the failure silently never runs. So every path here is wrapped,
a network problem is a log line and nothing more, and a machine with no
internet boots exactly as it does today -- just without the extra voices.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("OTR")

#: The upstream repo. Pinned by name, never by revision: a voice file is a
#: leaf asset and the repo's `main` is what `eng_kokoro`'s own KPipeline
#: resolves against, so pinning a different revision here would fetch files
#: the engine then cannot see.
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"

#: The ONNX model for the kokoro-onnx backend (Python 3.13 boxes, where the torch
#: `kokoro` package cannot install -- PBUG-20260901-04). One file, fp32, 326 MB:
#: the export kokoro-onnx targets, size parity with the torch `.pth`. Fetched here
#: at boot ONLY when that backend will be used (see `onnx_backend_wanted`), never
#: inside a render.
KOKORO_ONNX_REPO_ID = "onnx-community/Kokoro-82M-v1.0-ONNX"
KOKORO_ONNX_FILENAME = "onnx/model.onnx"
#: Where `eng_kokoro` looks for it, relative to the model dir. Duplicated from
#: that module ON PURPOSE (same reason as `_KOKORO_MODEL_SUBDIR` below); the
#: prefetch test pins the two equal.
_KOKORO_ONNX_REL_PATH = os.path.join("onnx", "model.onnx")

#: ENGLISH ONLY, and deliberately so. The repo ships 54 voices; 26 of them are
#: Spanish, French, Italian, Hindi, Japanese, Portuguese and Chinese, which
#: this show has no use for. Fetching the 28 English ones costs ~15 MB against
#: a 327 MB model -- about 4%, i.e. noise -- while fetching all 54 would be
#: ~28 MB of which half is dead weight.
#:
#: WHY NOT JUST THE FOUR ANNOUNCER VOICES: two characters never share a voice
#: (`_assert_unique_bark_voices` and the casting validator's `taken` set both
#: enforce it), and a cast runs to 6 on the legacy banks and 10 on
#: `scifi_news_pro`. Four voices cannot dress either, and a gender-balanced
#: cast halves the pool again. The British set alone is 4 male + 4 female,
#: which is tight at 6 and cannot serve 10.
#:
#: Both accents are kept on purpose: British for the announcer's house style
#: (the curated pool `eng_kokoro.ANNOUNCER_VOICE_POOL` draws from `b*`), and
#: American to widen character casting so episodes stop sounding like one
#: small repertory company.
ENGLISH_VOICES = (
    # British female / male -- the announcer's home accent.
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    # American female.
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # American male.
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
)

#: Where `eng_kokoro` looks. Duplicated from that module ON PURPOSE rather
#: than imported: this runs at PRESTARTUP, before ComfyUI has loaded any node
#: package, so importing the engine here would drag its dependency chain into
#: the boot path -- the exact weight this file exists to avoid. The constant
#: is small and the mismatch is guarded by `test_kokoro_prefetch_targets_the
#: _engine_s_own_dir`.
_KOKORO_MODEL_SUBDIR = os.path.join("TTS", "KokoroTTS")


def _models_dir() -> "str | None":
    """ComfyUI's models/ directory, the way the ENGINE will resolve it.

    ComfyUI imports `folder_paths` at the top of main.py and applies
    `--base-directory` / custom paths BEFORE it runs prestartup scripts
    (main.py: apply_custom_paths() then execute_prestartup_script()), so
    `folder_paths.models_dir` is importable and already final here -- and it is
    exactly what `eng_kokoro._kokoro_model_dir()` reads. That keeps a Desktop
    install with a relocated models tree coherent: the file lands where the
    engine looks. Fallback: three levels up from this file (this file sits at
    <comfy>/custom_nodes/<pack>/nodes/), the same derivation
    `prestartup_script.py` uses for HF_HOME.
    """
    try:
        import folder_paths  # type: ignore

        models = getattr(folder_paths, "models_dir", None)
        if models and os.path.isdir(models):
            return models
    except Exception:  # noqa: BLE001 -- not a Comfy process (tests / CLI)
        pass
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        comfy_base = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        models = os.path.join(comfy_base, "models")
        return models if os.path.isdir(models) else None
    except Exception:  # noqa: BLE001 -- boot must never die here
        return None


def _spec_exists(name: str) -> bool:
    """`importlib.util.find_spec` that never raises and never imports: a
    `sys.modules` fake without `__spec__` (the adapter tests do that) makes
    find_spec raise ValueError, and prestartup must not die for a test fixture."""
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def onnx_backend_wanted() -> bool:
    """Will `eng_kokoro` select the ONNX backend on this interpreter? True when
    forced by OTR_KOKORO_BACKEND=onnx, or when the torch `kokoro` package is
    absent and `kokoro_onnx` is present. Nothing is imported to decide."""
    mode = os.environ.get("OTR_KOKORO_BACKEND", "auto").strip().lower()
    if mode == "onnx":
        return True
    if mode == "torch":
        return False
    return (not _spec_exists("kokoro")) and _spec_exists("kokoro_onnx")


def prefetch_kokoro_onnx_model(*, force: bool = False) -> "dict":
    """Fetch the ONNX model once, at boot, when the ONNX backend will be used.

    Same contract as the voice prefetch: returns a receipt, never raises,
    honours HF_HUB_OFFLINE / OTR_SKIP_KOKORO_PREFETCH, and logs BEFORE the 326 MB
    download starts so a first boot never reads as a hang. `local_dir` is the
    engine's own KokoroTTS dir, so the file lands at
    <models>/TTS/KokoroTTS/onnx/model.onnx with no second copy in the hub cache.
    """
    receipt = {"attempted": 0, "fetched": 0, "skipped_offline": False,
               "present": False, "reason": "", "path": ""}
    models = _models_dir()
    if not models:
        receipt["reason"] = "could not locate ComfyUI models/ directory"
        return receipt
    kokoro_dir = os.path.join(models, _KOKORO_MODEL_SUBDIR)
    dest = os.path.join(kokoro_dir, _KOKORO_ONNX_REL_PATH)
    receipt["path"] = dest
    if os.path.exists(dest) and not force:
        receipt["present"] = True
        return receipt
    if not force and not onnx_backend_wanted():
        receipt["reason"] = "ONNX backend not selected on this interpreter"
        return receipt
    if not force and os.environ.get("HF_HUB_OFFLINE") == "1":
        receipt["skipped_offline"] = True
        receipt["reason"] = "HF_HUB_OFFLINE=1"
        return receipt
    if os.environ.get("OTR_SKIP_KOKORO_PREFETCH") == "1":
        receipt["skipped_offline"] = True
        receipt["reason"] = "OTR_SKIP_KOKORO_PREFETCH=1"
        return receipt
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        receipt["reason"] = f"huggingface_hub unavailable: {exc}"
        return receipt
    receipt["attempted"] = 1
    log.info(
        "OldTimeRadio: fetching the Kokoro ONNX model (%s %s, ~326 MB, one-time) "
        "into %s -- this can take a few minutes on a slow link",
        KOKORO_ONNX_REPO_ID, KOKORO_ONNX_FILENAME, kokoro_dir)
    try:
        os.makedirs(kokoro_dir, exist_ok=True)
        got = hf_hub_download(
            repo_id=KOKORO_ONNX_REPO_ID, filename=KOKORO_ONNX_FILENAME,
            local_dir=kokoro_dir)
        if not os.path.exists(dest):
            # A hub version that resolved elsewhere: copy, never symlink (the
            # engine checks this exact path with os.path.exists).
            import shutil

            shutil.copyfile(got, dest)
        receipt["fetched"] = 1
    except Exception as exc:  # noqa: BLE001 -- no internet, a 404, a full disk
        receipt["reason"] = f"{KOKORO_ONNX_FILENAME}: {exc}"
    return receipt


def missing_voices(voices_dir: str) -> "list[str]":
    """Which English voices are not on disk yet. Pure; no network, no I/O
    beyond `os.path.exists`, so the common case (everything present) costs
    28 stat calls and returns."""
    return [v for v in ENGLISH_VOICES
            if not os.path.exists(os.path.join(voices_dir, f"{v}.pt"))]


def prefetch_kokoro_voices(*, force: bool = False) -> "dict":
    """Fetch any missing English voice file. Returns a small receipt.

    Returns rather than raises, always. The caller is `prestartup_script.py`,
    where an exception costs far more than a missing voice: the boot banner
    reports failure and every statement below the raise is silently skipped.
    """
    receipt = {"attempted": 0, "fetched": 0, "skipped_offline": False,
               "reason": "", "voices_dir": ""}
    models = _models_dir()
    if not models:
        receipt["reason"] = "could not locate ComfyUI models/ directory"
        return receipt

    voices_dir = os.path.join(models, _KOKORO_MODEL_SUBDIR, "voices")
    receipt["voices_dir"] = voices_dir

    # An operator who has deliberately gone offline stays offline. The engine
    # still works for whatever is already on disk.
    if not force and os.environ.get("HF_HUB_OFFLINE") == "1":
        receipt["skipped_offline"] = True
        receipt["reason"] = "HF_HUB_OFFLINE=1"
        return receipt
    if os.environ.get("OTR_SKIP_KOKORO_PREFETCH") == "1":
        receipt["skipped_offline"] = True
        receipt["reason"] = "OTR_SKIP_KOKORO_PREFETCH=1"
        return receipt

    wanted = ENGLISH_VOICES if force else missing_voices(voices_dir)
    receipt["attempted"] = len(wanted)
    if not wanted:
        return receipt          # the ordinary case after first boot

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        receipt["reason"] = f"huggingface_hub unavailable: {exc}"
        return receipt

    try:
        os.makedirs(voices_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        receipt["reason"] = f"cannot create {voices_dir}: {exc}"
        return receipt

    for voice in wanted:
        try:
            src = hf_hub_download(
                repo_id=KOKORO_REPO_ID,
                filename=f"voices/{voice}.pt",
            )
            dest = os.path.join(voices_dir, f"{voice}.pt")
            if not os.path.exists(dest):
                # COPY, never symlink. The hub cache is a different tree with
                # its own eviction, and `eng_kokoro` checks this exact path
                # with `os.path.exists` before every episode; a dangling link
                # would read as "voice missing" and fail an episode closed.
                import shutil

                shutil.copyfile(src, dest)
            receipt["fetched"] += 1
        except Exception as exc:  # noqa: BLE001 -- one bad voice is not fatal
            # No internet, a 404, a read-only disk: all the same answer. Say
            # so once, keep whatever did land, and let the boot continue.
            receipt["reason"] = f"{voice}: {exc}"
            break
    return receipt


def prefetch_at_boot() -> None:
    """The prestartup entry point. Logs, never raises, never prints non-ASCII."""
    # The two fetches are independent: a voice-prefetch failure must not skip
    # the ONNX model, and vice versa.
    try:
        onnx_receipt = prefetch_kokoro_onnx_model()
        if onnx_receipt.get("fetched"):
            log.info("OldTimeRadio: fetched the Kokoro ONNX model into %s (one-time)",
                     onnx_receipt.get("path"))
        elif onnx_receipt.get("attempted") and onnx_receipt.get("reason"):
            log.info(
                "OldTimeRadio: Kokoro ONNX model prefetch incomplete (%s). The kokoro "
                "engine will name the fetch command at the first voice line; Bark "
                "needs nothing.", onnx_receipt["reason"])
    except Exception as exc:  # noqa: BLE001 -- belt and braces
        log.info("OldTimeRadio kokoro ONNX model prefetch skipped: %s", exc)
    try:
        receipt = prefetch_kokoro_voices()
    except Exception as exc:  # noqa: BLE001 -- belt and braces
        log.info("OldTimeRadio kokoro voice prefetch skipped: %s", exc)
        return
    if receipt.get("fetched"):
        log.info(
            "OldTimeRadio: fetched %d Kokoro voice file(s) into %s "
            "(one-time, ~523 KB each)",
            receipt["fetched"], receipt.get("voices_dir"))
    elif receipt.get("reason") and receipt.get("attempted"):
        log.info(
            "OldTimeRadio: Kokoro voice prefetch incomplete (%s). Kokoro will "
            "still use whatever voices are already on disk; Bark needs none.",
            receipt["reason"])
