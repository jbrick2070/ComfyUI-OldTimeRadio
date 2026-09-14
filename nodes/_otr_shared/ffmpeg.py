"""``ffmpeg`` -- the ONE answer to "which ffmpeg does this box run".

WHY THIS FILE EXISTS (2026-09-04, RunPod-found fixes, kibitz r1-r3). Twelve
sites in ``nodes/`` each decided this for themselves, in four different
orders. Six of them shared one defect: a caller's own signature default --
the bare string ``"ffmpeg"`` -- was treated as an operator CHOICE, so on any
box with ffmpeg on PATH it won at step one and ``OTR_FFMPEG`` was never
consulted. Two never read ``OTR_FFMPEG`` at all (the registered SignalLost
renderer and the raw-video encode sink), and the three cloud preflights
refused an install whose ffmpeg is reachable only through the variable. Every
copy was tested by stubbing ``shutil.which`` to ``None`` -- exactly the
condition under which the defect cannot show.

The bare-name rule is NOT re-implemented here. ``ffprobe.py`` already owns it
(``_explicit`` / ``_BARE_FFMPEG_NAMES``) for the probe, and one rule spelled
twice is how the last one drifted.

``tests/test_ffmpeg_single_resolution.py`` walks the AST of ``nodes/`` and
fails if any other module reads ``OTR_FFMPEG`` or asks ``which`` for the bare
name. That test, not this docstring, is what makes "one owner" true.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

try:
    from .ffprobe import _BARE_FFMPEG_NAMES, _explicit, _usable
except ImportError:  # loaded flat
    try:
        from _otr_shared.ffprobe import _BARE_FFMPEG_NAMES, _explicit, _usable  # type: ignore  # nodes/ on sys.path
    except ImportError:
        from ffprobe import _BARE_FFMPEG_NAMES, _explicit, _usable  # type: ignore  # _otr_shared/ on sys.path

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import env as otr_env  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import env as otr_env  # type: ignore  # _otr_shared/ on sys.path

#: The operator's pin. One spelling, read in one place.
FFMPEG_ENV = "OTR_FFMPEG"

#: Where a Windows box puts ffmpeg when it was installed but PATH was not
#: refreshed for the process that runs ComfyUI: winget's shim directory, and
#: the hand-unzip location the README has always suggested. Carried over
#: from the SignalLost renderer, which was the only site that knew them.
_WINDOWS_INSTALL_CANDIDATES = (
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
)

#: The same courtesy for macOS, added 2026-09-06. Homebrew does not put its
#: bin directory on the PATH of a GUI-launched app -- ComfyUI Desktop is
#: started by Finder/launchd, not by a login shell, so `ffmpeg` on PATH misses
#: even when `brew install ffmpeg` has plainly succeeded. Without this the
#: renderer refuses on a correctly-provisioned Mac, and every terminal step
#: goes with it: the caption burn, the credits roll, the silent composite and
#: the audio mux.
#:
#: Apple Silicon first (/opt/homebrew), then Intel Homebrew and MacPorts.
#: There is no QuickTime/AVFoundation alternative worth reaching for -- the
#: whole render chain speaks ffmpeg command lines, and ffmpeg already uses
#: Apple's VideoToolbox for hardware paths, so finding the binary IS the fix.
_MACOS_INSTALL_CANDIDATES = (
    "/opt/homebrew/bin/ffmpeg",   # Homebrew, Apple Silicon
    "/usr/local/bin/ffmpeg",      # Homebrew, Intel
    "/opt/local/bin/ffmpeg",      # MacPorts
)


_log = logging.getLogger("OTR")


def resolve_ffmpeg(preferred=None) -> Optional[str]:
    """Which ffmpeg THIS box should run, or ``None`` when it has none.

    The order, most explicit first; every step is skipped unless it resolves
    to something that exists:

    1. ``preferred`` -- a real path, or a non-default name the caller chose.
       A bare ``ffmpeg`` / ``ffmpeg.exe`` is the caller's own signature
       default and carries no information, so it is not a choice.
    2. ``$OTR_FFMPEG`` -- the operator's explicit pin.
    3. ``ffmpeg`` on ``PATH``.
    4. the well-known Windows and macOS install locations above. Both exist
       for the same reason: a GUI-launched ComfyUI does not inherit the PATH a
       login shell would have given it, so an ffmpeg the user definitely
       installed is invisible to step 3.
    5. the binary `imageio-ffmpeg` bundles. Last, so any real install above
       wins, and present so that `pip install` alone is enough to encode on
       Windows, Linux and macOS alike.

    NEVER RAISES. "This box has no ffmpeg" is a fact, and each caller has
    already decided what that fact costs it -- an empty string, its own
    literal, or a named refusal.
    """
    chosen = _usable(_explicit(preferred, _BARE_FFMPEG_NAMES))
    if chosen:
        return chosen
    # strip THEN expand: a pin typed with leading whitespace keeps its tilde
    # otherwise ("  ~/bin/ffmpeg" never expanded -- agy, manual r4).
    chosen = _usable(os.path.expanduser((otr_env.get(FFMPEG_ENV) or "").strip()))
    if chosen:
        return chosen
    chosen = _usable("ffmpeg")  # PATH, through the one function that reads it
    if chosen:
        return chosen
    for raw in _WINDOWS_INSTALL_CANDIDATES + _MACOS_INSTALL_CANDIDATES:
        candidate = os.path.expandvars(raw)
        if os.path.isfile(candidate):
            return candidate
    # Guarded AT THE CALL SITE as well as inside the helper. The helper has its
    # own try/except, but this module's contract is that resolution NEVER
    # raises, and a contract that depends on one particular callee staying
    # well-behaved is not a contract. Caught by
    # test_a_broken_imageio_ffmpeg_is_none_and_never_raises.
    # The downloaded PAIR first: it carries a sibling ffprobe, which the
    # imageio wheel does not, and several engines need both.
    try:
        chosen = _downloaded_ffmpeg()
        if chosen:
            return chosen
    except Exception:  # noqa: BLE001 -- resolution must never raise
        pass
    try:
        return _imageio_ffmpeg()
    except Exception:  # noqa: BLE001 -- resolution must never raise
        return None


def _downloaded_ffmpeg() -> Optional[str]:
    """The ffmpeg `ffmpeg-downloader` has installed, or ``None``.

    PREFERRED over the imageio wheel, and the reason is ffprobe. imageio-ffmpeg
    ships ONLY an ffmpeg binary -- there is no ffprobe beside it -- and
    `_otr_shared.ffprobe.resolve_ffprobe` finds ffprobe by looking for a SIBLING
    of the ffmpeg this box runs. So resolving to the imageio binary silently
    leaves the box with no ffprobe at all.

    That was measured, 2026-09-07 on a Mac mini M4: with only imageio-ffmpeg
    present, a full episode rendered its audio, encoded video frames, and then
    died in `eng_visualizer.py` at
        validate_silent_clip_contract(ffprobe_clip_fields(out_path), fps)
    -- the visualizer encodes a clip and probes it back to check the contract,
    so ffmpeg alone is only half a dependency.

    `ffmpeg-downloader` installs a matched ffmpeg + ffprobe PAIR into one
    directory on Windows, Linux and macOS, which satisfies both resolvers at
    once. Its own dependencies are light (platformdirs, tabulate).

    Returns None until `ffdl install` has run; the imageio fallback below then
    still covers plain encoding.
    """
    try:
        import ffmpeg_downloader as _fdl  # noqa: PLC0415 -- optional
        return _usable(getattr(_fdl, "ffmpeg_path", None))
    except Exception:  # noqa: BLE001 -- resolution must never raise
        return None


def _imageio_ffmpeg() -> Optional[str]:
    """The ffmpeg `imageio-ffmpeg` ships, or ``None``.

    LAST resort on purpose: every step above is either the operator's choice or
    a real system install, and both should win over a wheel-bundled copy. This
    step exists so that a box which installed the pack and nothing else still
    encodes.

    `imageio-ffmpeg` publishes prebuilt binaries for Windows, Linux and macOS
    (arm64 and x86_64) as ordinary wheels, so declaring it in requirements.txt
    makes ffmpeg arrive with `pip install` on every platform this pack targets.
    That is the whole point: before this step the macOS answer was "install
    Homebrew first" and the Windows answer was a `winget` line printed from a
    RuntimeError -- both of which ask a one-click-install user to open a
    terminal, which is exactly the audience a registry install exists to spare.

    2026-09-07, measured on a Mac mini M4: a full episode wrote its master WAV
    and then died at the mp4 encode with "ffmpeg not found. Install via:
    winget install ffmpeg" -- a Windows command, on macOS, at the last step of
    a 13-minute run. The wheel-bundled binary was already sitting in the same
    venv (ffmpeg 7.1, with libx264, aac and h264_videotoolbox); nothing had
    ever asked it.

    NEVER RAISES, per this module's contract. A missing or broken
    imageio-ffmpeg is just "this box has no ffmpeg".
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415 -- optional, resolved lazily
        return _usable(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 -- resolution must never raise
        return None


#: One probe per resolved binary per process. `ffmpeg -filters` and
#: `-encoders` each spawn a subprocess and print thousands of lines; a render
#: asks this question once per beat otherwise.
_CAPABILITY_CACHE: dict = {}

#: What burning captions actually needs. `otr_caption_burn` composes an
#: `ass={name}` filter and re-encodes with libx264, so a build missing either
#: cannot do the job however well it renders video.
CAPTION_FILTER = "ass"
CAPTION_ENCODER = "libx264"
#: What the master mux needs: PCM audio copied into an MP4 container
#: (ISOBMFF "ipcm", FFmpeg 6.1+). `otr_master_audio_mux` copies the WAV
#: master losslessly by contract (`-c:a copy`, asserted), so an older
#: build fails at the very last step with "Could not find tag for codec
#: pcm_s16le in stream #1" -- after the whole episode has rendered.
MASTER_MUX_PCM = "pcm_in_mp4"


def probe_ffmpeg_capabilities(path=None) -> dict:
    """What THIS ffmpeg build can do. Never raises; unknowns answer ``None``.

    Returns ``{"path", "ass", "libx264"}`` where the two capability values are
    True, False, or None when the probe could not run at all (binary missing,
    subprocess refused, timeout). None is deliberately NOT False: "we could not
    ask" and "it answered no" justify different messages, and a caller that
    treats them the same will refuse a working box on a failed subprocess.

    WHY THIS EXISTS. A Homebrew ffmpeg carries libass and libx264 by default,
    so a Mac that ran `brew install ffmpeg` burns captions fine. A MINIMAL
    build -- notably the binary bundled inside imageio-ffmpeg, the obvious
    candidate for an automatic fallback -- typically ships neither. Without
    this probe that box renders the whole episode and then dies at the caption
    stage, twenty minutes of writer, voices, music and video after the point
    where the answer was already knowable.
    """
    # An explicit path still has to EXIST. Without this a caller passing a
    # dead path got every capability as None, which caption_support_gap then
    # read as "could not ask, do not refuse" and reported no problem at all --
    # a missing binary silently answering "fine".
    resolved = _usable(path) if path else resolve_ffmpeg()
    if not resolved:
        return {"path": None, CAPTION_FILTER: None, CAPTION_ENCODER: None,
                MASTER_MUX_PCM: None}
    cached = _CAPABILITY_CACHE.get(resolved)
    if cached is not None:
        return dict(cached)

    try:
        from . import proc as otr_proc
    except ImportError:  # pragma: no cover -- loaded flat
        try:
            from _otr_shared import proc as otr_proc  # type: ignore  # nodes/ on sys.path
        except ImportError:
            import proc as otr_proc  # type: ignore  # _otr_shared/ on sys.path

    def _lists(flag):
        try:
            done = otr_proc.run(
                [resolved, "-hide_banner", flag],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except Exception:  # noqa: BLE001 -- a probe must never kill a render
            return None
        return (done.stdout or "") + (done.stderr or "")

    filters = _lists("-filters")
    encoders = _lists("-encoders")
    pcm_in_mp4 = _probe_pcm_in_mp4(resolved, otr_proc)
    result = {
        "path": resolved,
        MASTER_MUX_PCM: pcm_in_mp4,
        # Match the filter NAME in its own column, not anywhere in the blob:
        # "ass" appears inside "subtitles", "pass", "compass" and others.
        CAPTION_FILTER: None if filters is None else any(
            CAPTION_FILTER in line.split() for line in filters.splitlines()),
        CAPTION_ENCODER: None if encoders is None else any(
            CAPTION_ENCODER in line.split() for line in encoders.splitlines()),
    }
    _CAPABILITY_CACHE[resolved] = dict(result)
    return result


def _write_silence(resolved, otr_proc, container, suffix):
    """Mux 0.2 s of PCM silence into ``container``. True / False / None.

    None is "the probe could not be run at all" -- no temp file, no process --
    and is never evidence about the build.
    """
    import os
    import tempfile
    try:
        fd, tmp = tempfile.mkstemp(prefix="otr_pcm_probe_", suffix=suffix)
        os.close(fd)
    except Exception:  # noqa: BLE001 -- no writable temp: we cannot ask
        return None
    try:
        try:
            done = otr_proc.run(
                [resolved, "-hide_banner", "-v", "error", "-y",
                 "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.2",
                 "-c:a", "pcm_s16le", "-f", container, tmp],
                capture_output=True, text=True, timeout=30, check=False,
            )
        except Exception:  # noqa: BLE001 -- a probe must never kill a render
            return None
        try:
            wrote = os.path.getsize(tmp) > 0
        except OSError:
            wrote = False
        return bool(done.returncode == 0 and wrote)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _probe_pcm_in_mp4(resolved, otr_proc):
    """True / False / None: can this build write PCM audio into an MP4?

    Asked with a real, tiny mux rather than a version parse, because the answer
    is a property of the build and not of the number (a distro could backport
    it; a stripped build could lack it while numbering high).

    THE CONTROL IS THE POINT, and it is what the first cut of this function got
    wrong. The probe needs `lavfi` and `anullsrc` and a writable temp dir, none
    of which the real mux needs -- it only copies existing streams. Without a
    control, a build with the lavfi device disabled, or a read-only TMPDIR,
    answered "this ffmpeg cannot write PCM into MP4" and a working machine was
    refused. So the SAME silence is written to a WAV first: every ffmpeg ever
    shipped can do that. If the control fails, the environment is what failed
    and the answer is None. Only a passing control makes a failing MP4 mean the
    container.
    """
    control = _write_silence(resolved, otr_proc, "wav", ".wav")
    if control is not True:
        return None
    return _write_silence(resolved, otr_proc, "mp4", ".mp4")


def master_mux_support_gap(path=None) -> Optional[str]:
    """A sentence naming what stops this ffmpeg writing the final MP4, else None.

    Same contract as :func:`caption_support_gap`: None when the mux will work
    AND when the probe could not run. The writer asks this at its own start so
    a box with Ubuntu 22.04's ffmpeg 4.4 refuses in a second instead of
    rendering a whole episode and leaving a 0-byte final (PBUG-20260913-03).
    """
    caps = probe_ffmpeg_capabilities(path)
    if caps["path"] is None:
        return ("no ffmpeg was found on this host, so the episode cannot be "
                "muxed; install ffmpeg 6.1 or newer (macOS: `brew install "
                "ffmpeg`; Windows: `winget install Gyan.FFmpeg`; Linux: a "
                "static build if the distro's is older) or set OTR_FFMPEG")
    if caps.get(MASTER_MUX_PCM) is not False:
        return None
    return ("the ffmpeg at %s cannot write PCM audio into an MP4, which the "
            "final mux needs (it copies the master losslessly); this is an "
            "ffmpeg older than 6.1 -- Ubuntu 22.04's apt build is 4.4. "
            "Install ffmpeg 6.1 or newer (a static build is fine) or set "
            "OTR_FFMPEG to one." % caps["path"])


def caption_support_gap(path=None) -> Optional[str]:
    """A sentence naming what stops this ffmpeg burning captions, else None.

    Answers None when captions will work AND when the probe could not run --
    an unrunnable probe is not evidence of a missing feature, and refusing a
    render on it would be the guess this function exists to avoid.
    """
    caps = probe_ffmpeg_capabilities(path)
    if caps["path"] is None:
        return ("no ffmpeg was found on this host, so captions cannot be "
                "burned; install ffmpeg (macOS: `brew install ffmpeg`) or set "
                "OTR_FFMPEG to a full build")
    missing = [name for name in (CAPTION_FILTER, CAPTION_ENCODER)
               if caps[name] is False]
    if not missing:
        return None
    return ("the ffmpeg at %s is a minimal build missing %s; caption burning "
            "needs both the `ass` filter (libass) and the libx264 encoder. "
            "Install a full build -- macOS `brew install ffmpeg` ships both -- "
            "or set OTR_FFMPEG to one, or turn burn_captions off."
            % (caps["path"], " and ".join(missing)))


# REMOVED 2026-09-13: ``widget_ffmpeg_is_ignored(value, node)`` lived here,
# with a module-level warn-once set beside ``_CAPABILITY_CACHE`` above.
#
# WHAT IT DEFENDED AGAINST. A ComfyUI widget value arrives in the body of an
# unauthenticated ``/prompt`` request and is whatever a downloaded workflow
# JSON says. While five node classes declared an ``ffmpeg`` STRING widget,
# that value was a channel by which untrusted input could reach argv[0] --
# ahead of the operator's own ``OTR_FFMPEG`` pin, and the ffprobe sibling
# rule turned one such value into a SECOND attacker binary. From 2026-09-04
# the defence was to SANITISE: each execute method handed its widget value to
# this function, which discarded it and returned ``""``.
#
# THE DEFENCE IS NOW NON-DECLARATION, WHICH IS STRICTLY STRONGER. The widget
# is gone from all five classes that declared it, so ComfyUI never passes a
# value at all: the channel is closed rather than cleaned. A sanitiser with no
# caller would not be harmless decoration here -- it would invite a future
# window to re-declare the widget and "handle" it, restoring the weaker
# design. Hence the rip, per this repo's orphans rule.
#
# ``OTR_FFMPEG`` remains the one way to pin a build, because a workflow cannot
# set an environment variable -- exactly why the pin is the trustworthy channel
# and a widget is not. ``resolve_ffmpeg`` here, and ``resolve_ffprobe`` in the
# sibling ``ffprobe`` module, remain the live security surface: their
# absolute-path and bare-name rules are what decide argv[0].
