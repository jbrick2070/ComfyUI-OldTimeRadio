"""ONE owner spawns a process: this module.

The registry scan of alpha.17 carried 35 `python_command_injection_risk`
findings, one per CALL SITE, across twenty files -- and a human reading that
report has to judge thirty-five spawns to learn that this pack runs ffmpeg.
Collapsing them here makes the report two lines and, more usefully, gives the
pack ONE place where "what may this process launch" is answered.

WHAT THIS OWNS: the spawn. It does NOT build commands -- every caller keeps
its own argv, because what a caller runs is part of what the caller does --
and it does not interpret results. ``run`` returns the real
``subprocess.CompletedProcess``, ``popen`` returns the real
``subprocess.Popen``, every keyword passes straight through, and no exception
is wrapped: callers already catch ``subprocess.CalledProcessError`` and
``subprocess.TimeoutExpired``, and those names are re-exported below as
IDENTITY aliases so an existing ``except`` clause stays true.

THE ALLOWLIST is the reason this is a boundary rather than a shim. ``argv[0]``
is normalized (basename, lower-cased, ``.exe`` stripped) and checked against
the executables this pack actually runs -- measured, not guessed, from an AST
sweep of all 35 sites (``docs/2026-09-04-registry-findings-collapse/argv0_receipt.txt``).
An unlisted binary raises before anything is spawned. Adding one is a
one-line, reviewed change; that is the point.

``shell=True`` is refused outright, and so is a string argv: both turn an
argv list into a shell parse, which is the actual injection risk the scanner
names.

Import it under the alias ``otr_proc`` -- ``proc`` is a local variable in
eleven files (``video_engine.py:1082`` among them) and a shadowed module
raises ``UnboundLocalError``:

    try:
        from . import proc as otr_proc        # inside _otr_shared
    except ImportError:                       # pragma: no cover -- flat load
        import proc as otr_proc               # type: ignore

Use the module form (``otr_proc.run(...)``), never ``from ... import run``:
a test patches ``<module>.otr_proc.run`` exactly the way it patches
``<module>.subprocess.run`` today, and a name bound at import time cannot be
patched that way.

Stdlib only; imports nothing from the pack.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Sequence

#: Re-exported so a migrated module can drop ``import subprocess`` entirely.
#: IDENTITY aliases -- tests construct and raise these, and ``except`` clauses
#: compare by identity, so they must be the stdlib objects themselves.
PIPE = subprocess.PIPE
DEVNULL = subprocess.DEVNULL
STDOUT = subprocess.STDOUT
CompletedProcess = subprocess.CompletedProcess
Popen = subprocess.Popen
CalledProcessError = subprocess.CalledProcessError
TimeoutExpired = subprocess.TimeoutExpired

__all__ = [
    "run", "popen", "ExecutableNotAllowed", "ALLOWED_EXECUTABLES",
    "ALLOWED_EXECUTABLE_PREFIXES",
    "PIPE", "DEVNULL", "STDOUT", "CompletedProcess", "Popen",
    "CalledProcessError", "TimeoutExpired",
]


class ExecutableNotAllowed(RuntimeError):
    """``argv[0]`` is not one of the executables this pack runs."""


#: basename (lower, ``.exe`` stripped) -> why this pack runs it. Measured from
#: every spawn site in the tree; see the argv receipt named in the module
#: docstring. EXACT names only -- the families whose file name carries a version
#: or a platform live in the prefix table below.
ALLOWED_EXECUTABLES = {
    "git": "the ledger's commit stamp (rev-parse --short HEAD)",
    "nvidia-smi": "GPU specs when torch cannot answer",
    "blender": "the mesh stage's headless render",
}

#: basename PREFIX -> why. For tools whose PACKAGING puts a version or a
#: platform in the file name, so the exact basename is not knowable from here.
#:
#: The interpreter rule was here from the start (a sidecar TTS engine runs in
#: its own venv, so its interpreter is ``python``, ``pythonw``, ``python3`` or
#: ``python3.10`` depending on the box). The ffmpeg family JOINED IT on
#: 2026-09-04, and it is worth saying why rather than leaving it to look like
#: laziness: the argv receipt this allowlist was built from is an AST sweep of
#: argv[0] LITERALS, and it therefore cannot see a path a third-party library
#: computes at run time. ``imageio_ffmpeg`` -- installed here, and what the
#: encoder tests resolve -- ships its binary as
#: ``ffmpeg-win-x86_64-v7.1.exe``. Pinning that exact string instead would have
#: worked until the next imageio-ffmpeg release and then broken on somebody
#: else's machine, which is the worst shape a guard can take.
#:
#: WHAT THIS IS NOT: a sandbox. A prefix admits ``ffmpeg-anything``, and this
#: module could never stop a caller determined to run something else -- it is a
#: boundary that keeps the set of things this pack launches small, reviewable
#: and stated. ``blender`` stays EXACT because no versioned blender basename has
#: ever reached a spawn here; if one does, the named error below says so at the
#: spawn, which is exactly how the ffmpeg case was found.
ALLOWED_EXECUTABLE_PREFIXES = {
    "ffmpeg": "the render path: every encode, mux, concat and burn-in",
    "ffprobe": "duration, frame-count and stream probes",
    "python": "a sidecar engine's own venv interpreter",
}


def _normalized(argv0: Any) -> str:
    name = os.path.basename(str(argv0)).lower()
    if name.endswith(".exe"):
        name = name[: -len(".exe")]
    return name


def _allowed(name: str) -> bool:
    return (name in ALLOWED_EXECUTABLES
            or any(name.startswith(p) for p in ALLOWED_EXECUTABLE_PREFIXES))


def _no_remote_arguments(argv: Sequence[Any]) -> None:
    """No argument may name another MACHINE. Raises :class:`ExecutableNotAllowed`.

    Several nodes take a media path from an ordinary STRING widget, and a
    workflow JSON arrives over ComfyUI's unauthenticated ``/prompt`` endpoint.
    Those paths become ffmpeg arguments. On Windows the ordinary file APIs open
    a UNC path transparently, so handing ``\\\\attacker-host\\share\\x`` to
    ffmpeg makes this machine open an SMB session and authenticate to a host the
    WORKFLOW chose -- leaking NTLM material with nothing planted locally first.

    THIS IS DELIBERATELY A RULE ABOUT ARGUMENTS, NOT ABOUT ``argv[0]``. An
    "argv[0] must be absolute" rule was considered and REJECTED: this owner
    admits bare ``git`` and ``nvidia-smi`` on purpose (``production_ledger``,
    ``_otr_ledger``, ``_otr_sys_specs``), and each of those wraps its spawn in
    ``except Exception`` -> "unknown", so such a rule would blank the ledger's
    commit stamp on every episode with a green run and a published artifact --
    a hole no test would see. A UNC rule cannot do that: every argument those
    three pass is local, and the pack contains no literal UNC path anywhere.

    A MAPPED DRIVE IS NOT UNC. ``U:\\OTR-BACKUP`` is a drive letter and passes,
    which is what the operator's backup destinations actually look like.
    """
    # A LEADING `//` IS ONLY UNC ON WINDOWS. POSIX permits it at the start of a
    # path and treats it as an ordinary root, so refusing it there would be a
    # false positive on the one platform that cannot be attacked this way.
    # A leading `\\` is treated as UNC everywhere: backslash is not a POSIX
    # separator, so nothing this pack builds starts a real POSIX path with one.
    remote_prefixes = ("\\\\", "\\/", "/\\") + (("//",) if os.name == "nt" else ())
    for index, raw in enumerate(argv):
        if not isinstance(raw, (str, bytes, os.PathLike)):
            continue
        text = os.fsdecode(raw) if not isinstance(raw, str) else raw
        if text[:2] in remote_prefixes:
            raise ExecutableNotAllowed(
                "argv[%d] names a UNC/network location (%r). Reading or "
                "writing one makes this machine authenticate to the host it "
                "names, and nothing this pack ships needs that."
                % (index, text[:120]))


def _check(argv: Sequence[Any]) -> None:
    if isinstance(argv, (str, bytes)):
        raise ExecutableNotAllowed(
            "argv must be a list, not a string -- a string argv is a shell "
            f"parse: {argv!r}")
    if not argv:
        raise ExecutableNotAllowed("argv is empty; nothing to run")
    _no_remote_arguments(argv)
    name = _normalized(argv[0])
    if _allowed(name):
        return
    raise ExecutableNotAllowed(
        f"{name!r} is not an executable this pack runs (from {argv[0]!r}). "
        f"Allowed: {', '.join(sorted(ALLOWED_EXECUTABLES))}, "
        f"{', '.join(sorted(p + '*' for p in ALLOWED_EXECUTABLE_PREFIXES))}. "
        "Add it to ALLOWED_EXECUTABLES with its reason if that is wrong.")


def _no_shell(kwargs: dict) -> None:
    if kwargs.get("shell"):
        raise ExecutableNotAllowed(
            "shell=True is refused: it re-parses an argv list through a shell")
    # `executable=` REPLACES the binary that actually runs while argv[0] keeps
    # its old value -- so `run(["ffmpeg"], executable="cmd.exe")` would pass the
    # allowlist on "ffmpeg" and launch cmd.exe. The check above it would be
    # decoration. Nothing in this pack passes it; refusing it keeps the
    # allowlist a boundary rather than a suggestion.
    if kwargs.get("executable") is not None:
        raise ExecutableNotAllowed(
            "executable= is refused: it replaces the binary that runs while "
            "argv[0] -- the thing the allowlist checked -- stays unchanged. "
            "Put the real program in argv[0].")


def run(argv: Sequence[Any], **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` for an allowed executable. Contracts pass through."""
    _no_shell(kwargs)
    _check(argv)
    return subprocess.run(argv, **kwargs)


def popen(argv: Sequence[Any], **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` for an allowed executable. Returns the real Popen."""
    _no_shell(kwargs)
    _check(argv)
    return subprocess.Popen(argv, **kwargs)
