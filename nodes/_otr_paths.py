"""
_otr_paths.py  --  OTR canonical filesystem paths
==================================================

Single source of truth for ComfyUI output / OTR subdirectory
locations. All node code and CLI scripts should import from here
instead of hardcoding ``Path(r"C:\\Users\\...\\ComfyUI\\output")``.

Resolution order for ``comfy_output_dir()``:
  1. ``OTR_OUTPUT_DIR`` environment variable, if non-empty. Highest
     priority -- lets cloud deployments (RunPod, Linux/Mac, Docker)
     and CI override the output root without code edits.
  2. ``folder_paths.get_output_directory()`` -- ComfyUI's own canonical
     output dir API. Available when running inside the ComfyUI
     process; honors ``--output-directory`` and friends automatically.
  3. Walk up from this module's path: ``<repo>/../../../output``. With
     the standard ``custom_nodes/ComfyUI-OldTimeRadio/nodes/`` layout
     this yields ``ComfyUI/output`` -- the location every Jeffrey-on-
     Windows install has been hardcoding for the last six months.
  4. ``Path.cwd() / "output"`` as a final fallback. Last resort for
     ad-hoc CLI invocations from an unusual working dir.

The function is a plain dispatcher with no caching: a fresh resolution
on every call keeps pytest fixtures honest (they can monkey-patch
``OTR_OUTPUT_DIR`` per-test without leaking across cases).

Subdir helpers (``otr_audio_dir``, ``otr_stills_dir``, etc.) are thin
wrappers that compose ``comfy_output_dir()`` with the canonical layout
locked 2026-04-27. None of them ``mkdir`` on access -- callers do that
explicitly when they know they're about to write -- because read-only
consumers (e.g. ledger auto-discover) shouldn't accidentally create
empty dirs as a side effect.

OUTPUT TREE CONTRACT (operator law 2026-06-11; OH-1):
``<output>/otr/`` contains EXACTLY two top-level entries:

  * ``episodes/<episode_id>/...`` -- the asset of record for everything
    episode-scoped, plus the RESERVED system tier ``episodes/_shared/``
    holding ``cache/`` (content-addressed cross-episode copies; never
    the only copy), ``tmp/`` (scratch; TEMP/TMP/OTR_GPU_LEASE_DIR;
    janitor-swept) and ``state/`` (per-machine state).
  * ``obs/`` -- FINAL deliverable videos ONLY, flat, OBS-watched.

Every OUTPUT helper below validates its result against that contract
(``_validate_contract``) and the production write helpers RAISE LOUD on
an empty/invalid ``episode_id``; there is no silent ``_legacy_*`` fallback.
Input/models/log/HF resolvers are exempt (not OTR-output paths). Walkers over
``episodes/`` must skip ``_``-prefixed entries
(``_shared`` is never an episode) -- see ``is_reserved_episode_entry``.
"""
from __future__ import annotations

import os
import re as _re
from pathlib import Path
from typing import Optional

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

#: The environment variable that names the output root. It is read and pinned
#: here alone (tests/test_output_root_single_owner.py); a module that must only
#: NAME it -- the clip reuse key leaves it out, because the pack writes it while
#: a prompt runs -- imports this rather than spelling it.
OUTPUT_DIR_ENV = "OTR_OUTPUT_DIR"


class OtrPathContractError(ValueError):
    """An OTR output-path request violates the output-tree contract
    (empty/invalid episode_id, traversal, or a result outside
    ``otr/{episodes,obs}``). Raised LOUD -- never a silent legacy
    fallback (OH-1, operator law 2026-06-11)."""


#: A URL scheme -- `http://`, `file://`, `smb://`. Drive letters ("C:\\") are
#: NOT schemes: the pattern requires two or more leading letters before "://".
_URL_SCHEME = _re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def reject_remote_path(value, field: str) -> str:
    """``value`` unchanged when it names something LOCAL, else raise.

    WHY THIS EXISTS, and it is not traversal. A workflow JSON arrives over
    ComfyUI's unauthenticated ``/prompt`` endpoint, and several nodes take a
    path from it. On Windows the ordinary file APIs open a UNC path
    transparently, so merely STATTING ``\\\\attacker-host\\share\\x``
    makes the machine open an SMB session and authenticate to a host the
    workflow chose -- handing over NTLM material before any containment or
    digest check in this pack has run. The file need not exist and nothing need
    be planted locally first, which makes it strictly cheaper to reach than any
    of the pack's other path defects.

    Refused: UNC (``\\\\host\\share`` and its ``//host/share`` spelling),
    the Windows device namespaces ``\\\\?\\`` and ``\\\\.\\``, and
    anything carrying a URL scheme.

    NOT refused: every ordinary local path, absolute or relative, on any
    platform -- including ``C:\\...``, whose colon is a drive letter and not a
    scheme. Nothing this pack ships reads a remote path, so this cannot fire on
    a real render.
    """
    text = str(value or "").strip()
    if not text:
        return text
    normalized = text.replace("\\", "/")
    if normalized.startswith("//"):
        raise OtrPathContractError(
            "%s must name a local path; %r is a UNC/network path. Reading one "
            "makes this machine authenticate to the host it names."
            % (field, text))
    if _URL_SCHEME.match(text):
        raise OtrPathContractError(
            "%s must name a local path, not a URL (%r)." % (field, text))
    return text


# Walk-up math: this file lives at
# ``ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_paths.py``.
# parents[0] = nodes/, parents[1] = ComfyUI-OldTimeRadio/,
# parents[2] = custom_nodes/, parents[3] = ComfyUI/. The trailing
# ``output`` is what every BUG-LOCAL-* fix has been pinning.
_REPO_WALKUP_OUTPUT = Path(__file__).resolve().parents[3] / "output"


def pin_output_dir_from_comfy() -> Optional[str]:
    """Pin ``OTR_OUTPUT_DIR`` to ComfyUI's live output root when unset.

    Uses ``folder_paths.get_output_directory()`` so ``--output-directory``
    and Desktop remaps win. A registry or git clone then publishes to
    that install's ``output/otr/obs``. Does NOT walk up from this file:
    Comfy Desktop can load the pack through a ``custom_nodes`` junction
    under ``ComfyUI-Installs``, and ``os.path.abspath(__file__)`` keeps
    that install path -- which is how a finished episode missed the
    operator's ``Documents\\...\\output\\otr\\obs`` watch folder.

    Returns the pinned path, or ``None`` when the operator already set
    the knob, folder_paths is missing, or the API returned empty.
    """
    if (otr_env.get(OUTPUT_DIR_ENV) or "").strip():
        return None
    try:
        import folder_paths  # type: ignore

        api_dir = folder_paths.get_output_directory()
    except Exception:
        return None
    text = str(api_dir or "").strip()
    if not text:
        return None
    pinned = os.path.abspath(os.path.expanduser(text))
    otr_env.pin(OUTPUT_DIR_ENV, pinned)
    return pinned


def comfy_output_dir() -> Path:
    """Return the ComfyUI output root.

    See module docstring for resolution order. Caller is responsible
    for any ``mkdir(parents=True, exist_ok=True)`` before writing --
    this function is read-only and never creates directories.
    """
    # Tier 1: explicit env override
    env_override = (otr_env.get("OTR_OUTPUT_DIR") or "").strip()
    if env_override:
        return Path(env_override).expanduser()

    # Tier 2: ComfyUI's own canonical API
    try:
        import folder_paths  # type: ignore

        api_dir = folder_paths.get_output_directory()
        if api_dir:
            return Path(api_dir)
    except ImportError:
        # Running outside the ComfyUI process (CLI scripts, pytest);
        # drop to walk-up.
        pass
    except Exception:
        # Defensive: never let a folder_paths surprise break callers.
        pass

    # Tier 3: walk up from this module's path
    if _REPO_WALKUP_OUTPUT.parent.exists():
        return _REPO_WALKUP_OUTPUT

    # Tier 4: cwd fallback
    return Path.cwd() / "output"


def comfy_input_dir() -> Path:
    """Return the ComfyUI input root.

    Mirror of ``comfy_output_dir`` but for the input side. Used by
    nodes that drop temporary files (per-line portraits for HuMo
    smoke tests, etc.) where ComfyUI expects to read them. Same
    resolution order:

      1. ``OTR_INPUT_DIR`` env var
      2. ``folder_paths.get_input_directory()`` (ComfyUI's API)
      3. Walk up: ``<repo>/../../../input``
      4. ``Path.cwd() / "input"``
    """
    env_override = otr_env.get("OTR_INPUT_DIR")
    if env_override:
        return Path(env_override).expanduser()

    try:
        import folder_paths  # type: ignore

        api_dir = folder_paths.get_input_directory()
        if api_dir:
            return Path(api_dir)
    except ImportError:
        pass
    except Exception:
        pass

    walkup = Path(__file__).resolve().parents[3] / "input"
    if walkup.parent.exists():
        return walkup

    return Path.cwd() / "input"


# --------------------------------------------------------------------- #
# Output-tree contract plumbing (OH-1)
# --------------------------------------------------------------------- #

#: The ONLY entries allowed at the ``otr/`` top level.
_OTR_TOP_ALLOWED = ("episodes", "obs")

#: The reserved system tier under ``episodes/`` (never an episode).
SHARED_DIRNAME = "_shared"


def is_reserved_episode_entry(name: str) -> bool:
    """True for ``episodes/`` children that are NOT episodes (the
    ``_``-prefixed system tier, e.g. ``_shared``). EVERY walker over
    ``otr_episodes_root()`` must skip these (ledger auto-pick named --
    see ``_otr_ledger.find_most_recent_ledger``)."""
    return str(name or "").startswith("_")


def is_remote_path(value) -> bool:
    """True when ``value`` names something off this machine. NEVER raises.

    The non-raising twin of :func:`reject_remote_path`, for the places that must
    not raise: ComfyUI's ``IS_CHANGED`` fingerprint hooks. Those run BEFORE a
    node executes, with the RAW workflow inputs, so the execute-time refusal
    cannot protect them -- a UNC value reaches ``open()`` / ``isfile()`` /
    ``stat()`` during fingerprinting and Windows authenticates to the host the
    workflow named. Raising there is the wrong answer too: a fingerprint hook
    that throws can fail validation for an ordinary missing file. Callers
    instead take the answer they already have for "cannot read it".
    """
    try:
        return reject_remote_path(value, "_probe") is None
    except OtrPathContractError:
        return True
    except Exception:  # noqa: BLE001 -- a predicate must never raise
        return False


def reject_remote_paths(**named) -> None:
    """:func:`reject_remote_path` for several inputs at once.

    Called at a node's EXECUTE METHOD, where provenance is known: these values
    came from the workflow. Deliberately NOT enforced at the spawn -- a mapped
    network drive RESOLVES to a UNC path (measured: ``Path("U:\\").resolve()``
    -> ``\\\\10.55.0.2\\4060-TRANSFER\\``), so a rule applied to argv
    would refuse a legitimate render on any install whose output lives on one.
    The operator's own transfer drive is such a drive.
    """
    for field, value in named.items():
        reject_remote_path(value, field)


def _extra_output_roots() -> list:
    """Operator-declared extra write roots, from ``$OTR_EXTRA_OUTPUT_ROOTS``.

    Separator is ``;`` (a Windows path carries ``:`` in ``C:\\``, so ``os.pathsep``
    is right on Windows and ``;`` is unambiguous everywhere). Empty by default.
    An env value is OPERATOR configuration -- a workflow arriving over ``/prompt``
    cannot set one -- so this widens the contract without widening the attack
    surface. This is how an operator keeps writing to a destination outside the
    ComfyUI output tree after :func:`confine_to_output_tree` landed.
    """
    raw = (otr_env.get("OTR_EXTRA_OUTPUT_ROOTS") or "").strip()
    if not raw:
        return []
    out = []
    for chunk in raw.split(";"):
        text = chunk.strip()
        if not text:
            continue
        # As TYPED, not `.resolve()`d (2026-10-18): resolving turns a mapped
        # drive into its UNC spelling, so `U:\refs\x.wav` stopped matching the
        # root the operator named. `_confine` compares both the typed and the
        # real spelling of every root itself.
        try:
            out.append(Path(os.path.abspath(os.path.expanduser(text))))
        except Exception:  # noqa: BLE001 -- a bad entry must not break writing
            continue
    return out


#: ``C:`` not followed by a separator: a DRIVE-RELATIVE path, which resolves
#: against that drive's current directory -- wherever that happens to be.
_DRIVE_RELATIVE = _re.compile(r"^[A-Za-z]:(?![\\/])")
#: A drive letter that IS followed by a separator: the one colon a Windows path
#: may carry. Any other colon names an NTFS alternate data stream or a device.
_DRIVE_ABSOLUTE = _re.compile(r"^[A-Za-z]:[\\/]")


def _refuse_by_text(text: str, field: str) -> None:
    """The refusals that need no filesystem: decided on the value as typed.

    In this order because every later step TOUCHES the disk (``realpath``
    resolves links, and on Windows a UNC value opens an SMB session), so
    anything that can be refused from the string alone is refused first.
    """
    if "\x00" in text:
        raise OtrPathContractError("%s contains a NUL character." % field)
    reject_remote_path(text, field)
    if ".." in text.replace("\\", "/").split("/"):
        raise OtrPathContractError(
            "%s contains a '..' traversal segment (%r). Name the file directly; "
            "it must lie inside a permitted folder." % (field, text))
    if os.name == "nt":
        if _DRIVE_RELATIVE.match(text):
            raise OtrPathContractError(
                "%s is drive-relative (%r); give a full path such as C:\\..." % (field, text))
        body = text[2:] if _DRIVE_ABSOLUTE.match(text) else text
        if ":" in body:
            raise OtrPathContractError(
                "%s names an alternate data stream or device (%r)." % (field, text))


def _inside(path: str, root: str) -> bool:
    """Is ``path`` ``root`` or below it? Case-insensitive where the OS is."""
    try:
        path_n, root_n = os.path.normcase(path), os.path.normcase(root)
        return os.path.commonpath([path_n, root_n]) == root_n
    except ValueError:  # different drives, or a relative/absolute mix
        return False


def _confine(value, field: str, candidates, *, base=None, verb: str = "read") -> str:
    """The containment both directions share: ``value`` proven inside a root.

    The recipe the registry's reviewers accept (comfyui-darkroom went from
    Flagged to Active on it, 2026-10-01): refuse by text first, anchor a
    relative path, then require the path inside a root TWICE -- lexically
    (case-folded ``commonpath`` on the normalized absolute path, before any
    link is followed) and again after ``realpath`` (so a symlink or junction
    inside a root that points out of it is refused). Returns the LEXICAL
    absolute path: a mapped network drive's realpath is a UNC path, which the
    next node's remote check would refuse.
    """
    text = str(value or "").strip()
    if not text:
        return text
    _refuse_by_text(text, field)
    path = os.path.expanduser(text)
    if not os.path.isabs(path) and base is not None:
        path = os.path.join(str(base), path)
    lexical = os.path.normpath(os.path.abspath(path))
    roots = []
    for root in candidates:
        try:
            root_text = os.path.expanduser(str(root))
            roots.append((os.path.normpath(os.path.abspath(root_text)),
                          os.path.realpath(root_text)))
        except Exception:  # noqa: BLE001 -- an unusable root simply does not match
            continue
    # EITHER SPELLING OF A ROOT ADMITS, THE REAL LOCATION DECIDES (Sonnet QA,
    # 2026-10-18). Gating on the typed spelling alone refused paths the pack
    # itself produced: the mux and the caption burn hand on `.resolve()`d
    # episode paths, so a root typed through an 8.3 short name or a junction
    # no longer matched. So a value is considered when it lies lexically under
    # a root's typed OR real spelling -- and nothing else is ever resolved, so
    # realpath never touches a value outside every root (a mapped drive, a
    # device). The file read is the one at `real`, so `real` inside a root's
    # real location is the test; lexically inside but really outside is the
    # link escape, refused with its own message. (A mapped drive whose
    # `.resolve()`d spelling is UNC is still refused by the text check above.)
    if any(_inside(lexical, lex_root) or _inside(lexical, real_root)
           for lex_root, real_root in roots):
        try:
            real = os.path.realpath(lexical)
        except Exception as exc:  # noqa: BLE001 -- a malformed path is a refusal
            raise OtrPathContractError(
                "%s is not a usable path (%r): %s" % (field, text, exc))
        if any(_inside(real, real_root) for _lex, real_root in roots):
            return lexical
        raise OtrPathContractError(
            "%s (%s) leads through a link to %s, outside every permitted %s "
            "root." % (field, lexical, real, verb))
    raise OtrPathContractError(
        "%s resolves to %s, which is outside every permitted %s root (%s). "
        "Set OTR_EXTRA_OUTPUT_ROOTS to authorize another one."
        % (field, lexical, verb, ", ".join(lex for lex, _real in roots) or "none"))


def _comfy_temp_dir() -> Optional[Path]:
    try:
        import folder_paths  # type: ignore

        api_dir = folder_paths.get_temp_directory()
        return Path(api_dir) if api_dir else None
    except Exception:  # noqa: BLE001 -- outside ComfyUI there is no temp root
        return None


def read_roots() -> list:
    """Where a node may READ a path its inputs name -- the approved folders
    only (operator, 2026-10-18: "usually otr/episodes for most files"):
    ``otr/episodes``, the publication folder ``otr/obs``, ComfyUI's input
    folder (where a user drops media, and where ComfyUI's own upload writes),
    its temp folder, and ``$OTR_EXTRA_OUTPUT_ROOTS``. Not the rest of the
    output tree: other packs' files are not this pack's to read.

    Plus the two places the 3D mesh stage writes when they differ from the
    above (Sonnet QA, 2026-10-18): ``otr/episodes`` under ComfyUI's CONFIGURED
    output folder (SaveGLB must write there, and it can differ from
    ``OTR_OUTPUT_DIR``), and ``$OTR_MESH_CACHE_DIR``. Both are server
    configuration, never a workflow value."""
    candidates = []
    for _root in (otr_episodes_root, otr_obs_dir, comfy_input_dir, _comfy_temp_dir,
                  _configured_output_episodes, _mesh_cache_pin):
        try:
            found = _root()
        except Exception:  # noqa: BLE001 -- an unusable root simply does not match
            continue
        if found:
            candidates.append(found)
    return candidates + _extra_output_roots()


def _configured_output_episodes() -> Optional[str]:
    try:
        import folder_paths  # type: ignore

        out = folder_paths.get_output_directory()
        return os.path.join(out, "otr", "episodes") if out else None
    except Exception:  # noqa: BLE001 -- outside ComfyUI there is no configured output
        return None


def _mesh_cache_pin() -> Optional[str]:
    return (otr_env.get("OTR_MESH_CACHE_DIR") or "").strip() or None


def voice_read_roots() -> list:
    """The read roots plus the models roots, where a voice bank's reference
    WAVs live (``models/TTS/refs/...``): ComfyUI's ``models_dir``, the pack's
    models-root authority, and the migrated ``C:\\ComfyUI-Models`` tree the
    voice resolver also searches."""
    candidates = read_roots()
    try:
        import folder_paths  # type: ignore

        candidates.append(folder_paths.models_dir)
    except Exception:  # noqa: BLE001 -- outside ComfyUI there is no models_dir
        pass
    try:
        try:
            from ._otr_models_root import _models_root
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_models_root import _models_root  # type: ignore
        candidates.append(_models_root())
    except Exception:  # noqa: BLE001 -- ModelsRootUnresolved off Windows
        pass
    if os.name == "nt":
        candidates.append("C:\\ComfyUI-Models")
    return candidates


def confine_to_read_roots(value, field: str, *, roots=None, base=None) -> str:
    """``value`` as an absolute path PROVEN inside a read root, else raise.

    The read twin of :func:`confine_to_output_tree`. Registry review flagged
    2.3.21 and 2.3.22 "policy-v0.6: arbitrary-file-read, path-traversal": a
    path input reached ffmpeg / ``open()`` / ``json.load`` verbatim, so a
    workflow sent to ComfyUI's unauthenticated ``/prompt`` could read any
    file on the box into an output it then fetched. A relative value is
    anchored at ``base`` (default: the ComfyUI output folder, where every
    episode lives). An empty value returns ``""``: callers read it as "use my
    default", and the defaults are in-tree.
    """
    if base is None:
        base = comfy_output_dir()
    return _confine(value, field, read_roots() if roots is None else roots,
                    base=base, verb="read")


def confined_or_empty(value, *, roots=None, base=None) -> str:
    """The confined absolute path, or ``""`` -- never raises. For the places
    that DEGRADE rather than fail (ledger data, cache fingerprints): they must
    then open THIS path, not the value they were handed, or a relative value
    is checked against one folder and opened against the process CWD (Cursor
    review, 2026-10-18)."""
    try:
        return confine_to_read_roots(value, "_probe", roots=roots, base=base)
    except Exception:  # noqa: BLE001 -- a degrade path must never raise
        return ""


def refused_by_text(value) -> bool:
    """True when ``value`` is refused from its spelling alone (NUL, remote,
    ``..``, drive-relative, alternate data stream) -- for a search that would
    otherwise stat candidates built from it before any containment check.
    Never raises, never touches the disk."""
    try:
        _refuse_by_text(str(value or "").strip(), "_probe")
        return False
    except Exception:  # noqa: BLE001 -- a predicate must never raise
        return True


def is_readable_in_roots(value, *, roots=None, base=None) -> bool:
    """The non-raising twin, for ``IS_CHANGED`` hooks: they run before a node
    executes, on the raw workflow inputs, and must neither touch an
    out-of-root path nor raise. False means "do not stat it"."""
    try:
        return bool(confine_to_read_roots(value, "_probe", roots=roots, base=base))
    except Exception:  # noqa: BLE001 -- a predicate must never raise
        return False


def confine_to_output_tree(value, field: str, *, roots=None) -> str:
    """``value`` resolved, PROVEN to land under a permitted write root, else raise.

    WHY, and it is a different defect from :func:`reject_remote_path`. That one
    refuses a path that is off this MACHINE. This one refuses a path that is off
    the permitted TREE. A node declares ``output_path`` (or a source path whose
    directory decides where the output lands) in ``INPUT_TYPES``; ComfyUI's
    unauthenticated ``/prompt`` sets any literal for it -- ``forceInput`` is a
    frontend hint the server never reads, and a default value is irrelevant
    because the caller supplies their own. Without this the value reached
    ``ffmpeg -y``/``shutil.copy2``/``os.replace`` verbatim, so a caller chose an
    arbitrary local destination and overwrote whatever was there.

    THE PERMITTED ROOTS, in order: the ComfyUI output root
    (:func:`comfy_output_dir`, which honours ``$OTR_OUTPUT_DIR`` so the 4060 keeps
    working), the operator's declared publication root (:func:`otr_obs_dir` --
    ``$OTR_OBS_DIR`` may point anywhere and the mux publishes there by design),
    ComfyUI's own input directory, and anything in ``$OTR_EXTRA_OUTPUT_ROOTS``.
    Callers may pass ``roots`` to narrow it further.

    THE INPUT DIRECTORY IS DELIBERATE and worth the sentence. It is where a user
    drops media, this pack stages engine assets into it
    (``wrapper_bridge.stage_into_comfy_input``), and a graph that captions a
    video sitting there would otherwise have its output refused -- the default
    destination is computed BESIDE the input. Permitting it hands an attacker
    nothing: ComfyUI core's own unauthenticated ``POST /upload/image`` already
    writes arbitrary bytes into that directory. The ban class this function
    closes is a write to an arbitrary location on the box, not a write to a
    directory ComfyUI itself manages.

    RESOLVED-TO-RESOLVED, deliberately: a mapped network drive resolves to a UNC
    path (measured, ``Path("U:\\").resolve()`` -> ``\\\\10.55.0.2\\...``), so an
    output root that lives on one only matches itself after both sides resolve.
    Call this AFTER :func:`reject_remote_path`, never before -- the remote refusal
    is about the value as typed, and resolving first would launder a UNC spelling
    into a local-looking one.

    An empty value returns ``""`` unchanged: every caller reads it as "use my
    default", and the defaults are already in-tree. A ``..`` segment is refused
    textually as well as by containment, because the message is clearer and
    because Windows collapses ``..`` lexically in some spellings before the
    filesystem ever sees it.
    """
    # ONE CORE FOR BOTH DIRECTIONS (2026-10-18): the write guard now runs the
    # same text refusals and the same lexical-then-realpath containment as
    # `confine_to_read_roots`. Its roots are unchanged. A relative value still
    # anchors at the process CWD, as `Path.resolve()` did.
    if roots is None:
        candidates = [comfy_output_dir()]
        # `otr_obs_dir()` runs `_validate_contract`, and `comfy_input_dir()`
        # reads `folder_paths`; both can raise, and `otr_obs_dir` raises the
        # SAME exception type this function does. Collected defensively so a
        # contract problem in a root surfaces as "that root does not match"
        # rather than masquerading as "your path is outside the tree".
        for _root in (otr_obs_dir, comfy_input_dir):
            try:
                candidates.append(_root())
            except Exception:  # noqa: BLE001 -- an unusable root simply does not match
                continue
        candidates += _extra_output_roots()
    else:
        candidates = list(roots)
    return _confine(value, field, candidates, base=None, verb="write")


def _validate_episode_id(episode_id: str) -> str:
    """A production write helper's ``episode_id`` must be a non-empty
    single path component and not the reserved system tier. RAISES
    :class:`OtrPathContractError` LOUD -- the ``_legacy_*`` fallback
    era is over (OH-1)."""
    eid = str(episode_id or "").strip()
    if not eid:
        raise OtrPathContractError(
            "empty episode_id: per-episode output helpers REQUIRE a real "
            "episode_id (the _legacy_* fallbacks were removed by the "
            "output-tree contract, 2026-06-11)")
    # ":" too (2026-10-18): on Windows `Path(root) / "C:"` RESETS to a
    # drive-relative path, so an id of "C:" walked out of the episodes root.
    if any(t in eid for t in ("/", "\\", "..", "\x00", ":")):
        raise OtrPathContractError(
            "episode_id %r contains a path separator/traversal token" % eid)
    if set(eid) == {"."}:
        # A dots-only id is a NAVIGATION token, not a name. "." collapses
        # lexically, so `episodes/./clips` resolves to
        # `episodes/clips` -- a directory that still passes
        # _validate_contract (its first part is "episodes") while sitting
        # OUTSIDE any episode, where every ledger walker over
        # otr_episodes_root() would read it AS one. ".." is caught by the
        # separator/traversal test above; a single dot is not, because the
        # substring test looks for ".." and finds nothing. Measured 2026-09-11
        # (PBUG-20260911-03): before this guard, a per-episode helper handed
        # "." returned <output>/otr/episodes/<subdir> and was accepted.
        raise OtrPathContractError(
            "episode_id %r is a path-navigation token, not an episode name"
            % eid)
    if is_reserved_episode_entry(eid):
        raise OtrPathContractError(
            "episode_id %r is a reserved system entry (%s is never an "
            "episode)" % (eid, SHARED_DIRNAME))
    return eid


def _validate_contract(p: Path) -> Path:
    """Assert ``p`` lands under ``otr/{episodes,obs}`` (resolve() +
    relative_to -- the OH-1 output-helper guard). Scoped to OUTPUT
    helpers ONLY; input/models/log/HF resolvers never call this."""
    base = (comfy_output_dir() / "otr").resolve()
    try:
        rel = p.resolve().relative_to(base)
    except ValueError:
        raise OtrPathContractError(
            "output path %s escapes the otr/ root %s" % (p, base))
    parts = rel.parts
    if not parts or parts[0] not in _OTR_TOP_ALLOWED:
        raise OtrPathContractError(
            "output path %s violates the output-tree contract: otr/ top "
            "level is EXACTLY %r (got %r)"
            % (p, list(_OTR_TOP_ALLOWED), parts[:1]))
    return p


def otr_shared_root() -> Path:
    """The reserved system tier: ``<output>/otr/episodes/_shared/``.
    Never an episode; walkers skip it (``is_reserved_episode_entry``)."""
    return _validate_contract(otr_episodes_root() / SHARED_DIRNAME)


def otr_shared_cache_dir() -> Path:
    """Content-addressed cross-episode CACHE tier:
    ``<output>/otr/episodes/_shared/cache/``. Holds e.g. the AS-5
    content-addressed portrait/still pool (``{sha256}.png``). A cache
    entry is NEVER the only copy -- the episode dir holds the asset of
    record (ST-3 materialization)."""
    return _validate_contract(otr_shared_root() / "cache")


def otr_shared_tmp_dir() -> Path:
    """Scratch tier: ``<output>/otr/episodes/_shared/tmp/``. The launch
    scripts point TEMP/TMP/OTR_GPU_LEASE_DIR here; the janitor sweeps
    stale entries (the ONE sanctioned auto-delete, OH-3)."""
    return _validate_contract(otr_shared_root() / "tmp")


def otr_episodes_root() -> Path:
    """Root of the per-episode workspace tree: ``<output>/otr/episodes/``.

    Each episode has its own subdir under here:
    ``<output>/otr/episodes/<episode_id>/{audio,stills,portraits,videos}/``.
    The ledger auto-pick logic walks this root looking
    for ``*/audio/*_ledger.json`` files across all episodes.

    Added 2026-05-02 EVENING per Jeffrey directive: every per-episode
    working file nests inside that episode's own folder so the
    workspace is organized by episode, not by file type.
    """
    return comfy_output_dir() / "otr" / "episodes"


def otr_audio_dir(episode_id: str = "") -> Path:
    """Per-episode audio + ledger dir: ``<output>/otr/episodes/<episode_id>/audio/``.

    Holds the procgen mp4, the canonical ``<episode_id>_ledger.json``,
    Bark / Kokoro output wavs, and MusicGen + AudioGen cache files
    for this episode.

    The ``episode_id`` argument is REQUIRED: an empty/invalid value
    RAISES :class:`OtrPathContractError` LOUD (OH-1).

    For ledger auto-pick across all episodes, use
    ``otr_episodes_root()`` and walk ``*/audio/*_ledger.json``
    (skipping ``_``-prefixed entries -- ``is_reserved_episode_entry``).

    Cache trade-off (Jeffrey acknowledged 2026-05-02 EVENING):
    Per-episode audio dirs lose the cross-episode cache hit that a single
    shared ``otr/audio/`` dir gave MusicGen + AudioGen (SHA-keyed cache
    files, same prompt across two episodes) -- each episode re-renders
    music + sfx even when prompts are identical. Acceptable cost for the
    cleaner organization; revisit only if cache loss becomes a wallclock pain.
    """
    eid = _validate_episode_id(episode_id)
    return _validate_contract(otr_episodes_root() / eid / "audio")


def otr_stills_dir(episode_id: str = "") -> Path:
    """Per-episode FLUX cast / environment stills + radio bookend dir:
    ``<output>/otr/episodes/<episode_id>/stills/``.

    Holds ``full_env_NNNNN_.png`` cast environment portraits and the
    ``radio_bookend_<episode_id>.png`` LTX I2V reference still.
    An empty/invalid ``episode_id`` RAISES LOUD (OH-1). The cross-episode
    content-addressed pool lives at ``otr_shared_cache_dir()``.
    """
    eid = _validate_episode_id(episode_id)
    return _validate_contract(otr_episodes_root() / eid / "stills")


def otr_clips_dir(episode_id: str) -> Path:
    """Per-episode DURABLE per-beat motion-clip dir:
    ``<output>/otr/episodes/<episode_id>/clips/``.

    The final per-beat silent clip each video engine renders lands here (moved
    out of the janitor-swept ``episodes/_shared/tmp`` scratch tier -- operator
    directive: every rendered asset lives under ``otr/episodes/<ep>/``, the final
    episode only in ``otr/obs/``). The janitor sweeps ONLY ``_shared/tmp``, so
    ``clips/`` is never touched. OTR_SilentComposite reads these stable paths via
    the clip manifest. Distinct from ``videos/`` (the legacy per-line piece dir).
    """
    eid = _validate_episode_id(episode_id)
    return _validate_contract(otr_episodes_root() / eid / "clips")


def otr_state_dir() -> Path:
    """Per-machine OTR runtime-state dir:
    ``<output>/otr/episodes/_shared/state/``.

    Persistent runtime state (news_history.json, node run reports, future
    per-machine cursors) lives under the user's ComfyUI output tree, in the
    ``episodes/_shared`` system tier (output-tree contract, OH-1) -- the
    otr/ top level is EXACTLY ``episodes`` + ``obs``. State that's
    per-episode goes under ``otr/episodes/<episode_id>/``.

    Caller is responsible for ``mkdir(parents=True, exist_ok=True)``.
    """
    return _validate_contract(otr_shared_root() / "state")


def otr_runtime_log_path() -> Path:
    """Persistent OTR runtime heartbeat log (VRAM, orchestrator, video).

    Lives under ``otr_state_dir()`` so registry installs (often read-only)
    and ComfyUI-Manager pack replacements do not erase it. ``OTR_RUNTIME_LOG_PATH``
    overrides the destination verbatim."""
    pinned = (otr_env.get("OTR_RUNTIME_LOG_PATH") or "").strip()
    if pinned:
        return Path(pinned).expanduser()
    return _validate_contract(otr_state_dir() / "otr_runtime.log")


def otr_sidecar_stderr_path(basename: str) -> Path:
    """Captured stderr for Path-B audio sidecar workers.

    ``OTR_SIDECAR_STDERR_DIR`` overrides the directory; the basename is fixed
    per engine (``_otr_chatterbox_worker.err``, etc.)."""
    name = os.path.basename(str(basename or "").strip())
    if not name:
        raise OtrPathContractError("sidecar stderr basename must be non-empty")
    override = (otr_env.get("OTR_SIDECAR_STDERR_DIR") or "").strip()
    if override:
        return Path(override).expanduser() / name
    return _validate_contract(otr_state_dir() / name)


def otr_obs_dir() -> Path:
    """OBS-watched final-deliverable dir: ``$OTR_OBS_DIR`` when pinned
    (returned AS TYPED -- unresolved, and not held to the in-tree contract:
    it is the operator's declared publication root), else
    ``<output>/otr/obs/`` under the contract.

    Holds EXACTLY ONE mp4 per episode -- the final user-facing
    deliverable, published by OTR_MasterAudioMux under the name that
    ``_otr_shared/obs_name.py`` spells. OBS's directory_sorter
    watches this dir and queues each finished episode in turn.

    Render chain (one final mp4 per episode lands here):

      1. OTR_SilentComposite -> otr/episodes/<ep>/<ep>_silent.mp4
         (composite intermediate, already at delivery resolution;
         per-clip model enhancement happens inside SilentComposite via
         nodes/_otr_upscale_engines/)
      2. OTR_CaptionBurn and OTR_CreditsRoll add the captions and the
         credits tail beside it, in the same episode folder
      3. OTR_MasterAudioMux muxes the frozen master audio on LAST and
         publishes the cut to otr/obs/ (this dir)

    The "one mp4 per episode" rule is load-bearing: a second mp4 landing in
    obs/ per episode breaks the broadcast-folder contract. Intermediates
    live under their episode; only the broadcast cut lives here.
    """
    pinned = (otr_env.get("OTR_OBS_DIR") or "").strip()
    if pinned:
        # AN EXPLICIT PIN IS A DECLARED PUBLICATION ROOT, NOT AN ESCAPE
        # (kibitz runpod-found-fixes R-A, 2026-09-04). The headless launcher
        # renders into one tree while the operator watches another, and
        # `_otr_ledger._published_obs_path` already authorizes exactly this
        # root. The in-tree assert below guards ACCIDENTAL escapes; a path the
        # operator typed is not one. Returned unresolved, as typed, so the
        # ledger and the mux compare what was configured.
        return Path(pinned).expanduser()
    return _validate_contract(comfy_output_dir() / "otr" / "obs")


__all__ = [
    "pin_output_dir_from_comfy",
    "comfy_output_dir",
    "comfy_input_dir",
    "OtrPathContractError",
    "reject_remote_path",
    "reject_remote_paths",
    "is_remote_path",
    "confine_to_output_tree",
    "is_reserved_episode_entry",
    "otr_shared_root",
    "otr_shared_cache_dir",
    "otr_shared_tmp_dir",
    "otr_audio_dir",
    "otr_stills_dir",
    "otr_clips_dir",
    "otr_obs_dir",
    "otr_state_dir",
    "otr_runtime_log_path",
    "otr_sidecar_stderr_path",
]
