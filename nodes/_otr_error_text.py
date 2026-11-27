"""Keep filesystem locations out of the error text an OTR node hands back.

When a node raises, ComfyUI copies ``str(exception)`` into the prompt's
history and the websocket error event, which any client that can reach the
server can read, and into its log, which ``/internal/logs`` serves. An
absolute path in that text names the install folder and, on most machines,
the user's account. OTR builds paths into hundreds of messages, and Python's
own errors add more (``open()`` puts the full path on ``FileNotFoundError``,
``subprocess`` puts the whole command line on ``CalledProcessError``), so the
policy is applied once, at the node boundary, rather than at each raise:

* ``guard_node_class(cls)`` wraps the node's ``FUNCTION`` method. An exception
  leaving it has every absolute path in its text reduced to the last path
  component, in place, then propagates unchanged: same type, same traceback,
  same identity, so ComfyUI's interrupt and out-of-memory handling see exactly
  what they saw before.
  The text a node returns for the canvas (``ui["text"]``) is stored and
  served the same way, so the same wrapper applies the same rule to it.
* ``redact_paths(text)`` is the string rule, exposed for callers that build
  user-facing text themselves.

The file name survives, so a message still says which model or clip failed;
only the folders above it are dropped. URLs and HTTP routes are left alone.

Out of reach on purpose: the ``traceback`` ComfyUI adds beside the message
lists the source file of each frame. ComfyUI builds that from the code's own
location for every pack, and rewriting core's traceback from inside a pack
would be a larger intrusion than the leak it closes.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import re
import sys
from pathlib import Path

__all__ = ["redact_paths", "redact_exception", "guard_node_class", "guard_node_classes",
           "install_log_redaction"]

# A drive-letter path (C:\a\b, C:/a/b) or a UNC share (\\host\share\a). The
# drive letter must stand alone, so a URL scheme ("https:/") never matches.
_WINDOWS_PATH = r"(?:(?<![\w])[A-Za-z]:|\\\\[^\\/\s'\"<>|]+)[\\/][^\s'\"<>|,;()\[\]{}]*"
# A rooted POSIX path under a filesystem top level (/home/a/b, /workspace/x).
# Naming the top levels keeps HTTP routes (/v1/chat/completions, /api/prompt)
# intact, and the look-behind keeps relative paths (a/b) and the tail of a
# longer path out. A colon may precede it ("cache_dir:/home/a/b" is a path; a
# URL's host always stands between its scheme and its path), and so may a
# ``file://`` scheme, whose third slash IS the path's root.
_POSIX_ROOTS = (
    "home", "Users", "Volumes", "root", "tmp", "var", "opt", "srv", "mnt", "media",
    "private", "usr", "etc", "run", "data", "workspace", "content", "kaggle",
    "notebooks", "app", "teamspace", "runpod-volume", "comfyui", "ComfyUI",
    "Applications",
)
#: What may stand to the LEFT of a rooted POSIX path. Shared with the
#: known-folder cut below, so "/root" is never matched inside a URL
#: ("https://example.com/root/x") or inside a longer path ("/var/root/x").
_POSIX_LEFT = r"(?:(?<![\w./\\~-])|(?<=file://))"
# A backslash is NOT a path character in the POSIX rule: in JSON or repr text
# it is the escape before the closing quote ("...x.wav\"") or the start of an
# escaped newline, and taking it into the path corrupted that text.
_POSIX_PATH = (
    _POSIX_LEFT + r"/(?:" + "|".join(_POSIX_ROOTS) + r")/"
    r"(?:[^\s/\\'\"<>|,;()\[\]{}]+/)*[^\s/\\'\"<>|,;()\[\]{}]*"
)
_PATH_RE = re.compile(f"{_WINDOWS_PATH}|{_POSIX_PATH}")


def _home_prefixes() -> tuple[str, ...]:
    """The account's home folder in every spelling a message may carry.

    The path rule stops at whitespace, so on an account named "John Smith" it
    would leave "John Smith\\Documents\\x.wav" behind; the home prefix is cut
    first, whatever it contains.
    """
    try:
        home = str(Path.home()).rstrip("\\/")
    except Exception:  # noqa: BLE001 -- no resolvable home: rely on the path rule
        return ()
    if len(home) < 4:
        return ()
    spellings = {home, home.replace("\\", "/"), home.replace("/", "\\"), home.replace("\\", "\\\\")}
    return tuple(sorted(spellings, key=len, reverse=True))


_HOME_PREFIXES = _home_prefixes()


# What follows the home prefix, up to the end of the path token. The prefix must
# END a folder name: "/Users/jab" matches in "/Users/jab/x" and "/Users/jab", not
# inside "/Users/jabber/x" or "/Users/jab.smith/x", which the path rule handles
# as ordinary paths. A sentence's full stop after it still counts as its end.
_UNDER_HOME = r"(?:(?:\\\\|[\\/])[^\s'\"<>|,;()\[\]{}]*|(?!\.?[\w-]))"


def _spellings(folder: str) -> set:
    """One folder in every spelling a message may carry: either slash, and
    the doubled backslash of a ``repr`` or a JSON dump."""
    folder = folder.rstrip("\\/")
    return {folder, folder.replace("\\", "/"), folder.replace("/", "\\"),
            folder.replace("\\", "\\\\")}


#: A test seam: a tuple here is used as the answer of ``_install_prefixes``
#: as it stands. None (always, in production) means "work it out".
_INSTALL_PREFIXES = None
#: (the folders the answer was built from, the answer). Rebuilt whenever the
#: folders differ, so nothing is frozen: not the partial answer of a call at
#: prestartup, before ComfyUI's ``folder_paths`` is loaded, and not a stubbed
#: ``folder_paths`` a test left behind.
_INSTALL_CACHE: tuple = ((), ())


def _install_folders() -> tuple:
    """The ComfyUI folder this pack sits under, and ComfyUI's own base,
    models, output, input, temp and user folders once it is loaded."""
    folders = []
    pack_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent = os.path.dirname(pack_root)
    if os.path.basename(parent).lower() == "custom_nodes":
        folders.append(os.path.dirname(parent))
    comfy = sys.modules.get("folder_paths")
    if comfy is not None:
        for getter in ("get_output_directory", "get_input_directory",
                       "get_temp_directory", "get_user_directory"):
            try:
                folders.append(str(getattr(comfy, getter)()))
            except Exception:  # noqa: BLE001 -- an older or stubbed folder_paths
                pass
        for attr in ("base_path", "models_dir"):
            value = getattr(comfy, attr, None)
            if isinstance(value, str):
                folders.append(value)
    return tuple(folders)


def _install_prefixes() -> tuple:
    """This install's own folders that the path rule cannot cut whole.

    The path rule stops at whitespace, so an install under a folder with a
    space in its name -- ``D:\\AI Tools\\ComfyUI_windows_portable`` is a
    common one -- left everything after the first space in the text. Like the
    home folder, these are cut as known prefixes first. Only folders with a
    space are listed; every other path is already handled by the rule.
    """
    global _INSTALL_CACHE
    if _INSTALL_PREFIXES is not None:
        return _INSTALL_PREFIXES
    folders = _install_folders()
    if folders == _INSTALL_CACHE[0]:
        return _INSTALL_CACHE[1]
    found = set()
    for folder in folders:
        folder = folder.rstrip("\\/")
        if len(folder) < 4 or not re.search(r"\s", folder):
            continue
        bare = re.split(r"[\\/]", folder)[-1] or "<path>"
        for spelling in _spellings(folder):
            found.add((spelling, bare))
    answer = tuple(sorted(found, key=lambda pair: len(pair[0]), reverse=True))
    _INSTALL_CACHE = (folders, answer)
    return answer


def _cut_home(text: str) -> str:
    """Cut the known folders -- the account's home and this install's own
    folders that hold a space -- before the path rule runs.

    LONGEST FIRST, across both lists. An install under the home folder
    (``C:\\Users\\x\\AI Tools\\ComfyUI``) must be cut as the install folder:
    cutting the shorter home prefix first stops at the space and leaves the
    rest of the path, which the install prefix can then no longer match.

    WITH A LEFT BOUNDARY. A home of ``/root`` (a container, a pod) is also a
    segment of other text: without the boundary ``https://example.com/root/x``
    came out as ``https://example.comx``.
    """
    known = [(prefix, "~") for prefix in _HOME_PREFIXES]
    known += list(_install_prefixes())
    known.sort(key=lambda pair: len(pair[0]), reverse=True)
    lowered = text.lower()
    for prefix, bare in known:
        if prefix.lower() not in lowered:
            continue

        def _tail(match: re.Match, bare=bare) -> str:
            rest = (match.group(1) or "").rstrip("\\/.")
            return re.split(r"[\\/]", rest)[-1] or bare
        left = _POSIX_LEFT if prefix.startswith("/") else r"(?<![\w])"
        text = re.sub(left + re.escape(prefix) + "(" + _UNDER_HOME + ")", _tail, text,
                      flags=re.IGNORECASE)
        lowered = text.lower()
    return text

# Attributes some standard exceptions read their text from instead of args.
_PATH_ATTRS = ("filename", "filename2")
_COMMAND_ATTRS = ("cmd",)


def _last_component(match: re.Match) -> str:
    whole = match.group(0)
    path = whole.rstrip("\\/.")
    tail = re.split(r"[\\/]", path)[-1] if path else ""
    tail = tail or "<path>"
    # A Windows path inside JSON or repr text ends ``...a.wav\"``: the last
    # backslash escapes the closing quote, not the path. An ODD run of
    # backslashes right before a quote keeps one, so the text still parses.
    run = len(whole) - len(whole.rstrip("\\"))
    if run % 2 == 1 and match.string[match.end():match.end() + 1] in ("\"", "'"):
        tail += "\\"
    return tail


def redact_paths(text: str) -> str:
    """Return ``text`` with every absolute path cut to its last component."""
    if not isinstance(text, str) or ("/" not in text and "\\" not in text):
        return text
    return _PATH_RE.sub(_last_component, _cut_home(text))


def _redact_value(value):
    if isinstance(value, str):
        return redact_paths(value)
    if isinstance(value, os.PathLike):
        return redact_paths(os.fspath(value))
    if isinstance(value, (list, tuple)):
        items = [_redact_value(v) for v in value]
        try:
            return type(value)(items)
        except TypeError:
            # A namedtuple takes its fields one by one. Falling over here
            # used to leave the whole ``args`` unredacted.
            try:
                return type(value)(*items)
            except TypeError:
                return tuple(items) if isinstance(value, tuple) else items
    return value


def _redact_one(exc: BaseException) -> None:
    try:
        exc.args = tuple(_redact_value(a) for a in exc.args)
    except Exception:  # noqa: BLE001 -- an exotic args setter must not mask the real error
        pass
    for attr in _PATH_ATTRS + _COMMAND_ATTRS:
        value = getattr(exc, attr, None)
        if value is None:
            continue
        try:
            setattr(exc, attr, _redact_value(value))
        except Exception:  # noqa: BLE001 -- read-only attribute: leave it
            pass


def redact_exception(exc: BaseException) -> BaseException:
    """Redact paths in ``exc``'s text in place and return the same object.

    THE WHOLE CHAIN, not only the top. ``raise X(...) from err`` and a raise
    inside an ``except`` block keep the earlier exception on ``__cause__`` /
    ``__context__``, and the traceback ComfyUI formats for its log and for the
    prompt's history prints every link's own message. A top-level message cut
    to ``voice.pt`` under a ``FileNotFoundError`` still naming the whole path
    disclosed exactly what the cut was for.
    """
    seen: set = set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        _redact_one(current)
        pending.append(getattr(current, "__cause__", None))
        pending.append(getattr(current, "__context__", None))
        # An ExceptionGroup (Python 3.11+) prints every member too.
        members = getattr(current, "exceptions", None)
        if isinstance(members, (list, tuple)):
            pending.extend(m for m in members if isinstance(m, BaseException))
    return exc


def _redact_ui_text(result):
    """Cut absolute paths in the TEXT a node shows on the canvas; return
    ``result``.

    A node that returns ``{"ui": {"text": [...]}, "result": ...}`` has that
    text stored in the prompt's history and sent on the websocket, which any
    client that can reach the server reads -- the same audience as exception
    text. Only ``ui["text"]`` is touched: the file references beside it
    (``filename`` / ``subfolder`` / ``type``) are already relative to
    ComfyUI's own folders, and ``result`` is what the next node receives, so
    a path in it must stay whole.
    """
    if isinstance(result, dict):
        ui = result.get("ui")
        if isinstance(ui, dict) and ui.get("text"):
            try:
                ui["text"] = _redact_value(ui["text"])
            except Exception:  # noqa: BLE001 -- never fail a finished node over its label
                pass
    return result


def _guarded(func):
    if getattr(func, "__otr_redacts_paths__", False):
        return func
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def guarded(*args, **kwargs):
            try:
                return _redact_ui_text(await func(*args, **kwargs))
            except Exception as exc:
                redact_exception(exc)
                raise
    else:
        @functools.wraps(func)
        def guarded(*args, **kwargs):
            try:
                return _redact_ui_text(func(*args, **kwargs))
            except Exception as exc:
                redact_exception(exc)
                raise
    guarded.__otr_redacts_paths__ = True
    return guarded


def guard_node_class(cls) -> bool:
    """Wrap ``cls.FUNCTION`` so its exceptions carry no absolute paths.

    Returns True when the method is guarded (newly or already). The method is
    found where it is defined in the MRO, and a static or class method keeps
    its kind.
    """
    name = getattr(cls, "FUNCTION", None)
    if not isinstance(name, str):
        return False
    for owner in cls.__mro__:
        if name in vars(owner):
            raw = vars(owner)[name]
            break
    else:
        return False
    if isinstance(raw, staticmethod):
        wrapped = staticmethod(_guarded(raw.__func__))
    elif isinstance(raw, classmethod):
        wrapped = classmethod(_guarded(raw.__func__))
    elif callable(raw):
        wrapped = _guarded(raw)
    else:
        return False
    setattr(cls, name, wrapped)
    return True


def guard_node_classes(classes) -> int:
    """Guard every class in ``classes``; return how many are guarded."""
    return sum(1 for cls in classes if guard_node_class(cls))


# ---------------------------------------------------------------------------
# Log lines: the same rule, at the one place every OTR log record is born.
# ---------------------------------------------------------------------------
#
# ComfyUI serves its log at ``/internal/logs`` with no login, and copies it to
# the console. The node-boundary guard above keeps paths out of EXCEPTION text;
# it never saw ordinary log lines, and OTR logged full paths in hundreds of them
# (the Hugging Face cache, the output folder, model snapshots, every episode
# file). The registry's policy reviewer signed 2.3.51 "information-disclosure"
# for paths in exception text, and flagged 2.3.61-2.3.63 with no reason given;
# paths in the log were the leading candidate (Codex review, 2026-11-26).
#
# A logging RECORD FACTORY is the boundary: it runs for every record any logger
# creates, after the level check, whatever the logger's name (OTR uses
# "OTR", "OldTimeRadio", "OTR.*" and module names). Records whose source file
# lives in this pack get their message -- and any traceback or stack text --
# passed through ``redact_paths``. Every other pack's records pass untouched.

#: The pack's root folder (the parent of ``nodes/``), as imported: a junction
#: install reports the junction path in both ``__file__`` and the record, so
#: the plain absolute path is compared first and the resolved one second.
_PACK_ROOTS = tuple({
    os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep,
    os.path.normcase(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + os.sep,
})


def _from_this_pack(pathname) -> bool:
    if not isinstance(pathname, str) or not pathname:
        return False
    here = os.path.normcase(os.path.abspath(pathname))
    return any(here.startswith(root) for root in _PACK_ROOTS)


def _redact_record(record: logging.LogRecord) -> logging.LogRecord:
    try:
        message = record.getMessage()
    except Exception:  # noqa: BLE001 -- a malformed record is logging's to report
        return record
    clean = redact_paths(message)
    if clean != message:
        record.msg, record.args = clean, None
    if record.exc_info and record.exc_info[0] is not None and not record.exc_text:
        try:
            record.exc_text = redact_paths(logging.Formatter().formatException(record.exc_info))
        except Exception:  # noqa: BLE001 -- leave the formatter to try
            pass
    if record.stack_info:
        record.stack_info = redact_paths(record.stack_info)
    return record


def install_log_redaction() -> bool:
    """Wrap the logging record factory so OTR's own records carry no absolute
    paths. Idempotent: returns True when it installed, False when an OTR
    factory was already in place (prestartup installs first; ``__init__``
    calls again harmlessly)."""
    previous = logging.getLogRecordFactory()
    if getattr(previous, "__otr_redacts_paths__", False):
        return False

    def otr_record_factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if _from_this_pack(getattr(record, "pathname", None)):
            record = _redact_record(record)
        return record

    otr_record_factory.__otr_redacts_paths__ = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(otr_record_factory)
    return True
