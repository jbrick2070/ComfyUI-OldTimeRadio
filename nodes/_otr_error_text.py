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
    "notebooks", "app", "teamspace",
)
_POSIX_PATH = (
    r"(?:(?<![\w./\\~-])|(?<=file://))/(?:" + "|".join(_POSIX_ROOTS) + r")/"
    r"(?:[^\s/'\"<>|,;()\[\]{}]+/)*[^\s/'\"<>|,;()\[\]{}]*"
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


def _home_tail(match: re.Match) -> str:
    rest = (match.group(1) or "").rstrip("\\/.")
    return re.split(r"[\\/]", rest)[-1] or "~"


def _cut_home(text: str) -> str:
    for prefix in _HOME_PREFIXES:
        if prefix.lower() in text.lower():
            text = re.sub(re.escape(prefix) + "(" + _UNDER_HOME + ")", _home_tail, text, flags=re.IGNORECASE)
    return text

# Attributes some standard exceptions read their text from instead of args.
_PATH_ATTRS = ("filename", "filename2")
_COMMAND_ATTRS = ("cmd",)


def _last_component(match: re.Match) -> str:
    path = match.group(0).rstrip("\\/.")
    tail = re.split(r"[\\/]", path)[-1] if path else ""
    return tail or "<path>"


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
        return type(value)(_redact_value(v) for v in value)
    return value


def redact_exception(exc: BaseException) -> BaseException:
    """Redact paths in ``exc``'s text in place and return the same object."""
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
    return exc


def _guarded(func):
    if getattr(func, "__otr_redacts_paths__", False):
        return func
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def guarded(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                redact_exception(exc)
                raise
    else:
        @functools.wraps(func)
        def guarded(*args, **kwargs):
            try:
                return func(*args, **kwargs)
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
