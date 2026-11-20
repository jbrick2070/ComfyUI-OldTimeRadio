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
only the folders above it are dropped. URLs are left alone.
"""

from __future__ import annotations

import functools
import inspect
import os
import re

__all__ = ["redact_paths", "redact_exception", "guard_node_class", "guard_node_classes"]

# A drive-letter path (C:\a\b, C:/a/b) or a UNC share (\\host\share\a). The
# drive letter must stand alone, so a URL scheme ("https:/") never matches.
_WINDOWS_PATH = r"(?:(?<![\w])[A-Za-z]:|\\\\[^\\/\s'\"<>|]+)[\\/][^\s'\"<>|,;()\[\]{}]*"
# A rooted POSIX path with at least one folder (/a/b). The look-behind keeps
# URLs (scheme://host/path), relative paths (a/b) and fractions (1/2) out.
_POSIX_PATH = r"(?<![\w.:/\\~-])/(?:[^\s/'\"<>|,;()\[\]{}]+/)+[^\s/'\"<>|,;()\[\]{}]*"
_PATH_RE = re.compile(f"{_WINDOWS_PATH}|{_POSIX_PATH}")

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
    return _PATH_RE.sub(_last_component, text)


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
