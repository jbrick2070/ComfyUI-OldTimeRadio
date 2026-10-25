"""Phone ping for a finished or dead render, through ntfy (PLAN row 1.3).

    set OTR_NOTIFY_URL=https://ntfy.sh/<a-long-random-topic>
    python scripts/otr_notify.py                    # send a test ping
    python scripts/otr_notify.py --title T --body B

WHY. A leg runs for minutes to hours and the person who started it is usually
not watching. The leg log says what happened; nothing carries that to a phone.
(Idea taken from CreatorOS-Studio's ntfy notifications.)

HOW IT STAYS HONEST.
  * It is never a hidden switch. The runner prints `state_line()` at start:
    `notify: on -> <host>`, `notify: off (OTR_NOTIFY_URL unset)`, or, for a value
    that is set but cannot be used, a line that says no ping will be sent. Only
    the HOST (and port) is printed, never the topic path and never a password,
    because anyone who knows an ntfy.sh topic name can read it -- use a long
    random topic, or self-host.
  * It reports and never decides. `notify()` does not raise, does not retry and
    does not change a run's result: a failed POST prints one line and returns
    False, and the render and its exit code are exactly what they were.
  * It cannot hold the end of a run. One socket operation may take 10 s and the
    whole POST 15 s, so a server that answers a byte at a time still lets go.
  * "sent" means the server took the message. A redirect is never followed: urllib
    would answer a 301/302/303 to a POST by sending a GET with no body, which
    the server answers 200, so the message would be reported as sent and never
    delivered. It is reported as a failure that says to use the final URL.
  * Messages carry an episode id and the obs FILE NAME, no other local path.
  * A `user:password@` in the URL (a self-hosted ntfy with logins) is sent as
    an HTTP Basic `Authorization` header and never rides in the URL, so it is
    never printed. A failure line shows an error's message only when that
    message cannot carry the topic (a network error, scrubbed of the topic and
    credentials anyway); any other error shows its type alone.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

NOTIFY_ENV = "OTR_NOTIFY_URL"
#: Seconds one socket operation (connect, send, receive) may take.
_TIMEOUT_S = 10
#: Seconds the WHOLE post may take. `_TIMEOUT_S` restarts with every byte a
#: server sends, so on its own it lets a slow server hold a run open.
_DEADLINE_S = 15
#: ntfy turns a message past 4096 bytes into an attachment; stay well under it.
_MAX_BODY_BYTES = 3500
_HIDDEN = "<hidden>"
_UNUSABLE_STATE = (
    "notify: ON BUT UNUSABLE (%s is not an http:// or https:// URL with a "
    "host), no ping will be sent" % NOTIFY_ENV
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so the 3xx surfaces as an ``HTTPError``."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def notify_url() -> str:
    """The configured topic URL, or ``""`` when notifications are off."""
    return os.environ.get(NOTIFY_ENV, "").strip()


class _Target(NamedTuple):
    """The configured topic URL, taken apart."""

    url: str                      # what is POSTed to: never a user:password
    authorization: str | None     # "Basic ..." when the URL carried one
    host: str                     # host[:port] -- the only part ever printed
    secrets: tuple[str, ...]      # strings no printed message may contain


def _parse(url: str) -> _Target:
    """Take ``url`` apart. Raises ``ValueError`` with a FIXED message (never the
    URL) when it is not an http(s) URL with a host."""
    unusable = ValueError(
        "%s is not an http:// or https:// URL with a host" % NOTIFY_ENV)
    try:
        parts = urllib.parse.urlsplit(url)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        raise unusable from None
    if parts.scheme not in ("http", "https") or not hostname:
        raise unusable
    host = "[%s]" % hostname if ":" in hostname else hostname
    if port is not None:
        host += ":%d" % port
    secrets = [url, parts.path, urllib.parse.unquote(parts.path), parts.query]
    # The topic alone, too: a message may quote the last path segment without
    # its slash.
    segments = [seg for seg in parts.path.split("/") if seg]
    secrets += segments + [urllib.parse.unquote(seg) for seg in segments]
    authorization = None
    if parts.username is not None:
        user = urllib.parse.unquote(parts.username)
        password = urllib.parse.unquote(parts.password or "")
        pair = ("%s:%s" % (user, password)).encode("utf-8")
        authorization = "Basic " + base64.b64encode(pair).decode("ascii")
        secrets += [parts.username, parts.password or "", user, password]
    bare = urllib.parse.urlunsplit(
        (parts.scheme, host, parts.path, parts.query, ""))
    secrets.append(bare)
    # Longest first, so the whole URL is replaced before the pieces inside it.
    ordered = tuple(sorted({s for s in secrets if len(s) > 1},
                           key=len, reverse=True))
    return _Target(bare, authorization, host, ordered)


def state_line() -> str:
    """The one line a runner prints at start so the switch is never hidden."""
    url = notify_url()
    if not url:
        return "notify: off (%s unset)" % NOTIFY_ENV
    try:
        host = _parse(url).host
    except ValueError:
        return _UNUSABLE_STATE
    return "notify: on -> %s" % host


def _reason(exc: Exception, target: _Target) -> str:
    """Why a POST failed, in words that cannot carry the topic or a password.

    Network errors (refused, DNS, timeout, TLS, an HTTP error status) are
    ``OSError``s whose messages do not quote the URL, and are scrubbed anyway.
    Anything else is shown by type alone: ``http.client.InvalidURL`` quotes the
    request path, which IS the topic.
    """
    if (isinstance(exc, urllib.error.HTTPError) and 300 <= exc.code < 400):
        return ("HTTP %d: the server redirected the message instead of taking "
                "it, and a redirect is never followed (it would turn the POST "
                "into a GET that delivers nothing); put the final URL, usually "
                "https://, in %s" % (exc.code, NOTIFY_ENV))
    if not isinstance(exc, OSError):
        return ("%s (details withheld: this kind of message can carry the topic)"
                % type(exc).__name__)
    text = str(exc)
    for secret in target.secrets:
        text = text.replace(secret, _HIDDEN)
    return "%s: %s" % (type(exc).__name__, text[:200])


def _post(request: urllib.request.Request, outcome: dict) -> None:
    """Send ``request`` and record what happened in ``outcome``. Runs on a
    daemon thread so ``notify`` can give the whole post a deadline."""
    try:
        with _OPENER.open(request, timeout=_TIMEOUT_S) as response:
            outcome["status"] = response.status
    except Exception as exc:  # noqa: BLE001 -- handed back to notify()
        outcome["error"] = exc


def notify(title: str, body: str, priority: str | None = None) -> bool:
    """POST one message to the configured topic. True when the server took it.

    Returns False without printing when notifications are off (the start-up
    state line already said so), and False with one ``[notify] FAILED`` line
    when the POST did not go through. Never raises, and never takes longer than
    ``_DEADLINE_S`` to return.
    """
    url = notify_url()
    if not url:
        return False
    try:
        target = _parse(url)
    except ValueError as exc:
        print("[notify] FAILED: %s" % exc, flush=True)
        return False
    try:
        # HTTP header values are latin-1; a non-ASCII episode title would
        # raise before sending. The body carries the full text in UTF-8.
        headers = {
            "Title": title.encode("ascii", "replace").decode("ascii"),
            "Content-Type": "text/plain; charset=utf-8",
        }
        if priority:
            headers["Priority"] = priority
        if target.authorization:
            headers["Authorization"] = target.authorization
        data = body.encode("utf-8")[:_MAX_BODY_BYTES]
        request = urllib.request.Request(
            target.url, data=data, headers=headers, method="POST")
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        print("[notify] FAILED: %s" % _reason(exc, target), flush=True)
        return False
    outcome: dict = {}
    worker = threading.Thread(
        target=_post, args=(request, outcome), name="otr-notify", daemon=True)
    worker.start()
    worker.join(_DEADLINE_S)
    if worker.is_alive():
        print("[notify] FAILED: no answer within %g s, so the message may or "
              "may not have arrived" % _DEADLINE_S, flush=True)
        return False
    if "error" in outcome:
        print("[notify] FAILED: %s" % _reason(outcome["error"], target), flush=True)
        return False
    status = outcome.get("status")
    if status is not None and 200 <= status < 300:
        return True
    print("[notify] FAILED: %s" % (
        "HTTP %s" % status if status is not None
        else "the send ended without an answer"), flush=True)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send one ping to the OTR_NOTIFY_URL ntfy topic.")
    parser.add_argument("--title", default="OTR notify test")
    parser.add_argument("--body", default="If this reached your phone, the topic works.")
    parser.add_argument("--priority", default=None,
                        help="ntfy priority: min, low, default, high, max")
    args = parser.parse_args(argv)
    print(state_line(), flush=True)
    if not notify_url():
        return 2
    sent = notify(args.title, args.body, priority=args.priority)
    print("[notify] %s" % ("sent" if sent else "not sent"), flush=True)
    return 0 if sent else 1


if __name__ == "__main__":
    sys.exit(main())
