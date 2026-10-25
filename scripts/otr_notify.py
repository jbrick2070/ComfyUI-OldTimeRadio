"""Phone ping for a finished or dead render, through ntfy (PLAN row 1.3).

    set OTR_NOTIFY_URL=https://ntfy.sh/<a-long-random-topic>
    python scripts/otr_notify.py                    # send a test ping
    python scripts/otr_notify.py --title T --body B

WHY. A leg runs for minutes to hours and the person who started it is usually
not watching. The leg log says what happened; nothing carries that to a phone.
(Idea taken from CreatorOS-Studio's ntfy notifications.)

HOW IT STAYS HONEST.
  * It is never a hidden switch. The runner prints `state_line()` at start:
    `notify: on -> <host>` or `notify: off (OTR_NOTIFY_URL unset)`. Only the
    HOST is printed, never the topic path, because anyone who knows an ntfy.sh
    topic name can read it -- use a long random topic, or self-host.
  * It reports and never decides. `notify()` does not raise, does not retry and
    does not change a run's result: a failed POST prints one line and returns
    False, and the render and its exit code are exactly what they were.
  * Messages carry an episode id and the obs FILE NAME, no other local path.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

NOTIFY_ENV = "OTR_NOTIFY_URL"
_TIMEOUT_S = 10
#: ntfy turns a message past 4096 bytes into an attachment; stay well under it.
_MAX_BODY_BYTES = 3500


def notify_url() -> str:
    """The configured topic URL, or ``""`` when notifications are off."""
    return os.environ.get(NOTIFY_ENV, "").strip()


def state_line() -> str:
    """The one line a runner prints at start so the switch is never hidden."""
    url = notify_url()
    if not url:
        return "notify: off (%s unset)" % NOTIFY_ENV
    host = urllib.parse.urlsplit(url).netloc
    return "notify: on -> %s" % (host or "unreadable url")


def notify(title: str, body: str, priority: str | None = None) -> bool:
    """POST one message to the configured topic. True when the server took it.

    Returns False without printing when notifications are off (the start-up
    state line already said so), and False with one ``[notify] FAILED`` line
    when the POST did not go through. Never raises.
    """
    url = notify_url()
    if not url:
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
        data = body.encode("utf-8")[:_MAX_BODY_BYTES]
        request = urllib.request.Request(
            url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            if 200 <= response.status < 300:
                return True
            print("[notify] FAILED: HTTP %s" % response.status, flush=True)
            return False
    except Exception as exc:  # noqa: BLE001 -- reported, never raised
        # The message is scrubbed of the URL: its path is the topic, a secret.
        detail = str(exc).replace(url, urllib.parse.urlsplit(url).netloc)
        print("[notify] FAILED: %s: %s" % (type(exc).__name__, detail[:200]),
              flush=True)
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
