"""The 24/7 broadcast feed from the command line. (PLAN row 4.1,
planning/creatoros-concepts/PLAN.md; nodes/_otr_shared/broadcast_feed.py.)

Every published episode airs by default: it is hardlinked from ``otr/obs/`` into
``otr/episodes/_shared/broadcast/``, the folder OBS watches. This script is how
an episode is pulled from air or put back. ``otr/obs/`` itself is never touched.

  python scripts/otr_broadcast.py                 list every episode and its air state
  python scripts/otr_broadcast.py pull <name>     take it off air (removes only the link)
  python scripts/otr_broadcast.py put_back <name> put it back on air
  python scripts/otr_broadcast.py sync            make the feed match obs minus the pulled list

<name> is the episode's file name in otr/obs/ (the ComfyUI sidebar tab "On air"
shows it and copies the command). Pulling and putting back are not offered over
HTTP on purpose: ComfyUI routes need no login.

Exit 0 when done, 2 when the request is refused.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes._otr_shared import broadcast_feed  # noqa: E402


def _print_list() -> None:
    rows = broadcast_feed.list_episodes()
    print("[broadcast] feed: %s" % broadcast_feed.feed_dir())
    print("[broadcast] %d published, %d on air" % (
        len(rows), sum(1 for r in rows if r["on_air"])))
    for row in rows:
        when = datetime.datetime.fromtimestamp(row["published_at"]).strftime("%Y-%m-%d %H:%M")
        print("  %-12s %s  %s" % (row["state"], when, row["episode"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default="list",
                        choices=("list", "pull", "put_back", "sync"))
    parser.add_argument("episode", nargs="?", default=None)
    args = parser.parse_args(argv)
    if args.action == "list":
        _print_list()
        return 0
    if args.action in ("pull", "put_back") and not args.episode:
        parser.error("%s needs the episode's file name" % args.action)
    try:
        result = broadcast_feed.act(args.episode, args.action)
    except broadcast_feed.BroadcastError as exc:
        print("[broadcast] refused: %s" % exc)
        return 2
    if args.action == "sync":
        print("[broadcast] synced: %(linked)d linked, %(relinked)d relinked, "
              "%(removed)d removed, %(pending)d pull pending, %(errors)d errors" % result)
        return 0
    line = "[broadcast] %s: %s" % (result["episode"], result["state"])
    if result.get("reason"):
        line += " -- " + result["reason"]
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
