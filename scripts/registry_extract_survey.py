"""Has Comfy-Org's node-pack-extract pipeline produced a success for ANYONE lately?

The registry card's "N Nodes" badge is fed by Algolia's ``comfy_nodes`` array,
which ``node-pack-extract`` populates by booting a headless CPU ComfyUI on Linux,
running ``pip install -r requirements.txt``, and reading ``/object_info``.  When a
version's ``comfy_nodes`` reads ``null`` it is tempting to blame the pack's own
dependencies -- that reading cost this project a near-miss on 2026-09-03, when the
plan's next suspect was kokoro's spacy/thinc/blis chain.

The check that settles it takes one run: ``?include_status_reason=true`` exposes
``comfy_node_extract_status``, which carries terminal values (``success``,
``invalid_format``).  Survey a broad sample and read the NEWEST success anywhere.
Measured 2026-09-03 over 360 packs: 64 had ever succeeded and the most recent
success anywhere was 2026-04-28, while rgthree (Active, 47 prior successes) had
versions pending since 2026-05-09.  The pipeline is stalled, not picky -- so no
dependency change reaches the badge, and a null is not evidence about our deps.

Re-run this before re-opening the node-count question.  If the newest success is
recent again, the pipeline is back and OUR dependencies become worth testing.
"""

import argparse
import concurrent.futures
import json
import urllib.request

API = "https://api.comfy.org"


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def sample_node_ids(pages, per_page):
    """Node ids from the registry listing, de-duplicated, in listing order."""
    ids = []
    for page in range(1, pages + 1):
        try:
            payload = _get("%s/nodes?page=%d&limit=%d" % (API, page, per_page))
        except Exception as err:  # noqa: BLE001 -- a short sample still answers
            print("  listing page %d unavailable: %s" % (page, err))
            break
        rows = payload.get("nodes") or payload.get("data") or []
        if not rows:
            break
        for row in rows:
            node_id = row.get("id") or row.get("node_id")
            if node_id:
                ids.append(node_id)
    return list(dict.fromkeys(ids))


def newest_success(node_id):
    """``(node_id, newest success timestamp or None, oldest pending or None)``."""
    try:
        versions = _get("%s/nodes/%s/versions?include_status_reason=true" % (API, node_id))
    except Exception:  # noqa: BLE001 -- an unreachable pack is simply not counted
        return None
    successes, pendings = [], []
    for version in versions:
        created = version.get("createdAt")
        status = version.get("comfy_node_extract_status")
        if not created:
            continue
        if status == "success":
            successes.append(created)
        elif status == "pending":
            pendings.append(created)
    return (node_id,
            max(successes) if successes else None,
            min(pendings) if pendings else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pages", type=int, default=6, help="listing pages to sample")
    parser.add_argument("--per-page", type=int, default=60, help="packs per listing page")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--top", type=int, default=15, help="most-recent successes to print")
    parser.add_argument("--node-id", default="comfyui-old-time-radio",
                        help="the pack whose first publish the verdict compares against")
    args = parser.parse_args()

    ids = sample_node_ids(args.pages, args.per_page)
    if args.node_id not in ids:
        ids.append(args.node_id)
    print("sampling %d packs" % len(ids))

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(newest_success, ids):
            if row:
                rows.append(row)

    succeeded = sorted([r for r in rows if r[1]], key=lambda r: r[1], reverse=True)
    print("probed %d packs | %d have ever recorded a successful extract"
          % (len(rows), len(succeeded)))

    if not succeeded:
        print("\nNo pack in this sample has ever extracted successfully.")
        return

    print("\n=== %d most recent successful extracts anywhere ===" % args.top)
    for node_id, created, _pending in succeeded[:args.top]:
        print("  %s   %s" % (created[:19], node_id))

    newest = succeeded[0][1][:19]
    print("\nNEWEST SUCCESS ANYWHERE: %s  (%s)" % (newest, succeeded[0][0]))

    ours = [r for r in rows if r[0] == args.node_id]
    if ours:
        _node_id, our_success, our_pending = ours[0]
        print("%s: newest success=%s  oldest pending=%s"
              % (args.node_id, our_success or "never", (our_pending or "n/a")[:19]))

    # A version published AFTER that pack's newest success and still pending is the
    # signature of a stalled pipeline; the oldest such version says how long ago it
    # stopped keeping up.
    overdue = sorted(pending[:19] for _id, success, pending in rows
                     if pending and (success is None or pending > success))
    if overdue:
        print("oldest version still waiting on an extract: %s" % overdue[0])
    print("\nA newest-success months in the past means the pipeline is stalled for\n"
          "EVERYONE -- a null `comfy_nodes` is then no evidence about our dependencies.")


if __name__ == "__main__":
    main()
