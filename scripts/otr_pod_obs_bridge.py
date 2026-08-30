#!/usr/bin/env python
"""Pull PUBLISHED episodes from a remote ComfyUI (a rented pod) into local obs.

    python scripts/otr_pod_obs_bridge.py <podId> [--watch] [--dest <dir>]

obs is the operator's success signal, so two rules bind this tool and both are
enforced in code rather than by care:

**ADD-ONLY.** It writes new files and never moves, renames, sorts or deletes
anything already in obs. A name that already exists is SKIPPED, never
overwritten.

**PUBLISHED ONLY, and this is the correction that created this file.** The first
version took "the largest video in /history" and put a 27 MB intermediate into
obs -- the pre-post-processing mux out of an episode's `audio/` folder, from a
leg that died at OTR_SceneAwareScopes. It had no credits, no captions and no
procgen blend, and it sat in the broadcast folder looking like a finished
episode. "Biggest" only means "published" when the render succeeded, which is
exactly the case where you do not need a heuristic.

A published episode is identifiable, so identify it: obs_publish writes into
the `otr/obs` subfolder, and the finished artifact carries `_final` in its name.
Require BOTH. An intermediate satisfies neither.

**WHY THE HTTP ROUTE CANNOT WORK, AND WHY SSH IS THE DEFAULT (PBUG-20260830-24).**
Proven on a live pod leg that returned RESULT SUCCESS and published a 77 MB
episode to `otr/obs`: `/history` recorded exactly one video, the INTERMEDIATE in
the episode's `audio/` folder. `OTR_MasterAudioMux` publishes the deliverable
but returns a bare tuple with no `ui` payload, so ComfyUI never records the
published copy -- it is not in `/history` under any key, and no amount of
filtering can find what was never written down. The `--http` route below is
therefore correct AND permanently empty on a healthy render; it is kept only
because it is the honest thing to report when SSH is unavailable.

The directory obs_publish writes IS the record, so read that directory. SSH is
the default for exactly that reason.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

DEFAULT_DEST = os.path.join(
    os.path.expanduser("~"), "Documents", "ComfyUI", "output", "otr", "obs")

#: Cloudflare fronts the pod proxy and rejects urllib's default user-agent
#: outright -- every poll returns HTTPError while the identical curl succeeds.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

#: A published episode lives here and carries this marker. Both, not either.
_PUBLISHED_SUBFOLDER = "otr/obs"
_PUBLISHED_MARKER = "_final"


def _open(url: str, timeout: int):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(base: str, path: str, timeout: int = 60):
    with _open(base + path, timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def published_videos(history: dict) -> list:
    """Only artifacts obs_publish actually published. Intermediates excluded."""
    out = []
    for _pid, entry in (history or {}).items():
        for _node, payload in (entry.get("outputs") or {}).items():
            for key in ("gifs", "videos", "images", "files"):
                for item in (payload.get(key) or []):
                    fn = str(item.get("filename") or "")
                    sub = str(item.get("subfolder") or "").replace("\\", "/")
                    if not fn.lower().endswith((".mp4", ".mkv", ".webm")):
                        continue
                    if _PUBLISHED_SUBFOLDER not in sub:
                        continue
                    if _PUBLISHED_MARKER not in fn:
                        continue
                    out.append((fn, sub, str(item.get("type") or "output")))
    return out


def fetch(base: str, fn: str, sub: str, typ: str, dest_dir: str):
    dest = os.path.join(dest_dir, fn)
    if os.path.exists(dest):
        print("  SKIP (add-only, already present): %s" % fn)
        return None
    q = urllib.parse.urlencode({"filename": fn, "subfolder": sub, "type": typ})
    tmp = dest + ".part"
    with _open("%s/view?%s" % (base, q), 1800) as r, io.open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)          # atomic: a dropped transfer leaves .part
    return dest


def sync_over_ssh(args) -> int:
    """Pull every published episode the pod has and this box does not.

    ADD-ONLY, like the HTTP route: an existing name is skipped, never
    overwritten, and nothing already in obs is moved, renamed or removed.
    """
    import subprocess
    if not args.host:
        print("  --host <ip> is required for the SSH route "
              "(or pass --http to use the empty /history route)")
        return 2
    common = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
              "-o", "BatchMode=yes", "-i", args.key]
    listing = subprocess.run(
        ["ssh"] + common + ["-p", str(args.port), "root@" + args.host,
         "ls -1 %s/*%s.mp4 2>/dev/null" % (args.pod_obs, _PUBLISHED_MARKER)],
        capture_output=True, text=True, timeout=120)
    names = [ln.strip() for ln in listing.stdout.splitlines() if ln.strip()]
    print("  published episodes on the pod: %d" % len(names))
    os.makedirs(args.dest, exist_ok=True)
    pulled = 0
    for remote in names:
        fn = remote.rsplit("/", 1)[-1]
        dest = os.path.join(args.dest, fn)
        if os.path.exists(dest):
            print("  SKIP (add-only, already present): %s" % fn[:70])
            continue
        tmp = dest + ".part"
        rc = subprocess.run(
            ["scp"] + common + ["-P", str(args.port),
             "root@%s:%s" % (args.host, remote), tmp],
            capture_output=True, text=True, timeout=1800).returncode
        if rc == 0 and os.path.getsize(tmp) > 0:
            os.replace(tmp, dest)
            print("  PULLED %8.1f MB  %s"
                  % (os.path.getsize(dest) / 1048576.0, fn[:60]))
            pulled += 1
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            print("  FAILED %s" % fn[:60])
    print("  %d new episode(s) in obs" % pulled)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pod_id")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--watch", action="store_true",
                    help="poll until the pod's queue is idle, then pull")
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--host", help="pod IP for the SSH route (default route)")
    ap.add_argument("--port", default="22", help="pod SSH port")
    ap.add_argument("--key", default=os.path.join(
        os.path.expanduser("~"), ".ssh", "runpod_otr"))
    ap.add_argument("--pod-obs",
                    default="/workspace/runpod-slim/ComfyUI/output/otr/obs")
    ap.add_argument("--http", action="store_true",
                    help="use the /history route instead of SSH. It is honest "
                         "and it finds nothing -- see the module docstring.")
    ap.add_argument("--max-wait-s", type=int, default=4 * 60 * 60)
    args = ap.parse_args(argv)

    if not args.http:
        return sync_over_ssh(args)

    base = "https://%s-8188.proxy.runpod.net" % args.pod_id
    print("pod  : %s" % base)
    print("dest : %s" % args.dest)
    print("NOTE : the HTTP route reads /history, which does NOT record the")
    print("       published episode (PBUG-20260830-24). Expect zero. Use SSH.")

    if args.watch:
        t0 = time.time()
        while time.time() - t0 < args.max_wait_s:
            try:
                q = get_json(base, "/queue", timeout=30)
                run = len(q.get("queue_running", []))
                pend = len(q.get("queue_pending", []))
                if run == 0 and pend == 0:
                    print("  queue idle after %.0f min" % ((time.time()-t0)/60))
                    break
                print("  running=%d pending=%d (%.0f min)"
                      % (run, pend, (time.time()-t0)/60), flush=True)
            except Exception as exc:
                print("  poll failed: %s" % type(exc).__name__, flush=True)
            time.sleep(args.poll_s)

    try:
        hist = get_json(base, "/history", timeout=180)
    except Exception as exc:
        print("  could not read /history: %s" % type(exc).__name__)
        return 1

    vids = published_videos(hist)
    print("  published episodes in history: %d" % len(vids))
    if not vids:
        print("  nothing to pull. An intermediate is NOT an episode -- if a leg")
        print("  died before obs_publish, there is correctly nothing here.")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    pulled = 0
    for fn, sub, typ in vids:
        try:
            p = fetch(base, fn, sub, typ, args.dest)
            if p:
                print("  PULLED %8.1f MB  %s" % (os.path.getsize(p)/1048576.0, fn))
                pulled += 1
        except Exception as exc:
            print("  FAILED %s: %s" % (fn[:60], type(exc).__name__))
    print("  %d new episode(s) in obs" % pulled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
