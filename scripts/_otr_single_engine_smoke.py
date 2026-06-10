"""_otr_single_engine_smoke.py -- live one-shot engine render via /prompt.

Queues a ONE-node OTR_VideoRenderBatch ``mode=single`` graph on the live
headless ComfyUI (:8000) so the heavy in-process forward runs in the EXECUTOR
THREAD (the A-ship finding: never a background route thread), then polls
history and prints the engine's render report. Used for the v1.4 capstone
LTX / Wan opt-in lane proofs:

  * LIVE PASS: the engine is enabled + installed -> report ok=true with a real
    clip path + frame_count.
  * FAIL-CLOSED PASS (--expect-fail): the engine is gated/missing/Sage-blocked
    -> the report (or the node error) NAMES the EngineUnusable reason; exit 0
    only when the failure is the named fail-closed kind.

Usage:
  python scripts/_otr_single_engine_smoke.py --engine ltx_video \
      [--frames 25] [--portrait P] [--audio A] [--expect-fail REASON_SUBSTR]

Exit 0 = the expectation held. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import requests  # noqa: E402

from otr_api import COMFYUI_URL, submit_prompt, poll_history  # noqa: E402

REPORT_DIR = os.environ.get(
    "OTR_SOAK_SERVER_OUTPUT",
    r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\output")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--portrait", default="")
    ap.add_argument("--audio", default="")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--expect-fail", default="",
                    help="substring the named failure must carry; the render "
                         "must NOT succeed")
    args = ap.parse_args(argv)

    started = time.time()
    api = {"1": {"class_type": "OTR_VideoRenderBatch", "inputs": {
        "mode": "single", "beats": 1, "oom_index": 0,
        "frame_count": int(args.frames), "engine": args.engine,
        "portrait_path": args.portrait, "audio_path": args.audio,
    }}}
    print("[single] %s engine=%s frames=%d expect_fail=%r"
          % (COMFYUI_URL, args.engine, args.frames, args.expect_fail),
          flush=True)
    prompt_id = submit_prompt(api)
    print("[single] QUEUED prompt_id=%s" % prompt_id, flush=True)
    status, err = poll_history(prompt_id, timeout_s=args.timeout)
    print("[single] history status=%s %s" % (status, err[:300]), flush=True)

    report = {}
    try:
        hist = requests.get("%s/history/%s" % (COMFYUI_URL, prompt_id),
                            timeout=15).json().get(prompt_id, {})
        for node_out in (hist.get("outputs") or {}).values():
            for txt in (node_out.get("text") or []):
                try:
                    report = json.loads(txt)
                except ValueError:
                    pass
    except Exception as exc:  # noqa: BLE001
        print("[single] history fetch failed: %s" % exc, flush=True)
    # Durable fallback: the node also writes otr/aship/node_single_<engine>.json
    if not report:
        p = os.path.join(REPORT_DIR, "otr", "aship",
                         "node_single_%s.json" % args.engine)
        if os.path.isfile(p) and os.path.getmtime(p) > started:
            report = json.load(open(p, encoding="utf-8"))
    print("[single] report: %s" % json.dumps(report)[:1200], flush=True)

    ok = bool(report.get("ok"))
    if args.expect_fail:
        blob = json.dumps(report) + " " + err
        if ok:
            print("[single] FAIL: expected fail-closed but the render "
                  "SUCCEEDED", flush=True)
            return 1
        if args.expect_fail.lower() not in blob.lower():
            print("[single] FAIL: failure did not NAME %r" % args.expect_fail,
                  flush=True)
            return 1
        print("[single] PASS: fail-closed NAMED (%r)" % args.expect_fail,
              flush=True)
        return 0
    if status != "SUCCESS" or not ok:
        print("[single] FAIL: live render did not succeed", flush=True)
        return 1
    print("[single] PASS: live %s render ok (prompt %s)"
          % (args.engine, prompt_id), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
