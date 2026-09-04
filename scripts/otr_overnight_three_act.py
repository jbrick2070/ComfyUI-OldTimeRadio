"""THREE-ACT overnight renders on the proven lanes, with a hard stop time.

WHY THIS EXISTS (operator, 2026-09-03 23:00): *"keep it going until 7am with
additional 3 act renders, our picks."* The 1-act matrix answers "does this lane
run at all". A 3-act episode is the real product -- a full arc, the length he
actually watches -- and nothing in the harness rendered one unattended.

THE DEADLINE IS A START GATE, NOT A KILL. A leg already rendering is allowed to
finish: killing a 40-minute render at minute 39 wastes the GPU and publishes
nothing, which is the opposite of the point. The runner simply stops STARTING
legs once the cutoff passes, so the last episode lands shortly after it.

LANES ARE ROTATED AND SO ARE BANKS. Every lane here has PASSED a 1-act leg, so
a failure at 3 acts is news rather than noise. Banks rotate deliberately --
`media_archive` is first in every roster in this repo and had taken 27% of three
weeks of renders by simply being the thing the eye grabs.

No --title is passed: the harness label becomes the published title card.
"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time

REPO = os.environ.get(
    "OTR_REPO_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER = os.path.join(REPO, "scripts", "otr_canonical_api_run.py")
if not os.path.isfile(RUNNER):
    raise SystemExit(
        "cannot find the canonical runner at %s -- set OTR_REPO_ROOT to the "
        "ComfyUI-OldTimeRadio checkout" % RUNNER)

WRITER = "google/gemma-4-12b-it (11.9 GB)"
STYLE = "roll (any style)"        # the randomizer stays ON
BANKS = ("public_domain", "shakespeare", "original", "media_archive")

#: (profile, engine). Real VIDEO lanes only -- a 3-act still lane is just a
#: longer slideshow and proves nothing a 1-act did not.
LANES = (
    ("otr_w45_animatediff15_v3_haunted_video", "animatediff15_v3_haunted_video"),
    ("otr_w45_wan_ti2v", "wan_ti2v"),
    ("otr_w45_ltx_video", "ltx_video"),
    ("otr_w45_fastwan", "fastwan_8gb"),
    ("otr_w45_ltx_8gb", "ltx_8gb"),
)


def _now():
    return datetime.datetime.now()


def _stamp():
    return _now().strftime("%I:%M %p").lstrip("0")


def _deadline(hhmm: str) -> datetime.datetime:
    """Next occurrence of HH:MM local -- tomorrow if it has already passed."""
    hh, mm = (int(x) for x in hhmm.split(":", 1))
    now = _now()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    return target


def run_leg(idx, profile, engine, bank, acts, timeout_s, logdir, comfy_url):
    log = os.path.join(logdir, "act%s_%02d_%s.log" % (acts, idx, engine))
    cmd = [sys.executable, RUNNER,
           "--profile", profile,
           "--source-bank", bank,
           "--visual-style", STYLE,
           "--creative-model", WRITER,
           "--technical-model", WRITER,
           "--act-count", str(acts),
           "--comfyui-url", comfy_url,
           "--timeout", str(timeout_s)]
    print("[%s] leg %d START %s acts=%s bank=%s"
          % (_stamp(), idx, engine, acts, bank), flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write("### engine=%s bank=%s acts=%s profile=%s writer=%s\n"
                 % (engine, bank, acts, profile, WRITER))
        fh.flush()
        try:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=REPO, env=env, timeout=timeout_s + 600)
        except subprocess.TimeoutExpired:
            rc = 124
    mins = (time.time() - t0) / 60.0
    print("[%s] leg %d %s %.1f min rc=%d %s"
          % (_stamp(), idx, "PASS" if rc == 0 else "FAIL", mins, rc, engine),
          flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="07:00",
                    help="stop STARTING legs at this local time (HH:MM)")
    ap.add_argument("--acts", default="3")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--logdir", default=os.path.join(REPO, "tmp", "three_act"))
    ap.add_argument("--comfyui-url", default="http://127.0.0.1:8000")
    ap.add_argument("--only", default=None,
                    help="comma-separated engine ids to restrict to")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    stop = _deadline(args.until)
    lanes = list(LANES)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        lanes = [x for x in lanes if x[1] in want]

    print("=== 3-act overnight: %d lane(s), acts=%s, until %s ==="
          % (len(lanes), args.acts, stop.strftime("%I:%M %p").lstrip("0")),
          flush=True)
    idx = 0
    passed = failed = 0
    while _now() < stop:
        profile, engine = lanes[idx % len(lanes)]
        bank = BANKS[idx % len(BANKS)]
        idx += 1
        left = (stop - _now()).total_seconds() / 60.0
        print("[%s] %.0f min before the cutoff" % (_stamp(), left), flush=True)
        if run_leg(idx, profile, engine, bank, args.acts,
                   args.timeout, args.logdir, args.comfyui_url):
            passed += 1
        else:
            failed += 1
    print("=== cutoff reached: %d passed, %d failed, %d legs ==="
          % (passed, failed, idx), flush=True)


if __name__ == "__main__":
    main()
