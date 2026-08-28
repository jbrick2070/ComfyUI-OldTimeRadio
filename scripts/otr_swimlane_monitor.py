"""A live swim-lane view of the render pipeline, for the operator's browser.

Writes ``tmp/otr_swimlane.html`` every couple of seconds: the water-main swim
lane with the CURRENTLY ACTIVE block lit, plus a small log window showing the
active stage's most recent lines. Open the file in any browser; it refreshes
itself. No server, no port, no pipeline hooks -- this READS the same logs the
render already writes and touches nothing else.

WHY A LOCAL FILE AND NOT THE HOSTED DIAGRAM: the claude.ai artifact runs under
a strict CSP that cannot reach localhost, so a live view must live on the box
that renders. This is the operator-tool answer: zero dependencies, zero risk
to a running leg.

Usage:
    python scripts/otr_swimlane_monitor.py            # watch, ~8h cap
    python scripts/otr_swimlane_monitor.py --once     # single snapshot
"""
from __future__ import annotations

import argparse
import glob
import html
import io
import os
import re
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(REPO, "tmp")
OUT = os.path.join(TMP, "otr_swimlane.html")
OBS = r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: Stage classifier -- ORDER MATTERS, first match on a line wins, and the last
#: classified line in the tail decides the active block. Patterns come from
#: the real log vocabulary, not guesses; extend here when a stage's wording
#: changes.
STAGES = (
    ("publish",  re.compile(r"obs_publish|RESULT SUCCESS")),
    ("assembly", re.compile(r"caption|credits|_with_credits|blend", re.I)),
    ("upscale",  re.compile(r"spandrel|upscal", re.I)),
    ("foley",    re.compile(r"MasterAudioMux|FOLEY decode|foley_bed|foley stem", re.I)),
    ("video",    re.compile(r"\[OTR video\]|prompt source=|TWO-STAGE|render-window "
                            r"VRAM|LTX MOTION|graph-exec.*(LTXV|Wan|sampler)", re.I)),
    ("stills",   re.compile(r"still|z_image|flux|lumina|ideogram|portrait", re.I)),
    ("freeze",   re.compile(r"LFC:phase_10|frozen_clean|gap_audit")),
    ("tts",      re.compile(r"bark|kokoro|indextts|chatterbox|\[OTR audio\]|voice", re.I)),
    ("writer",   re.compile(r"LedgerScriptWriter|\[OTR\.writer\]|pass 'script'|story_brief", re.I)),
)

#: Where each stage sits in the diagram. (band, label): band "plant" is the
#: treatment row, band "lane" is a swim lane fed by the main.
LAYOUT = (
    ("writer",   "plant", "Writer"),
    ("tts",      "plant", "TTS"),
    ("freeze",   "plant", "FREEZE"),
    ("stills",   "lane",  "Image"),
    ("video",    "lane",  "Video"),
    ("foley",    "lane",  "Foley mux"),
    ("upscale",  "lane",  "Upscale"),
    ("assembly", "lane",  "Assembly"),
    ("publish",  "lane",  "Publish"),
)


def newest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def read_tail(path, n=500):
    if not path or not os.path.exists(path):
        return []
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()[-n:]
    return [_ANSI.sub("", ln).replace("\x00", "").rstrip() for ln in lines]


def classify(lines):
    """(active_stage, {stage: recent lines}) from a log tail."""
    per = {name: [] for name, _rx in STAGES}
    active = None
    for ln in lines:
        if not ln.strip():
            continue
        for name, rx in STAGES:
            if rx.search(ln):
                per[name].append(ln)
                active = name
                break
    return active, per


def leg_context():
    """Profile + elapsed from the newest queue/leg log, best effort."""
    leg = newest(os.path.join(TMP, "overnight_rest*.log")) or \
        newest(os.path.join(TMP, "*leg*.log"))
    profile = elapsed = ""
    for ln in reversed(read_tail(leg, 200)):
        if not elapsed:
            m = re.search(r"t=(\d+)s", ln)
            if m:
                s = int(m.group(1))
                elapsed = "%dm%02ds" % divmod(s, 60)[0:2] if s < 3600 else \
                    "%dh%02dm" % (s // 3600, (s % 3600) // 60)
        if not profile:
            m = re.search(r"--profile (\S+)|LEG \d+/\d+\s+(\S+)", ln)
            if m:
                profile = m.group(1) or m.group(2)
        if profile and elapsed:
            break
    return profile, elapsed


def render(active, per, profile, elapsed, server_log):
    obs_n = len(glob.glob(os.path.join(OBS, "*.mp4")))
    stamp = time.strftime("%H:%M:%S")
    log_lines = per.get(active, [])[-14:] if active else []
    idle = active is None

    def block(name, band, label):
        cls = "blk" + (" active" if name == active else "")
        return '<div class="%s %s" id="%s">%s</div>' % (cls, band, name,
                                                        html.escape(label))

    plant = "".join(block(n, b, l) for n, b, l in LAYOUT if b == "plant")
    lanes = "".join(block(n, b, l) for n, b, l in LAYOUT if b == "lane")
    logw = "\n".join(html.escape(ln[-160:]) for ln in log_lines) or \
        ("-- idle: no active render detected --" if idle else "...")

    return """<!-- generated; do not edit --><title>OTR Swim Lane (live)</title>
<meta http-equiv="refresh" content="2">
<style>
 body{background:#0B0E0D;color:#E6EDE9;font-family:Segoe UI,system-ui,sans-serif;
      margin:0;padding:22px 18px}
 h1{font-size:17px;margin:0 0 2px;letter-spacing:.06em;text-transform:uppercase;
    color:#5BE58C}
 .meta{font-family:Consolas,monospace;font-size:12px;color:#78857F;margin:0 0 16px}
 .band{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
 .rail{height:8px;border-radius:4px;margin:2px 0 12px;
       background:linear-gradient(90deg,#2F6B47,#5BE58C 45%%,#2F6B47)}
 .blk{flex:1;min-width:86px;text-align:center;padding:10px 6px;border-radius:3px;
      border:1px solid #2C3835;background:#141A18;font-size:12.5px;
      text-transform:uppercase;letter-spacing:.07em;color:#A6B3AE}
 .blk.active{border-color:#5BE58C;color:#0B0E0D;background:#5BE58C;
      font-weight:600;animation:pulse 1.2s ease-in-out infinite}
 @keyframes pulse{50%%{filter:brightness(.82)}}
 @media(prefers-reduced-motion:reduce){.blk.active{animation:none}}
 .logbox{background:#0E1311;border:1px solid #2C3835;border-left:3px solid #5BE58C;
      border-radius:3px;padding:10px 12px;margin-top:14px}
 .logbox h2{font-size:11px;margin:0 0 6px;color:#5BE58C;text-transform:uppercase;
      letter-spacing:.12em}
 pre{margin:0;font-family:Consolas,monospace;font-size:11.5px;line-height:1.55;
     color:#9FD8B4;white-space:pre-wrap;word-break:break-all}
</style>
<h1>Signal Lost -- render swim lane</h1>
<p class="meta">%s &middot; profile %s &middot; elapsed %s &middot; obs %d episodes
 &middot; refreshed %s</p>
<div class="band">%s</div>
<div class="rail"></div>
<div class="band">%s</div>
<div class="logbox"><h2>%s</h2><pre>%s</pre></div>
<p class="meta">source: %s</p>
""" % ("RENDERING" if not idle else "IDLE", html.escape(profile or "?"),
       elapsed or "?", obs_n, stamp, plant, lanes,
       ("current activity -- " + (active or "none")).upper(), logw,
       html.escape(os.path.basename(server_log or "no server log found")))


def snapshot():
    server_log = newest(os.path.join(TMP, "otr_headless_*.log"))
    lines = read_tail(server_log)
    active, per = classify(lines)
    # a server log idle for >10 min is a finished/dead leg, not activity
    if server_log and time.time() - os.path.getmtime(server_log) > 600:
        active = None
    profile, elapsed = leg_context()
    page = render(active, per, profile, elapsed, server_log)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write(page)
    return active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--minutes", type=int, default=480,
                    help="watch duration cap (default 8h)")
    args = ap.parse_args()
    active = snapshot()
    print("wrote %s (active: %s)" % (OUT, active or "idle"))
    if args.once:
        return
    end = time.time() + args.minutes * 60
    while time.time() < end:
        time.sleep(2)
        snapshot()


if __name__ == "__main__":
    main()
