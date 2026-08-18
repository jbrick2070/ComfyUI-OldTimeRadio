"""Build the BLINDED listen page for a G1 Lemmy audition.

WHY THIS EXISTS. `otr_g1_lemmy_audition.py` renders the clips and writes a
MANIFEST, but nothing turned them into something a person can actually sit down
and listen to; the historical `G1-LISTEN.html` in the 2026-08-10 directory has
no generator in the tree. `otr_lemmy_listen_page.py` is a different instrument
for the CROSS-ENGINE campaign and reads a different source.

BLINDED, AND THAT IS THE WHOLE POINT. The arms are shuffled at render time and
this page never names which reference is which -- it cannot, because the answer
lives in the `_KEY` directory the audition writes separately. The operator has
independently heard the incumbent as "Indian rather than Cockney" without seeing
its label, which is exactly the kind of judgement a labelled page destroys.

SELF-CONTAINED ON PURPOSE. Every clip is embedded as a data URI so the page
works from disk with no server, no network and no cloud -- this project is
offline-first and the audio is the operator's own.

Usage:
    python scripts/otr_g1_listen_page.py --dir g1_lemmy_2026-08-18
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EPISODES = pathlib.Path(
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")


def _audio_tag(path: pathlib.Path) -> str:
    """One clip, embedded. A file:// page cannot fetch a sibling wav in every
    browser, and half a listen page is worse than none."""
    if not path.exists():
        return '<p class="missing">MISSING: %s</p>' % html.escape(path.name)
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    uri = "data:audio/wav;base64,%s" % b64
    # preload="metadata", not "none": some embedded viewers never issue the
    # on-demand fetch for a data: URI, so the player renders and the play
    # button does nothing. Also ship a direct link to the wav on disk, so a
    # sandbox that blocks inline playback still cannot stop the audition.
    return ('<audio controls preload="metadata" src="%s"></audio>'
            '<div class="fallback">no sound? '
            '<a href="%s" download="%s">open %s directly</a></div>'
            % (uri, html.escape(path.name), html.escape(path.name),
               html.escape(path.name)))


#: Clip-suffix -> the label a listener wants to read. Anything not listed keeps
#: its own suffix, so a new audition shape needs no code change here.
_CLIP_LABELS = {
    "neutral": "Neutral line",
    "emotional": "Emotional line",
    "line1": "Line 1 (neutral)",
    "line2": "Line 2 (emotional)",
    "line3": "Line 3 (neutral)",
}


def build(out_dir: pathlib.Path, live_fingerprint: str) -> str:
    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    clips = sorted(out_dir.glob("arm*_*.wav"))
    arms = sorted({c.name.split("_")[0] for c in clips})

    # DISCOVERED, NOT HARDCODED. The G1 audition names its clips
    # <arm>_neutral / <arm>_emotional; the production audition names them
    # <arm>_line1..3. A page that only knew one shape would silently render
    # zero players for the other, which is the failure mode hardest to notice.
    blocks = []
    for arm in arms:
        rows = []
        for clip in [c for c in clips if c.name.startswith(arm + "_")]:
            suffix = clip.stem.split("_", 1)[1]
            label = _CLIP_LABELS.get(suffix, suffix.replace("_", " ").title())
            rows.append('<div class="clip"><h3>%s</h3>%s</div>'
                        % (html.escape(label), _audio_tag(clip)))
        blocks.append('<section class="arm"><h2>%s</h2>%s</section>'
                      % (html.escape(arm.upper()), "".join(rows)))

    return TEMPLATE % {
        "title": html.escape(str(manifest.get("test") or "Lemmy audition")),
        "generated": html.escape(str(manifest.get("generated_utc") or "")),
        "engine": html.escape(str(manifest.get("engine") or "")),
        "seed": html.escape(str(manifest.get("render_seed") or "")),
        "neutral_line": html.escape(str(manifest.get("neutral_line") or "")),
        "emotional_line": html.escape(str(manifest.get("emotional_line") or "")),
        "engine": html.escape(str(manifest.get("engine") or "")),
        "arms": "\n".join(blocks),
        "dirname": html.escape(out_dir.name),
        "fingerprint": html.escape(live_fingerprint or "(unavailable)"),
        "key_dir": html.escape(out_dir.name + "_KEY"),
    }


TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>%(title)s</title>
<style>
 :root{--bg:#faf8f4;--fg:#1a1815;--mut:#6b6560;--card:#fff;--line:#e2ddd4;--accent:#8a5a2b}
 @media (prefers-color-scheme:dark){:root{--bg:#16140f;--fg:#ece7dd;--mut:#9c948a;--card:#211d17;--line:#332d24;--accent:#d8a05c}}
 *{box-sizing:border-box}
 body{margin:0;padding:2rem 1rem 4rem;background:var(--bg);color:var(--fg);
      font:16px/1.6 -apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:52rem;margin:0 auto}
 h1{font-size:1.6rem;margin:0 0 .3rem}
 .sub{color:var(--mut);margin:0 0 2rem;font-size:.9rem}
 .lines{background:var(--card);border:1px solid var(--line);border-radius:10px;
        padding:1rem 1.25rem;margin-bottom:2rem}
 .lines p{margin:.4rem 0}
 .lines .lbl{color:var(--accent);font-weight:600;font-size:.8rem;
             text-transform:uppercase;letter-spacing:.05em}
 .arm{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:1.25rem;margin-bottom:1.25rem}
 .arm h2{margin:0 0 .75rem;font-size:1.1rem;color:var(--accent)}
 .clip{margin:.75rem 0}
 .clip h3{margin:0 0 .35rem;font-size:.8rem;font-weight:600;color:var(--mut);
          text-transform:uppercase;letter-spacing:.05em}
 audio{width:100%%}
 .missing{color:#b3261e;font-weight:600}
 .fallback{font-size:.8rem;color:var(--mut);margin-top:.3rem}
 .fallback a{color:var(--accent)}
 .note{border-left:3px solid var(--accent);padding:.75rem 1rem;margin:2rem 0;
       background:var(--card);border-radius:0 8px 8px 0}
 code{background:var(--bg);padding:.1rem .35rem;border-radius:4px;
      font-size:.85em;word-break:break-all}
 footer{color:var(--mut);font-size:.85rem;margin-top:2.5rem;
        border-top:1px solid var(--line);padding-top:1rem}
</style>
<div class="wrap">
<h1>%(title)s</h1>
<p class="sub">%(engine)s &middot; render seed %(seed)s &middot; generated %(generated)s</p>

<div class="note">
<strong>This is blinded on purpose.</strong> The arms are shuffled and nothing here
says which reference is which. Pick the one that sounds most like Lemmy &mdash;
gravelly, Cockney, intelligible &mdash; and only then open
<code>%(key_dir)s</code> to see what you chose.
</div>

<div class="lines">
 <p class="lbl">Neutral line</p><p>%(neutral_line)s</p>
 <p class="lbl">Emotional line</p><p>%(emotional_line)s</p>
</div>

%(arms)s

<div class="note">
<strong>If one of these passes</strong>, the new qualification record needs the
runtime it was heard on:<br>
<code>runtime.engine_impl_version = %(fingerprint)s</code><br>
<code>audition_manifest.path = otr/episodes/%(dirname)s/MANIFEST.json</code>
<br><br>
Until a verdict exists, Lemmy's route stays demoted and he renders on the
ordinary draw. Episodes keep publishing either way.
</div>

<footer>Clips embedded in this file &mdash; no server, no network. Source
directory: otr/episodes/%(dirname)s/</footer>
</div>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True,
                    help="audition directory (a bare name resolves under "
                         "otr/episodes/)")
    ap.add_argument("--out", default=None, help="output html path")
    args = ap.parse_args(argv)

    out_dir = pathlib.Path(args.dir)
    if not out_dir.is_absolute():
        out_dir = EPISODES / args.dir
    if not (out_dir / "MANIFEST.json").exists():
        print("no MANIFEST.json in %s" % out_dir)
        return 2

    try:
        from nodes import _otr_voice_route as ROUTE
        ROUTE._LIVE_FINGERPRINT_CACHE.clear()
        fingerprint = ROUTE.live_engine_impl_version("indextts2")
    except Exception:                      # noqa: BLE001 -- page still builds
        fingerprint = ""

    page = build(out_dir, fingerprint)
    out = pathlib.Path(args.out) if args.out else (out_dir / "LISTEN.html")
    out.write_text(page, encoding="utf-8")
    print("listen page -> %s  (%.1f MB)" % (out, out.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
