"""Build the ONE Lemmy listen page -- every engine, every claim, in one place.

WHAT THIS IS FOR. The operator is remote and his ears are deferred, so every
artifact that wants a listening session ACCUMULATES into one batch review for his
return. This assembles that page: the four freshly rendered local engines, the
already-qualified IndexTTS2 clips referenced rather than re-rendered, the historic
Google Algenib clips labelled for exactly what they are, and the two cloud rows
that were configured and never rendered.

IT DECIDES NOTHING. A page cannot produce a verdict and this script does not
pretend otherwise. Beside it sits DECISIONS.json -- one row per route, every one
`pending` -- so a listening session produces DATA rather than an impression that
evaporates. Promotion out of the provisional tier still means a human writing a
qualified route by hand; there is no automated path and there should not be.

A HASH MISMATCH WARNS BESIDE THE CLIP AND NEVER HALTS THE PAGE. A page that
refuses to build because one arm went stale costs the whole session, and the
session is the scarce thing here.

    python scripts/otr_lemmy_listen_page.py
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

_EPISODES = r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
_CAMPAIGN_DIR = os.path.join(_EPISODES, "lemmy_cross_engine")
_G1_DIR = os.path.join(_EPISODES, "g1_lemmy_test_a")
_G1_KEY_DIR = os.path.join(_EPISODES, "g1_lemmy_test_a_KEY")
_HISTORIC_DIR = os.path.join(_EPISODES, "voice_audition_cockney")

#: The historic Google clips, BY EXACT FILENAME. Globbing this directory would
#: sweep in `4_charon_plain_control.wav`, which is a CONTROL VOICE and not Lemmy
#: at all -- an earlier draft of the plan called all four of them Algenib.
_HISTORIC_CLIPS = (
    ("1_algenib_plain.wav", "plain -- no accent direction"),
    ("2_algenib_cockney.wav", "Cockney -- this is the approved clone reference"),
    ("3_algenib_cockney_angry.wav", "Cockney under emotion -- the accent held"),
)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                   # noqa: BLE001
        return None


def _verify(path, expected_sha):
    """(exists, warning). Never raises, never halts the build."""
    if not os.path.isfile(path):
        return False, "the file named here is not on disk"
    if not expected_sha:
        return True, ""
    actual = sha256_of(path)
    if actual.lower() != str(expected_sha).lower():
        return True, ("the bytes on disk do not match the manifest -- this clip "
                      "has been re-rendered or replaced since it was recorded")
    return True, ""


def _clip_block(rel_path, label, note, warning=""):
    warn = ('<p class="warn">%s</p>' % html.escape(warning)) if warning else ""
    return """
      <div class="clip">
        <div class="clip-label">%s</div>
        <audio controls preload="none" src="%s"></audio>
        <div class="note">%s</div>%s
      </div>""" % (html.escape(label), html.escape(rel_path),
                   html.escape(note), warn)


def _rel(path):
    """Path relative to the page, which lives in the campaign directory."""
    return os.path.relpath(path, _CAMPAIGN_DIR).replace("\\", "/")


def build_rows():
    """Every row the page shows, in listening order, with its decision record."""
    from config.cast_pools import LEMMY_VOICE_POLICY as POLICY

    rows = []
    manifest = _read_json(os.path.join(_CAMPAIGN_DIR, "MANIFEST.json")) or {}
    engines = manifest.get("engines") or {}

    # 1. The QUALIFIED route, referenced and never re-rendered. Its manifest is
    #    cited by sha inside the route itself; re-rendering it would invalidate
    #    the only real audition evidence this project has.
    key = _read_json(os.path.join(_G1_KEY_DIR, "KEY.json")) or {}
    g1 = _read_json(os.path.join(_G1_DIR, "MANIFEST.json")) or {}
    candidate_label = next(
        (label for label, spec in (key.get("mapping") or {}).items()
         if spec.get("arm") == "A"), None)
    clips = []
    if candidate_label:
        by_file = {c.get("file"): c for c in (g1.get("clips") or [])}
        for kind in ("neutral", "emotional"):
            name = "%s_%s.wav" % (candidate_label, kind)
            path = os.path.join(_G1_DIR, name)
            exists, warning = _verify(path, (by_file.get(name) or {}).get("sha256"))
            if exists:
                clips.append((_rel(path), "IndexTTS2 -- %s" % kind,
                              "the qualified route, auditioned blind 2026-08-10",
                              warning))
    rows.append({
        "engine": "indextts2",
        "heading": "IndexTTS2 -- QUALIFIED (nothing to decide)",
        "state": "qualified",
        "route_id": (POLICY["approved_native_routes"]["indextts2"]["route_id"]),
        "identity": "idx_lemmy_algenib_cockney_v1",
        "blurb": "Already auditioned and already approved -- the operator picked "
                 "this arm blind, over the historic incumbent, on 2026-08-10. It "
                 "is here for reference so every other engine can be compared "
                 "against the one that is settled. No decision is needed.",
        "clips": clips,
        "decidable": False,
    })

    # 2. The four locally rendered engines.
    _BLURB = {
        "bark": "His shipped preset, pinned at writer time. Bark never had the "
                "per-episode redraw defect -- this clip is here so the others "
                "have something familiar to be compared against.",
        "kokoro": "The warmest British male in the bank. A friendly-broadcaster "
                  "cousin, NOT Cockney -- accepted under the no-skip direction "
                  "rather than claimed as a match. If this one is wrong, it is "
                  "wrong in a predictable direction.",
        "chatterbox": "The approved Cockney reference, cloned. Same bytes as the "
                      "qualified IndexTTS2 route -- so this is the same mould in "
                      "a different machine.",
        "dia": "The same reference again, on the other clone engine. Comparing "
               "this against chatterbox is comparing the ENGINES, not the voice.",
    }
    for engine in ("bark", "kokoro", "chatterbox", "dia"):
        row = engines.get(engine) or {}
        clips = []
        for kind in ("neutral", "emotional"):
            clip = (row.get("clips") or {}).get(kind) or {}
            name = clip.get("file")
            if not name:
                continue
            path = os.path.join(_CAMPAIGN_DIR, name)
            exists, warning = _verify(path, clip.get("sha256"))
            if exists:
                clips.append((_rel(path), "%s -- %s" % (engine, kind),
                              "%s / %s" % (row.get("identity_kind", "?"),
                                           row.get("identity_id", "?")),
                              warning))
        provisional = (POLICY.get("provisional_native_routes") or {}).get(engine)
        rows.append({
            "engine": engine,
            "heading": "%s -- %s" % (
                engine, "PROVISIONAL, rendered, awaiting your ears"
                if clips else "NOT RENDERED"),
            "state": "rendered_pending_listen" if clips else "missing",
            "route_id": (provisional or {}).get("route_id", "-- no route --"),
            "identity": row.get("identity_id", "?"),
            "blurb": _BLURB[engine] + (
                "" if clips else "  NO CLIPS FOUND -- re-run the harness with "
                "--engine %s before this session." % engine),
            "clips": clips,
            "decidable": bool(clips),
            "error": row.get("error", ""),
        })

    # 3. The historic Google clips: right voice, different words, prior approval.
    clips = []
    for name, note in _HISTORIC_CLIPS:
        path = os.path.join(_HISTORIC_DIR, name)
        if os.path.isfile(path):
            clips.append((_rel(path), name, "%s  [sha %s]"
                          % (note, sha256_of(path)[:16]), ""))
    rows.append({
        "engine": "google_tts",
        "heading": "google_tts -- CONFIGURED, never rendered here",
        "state": "configured_unrendered",
        "route_id": ((POLICY.get("provisional_native_routes") or {})
                     .get("google_tts", {}).get("route_id", "")),
        "identity": "gt_algenib  (provider voice: Algenib)",
        "blurb": "Rendering this engine needs an API key and spends money, so "
                 "nothing was rendered for it. What IS here is older and worth "
                 "hearing: three real Algenib clips from 2026-08-08, the voice "
                 "the approved clone reference was made FROM. You approved that "
                 "accent, including under emotion. They cannot complete a receipt "
                 "because the LINES were never written down -- so they are shown "
                 "as what they are: right voice, different words, prior approval. "
                 "(A fourth file in that directory, 4_charon_plain_control.wav, "
                 "is a different voice entirely and is deliberately not here.)",
        "clips": clips,
        "decidable": True,
        "qualifiable": False,
    })

    # 4. ElevenLabs -- and the data question that does not need ears.
    rows.append({
        "engine": "elevenlabs",
        "heading": "elevenlabs -- CONFIGURED, never rendered (a DATA question)",
        "state": "configured_unrendered",
        "route_id": ((POLICY.get("provisional_native_routes") or {})
                     .get("elevenlabs", {}).get("route_id", "")),
        "identity": "el_daniel  (provider voice: onwK4e9ZLuTAKqWW03F9)",
        "blurb": "Same reason -- credits and auth, so no clip. This one you can "
                 "settle from the metadata without hearing anything: the bank "
                 "tags el_daniel BRITISH and el_harry AMERICAN. el_harry is the "
                 "name that came up as the candidate, so either the tag is wrong "
                 "or the recollection is. el_daniel was taken provisionally "
                 "because he is the only bank-tagged British male on the engine.",
        "clips": [],
        "decidable": True,
        "qualifiable": False,
    })
    return rows


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>LEMMY on every engine -- listen page</title>
<style>
 body {{ font: 16px/1.55 -apple-system, Segoe UI, Roboto, sans-serif;
        max-width: 60rem; margin: 2rem auto; padding: 0 1.25rem;
        background: #14140f; color: #ece6d8; }}
 h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
 h2 {{ font-size: 1.15rem; margin: 0 0 .35rem; }}
 .sub {{ color: #b3a98f; margin-top: 0; }}
 .lines {{ background: #1e1e16; border-left: 3px solid #7a6a3a;
          padding: .8rem 1rem; margin: 1.5rem 0; }}
 .lines p {{ margin: .35rem 0; }}
 section {{ background: #1b1b14; border: 1px solid #33301f; border-radius: 8px;
           padding: 1rem 1.25rem; margin: 1.25rem 0; }}
 .meta {{ color: #9c937c; font-size: .85rem; margin-bottom: .6rem; }}
 .blurb {{ margin: .5rem 0 1rem; }}
 .clip {{ margin: .7rem 0; }}
 .clip-label {{ font-size: .9rem; color: #cbbf9e; }}
 .note {{ font-size: .8rem; color: #8d856f; }}
 audio {{ width: 100%; max-width: 34rem; margin: .25rem 0; }}
 .warn {{ color: #e0a03c; font-size: .85rem; margin: .2rem 0 0; }}
 .decide {{ margin-top: .9rem; padding-top: .7rem; border-top: 1px dashed #3a3626;
           font-size: .9rem; color: #b3a98f; }}
 code {{ background: #26241a; padding: .1rem .35rem; border-radius: 3px; }}
 .settled {{ color: #8fbf7a; }}
</style></head><body>
<h1>LEMMY on every TTS engine</h1>
<p class="sub">Built {built}. Nothing on this page is a verdict.</p>

<div class="lines">
 <p><strong>Every clip below says the same two lines</strong>, so the only thing
 that differs is the engine.</p>
 <p><em>Neutral:</em> {neutral}</p>
 <p><em>Emotional:</em> {emotional}</p>
</div>

<p>Six engines, three kinds of claim. <strong>One is settled</strong> -- IndexTTS2
was auditioned blind and you picked it. <strong>Four are provisional</strong>: a
deliberate identity, rendered where it could be, that nobody has listened to.
<strong>Two were configured and never rendered</strong>, because rendering them
costs an API key and money.</p>

<p>A provisional route is not a weak qualification -- it is a different kind of
claim. It says "this is who Lemmy is on this engine, chosen on purpose", and it
exists because the alternative was a different stranger every episode. Your ears
can demote any of them later without any archaeology: each row names its route id
below.</p>

{sections}

<h2>Recording what you decide</h2>
<p><code>DECISIONS.json</code> sits beside this page with one row per route, every
one <code>pending</code>. Set each to <code>keep_provisional</code> or
<code>demote</code> as you go. <strong>Nothing here can be promoted to qualified
by editing a file</strong> -- promotion means a human listened, and it is written
by hand as a qualified route with a real verdict. Demoting deletes the POLICY row
only; the voice-bank row stays, because it is reusable and it did nothing wrong.</p>
</body></html>
"""


def render_page(rows, manifest):
    sections = []
    for row in rows:
        clips = "".join(_clip_block(*c) for c in row["clips"]) or (
            '<p class="note">No audio for this row -- see above.</p>')
        decide = ""
        if row["decidable"]:
            decide = (
                '<div class="decide">Decision for <code>%s</code>: '
                '<code>keep_provisional</code> or <code>demote</code>%s</div>'
                % (html.escape(row["route_id"] or "-- no route --"),
                   "" if row.get("qualifiable", True) else
                   " &mdash; this row cannot be qualified from here, "
                   "there is no audio of it saying the frozen lines"))
        else:
            decide = ('<div class="decide settled">Settled 2026-08-10. '
                      'Nothing to decide.</div>')
        error = ('<p class="warn">The harness recorded an error for this engine: '
                 '%s</p>' % html.escape(row["error"])) if row.get("error") else ""
        sections.append(
            '<section><h2>%s</h2><div class="meta">route %s &middot; identity %s '
            '&middot; state %s</div><div class="blurb">%s</div>%s%s%s</section>'
            % (html.escape(row["heading"]),
               html.escape(row["route_id"] or "--"),
               html.escape(str(row["identity"])),
               html.escape(row["state"]), html.escape(row["blurb"]),
               error, clips, decide))
    return _PAGE.format(
        built=datetime.datetime.now(datetime.timezone.utc)
                       .strftime("%Y-%m-%d %H:%M UTC"),
        neutral=html.escape(manifest.get("neutral_line", "")),
        emotional=html.escape(manifest.get("emotional_line", "")),
        sections="\n".join(sections))


def write_decisions(rows):
    """The checklist as DATA. NEVER overwrites an existing file -- that file may
    already hold a listening session's answers, and rebuilding a page is not a
    reason to lose them."""
    path = os.path.join(_CAMPAIGN_DIR, "DECISIONS.json")
    if os.path.isfile(path):
        print("DECISIONS.json exists -- left untouched (it may hold answers)")
        return path
    payload = {
        "schema_version": 1,
        "campaign": "lemmy_cross_engine",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "how_to_use": "Set each decision to keep_provisional or demote. "
                      "`qualify` is NOT available here: a qualified route needs "
                      "an operator verdict written into config/cast_pools.py by "
                      "hand, with the artifacts it listened to. Demoting removes "
                      "the POLICY row only, never the voice-bank row.",
        "decisions": [
            {
                "engine": row["engine"],
                "route_id": row["route_id"],
                "state": row["state"],
                "listened_artifacts": [
                    {"file": os.path.basename(c[0])} for c in row["clips"]],
                "decision": "settled" if not row["decidable"] else "pending",
                "decided_utc": None,
            }
            for row in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=True)
        fh.write("\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    if not os.path.isdir(_CAMPAIGN_DIR):
        raise SystemExit("no campaign directory at %s -- run the audition first"
                         % _CAMPAIGN_DIR)
    manifest = _read_json(os.path.join(_CAMPAIGN_DIR, "MANIFEST.json")) or {}
    rows = build_rows()
    page = os.path.join(_CAMPAIGN_DIR, "LISTEN.html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(render_page(rows, manifest))
    decisions = write_decisions(rows)

    missing = [r["engine"] for r in rows if r["state"] == "missing"]
    print("page      -> %s" % page)
    print("decisions -> %s" % decisions)
    for row in rows:
        print("  %-11s %-24s %d clip(s)"
              % (row["engine"], row["state"], len(row["clips"])))
    if missing:
        print("\nNOT RENDERED: %s -- the page says so rather than hiding it."
              % ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
