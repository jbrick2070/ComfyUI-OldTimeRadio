"""Emit docs/ENGINE_MATRIX.md -- the per-engine requirements record.

    python tools/engine_matrix.py            # rewrite the doc
    python tools/engine_matrix.py --check    # fail if the doc has drifted

WHY THIS IS GENERATED AND NEVER HAND-KEPT. The operator's ask (2026-07-26) was
to record, per model, "is it portrait or landscape, what the resolution is, how
many seconds each clip is... and the requirements for the stills", so the new
architecture can be checked against it. A hand-written table answers that once
and then rots, and a rotted requirements table is worse than none: it reads
authoritative while describing an engine that no longer exists. So every number
below is read from the LIVE registry at generation time, and ``--check`` is
wired into the suite so the doc cannot drift from the adapters silently.

WHAT THIS DELIBERATELY DOES NOT RECORD. Not the composed prompt text. That is
per-episode -- it is written by the story pass, varies per beat, and is not a
per-model requirement at all. What IS recorded is each engine's prompt
CONTRACT: whether it takes text at all, whether that text is required, and
which conditioner rewrites it. Blurring the two would put a sample of one
episode's output into a document the next reader would take as a spec.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "ENGINE_MATRIX.md"

#: Generated with the box in test mode so no adapter reaches for a GPU while
#: the matrix is only asking it what it declares.
os.environ.setdefault("OTR_TEST_MODE", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MISSING = object()


def _get(engine, attr, default=None):
    value = getattr(engine, attr, MISSING)
    return default if value is MISSING else value


def _seconds(frames, fps):
    """Frames -> a seconds string, or '-' when either side is unknown."""
    if not frames or not fps:
        return "-"
    secs = float(frames) / float(fps)
    return ("%.0f" % secs) if abs(secs - round(secs)) < 0.005 else ("%.2f" % secs)


def _clip_window(contract):
    """The legal clip length as both frames and seconds, honestly."""
    if contract.discrete_frames:
        menu = ", ".join(str(int(d)) for d in contract.discrete_frames)
        secs = ", ".join(_seconds(int(d), contract.native_fps)
                         for d in contract.discrete_frames)
        return ("menu: %s" % menu, "menu: %s s" % secs)
    if not contract.max_frames:
        return ("%d.. (no ceiling)" % contract.min_frames, "unbounded")
    span = "%d-%d" % (contract.min_frames, contract.max_frames)
    if contract.quantum > 1:
        span += " step %d" % contract.quantum
    return (span, "%s-%s s" % (_seconds(contract.min_frames, contract.native_fps),
                               _seconds(contract.max_frames, contract.native_fps)))


def _resolution(engine, name):
    """What the engine promises about output size.

    Local adapters do NOT have a static resolution: ``_aspect_plan`` /
    ``_aspect_policy`` negotiate it per render from the canvas and the profile.
    Recording a number for them would be recording a number the code never
    promised, so they record the mechanism instead.
    """
    from nodes._otr_video_engines import eng_cloud_video as _cv
    if name.startswith("cloud_vidu"):
        return _cv._VIDU_Q2_RESOLUTION + " (fixed)"
    if name == "cloud_seedance_2":
        return "env OTR_CLOUD_SEEDANCE_RESOLUTION, default 720p"
    if name.startswith("cloud_wan"):
        return "env OTR_CLOUD_WAN_RESOLUTION, default 720P"
    if name == "word_razzle":
        return "env OTR_CLOUD_PIXVERSE_QUALITY, default 1080p"
    if name == "cloud_kling_avatar":
        return "provider default (none sent)"
    if name.startswith("google_") and "omni" in name:
        return "720p (fixed)"
    if name.startswith("google_"):
        return "env OTR_GOOGLE_VEO_RESOLUTION, default 720p"
    if hasattr(engine, "_aspect_plan") or hasattr(engine, "_aspect_policy"):
        return "canvas-negotiated (_aspect_plan)"
    return "canvas"


def _prompt_contract(engine):
    """Whether this lane takes text, and who rewrites it before it is sent.

    ``required_inputs`` alone is not the answer. ``cloud_kling_avatar`` does
    not require a text_prompt and DOES send one -- ``_condition_kling_avatar_
    prompt`` builds it, falling back to a standing broadcast clause when the
    beat supplies nothing. Reading only the required list would record "no text
    input" for a lane that sends a conditioned prompt on every call, so the
    adapter's own source is asked whether it ever reaches for the field.

    THE WHOLE MRO, not just the class body. ``humo_1.7B`` is a four-line
    subclass of ``HuMoEngine`` that changes a checkpoint and inherits every
    graph method. Reading only its own source recorded "no text input" for it
    while recording "OPTIONAL" for its parent -- three rows wrong, and wrong in
    the direction that makes a lane look simpler than it is.
    """
    import inspect

    required = tuple(_get(engine, "required_inputs", ()) or ())
    if "text_prompt" in required:
        return "text_prompt REQUIRED"
    for klass in type(engine).__mro__:
        if klass is object:
            continue
        try:
            source = inspect.getsource(klass)
        except (OSError, TypeError):    # pragma: no cover -- exotic adapters
            continue
        if "text_prompt" in source:
            return "text_prompt OPTIONAL (sent when present)"
    return "no text input"


def _stills(engine):
    """The still requirements, summarised from the adapter's own still_plan."""
    plan = _get(engine, "still_plan", ()) or ()
    if not plan:
        return "none"
    parts = []
    for row in plan:
        kind = getattr(row, "kind", "?")
        req = getattr(row, "required", "?")
        aspect = getattr(row, "aspect", "?")
        parts.append("%s/%s/%s" % (kind, aspect, req))
    return "; ".join(parts)


def rows():
    import nodes._otr_video_engines  # noqa: F401  -- populate the registry
    from nodes._otr_video_engines import frame_contract as fc
    from nodes._otr_video_engines import registry as vreg

    out = []
    for name in sorted(vreg.all_engine_names()):
        engine = vreg.get_engine(name)
        c = fc.frame_contract_for(engine)
        frames, secs = _clip_window(c)
        out.append({
            "engine": name,
            "family": str(_get(engine, "family", "-")),
            "aspect": str(_get(engine, "render_aspect", "-")),
            "resolution": _resolution(engine, name),
            "clip_frames": frames,
            "clip_seconds": secs,
            "fps": str(c.native_fps or "canvas"),
            "continuity": c.continuity,
            "trim": "yes" if c.allow_tail_trim else "no",
            "side": "provider" if _get(engine, "provider_side", False) or
                    name.startswith(("cloud_", "google_")) or name == "word_razzle"
                    else "local",
            "inputs": ", ".join(_get(engine, "required_inputs", ()) or ()) or "-",
            "prompt": _prompt_contract(engine),
            "stills": _stills(engine),
        })
    return out


HEADER = """# ENGINE MATRIX -- the per-model requirements record

<!-- GENERATED FILE. Do not edit by hand.
     Regenerate:  python tools/engine_matrix.py
     Drift gate:  python tools/engine_matrix.py --check  (also a suite test)
-->

Every number here is read from the LIVE engine registry, so it cannot drift
from the adapters without the suite noticing. Written for multi-clip coverage
chunk 7a (2026-07-26), when every registered engine gained a declared
`FrameContract` and the per-engine opt-in was removed.

## How to read the clip window

`clip frames` is what ONE render call may legally produce. `step N` means the
ladder is arithmetic -- `min + k*N` -- so lengths off that grid have no legal
render and the planner renders the next length up and trims. `menu:` means the
provider serves a fixed set of lengths and nothing between them.

`clip seconds` is that window divided by `fps`. Where `fps` reads `canvas`, the
engine renders at whatever rate the canvas asks for and the seconds column is
meaningless rather than merely unknown -- it is marked `unbounded`.

**Google runs at 24 fps against a 25 fps canvas.** Veo's published menu is 4/6/8
SECONDS, which is 96/144/192 frames. The contract counts frames.

## What is NOT here, and why

* **The prompt text.** It is composed per episode by the story pass and varies
  per beat, so it is not a per-model requirement. What is recorded is the
  prompt CONTRACT: whether the lane takes text, and which conditioner rewrites
  it before it is sent.
* **A resolution number for the local lanes.** They negotiate size per render
  from the canvas and the profile (`_aspect_plan` / `_aspect_policy`). Printing
  a number here would be printing one the code never promised.
* **The rate the cloud providers actually DELIVER at.** No adapter declares it
  and nothing in the tree reads it back; the cloud rows convert seconds at the
  canvas's 25 fps because that is what `_CloudVideoBase._duration_seconds`
  itself assumes. This is a real open gap, not an omission.

"""


def render() -> str:
    data = rows()
    cols = [("engine", "engine"), ("side", "side"), ("family", "family"),
            ("aspect", "aspect"), ("resolution", "resolution"),
            ("clip_frames", "clip frames"), ("clip_seconds", "clip seconds"),
            ("fps", "fps"), ("continuity", "continuity"), ("trim", "tail trim")]
    lines = [HEADER, "## The matrix", ""]
    lines.append("| " + " | ".join(label for _k, label in cols) + " |")
    lines.append("|" + "|".join("---" for _c in cols) + "|")
    for row in data:
        lines.append("| " + " | ".join(row[key] for key, _l in cols) + " |")

    lines += ["", "## Inputs and prompt contract", "",
              "| engine | required inputs | prompt contract |",
              "|---|---|---|"]
    for row in data:
        lines.append("| %s | %s | %s |"
                     % (row["engine"], row["inputs"], row["prompt"]))

    lines += ["", "## Still requirements", "",
              "Read as `kind/aspect/when-required`, straight off each adapter's",
              "own `still_plan`. `inherit_engine` means the still is minted at",
              "the engine's own `aspect` column above.", "",
              "| engine | stills |", "|---|---|"]
    for row in data:
        lines.append("| %s | %s |" % (row["engine"], row["stills"]))

    lines += ["", "## Counts", "",
              "* registered engine names: **%d**" % len(data),
              "* provider-side: **%d**" % sum(1 for r in data if r["side"] == "provider"),
              "* local: **%d**" % sum(1 for r in data if r["side"] == "local"),
              "* can chain (strict_first_frame): **%d**"
              % sum(1 for r in data if r["continuity"] == "strict_first_frame"),
              ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the doc differs from the live registry")
    args = ap.parse_args(argv)

    fresh = render()
    if not args.check:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(fresh, encoding="utf-8", newline="\n")
        print("wrote %s (%d bytes)" % (DOC, len(fresh.encode("utf-8"))))
        return 0

    if not DOC.exists():
        print("DRIFT: %s does not exist. Run: python tools/engine_matrix.py" % DOC)
        return 1
    current = DOC.read_text(encoding="utf-8")
    if current == fresh:
        print("OK: %s matches the live registry" % DOC.name)
        return 0

    import difflib
    print("DRIFT: %s no longer matches the live registry." % DOC.name)
    print("Run: python tools/engine_matrix.py")
    print()
    diff = difflib.unified_diff(current.splitlines(), fresh.splitlines(),
                                fromfile="docs (on disk)", tofile="live registry",
                                lineterm="", n=1)
    for line in list(diff)[:60]:
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
