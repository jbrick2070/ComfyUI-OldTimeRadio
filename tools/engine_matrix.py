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
import re
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


#: The reference beat every multi-clip column is computed at: 442 frames =
#: 17.68 s at 25 fps. Long enough that every bounded engine must split it, so
#: the columns show real partitions rather than a row of "1 segment".
REFERENCE_BEAT_FRAMES = 442

#: Matches a docs/ citation inside an adapter's own source, so the evidence
#: column can say whether the receipt a cap cites is actually IN the repo. This
#: exists because `eng_humo.py` justified its 49-frame ceiling with
#: "docs/2026-06-27-humo-bakeoff", a file that is not in the tree -- the load-
#: bearing safety number for the heaviest engine cited a receipt nobody could
#: open, and nothing surfaced that until a human went looking.
_DOC_CITATION = re.compile(r"docs/[A-Za-z0-9._-]+")


def _effective_canvas(engine, name):
    """The canvas this engine ACTUALLY renders at, and where that comes from.

    The old matrix refused to print a local resolution ("printing a number the
    code never promised"). That refusal is what let `wan_i2v` sit on the shared
    1472x832 landscape default with no opinion of its own. The number IS
    resolvable -- it is just resolved in three different places -- so this
    function walks the same precedence the driver walks and NAMES the winner.
    """
    from nodes._otr_video_engines.render_driver import declared_render_canvas

    # An engine that computes its own dims ignores the request canvas entirely
    # (HuMo does: `_native_dims` off `render_aspect`), so it wins outright.
    native = getattr(engine, "_native_dims", None)
    if callable(native):
        try:
            w, h = native()
            return "%dx%d" % (int(w), int(h)), "engine _native_dims"
        except Exception:  # noqa: BLE001 -- an engine that cannot self-size
            pass                                    # falls through to the chain

    declared = declared_render_canvas(name)
    if declared is not None:
        return "%dx%d" % (int(declared[0]), int(declared[1])), "declared"

    # A provider lane has NO local render canvas: the service renders remotely
    # and the `resolution` column above already records what it is asked for.
    # Printing a canvas here invited exactly the misread it should prevent --
    # "cloud declares no canvas" was written into a review doc when the adapters
    # declare provider resolution perfectly well. Two different questions, kept
    # visibly apart.
    if str(_get(engine, "provider_side", False)) == "True" or name.startswith(
            ("cloud_", "google_")):
        return "n/a -- renders remotely", "see `resolution` column"

    # `ltx_audio_in` resolves its canvas in the driver rather than declaring it
    # (the O1 judgment scoped it that way until a general resolver lands), and
    # the value depends on whether the talking register is active.
    if name == "ltx_audio_in":
        return ("832x480 talking / 512x288 otherwise",
                "driver env branch OTR_LTX_AV_RENDER_CANVAS")

    # The driver's own family branch: face families keep the portrait default,
    # everything else is given the landscape composite canvas.
    family = str(_get(engine, "family", "") or "")
    wide = str(_get(engine, "render_aspect", "portrait")).lower() in (
        "wide", "landscape", "169", "16:9")
    if family in ("audio_driven_face", "lipsync_overlay", "character_3d") \
            and not wide:
        return "480x832", "family portrait default"

    # Landscape is CORRECT BY DESIGN for the still / procedural floor families:
    # the 2026-06-14 operator catch was that they had inherited the face
    # PORTRAIT and were rendering skinny pillarboxed b-roll. Only a generative
    # VIDEO engine landing here is a gap, so the two cases are labelled apart --
    # a matrix that flags the intentional case as a defect teaches the reader to
    # ignore the column.
    if family in ("static_motion", "static_image_gen", "abstract",
                  "text_overlay", "mesh_render"):
        return "1472x832", "shared landscape (by design for this family)"
    return "1472x832", "SHARED LANDSCAPE DEFAULT (unclaimed)"


def _multiclip(name, engine, contract):
    """Join mode, partition, and per-segment still behaviour at the reference
    beat -- the numbers four separate review panels had to re-derive by hand."""
    from nodes import otr_shot_lock as sl
    from nodes._otr_video_engines import coverage_plan as cp
    from nodes._otr_video_engines.render_driver import engine_family

    blank = {"join": "-", "segments": "-", "render_total": "-",
             "visible_total": "-", "remints": "-"}
    try:
        plan = cp.partition_beat(REFERENCE_BEAT_FRAMES, contract)
    except Exception as exc:  # noqa: BLE001 -- an engine that cannot plan this
        blank["join"] = "cannot plan (%s)" % type(exc).__name__
        return blank

    render = [int(s.render_frames) for s in plan.segments]
    visible = [int(s.visible_frames) for s in plan.segments]
    family = engine_family(name, str(_get(engine, "family", "") or ""))
    try:
        consumes = bool(sl._lane_consumes_a_still(
            {"engine_id": name, "family": family, "role": "character"}, name))
    except Exception:  # noqa: BLE001 -- unknown lane -> assume it mints none
        consumes = False

    # A per-segment still is only RE-MINTED when the plan jumps AND the lane
    # consumes a scene still. A chained successor begins on its predecessor's
    # real terminal frame, so it owns no still by design.
    remints = (len(render) - 1 if consumes
               and plan.join_mode == cp.JOIN_JUMP and len(render) > 1 else 0)
    return {
        "join": plan.join_mode,
        "segments": "%d: %s" % (len(render), ", ".join(str(f) for f in render)),
        "render_total": str(sum(render)),
        "visible_total": str(sum(visible)),
        "remints": str(remints),
    }


def _cap_and_evidence(engine, name):
    """The render ceiling, where it is set, and whether its cited receipt is
    real. A cap whose evidence file is missing reads MISSING here rather than
    looking exactly like a measured one."""
    cap = _get(engine, "safe_render_frames", None)
    source = "safe_render_frames" if cap else "contract max"

    module = sys.modules.get(type(engine).__module__)
    path = getattr(module, "__file__", None)
    cited, missing = set(), []
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                cited = set(_DOC_CITATION.findall(handle.read()))
        except OSError:
            cited = set()
    for ref in sorted(cited):
        target = ROOT / ref
        if not (target.exists() or list(ROOT.glob(ref + "*"))):
            missing.append(ref)

    if missing:
        evidence = "**MISSING: %s**" % ", ".join(missing)
    elif cited:
        evidence = ", ".join(sorted(cited))
    else:
        evidence = "none cited"
    return (str(cap) if cap else "-"), source, evidence


def rows():
    import nodes._otr_video_engines  # noqa: F401  -- populate the registry
    from nodes._otr_video_engines import frame_contract as fc
    from nodes._otr_video_engines import registry as vreg

    out = []
    for name in sorted(vreg.all_engine_names()):
        engine = vreg.get_engine(name)
        c = fc.frame_contract_for(engine)
        frames, secs = _clip_window(c)
        canvas, canvas_src = _effective_canvas(engine, name)
        cap, cap_src, evidence = _cap_and_evidence(engine, name)
        multi = _multiclip(name, engine, c)
        out.append({
            "engine": name,
            "canvas": canvas,
            "canvas_src": canvas_src,
            "cap": cap,
            "cap_src": cap_src,
            "evidence": evidence,
            "join": multi["join"],
            "mc_segments": multi["segments"],
            "mc_render": multi["render_total"],
            "mc_visible": multi["visible_total"],
            "mc_remints": multi["remints"],
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
* ~~**A resolution number for the local lanes.**~~ **RETRACTED 2026-08-02.**
  This omission was itself the defect. Refusing to print a local resolution
  "because the code never promised one" is exactly how `wan_i2v` came to sit on
  the shared 1472x832 landscape default with no opinion of its own, and how a
  profile asking for 832x480 could fail to reach the render. The number IS
  resolvable -- it is just resolved in three different places -- so the
  **effective canvas** column now walks the same precedence the driver walks and
  names which authority won.
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

    lines += ["", "## Effective render canvas", "",
              "The canvas each engine ACTUALLY renders at, and which authority",
              "decided it. `SHARED LANDSCAPE DEFAULT` means the engine expressed",
              "no preference and inherited the composite canvas -- the dead",
              "channel that cost `wan_8gb` a 268-minute leg. `engine _native_dims`",
              "means the adapter sizes itself and IGNORES the request canvas.", "",
              "| engine | effective canvas | decided by |", "|---|---|---|"]
    for row in data:
        lines.append("| %s | %s | %s |"
                     % (row["engine"], row["canvas"], row["canvas_src"]))

    lines += ["", "## Multi-clip behaviour at a %d-frame beat"
              % REFERENCE_BEAT_FRAMES, "",
              "A %d-frame beat is %.2f s at 25 fps -- long enough that every"
              % (REFERENCE_BEAT_FRAMES, REFERENCE_BEAT_FRAMES / 25.0),
              "bounded engine must split it. `render` is what the GPU or the",
              "provider is asked to produce; `visible` is what survives to the",
              "cut. They differ when a ladder has no legal length at the target",
              "and the tail is trimmed -- so `render` is what VRAM must afford,",
              "and `visible` is what the audio needs. `re-mints` counts stills",
              "generated fresh per cut: a chained successor begins on its",
              "predecessor's real terminal frame and owns no still, so only a",
              "JUMP plan on a still-consuming lane ever re-mints.", "",
              "| engine | join | segments (render frames) | render | visible | re-mints |",
              "|---|---|---|---|---|---|"]
    for row in data:
        lines.append("| %s | %s | %s | %s | %s | %s |"
                     % (row["engine"], row["join"], row["mc_segments"],
                        row["mc_render"], row["mc_visible"], row["mc_remints"]))

    lines += ["", "## Frame caps and the evidence behind them", "",
              "`evidence` lists every `docs/` receipt the adapter's own source",
              "cites. **MISSING** means the adapter cites a document that is not",
              "in this repo -- a safety number nobody can check. This column",
              "exists because the HuMo 49-frame ceiling cited",
              "`docs/2026-06-27-humo-bakeoff`, which has never been in the tree,",
              "and it read exactly like a measured number until someone looked.",
              "", "| engine | cap | set by | evidence |", "|---|---|---|---|"]
    for row in data:
        lines.append("| %s | %s | %s | %s |"
                     % (row["engine"], row["cap"], row["cap_src"],
                        row["evidence"]))

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
