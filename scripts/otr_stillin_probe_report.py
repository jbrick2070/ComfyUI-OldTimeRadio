"""The still-in probe report (campaign item 2, 2026-09-02): motion energy per beat,
the null band, damping flags and one triptych card per beat.

    python scripts/otr_stillin_probe_report.py --source <ep> --baseline <ep> \
        --null <ep> --null <ep> [--arm label=<ep> ...] [--out DIR]

Every episode argument is an episode directory (or its ledger path) that was
published by the canonical graph. Motion energy is ``scripts/otr_ltx_mad.py::
mad_of`` (mean inter-frame absolute difference), one formula for every arm.

THE NULL BAND is the interval between the two A/A nulls' per-beat MAD, widened
by 10% of its width on each side; a beat whose arm MAD falls BELOW the band is
flagged DAMPED (the 2026-08-30 disqualification: a repeated init latent that
suppresses the trajectory), one above the band is flagged HOT. The report never
grades looks -- that is the operator's eye on the triptych (plate | baseline
frame 0 | arm frame 0) -- it only says where the numbers sit.

Fail-loud: any missing trace row, clip, plate or unreadable file exits non-zero
with the name of what was missing. Rows are matched to clips by ``shot_id``
through ``meta.render_trace`` (the instrument's durable trace).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.otr_ltx_mad import mad_of  # noqa: E402

BAND_WIDEN = 0.10


class ProbeError(SystemExit):
    pass


def load_ledger(arg: str) -> tuple[dict, pathlib.Path]:
    p = pathlib.Path(arg)
    if p.is_dir():
        hits = sorted((p / "audio").glob("*_ledger.json"))
        named = [h for h in hits if h.name.startswith(p.name)]
        hits = named or hits
        if not hits:
            raise ProbeError("no ledger under %s" % p)
        p = hits[0]
    return json.loads(p.read_text(encoding="utf-8")), p.parent.parent


def trace_rows(led: dict) -> dict:
    meta = led.get("meta") if isinstance(led.get("meta"), dict) else {}
    rows = [r for r in (meta.get("render_trace") or []) if isinstance(r, dict)]
    if not rows:
        raise ProbeError("episode %s carries no meta.render_trace" % led.get("episode_id"))
    out = {}
    for r in rows:
        key = (str(r.get("shot_id")), int(r.get("segment_index") or 0))
        out[key] = r
    return out


def clip_path_for(row: dict, ep_dir: pathlib.Path) -> pathlib.Path:
    name = str(row.get("clip_path") or "")
    if not name:
        raise ProbeError("trace row %s carries no clip_path" % row.get("shot_id"))
    hits = list(ep_dir.rglob(name))
    if not hits:
        raise ProbeError("clip %s not found under %s" % (name, ep_dir))
    return hits[0]


def plate_path_for(row: dict, ep_dir: pathlib.Path) -> pathlib.Path | None:
    name = str(row.get("plate_name") or "")
    if not name:
        return None
    hits = list((ep_dir / "stills").rglob(name))
    if not hits:
        raise ProbeError("plate %s named by the trace is not under %s" % (name, ep_dir / "stills"))
    return hits[0]


def first_frame(path: pathlib.Path):
    """The first decoded frame of a clip as an RGB image (PIL), via PyAV."""
    import av  # ComfyUI core dependency, bundled ffmpeg libs
    from PIL import Image
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            return Image.fromarray(frame.to_ndarray(format="rgb24"))
    raise ProbeError("clip %s decoded no frame" % path)


def triptych(plate, baseline, arm, out_path: pathlib.Path, label: str) -> None:
    from PIL import Image, ImageDraw
    tiles = [t for t in (plate, baseline, arm) if t is not None]
    h = max(t.height for t in tiles)
    w = sum(t.width for t in tiles)
    card = Image.new("RGB", (w, h + 24), (16, 16, 16))
    x = 0
    for t in tiles:
        card.paste(t, (x, 24))
        x += t.width
    ImageDraw.Draw(card).text((6, 4), label, fill=(230, 230, 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    card.save(out_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="the frozen source episode (rendered on the peer)")
    ap.add_argument("--baseline", required=True, help="the same bundle replayed on the shipping lane")
    ap.add_argument("--null", action="append", required=True,
                    help="a peer replay at the source denoise; give exactly two for the A/A band")
    ap.add_argument("--arm", action="append", default=[],
                    help="label=<episode> for each denoise-grid leg (e.g. d035=<dir>)")
    ap.add_argument("--out", default="", help="report directory (default: <source>/probe_report)")
    args = ap.parse_args(argv)
    if len(args.null) != 2:
        raise ProbeError("exactly two --null episodes define the A/A band (got %d)" % len(args.null))

    src_led, src_dir = load_ledger(args.source)
    base_led, base_dir = load_ledger(args.baseline)
    nulls = [load_ledger(n) for n in args.null]
    arms = []
    for spec in args.arm:
        if "=" not in spec:
            raise ProbeError("--arm wants label=<episode>, got %r" % spec)
        label, ep = spec.split("=", 1)
        arms.append((label.strip(), load_ledger(ep)))
    out_dir = pathlib.Path(args.out) if args.out else (src_dir / "probe_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_rows = trace_rows(src_led)
    base_rows = trace_rows(base_led)
    null_rows = [trace_rows(led) for led, _ in nulls]
    arm_rows = [(label, trace_rows(led), d) for label, (led, d) in arms]

    report = {"source": src_led.get("episode_id"), "baseline": base_led.get("episode_id"),
              "nulls": [led.get("episode_id") for led, _ in nulls],
              "arms": {label: led.get("episode_id") for label, (led, _) in arms},
              "beats": []}
    damped = hot = 0
    print("%-28s %9s %9s %9s %9s  %s" % ("shot", "baseline", "null_lo", "null_hi", "source", "arms"))
    for key in sorted(src_rows):
        shot_id, seg = key
        for name, rows in (("baseline", base_rows), ("null 1", null_rows[0]), ("null 2", null_rows[1])):
            if key not in rows:
                raise ProbeError("%s carries no trace row for %s#%d" % (name, shot_id, seg))
        src_mad = mad_of(str(clip_path_for(src_rows[key], src_dir)))
        base_mad = mad_of(str(clip_path_for(base_rows[key], base_dir)))
        n_mads = [mad_of(str(clip_path_for(rows[key], d))) for rows, (_, d) in zip(null_rows, nulls)]
        lo, hi = min(n_mads), max(n_mads)
        widen = max(hi - lo, 1e-9) * BAND_WIDEN
        band = (lo - widen, hi + widen)
        beat = {"shot_id": shot_id, "segment_index": seg, "source_mad": src_mad,
                "baseline_mad": base_mad, "null_mads": n_mads, "band": list(band), "arms": {}}
        arm_cells = []
        for label, rows, d in arm_rows:
            if key not in rows:
                raise ProbeError("arm %s carries no trace row for %s#%d" % (label, shot_id, seg))
            m = mad_of(str(clip_path_for(rows[key], d)))
            verdict = "DAMPED" if m < band[0] else ("HOT" if m > band[1] else "in-band")
            damped += verdict == "DAMPED"
            hot += verdict == "HOT"
            beat["arms"][label] = {"mad": m, "verdict": verdict}
            arm_cells.append("%s=%.3f %s" % (label, m, verdict))
        # the source itself against the band it seeded (a sanity line, not a verdict)
        src_verdict = "DAMPED" if src_mad < band[0] else ("HOT" if src_mad > band[1] else "in-band")
        beat["source_verdict"] = src_verdict
        print("%-28s %9.3f %9.3f %9.3f %9.3f  %s" % (shot_id, base_mad, band[0], band[1], src_mad,
                                                    "; ".join(arm_cells) or "-"))
        # the triptych: plate | baseline frame 0 | source (peer) frame 0
        plate_p = plate_path_for(src_rows[key], src_dir)
        try:
            from PIL import Image
            plate_img = Image.open(plate_p).convert("RGB") if plate_p else None
            triptych(plate_img,
                     first_frame(clip_path_for(base_rows[key], base_dir)),
                     first_frame(clip_path_for(src_rows[key], src_dir)),
                     out_dir / ("%s_%d_triptych.png" % (shot_id, seg)),
                     "%s  plate | baseline | peer" % shot_id)
            beat["triptych"] = str(out_dir / ("%s_%d_triptych.png" % (shot_id, seg)))
        except ProbeError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a card that cannot be drawn is a report defect
            raise ProbeError("triptych for %s failed: %s: %s" % (shot_id, type(exc).__name__, exc))
        report["beats"].append(beat)

    report["damped_cells"] = damped
    report["hot_cells"] = hot
    (out_dir / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print("beats: %d | damped arm cells: %d | hot arm cells: %d | report: %s"
          % (len(report["beats"]), damped, hot, out_dir / "probe_report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
