"""Replay an episode's FOLEY MIX -- the terminal node -- without re-rendering it.

WHY THIS EXISTS. On 2026-08-26 a foley leg rendered for **3h17m22s**, decoded
every stem correctly, and then raised at the LAST node and published nothing
(PBUG-20260826-02). The feedback loop on a terminal-node fault is the whole
render. This collapses it to about two seconds by rebuilding the mux's real
inputs from the artifacts a finished -- or failed -- episode leaves on disk:

    <ep>/<ep>_silent.mp4.qa.json            beat_id, shot_id, frame_count
    <ep>/audio/<ep>_ledger.json             start_s per beat
    <ep>/audio/foley/beat_<shot_id>_foley.wav   the durable per-beat stem
    <ep>/audio/pending_*_master.wav         the master to mix under

ONLY THE DURABLE `beat_shot_*` STEMS ARE ROWS. The `otr_<engine>_<hash>_foley`
files beside them are per-segment scratch, not manifest rows. An earlier cut of
this tool swept them in, left them with no `start_s`, and produced a failure
that said nothing about the code -- a harness bug wearing a product bug's
clothes. If you change the row builder, keep that distinction.

`--inject-unpositioned` adds a synthetic bearing row with `start_s=None` -- the
`music_inter` bridge case that caused the original fault, where a beat renders a
picture and produces a stem but occupies NO time in the master mix. Use it to
prove the SKIP path, because a real episode may not contain such a beat and a
replay that never exercises the fault proves only that the happy path works.

Usage:
    python scripts/otr_replay_foley_mix.py <episode_dir> [--inject-unpositioned]

Exit 0 = the mix survived. Exit 1 = it did not, and the reason is printed.
CPU-only; reads the episode, writes nothing.
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
os.environ.setdefault("OTR_TEST_MODE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def build_rows(episode_dir):
    """The bearing rows the mux would have been handed, from disk artifacts."""
    name = os.path.basename(episode_dir.rstrip("\\/"))
    qa_path = os.path.join(episode_dir, name + "_silent.mp4.qa.json")
    led_path = os.path.join(episode_dir, "audio", name + "_ledger.json")
    foley_dir = os.path.join(episode_dir, "audio", "foley")

    beats = json.load(open(qa_path, encoding="utf-8"))["beats"]
    ledger = json.load(open(led_path, encoding="utf-8"))
    start_by_beat = {str(line.get("line_id") or ""): line.get("start_s")
                     for line in (ledger.get("lines") or [])}

    rows, missing_stem = [], []
    for beat in beats:
        shot_id = str(beat.get("shot_id") or "")
        beat_id = str(beat.get("beat_id") or "")
        stem = os.path.join(foley_dir, "beat_%s_foley.wav" % shot_id)
        if not os.path.isfile(stem):
            missing_stem.append(beat_id)
            continue
        rows.append({
            "beat_id": beat_id,
            "engine_id": str(beat.get("engine_id") or ""),
            "foley_path": stem,
            "start_s": start_by_beat.get(beat_id),
            "start_s_space": "master_mix",
            "frame_count": int(beat.get("frame_count") or 0),
        })
    return rows, missing_stem


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    inject = "--inject-unpositioned" in argv
    if not args:
        print(__doc__.strip().splitlines()[-4])
        return 2
    episode_dir = args[0]

    from nodes._otr_video_engines import foley_stems as fs

    rows, missing_stem = build_rows(episode_dir)
    if inject and rows:
        bridge = dict(rows[min(5, len(rows) - 1)])
        bridge["beat_id"] = "synthetic_bridge"
        bridge["start_s"] = None
        rows.insert(min(6, len(rows)), bridge)

    audio_dir = os.path.join(episode_dir, "audio")
    masters = sorted(f for f in os.listdir(audio_dir)
                     if f.endswith("_master.wav"))
    if not masters:
        print("no *_master.wav in %s -- the leg never reached the mix" % audio_dir)
        return 1
    master_path = os.path.join(audio_dir, masters[0])

    unpositioned = [r["beat_id"] for r in rows if r["start_s"] is None]
    print("episode        :", os.path.basename(episode_dir.rstrip("\\/")))
    print("bearing rows   :", len(rows),
          ("| no stem: %s" % missing_stem) if missing_stem else "")
    print("NO start_s     :", unpositioned or "(none)")
    if inject and not unpositioned:
        print("  !! injection asked for but produced no unpositioned row")

    master, rate = fs.read_pcm16_wav(master_path)
    print("master         : %s  %.1f s @ %d Hz"
          % (masters[0], master.shape[-1] / float(rate), rate))

    lane_ids = {str(r.get("engine_id") or "") for r in rows} & set(
        getattr(fs, "FOLEY_LANE_GAINS", {}))
    print("lanes          :", sorted(lane_ids) or "(none detected)")

    print("\n--- mix_foley_under_master ---")
    try:
        mixed, stats = fs.mix_foley_under_master(
            master, rate, rows, fps=25, lane_ids=lane_ids)
    except Exception as exc:  # noqa: BLE001 -- the verdict is the point
        print("FAILED (%s): %s" % (type(exc).__name__, exc))
        return 1

    print("SURVIVED.  mixed:", getattr(mixed, "shape", "?"))
    for key in sorted(stats):
        value = stats[key]
        if isinstance(value, list) and len(value) > 4:
            value = "%d item(s)" % len(value)
        print("   %-18s %s" % (key, value))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
