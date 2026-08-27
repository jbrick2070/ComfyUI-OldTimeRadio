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
    python scripts/otr_replay_foley_mix.py <episode_dir>
        [--inject-unpositioned] [--audition]

Exit 0 = the mix survived. Exit 1 = it did not, and the reason is printed.
CPU-only; reads the episode, writes nothing.
"""
from __future__ import annotations

import json
import math
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




#: Every OTR canvas to date. Used only when the episode cannot say otherwise.
DEFAULT_FPS = 25


def detect_fps(episode_dir, default=DEFAULT_FPS):
    """Frames per second, DERIVED from the episode rather than assumed.

    The r1 panel flagged the hardcoded 25 the moment this harness became the
    proof tool, and it was right to -- a literal that happens to match is not
    a measurement. It also asked for fps "from the QA/ledger", which turns out
    not to be satisfiable: NEITHER file records it anywhere. So it is recovered
    from where it does survive, ``frame_count / dur_s`` per beat, which agrees
    to within 0.05 fps across a real episode.

    Returns ``(fps, note)``. The note is non-empty whenever the derivation was
    not unanimous or could not run at all, because a proof tool that quietly
    picks a number is the exact failure being fixed here.
    """
    name = os.path.basename(episode_dir.rstrip("\\/"))
    try:
        beats = json.load(open(
            os.path.join(episode_dir, name + "_silent.mp4.qa.json"),
            encoding="utf-8"))["beats"]
        lines = json.load(open(
            os.path.join(episode_dir, "audio", name + "_ledger.json"),
            encoding="utf-8")).get("lines") or []
    except Exception as exc:  # noqa: BLE001 -- a note, never a crash
        return default, "artifacts unreadable (%s); assumed %d" % (
            type(exc).__name__, default)

    dur_by_beat = {str(line.get("line_id") or ""): line.get("dur_s")
                   for line in lines}
    votes = []
    for beat in beats:
        frames = beat.get("frame_count")
        secs = dur_by_beat.get(str(beat.get("beat_id") or ""))
        try:
            if float(secs) > 0 and int(frames) > 0:
                votes.append(int(frames) / float(secs))
        except (TypeError, ValueError):
            continue
    if not votes:
        return default, "no beat carried both a frame count and a duration; assumed %d" % default

    rounded = sorted({int(round(v)) for v in votes})
    if len(rounded) != 1:
        return default, ("beats disagree on fps %s -- assumed %d; the mix "
                         "placement in this replay is only as good as that"
                         % (rounded, default))
    spread = max(votes) - min(votes)
    return rounded[0], ("" if spread < 0.25 else
                        "derived from %d beat(s), spread %.3f fps" % (
                            len(votes), spread))

def rms_db(block):
    """RMS of a float block in dBFS. ``-inf`` for digital silence."""
    import numpy as np

    if getattr(block, "size", 0) == 0:
        return float("-inf")
    value = float(np.sqrt(np.mean(np.asarray(block, dtype=np.float64) ** 2)))
    return 20.0 * math.log10(value) if value > 0.0 else float("-inf")


def audibility_rows(fs, master, rate, rows, lane_ids, fps):
    """Per-beat ``RMS(bed in window) - RMS(programme in window)``, in dB.

    THE RECEIPT THIS BUG ACTUALLY NEEDS. ``foley_bed=mixed`` says the stem was
    decoded, placed and levelled; it says nothing about whether a listener can
    hear it. PBUG-20260827-03 shipped green on that receipt and inaudible in
    the ear, so the plumbing receipt must not be the one that closes it.

    BOTH SIDES COME FROM :func:`mix_foley_under_master` ITSELF, never from
    re-implemented arithmetic -- the bed by mixing the same rows under a SILENT
    master (``0 * envelope + bed`` is the bed exactly), the programme by mixing
    NO rows under the real one (``master * envelope``, keeping the lane global
    floor because ``lane_ids`` still carries it). Neither side can drift from
    what production did, which is the entire point of measuring here rather
    than in a fresh script.

    KNOWN LIMIT, stated rather than hidden: a PER-WINDOW lane (``ltx25_mime``,
    master gain 0.00) punches its envelope down from the rows themselves, so
    the no-rows programme baseline does not carry that punch. On such a lane
    the programme in-window is really silence and the delta is meaningless.
    Those rows are labelled rather than dressed up as a number. The global
    lane -- ``ltx25_foley_plus``, the one under investigation -- is exact.
    """
    import numpy as np

    programme, _ = fs.mix_foley_under_master(
        master, rate, [], fps=fps, lane_ids=lane_ids)
    bed, _ = fs.mix_foley_under_master(
        np.zeros_like(np.asarray(master, dtype=np.float32)), rate, rows,
        fps=fps, lane_ids=lane_ids)

    programme = np.asarray(programme)
    step = fs.samples_per_frame(rate, fps)
    length = int(programme.shape[-1])
    channels = int(programme.shape[0])
    out = []
    for row in rows:
        path = str((row or {}).get("foley_path") or "")
        beat_id = (row or {}).get("beat_id")
        if not path or not os.path.isfile(path):
            continue
        try:
            start_s = float((row or {}).get("start_s"))
        except (TypeError, ValueError):
            out.append({"beat_id": beat_id, "state": "unpositioned"})
            continue
        stem, stem_rate = fs.read_pcm16_wav(path)
        stem, _notes = fs.conform_to_master(stem, stem_rate, rate, channels)
        offset = int(round(start_s * float(fps))) * step
        if offset < 0 or offset >= length:
            out.append({"beat_id": beat_id, "state": "outside"})
            continue
        end = min(offset + int(stem.shape[-1]), length)
        bed_db = rms_db(bed[:, offset:end])
        programme_db = rms_db(programme[:, offset:end])
        out.append({
            "beat_id": beat_id,
            "lane": str((row or {}).get("engine_id") or ""),
            "state": "placed",
            "raw_stem_db": rms_db(stem),
            "bed_db": bed_db,
            "programme_db": programme_db,
            "delta_db": bed_db - programme_db,
        })
    return out


#: The band the operator ear calls a bed rather than a rumour, recorded in
#: PBUG-20260827-03: audible foley sits 15-25 dB under the programme.
AUDIBLE_BAND_DB = (-25.0, -15.0)


def print_audibility(rows):
    """The table, plus a one-line verdict against :data:`AUDIBLE_BAND_DB`."""
    print("")
    print("--- audibility: bed vs programme, per beat ---")
    print("%-24s %-18s %9s %9s %10s %9s"
          % ("beat", "lane", "raw_stem", "bed", "programme", "delta"))
    judged = []
    for row in rows:
        if row.get("state") != "placed":
            print("%-24s %-18s %s"
                  % (row.get("beat_id"), "", row.get("state")))
            continue
        judged.append((row["beat_id"], row["delta_db"]))
        print("%-24s %-18s %9.2f %9.2f %10.2f %+9.2f"
              % (row["beat_id"], row["lane"], row["raw_stem_db"],
                 row["bed_db"], row["programme_db"], row["delta_db"]))
    if not judged:
        print("no placed rows to judge")
        return

    # A NON-FINITE DELTA IS REPORTED, NEVER AVERAGED IN.
    #
    # Digital silence gives ``rms_db`` a true -inf, and the two ways that
    # reaches this line are not the same finding:
    #   bed AND programme silent -> -inf - -inf = NAN, meaning "no evidence
    #     either way", and
    #   programme silent under a live bed -> +INF, meaning the bed is the only
    #     thing in that window.
    # Neither belongs in a min/max. NaN in particular POISONS both: a running
    # comparison against NaN is always False, so a single NaN landing first
    # makes ``min``/``max`` return NaN for a list of otherwise normal beats --
    # a quietly wrong summary over correct rows, which is precisely the failure
    # this whole tool exists to prevent. Named, then set aside.
    finite = [(bid, d) for bid, d in judged if math.isfinite(d)]
    odd = [(bid, d) for bid, d in judged if not math.isfinite(d)]
    print("")
    if odd:
        for bid, value in odd:
            print("%-15s: %s -- %s"
                  % ("silent window", bid,
                     "bed and programme both digitally silent; no evidence"
                     if value != value else
                     "programme digitally silent under a live bed"))
        print("%-15s: %d beat(s) above are held out of the range and the band"
              % ("held out", len(odd)))
    if not finite:
        print("VERDICT        : NO JUDGEABLE BEAT -- every placed beat had a "
              "digitally silent window")
        return

    low, high = AUDIBLE_BAND_DB
    inside = [d for _bid, d in finite if low <= d <= high]
    values = [d for _bid, d in finite]
    print("delta range    : %+.2f dB (quietest) .. %+.2f dB (loudest)"
          % (min(values), max(values)))
    print("audible band   : %+.1f .. %+.1f dB -- %d of %d judgeable beat(s) "
          "inside" % (low, high, len(inside), len(values)))
    print("VERDICT        : %s"
          % ("AUDIBLE on every judgeable beat" if len(inside) == len(values)
             else "NOT audible on %d of %d judgeable beat(s)"
                  % (len(values) - len(inside), len(values))))

def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    inject = "--inject-unpositioned" in argv
    audition = "--audition" in argv
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

    fps, fps_note = detect_fps(episode_dir)
    print("fps            : %d %s" % (fps, ("(%s)" % fps_note) if fps_note
                                      else "(derived, unanimous)"))

    lane_ids = {str(r.get("engine_id") or "") for r in rows} & set(
        getattr(fs, "FOLEY_LANE_GAINS", {}))
    print("lanes          :", sorted(lane_ids) or "(none detected)")

    print("\n--- mix_foley_under_master ---")
    try:
        mixed, stats = fs.mix_foley_under_master(
            master, rate, rows, fps=fps, lane_ids=lane_ids)
    except Exception as exc:  # noqa: BLE001 -- the verdict is the point
        print("FAILED (%s): %s" % (type(exc).__name__, exc))
        return 1

    print("SURVIVED.  mixed:", getattr(mixed, "shape", "?"))
    for key in sorted(stats):
        value = stats[key]
        if isinstance(value, list) and len(value) > 4:
            value = "%d item(s)" % len(value)
        print("   %-18s %s" % (key, value))

    print_audibility(audibility_rows(fs, master, rate, rows,
                                     lane_ids, fps))

    if audition:
        out_path = os.path.join(audio_dir, "foley_audition.wav")
        fs.write_pcm16_wav(out_path, mixed, rate)
        print("")
        print("audition WAV   :", out_path)
        print("  NO RECEIPT CLOSES THIS BUG -- the operator listens to "
              "that file. The table above says the bed is present at a "
              "measurable level, which is a different claim.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
