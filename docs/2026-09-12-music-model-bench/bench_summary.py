"""Summarise the music-model bench: per arm and family, plus an honest repeat
metric computed after the fact from the FLACs.

`loopiness` in scripts/otr_organ_bench.py is the max envelope autocorrelation
over 0.25-6 s, and on sustained material the max sits AT the 0.25 s floor --
that is envelope smoothness, not a repeat. `repeat` below is the strongest
LOCAL maximum of the same autocorrelation between 1.0 s and 6.0 s, i.e. a bar
or phrase that actually comes back, with the lag it comes back at.
"""
import json
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf


def repeat_metric(path):
    data, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    step = max(1, int(sr * 0.01))
    n = len(mono) // step
    if n < 200:
        return 0.0, 0.0
    env = np.sqrt(np.mean(mono[:n * step].reshape(n, step) ** 2, axis=1))
    env = env - env.mean()
    if np.dot(env, env) <= 0:
        return 0.0, 0.0
    ac = np.correlate(env, env, mode="full")[n - 1:]
    ac = ac / ac[0]
    lo, hi = 100, min(600, ac.size - 2)
    best, lag = 0.0, 0.0
    for k in range(lo, hi):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > best:
            best, lag = float(ac[k]), k * 0.01
    return round(best, 3), round(lag, 2)


def main(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows += json.load(fh)
    rows = [r for r in rows if "error" not in r]
    for r in rows:
        r["repeat"], r["repeat_lag_s"] = repeat_metric(r["flac"])
    fams = []
    for r in rows:
        if r["family"] not in fams:
            fams.append(r["family"])
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    keys = ["bursts", "loopiness", "repeat", "repeat_lag_s", "pulse", "tempo_bpm",
            "onsets_per_min", "peak_dbfs", "rms_dbfs", "clipped", "wall_s", "vram_peak_mib"]
    print("%-12s %-22s n  " % ("arm", "family") + " ".join("%9s" % k[:9] for k in keys))
    for fam in fams:
        for arm in arms:
            sub = [r for r in rows if r["arm"] == arm and r["family"] == fam]
            if not sub:
                continue
            vals = []
            for k in keys:
                xs = [float(r[k]) for r in sub]
                if k in ("bursts", "clipped"):
                    vals.append("%9d" % int(sum(xs)))
                elif k == "vram_peak_mib":
                    vals.append("%9d" % int(max(xs)))
                else:
                    vals.append("%9.2f" % float(np.mean(xs)))
            print("%-12s %-22s %d  " % (arm, fam, len(sub)) + " ".join(vals))
        print()
    print("== per-arm totals")
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        print("%-12s n=%d bursts=%d clipped=%d mean_wall=%.1fs max_vram=%d MiB mean_repeat=%.3f mean_pulse=%.2f"
              % (arm, len(sub), sum(r["bursts"] for r in sub), sum(r["clipped"] for r in sub),
                 np.mean([r["wall_s"] for r in sub]), max(r["vram_peak_mib"] for r in sub),
                 np.mean([r["repeat"] for r in sub]), np.mean([r["pulse"] for r in sub])))
    print("\n== every render (repeat@lag, pulse, tempo)")
    for r in rows:
        print("  %-11s %-22s s%d repeat=%.3f@%.2fs pulse=%.2f tempo=%5.1f wall=%5.1f peak=%6.2f"
              % (r["arm"], r["family"], r["seed_index"], r["repeat"], r["repeat_lag_s"],
                 r["pulse"], r["tempo_bpm"], r["wall_s"], r["peak_dbfs"]))
    with open(paths[0].replace(".json", "_with_repeat.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
