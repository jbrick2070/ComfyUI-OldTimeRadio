"""otr_ltx_mad -- quantify MOTION in a clip via mean inter-frame difference.

MAD = mean absolute difference between consecutive frames (0-255 scale),
averaged over the clip. Higher = more motion. A pure freeze ~0; a slow pan is
small-but-nonzero; real dynamic animation is clearly higher. Mirrors the 6/5
MAD static-motion analysis so euler vs euler_cfg_pp etc. are comparable.

Usage: python otr_ltx_mad.py <clip1.webm> [clip2.webm ...]
"""
import sys
import os

import numpy as np

try:
    import imageio.v3 as iio
    def read_frames(path):
        return np.asarray(iio.imread(path, plugin="pyav"))  # (T,H,W,C)
except Exception:
    import imageio
    def read_frames(path):
        rd = imageio.get_reader(path)
        return np.stack([f for f in rd])


def mad_of(path):
    frames = read_frames(path).astype(np.float32)
    if frames.ndim == 4 and frames.shape[-1] >= 3:
        frames = frames[..., :3].mean(axis=-1)   # grayscale (T,H,W)
    diffs = np.abs(frames[1:] - frames[:-1])
    per_frame = diffs.reshape(diffs.shape[0], -1).mean(axis=1)
    return {
        "frames": int(frames.shape[0]),
        "mad_mean": float(per_frame.mean()),
        "mad_max": float(per_frame.max()),
        "mad_p90": float(np.percentile(per_frame, 90)),
    }


def main():
    print(f"{'clip':<34} {'frames':>6} {'MAD_mean':>9} {'MAD_p90':>8} {'MAD_max':>8}  motion?")
    for path in sys.argv[1:]:
        try:
            m = mad_of(path)
        except Exception as e:
            print(f"{os.path.basename(path):<34} ERROR: {e}")
            continue
        mm = m["mad_mean"]
        verdict = "none/freeze" if mm < 0.6 else ("pan" if mm < 2.0 else "REAL")
        print(f"{os.path.basename(path):<34} {m['frames']:>6} {mm:>9.3f} "
              f"{m['mad_p90']:>8.3f} {m['mad_max']:>8.3f}  {verdict}")


if __name__ == "__main__":
    main()
