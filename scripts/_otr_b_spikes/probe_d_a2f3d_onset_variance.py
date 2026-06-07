"""B-S1 probe (d) -- A2F-3D lip-sync onset: CONSTANT vs VARIABLE (BUG-102).

The driver's first-frame onset (warm-up) is absorbed by TRIMMING/offsetting the
rendered VIDEO -- the FROZEN master audio is NEVER padded or trimmed (V-1). That
only works if the onset is engine-CONSTANT. A VARIABLE onset is a v1 GO/NO-GO
blocker (you cannot fix-trim a moving target).

The operator runs A2F-3D on N clips on the 5080, records the per-clip onset (in
VIDEO frames), and feeds them here via:
  - OTR_B_ONSETS="12,12,13,12"   (inline, comma-separated), or
  - OTR_B_ONSET_JSON=path        (a JSON array of numbers).
Tolerance: OTR_B_ONSET_TOL frames (default 1).

PASS = constant onset (GO; prints the fixed VIDEO trim). NO-GO = variable.
With no data it prints a SMOKE result and asks for measurements.

OPERATOR-RUN (the onsets come from A2F-3D on the 5080).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import _b_harness as H  # noqa: E402

TOL = int(os.getenv("OTR_B_ONSET_TOL", "1"))


def _load_onsets() -> list | None:
    j = os.getenv("OTR_B_ONSET_JSON", "")
    if j and pathlib.Path(j).is_file():
        return [float(x) for x in json.loads(pathlib.Path(j).read_text(encoding="utf-8"))]
    inline = os.getenv("OTR_B_ONSETS", "").strip()
    if inline:
        return [float(x) for x in inline.replace(" ", "").split(",") if x]
    return None


def main() -> int:
    onsets = _load_onsets()
    if onsets is None:
        demo = H.classify_onset([12, 12, 12], tol_frames=TOL)
        print("PROBE d: no onset data (set OTR_B_ONSETS or OTR_B_ONSET_JSON)")
        print(f"PROBE d: SMOKE example {demo['recommendation']}")
        print("PROBE d: SMOKE-ONLY -- supply A2F-3D onset measurements to gate B-S1")
        return 0
    rep = H.classify_onset(onsets, tol_frames=TOL)
    print(f"PROBE d: n={rep['n']} min={rep['min']} max={rep['max']} "
          f"mean={rep['mean']:.2f} spread={rep['spread']:.2f} (tol {TOL})")
    print(f"PROBE d: {rep['recommendation']}")
    if not rep["constant"]:
        return 1
    print("PROBE d: PASS -- engine-constant onset; fix-trim the VIDEO (audio frozen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
