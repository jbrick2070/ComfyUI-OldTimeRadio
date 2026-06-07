"""B-S2 probe (c) -- the KEYSTONE: mesh -> ARKit-52 topology WRAP.

Transfers a 52-blendshape ARKit template onto each candidate mesh and validates
the result is finite + bounded (a torn correspondence shows up as a NaN or a
delta blowup). The KEYSTONE GATE is strict: NO-GO if wrap-failure exceeds 20%
(5 of 25 meshes == NO-GO). A NO-GO means character_3d does not ship -- HuMo-2D
stays the v1 character path.

Source meshes: OTR_B_MESH_DIR (OBJ) or a synthetic batch.
ARKit template: OTR_B_ARKIT_TEMPLATE_NPZ -- an .npz with arrays `verts`,
`faces`, `mouth_idx`, and one `delta_<blendshapeName>` per ARKit-52 coefficient.
With none supplied a SYNTHETIC stand-in template is used (logic check only;
the real keystone needs the operator's real ARKit template + real meshes).

PASS = the keystone gate is GO (failure-rate < 20%). NO-GO otherwise.

OPERATOR-RUN (CPU wrap math; real meshes/template come from the 5080 pipeline).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import _b_harness as H  # noqa: E402

MESH_DIR = os.getenv("OTR_B_MESH_DIR", "")
TEMPLATE_NPZ = os.getenv("OTR_B_ARKIT_TEMPLATE_NPZ", "")
_EXTS = (".obj",)


def _load_template() -> tuple[dict, bool]:
    if TEMPLATE_NPZ and pathlib.Path(TEMPLATE_NPZ).is_file():
        z = np.load(TEMPLATE_NPZ, allow_pickle=False)
        deltas = {n: z[f"delta_{n}"] for n in H.ARKIT_52 if f"delta_{n}" in z}
        missing = [n for n in H.ARKIT_52 if n not in deltas]
        if missing:
            raise ValueError(f"template missing {len(missing)} blendshapes: {missing[:3]}...")
        return {"verts": z["verts"], "faces": z["faces"],
                "deltas": deltas, "mouth_idx": z["mouth_idx"]}, True
    return H.make_arkit_template(2), False


def _source_meshes() -> tuple[list, bool]:
    if MESH_DIR and pathlib.Path(MESH_DIR).is_dir():
        out = []
        for p in sorted(pathlib.Path(MESH_DIR).iterdir()):
            if p.suffix.lower() in _EXTS:
                out.append((p.name, H.load_obj(str(p))))
        return out, True
    batch = [(f"icosphere_{i}", (H.perturb(H.make_icosphere(2)[0], 0.01, i),
                                 H.make_icosphere(2)[1])) for i in range(23)]
    batch += [("nonmanifold_a", H.make_nonmanifold()),
              ("nonmanifold_b", H.make_nonmanifold())]  # 2/25 -> GO
    return batch, False


def main() -> int:
    template, real_tmpl = _load_template()
    meshes, real_mesh = _source_meshes()
    if not meshes:
        print(f"PROBE c: NO-GO: no source meshes (OTR_B_MESH_DIR={MESH_DIR or 'unset'})")
        return 1
    tag = "REAL" if (real_tmpl and real_mesh) else "SYNTHETIC-smoke"
    print(f"PROBE c: {tag}; template real={real_tmpl} meshes real={real_mesh}; "
          f"wrapping {len(meshes)} mesh(es)")
    results = []
    for name, (v, f) in meshes:
        r = H.wrap_mesh_to_arkit(v, f, template)
        results.append(r)
        if not r["ok"]:
            print(f"  WRAP-FAIL {name}: {r['reason']}")
    gate = H.keystone_gate(len(results), sum(1 for r in results if not r["ok"]))
    print(f"PROBE c: wrap-failure {gate['n_failed']}/{gate['n_total']} = "
          f"{gate['fail_fraction']:.3f} (threshold {gate['threshold']})")
    if not gate["go"]:
        print("PROBE c: NO-GO: wrap-failure >= 20% -- character_3d does not ship; "
              "HuMo-2D stays the v1 character path")
        return 1
    print(f"PROBE c: PASS -- keystone GO ({tag}); mesh -> ARKit-52 WRAP is viable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
