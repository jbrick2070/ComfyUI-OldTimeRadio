"""B-S2 probe (b) -- the ~2-hour mesh-topology PRE-SCREEN (keystone-first).

Non-manifold mouth/eye geometry makes deformation-transfer NaN out silently, so
before sinking the flash_attn/TRT toolchain we screen the candidate meshes:
manifold edges, winding consistency, degenerate/sliver faces, watertightness.

Point it at a directory of generated meshes with OTR_B_MESH_DIR (OBJ loaded
natively; other formats via trimesh if installed). With no dir it runs a SYNTHETIC
smoke (clean icospheres + injected non-manifold meshes) so the script is
self-verifying anywhere -- the REAL gate needs the operator's Hunyuan3D/TRELLIS
mesh outputs.

PASS = the screen ran and produced a manifold census (and, on real meshes, at
least one mesh is wrap-eligible). The census feeds the B-S2 keystone (probe c).

OPERATOR-RUN (CPU is fine; meshes come from the 5080 mesh-gen).
"""
from __future__ import annotations

import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import _b_harness as H  # noqa: E402

MESH_DIR = os.getenv("OTR_B_MESH_DIR", "")
_EXTS = (".obj", ".ply", ".glb", ".gltf", ".stl", ".off")


def _load_any(path: pathlib.Path):
    if path.suffix.lower() == ".obj":
        return H.load_obj(str(path))
    import trimesh  # operator installs it for non-OBJ formats
    m = trimesh.load(str(path), force="mesh")
    return m.vertices.view("float64"), m.faces.view("int64")


def _screen(name: str, verts, faces) -> bool:
    rep = H.manifold_report(verts, faces)
    verdict = "wrap-eligible" if rep["manifold_ok"] else "NEEDS-REPAIR"
    print(f"  {name}: {verdict} :: verts={rep['n_verts']} faces={rep['n_faces']} "
          f"nonmanifold={rep['non_manifold_edges']} winding_bad="
          f"{rep['winding_inconsistent_edges']} degenerate={rep['degenerate_faces']} "
          f"watertight={rep['watertight']}")
    return rep["manifold_ok"]


def main() -> int:
    eligible = 0
    total = 0
    if MESH_DIR and pathlib.Path(MESH_DIR).is_dir():
        files = [p for p in sorted(pathlib.Path(MESH_DIR).iterdir())
                 if p.suffix.lower() in _EXTS]
        if not files:
            print(f"PROBE b: NO-GO: no meshes ({_EXTS}) under {MESH_DIR}")
            return 1
        print(f"PROBE b: screening {len(files)} mesh(es) from {MESH_DIR}")
        for p in files:
            total += 1
            try:
                v, f = _load_any(p)
                eligible += 1 if _screen(p.name, v, f) else 0
            except Exception as e:  # noqa: BLE001
                print(f"  {p.name}: NEEDS-REPAIR :: load/screen error: {e}")
        print(f"PROBE b: {eligible}/{total} wrap-eligible")
        if eligible == 0:
            print("PROBE b: NO-GO: zero wrap-eligible meshes -- keystone cannot proceed")
            return 1
        print("PROBE b: PASS -- pre-screen census produced; feed probe c")
        return 0
    # synthetic smoke
    print("PROBE b: OTR_B_MESH_DIR unset -- SYNTHETIC smoke (supply real meshes to gate)")
    cases = [("icosphere_clean", H.make_icosphere(2)),
             ("icosphere_clean_2", H.make_icosphere(3)),
             ("nonmanifold", H.make_nonmanifold()),
             ("degenerate", H.make_degenerate())]
    for name, (v, f) in cases:
        total += 1
        eligible += 1 if _screen(name, v, f) else 0
    ok = eligible == 2  # exactly the two clean spheres
    print(f"PROBE b: {eligible}/{total} wrap-eligible (expected 2)")
    print("PROBE b: PASS -- screen logic verified on synthetic meshes" if ok
          else "PROBE b: NO-GO: synthetic screen miscounted")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
