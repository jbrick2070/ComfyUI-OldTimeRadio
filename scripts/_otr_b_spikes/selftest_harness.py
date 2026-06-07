"""CPU self-test for the B-spike harness (_b_harness.py).

Runs with plain python in the ComfyUI venv -- intentionally NOT pytest, so it
stays fully decoupled from the repo conftest / test suite (which a parallel
A-lane teardown may have intentionally red). Exercises ONLY the pure logic on
synthetic data. Prints PASS lines; exits 0 (all pass) / 1 (any fail).

    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe ^
        scripts\\_otr_b_spikes\\selftest_harness.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _b_harness as H  # noqa: E402

_FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    if not cond:
        _FAILS.append(name)
    print(f"  [{tag}] {name}{(' :: ' + detail) if detail else ''}")


# --- ARKit table ----------------------------------------------------------- #
print("ARKit-52 table")
check("arkit_52_count", len(H.ARKIT_52) == 52, f"len={len(H.ARKIT_52)}")
check("arkit_52_unique", len(set(H.ARKIT_52)) == 52)

# --- topology pre-screen --------------------------------------------------- #
print("topology pre-screen")
cv, cf = H.make_cube()
cube = H.manifold_report(cv, cf)
check("cube_manifold_ok", cube["manifold_ok"], str(cube))
check("cube_watertight", cube["watertight"])
check("cube_no_degenerate", cube["degenerate_faces"] == 0)
check("cube_winding_ok", cube["winding_inconsistent_edges"] == 0)

for sd in (0, 2):
    sv, sf = H.make_icosphere(sd)
    rep = H.manifold_report(sv, sf)
    check(f"icosphere{sd}_manifold_ok", rep["manifold_ok"], str(rep))
    check(f"icosphere{sd}_watertight", rep["watertight"])

nm = H.manifold_report(*H.make_nonmanifold())
check("nonmanifold_detected", nm["non_manifold_edges"] > 0 and not nm["manifold_ok"], str(nm))
dg = H.manifold_report(*H.make_degenerate())
check("degenerate_detected", dg["degenerate_faces"] > 0 and not dg["manifold_ok"], str(dg))

# --- OBJ roundtrip --------------------------------------------------------- #
print("OBJ roundtrip")
with tempfile.TemporaryDirectory() as td:
    p = str(pathlib.Path(td) / "cube.obj")
    H.write_obj(p, cv, cf)
    rv, rf = H.load_obj(p)
    rr = H.manifold_report(rv, rf)
    check("obj_roundtrip_counts", rv.shape == cv.shape and rf.shape == cf.shape,
          f"{rv.shape}/{rf.shape}")
    check("obj_roundtrip_manifold", rr["manifold_ok"])


# --- mesh -> ARKit WRAP (keystone) ----------------------------------------- #
print("mesh -> ARKit WRAP")
tmpl = H.make_arkit_template(2)
clean = H.make_icosphere(3)
w_ok = H.wrap_mesh_to_arkit(clean[0], clean[1], tmpl)
check("wrap_clean_ok", w_ok["ok"], str(w_ok))
check("wrap_clean_52_targets", w_ok["n_targets"] == 52)
check("wrap_clean_mouth", w_ok["mouth_ok"])
check("wrap_clean_bounded", 0.0 < w_ok["max_delta_ratio"] <= 0.5, str(w_ok["max_delta_ratio"]))

w_bad = H.wrap_mesh_to_arkit(*H.make_nonmanifold(), template=tmpl)
check("wrap_nonmanifold_fails", (not w_bad["ok"]) and "manifold" in w_bad["reason"], str(w_bad))

# batch failure-rate + keystone gate (strict 20% boundary: 5/25 == NO-GO)
def _batch(n_clean: int, n_bad: int):
    res = []
    for i in range(n_clean):
        m = H.make_icosphere(2)
        res.append(H.wrap_mesh_to_arkit(H.perturb(m[0], 0.01, i), m[1], tmpl))
    for _ in range(n_bad):
        res.append(H.wrap_mesh_to_arkit(*H.make_nonmanifold(), template=tmpl))
    return res

r_16 = _batch(21, 4)   # 4/25 = 0.16 -> GO
r_20 = _batch(20, 5)   # 5/25 = 0.20 -> NO-GO (strict)
g16 = H.keystone_gate(25, sum(1 for r in r_16 if not r["ok"]))
g20 = H.keystone_gate(25, sum(1 for r in r_20 if not r["ok"]))
check("keystone_rate_16_go", abs(H.wrap_failure_rate(r_16) - 0.16) < 1e-9 and g16["go"], str(g16))
check("keystone_rate_20_nogo", abs(H.wrap_failure_rate(r_20) - 0.20) < 1e-9 and not g20["go"], str(g20))

# --- onset variance (BUG-102) ---------------------------------------------- #
print("A2F-3D onset variance")
c = H.classify_onset([10, 10, 10])
check("onset_constant_go", c["constant"] and c["video_trim_frames"] == 10 and "GO" in c["recommendation"], str(c))
vv = H.classify_onset([5, 9, 14])
check("onset_variable_nogo", (not vv["constant"]) and "NO-GO" in vv["recommendation"], str(vv))
check("onset_empty_nogo", "NO-GO" in H.classify_onset([])["recommendation"])

# --- nvcc / cu-toolkit detection ------------------------------------------- #
print("cu-toolkit detection")
check("nvcc_parse_128", H.parse_nvcc_version("Cuda compilation tools, release 12.8, V12.8.89") == "12.8")
check("nvcc_parse_130", H.parse_nvcc_version("release 13.0, V13.0.1") == "13.0")
check("nvcc_parse_junk", H.parse_nvcc_version("no version here") is None)

fake = {r"C:\cu128\bin\nvcc.exe": True, r"C:\cu130\bin\nvcc.exe": True}
found = H.detect_nvcc_on_path(r"C:\cu128\bin;C:\cu130\bin;C:\other",
                              lambda p: fake.get(p, False))
check("nvcc_on_path_finds_two", len(found) == 2, str(found))
mm = H.cu_toolkit_mismatch("12.8", ["12.8", "13.0"])
check("cu_mismatch_flags_130", mm["risk"] and mm["major_conflict"] == ["13.0"], str(mm))
check("cu_match_clean", not H.cu_toolkit_mismatch("12.8", ["12.8"])["risk"])

# --- sidecar env sanitization ---------------------------------------------- #
print("sidecar env sanitization")
base = {"PATH": r"C:\windows;C:\old", "TORCH_HOME": r"C:\host\torch",
        "HF_HOME": r"C:\host\hf", "CUDA_VISIBLE_DEVICES": "1", "KEEP": "yes"}
env = H.build_sidecar_env(base, r"C:\cu128", r"C:\cu128\ext")
check("env_drops_leaks", all(k not in env for k in ("TORCH_HOME", "HF_HOME", "CUDA_VISIBLE_DEVICES")), str(env))
check("env_keeps_unrelated", env.get("KEEP") == "yes")
check("env_nousersite", env.get("PYTHONNOUSERSITE") == "1")
check("env_cuda_home", env.get("CUDA_HOME") == r"C:\cu128")
check("env_path_leads_cu128", H.path_leads_with(env["PATH"], r"C:\cu128"))

# --- VRAM sub-ceiling ------------------------------------------------------ #
print("VRAM 3D sub-ceiling")
check("budget_is_14000", H.SIDECAR_3D_BUDGET_MB == 14000)
check("vram_ok_under", H.sidecar_vram_ok(13999))
check("vram_nogo_over", not H.sidecar_vram_ok(14001))

# --- summary --------------------------------------------------------------- #
print("-" * 60)
if _FAILS:
    print(f"SELFTEST: FAIL ({len(_FAILS)}): {', '.join(_FAILS)}")
    sys.exit(1)
print("SELFTEST: PASS -- all B-spike harness checks green")
sys.exit(0)
