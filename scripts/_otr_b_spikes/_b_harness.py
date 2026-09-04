"""Subproject B (3D character) de-risk harness -- CPU-pure shared logic.

This module holds the CPU-testable core that the B-spike probe scripts import:
  - the canonical 52 ARKit blendshape names,
  - minimal dependency-light OBJ mesh I/O + synthetic mesh generators,
  - mesh topology pre-screen (manifold / winding / degenerate),
  - a screening-grade mesh -> ARKit-topology WRAP (deformation transfer proxy)
    with a finite/bounded validity gate and a batch failure-rate vs the EXACT
    20% keystone NO-GO,
  - A2F-3D onset constant-vs-variable classification (BUG-102),
  - cu-toolkit / nvcc-on-PATH detection + sidecar env sanitization logic.

It imports numpy ONLY (scipy used if present, with a numpy fallback). No torch,
no heavy libs -- it must stay cold-import clean so the probes that wrap it can be
reasoned about without a GPU. The probe scripts (operator-run on the 5080) layer
the real GPU work on top of these pure functions.

SFW; UTF-8 no BOM; ASCII quotes only.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Optional

import numpy as np

# Apple ARKit ARFaceAnchor.BlendShapeLocation -- the canonical 52 coefficients a
# viseme / audio driver (A2F-3D) animates. The WRAP keystone must expose ALL 52.
ARKIT_52 = (
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
)


# --------------------------------------------------------------------------- #
# Minimal OBJ I/O (numpy only -- no trimesh dependency required)               #
# --------------------------------------------------------------------------- #
def load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an OBJ. Returns (verts Nx3 float64, faces Mx3 int64, 0-based).

    Triangulates polygon faces with a simple fan. Ignores normals/uv/groups.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line[0] == "#":
                continue
            tok = line.split()
            if tok[0] == "v" and len(tok) >= 4:
                verts.append([float(tok[1]), float(tok[2]), float(tok[3])])
            elif tok[0] == "f" and len(tok) >= 4:
                # face indices may be "i", "i/j", "i/j/k", and may be negative.
                idx = []
                for t in tok[1:]:
                    s = t.split("/")[0]
                    i = int(s)
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return v, f


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle OBJ (1-based indices)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for x, y, z in np.asarray(verts, dtype=np.float64):
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in np.asarray(faces, dtype=np.int64):
            fh.write(f"f {a + 1} {b + 1} {c + 1}\n")


def make_cube() -> tuple[np.ndarray, np.ndarray]:
    """A closed, manifold, consistently-wound unit cube (12 triangles)."""
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    f = np.array([
        [0, 2, 1], [0, 3, 2],   # -z
        [4, 5, 6], [4, 6, 7],   # +z
        [0, 1, 5], [0, 5, 4],   # -y
        [2, 3, 7], [2, 7, 6],   # +y
        [1, 2, 6], [1, 6, 5],   # +x
        [0, 4, 7], [0, 7, 3],   # -x
    ], dtype=np.int64)
    return v, f


def make_icosphere(subdiv: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A closed, manifold, consistently-wound icosphere on the unit sphere.

    subdiv=0 is a 12-vertex / 20-face icosahedron; each subdivision quadruples
    the face count, sharing midpoint vertices (stays edge-manifold).
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0
    v = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    f = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    verts = [row.tolist() for row in v]
    for _ in range(max(0, int(subdiv))):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            hit = cache.get(key)
            if hit is not None:
                return hit
            mp = [(verts[a][i] + verts[b][i]) * 0.5 for i in range(3)]
            verts.append(mp)
            idx = len(verts) - 1
            cache[key] = idx
            return idx

        new_faces = []
        for a, b, c in f:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = np.asarray(new_faces, dtype=np.int64)
    out = np.asarray(verts, dtype=np.float64)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms, f


def make_nonmanifold() -> tuple[np.ndarray, np.ndarray]:
    """Three triangles sharing one edge -> a non-manifold edge (>2 faces)."""
    v = np.array([
        [0, 0, 0], [1, 0, 0],   # the shared edge 0-1
        [0, 1, 0], [0, -1, 0], [0, 0, 1],
    ], dtype=np.float64)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int64)
    return v, f


def make_degenerate() -> tuple[np.ndarray, np.ndarray]:
    """A cube with one zero-area (repeated-index) face appended."""
    v, f = make_cube()
    bad = np.array([[0, 0, 1]], dtype=np.int64)  # repeated vertex index -> area 0
    return v, np.vstack([f, bad])


def perturb(verts: np.ndarray, scale: float, seed: int = 0) -> np.ndarray:
    """Add deterministic gaussian noise (fraction of the bbox diagonal)."""
    rng = np.random.default_rng(seed)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0))) or 1.0
    return verts + rng.normal(0.0, scale * diag, size=verts.shape)


# --------------------------------------------------------------------------- #
# Topology pre-screen (the ~2-hour B-S2 gate; pure numpy)                      #
# --------------------------------------------------------------------------- #
def bbox_diagonal(verts: np.ndarray) -> float:
    v = np.asarray(verts, dtype=np.float64)
    if v.size == 0:
        return 0.0
    return float(np.linalg.norm(v.max(0) - v.min(0)))


def _face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = verts[faces]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def manifold_report(verts: np.ndarray, faces: np.ndarray) -> dict:
    """Screen a mesh for the failure modes that make deformation-transfer NaN.

    Reports non-manifold edges (shared by >2 faces), boundary edges, winding
    inconsistencies, degenerate (zero-area / repeated-index) faces and sliver
    faces. ``manifold_ok`` is the keystone-relevant gate: no non-manifold edges,
    no degenerate faces, consistent winding. True self-intersection is left to a
    trimesh-grade operator check; this is the cheap pre-screen.
    """
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    rep = {
        "n_verts": int(verts.shape[0]), "n_faces": int(faces.shape[0]),
        "degenerate_faces": 0, "non_manifold_edges": 0, "boundary_edges": 0,
        "winding_inconsistent_edges": 0, "sliver_faces": 0, "manifold_ok": False,
        "watertight": False,
    }
    if faces.shape[0] == 0:
        return rep
    # repeated-index faces
    repeated = np.array([len(set(f.tolist())) < 3 for f in faces])
    areas = _face_areas(verts, faces)
    zero_area = areas <= 1e-12
    rep["degenerate_faces"] = int(np.count_nonzero(repeated | zero_area))
    med = float(np.median(areas[areas > 0])) if np.any(areas > 0) else 0.0
    if med > 0:
        rep["sliver_faces"] = int(np.count_nonzero((areas > 0) & (areas < 1e-4 * med)))
    undirected: dict[tuple[int, int], int] = {}
    directed: dict[tuple[int, int], int] = {}
    good = faces[~(repeated | zero_area)]
    for a, b, c in good:
        for u, w in ((a, b), (b, c), (c, a)):
            directed[(int(u), int(w))] = directed.get((int(u), int(w)), 0) + 1
            key = (int(u), int(w)) if u < w else (int(w), int(u))
            undirected[key] = undirected.get(key, 0) + 1
    for key, cnt in undirected.items():
        if cnt > 2:
            rep["non_manifold_edges"] += 1
        elif cnt == 1:
            rep["boundary_edges"] += 1
        if cnt == 2:
            u, w = key
            if directed.get((u, w), 0) != 1 or directed.get((w, u), 0) != 1:
                rep["winding_inconsistent_edges"] += 1
    rep["watertight"] = (rep["boundary_edges"] == 0 and rep["non_manifold_edges"] == 0)
    rep["manifold_ok"] = (
        rep["non_manifold_edges"] == 0
        and rep["degenerate_faces"] == 0
        and rep["winding_inconsistent_edges"] == 0
    )
    return rep


# --------------------------------------------------------------------------- #
# Mesh -> ARKit-topology WRAP (the KEYSTONE; screening-grade transfer)         #
# --------------------------------------------------------------------------- #
def _unit_normalize(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center at origin, scale so bbox-diagonal == 1. Returns (v', center, scale)."""
    center = (v.max(0) + v.min(0)) * 0.5
    scale = float(np.linalg.norm(v.max(0) - v.min(0))) or 1.0
    return (v - center) / scale, center, scale


def _nearest(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Index into tgt of the nearest point for every row of src."""
    try:
        from scipy.spatial import cKDTree  # type: ignore
        return cKDTree(tgt).query(src, k=1)[1].astype(np.int64)
    except Exception:  # noqa: BLE001 -- numpy brute-force fallback
        out = np.empty(src.shape[0], dtype=np.int64)
        for i in range(src.shape[0]):
            out[i] = int(np.argmin(((tgt - src[i]) ** 2).sum(1)))
        return out


def make_arkit_template(subdiv: int = 2) -> dict:
    """A synthetic stand-in ARKit-topology template for CPU self-test.

    Operator-run probes load a REAL ARKit-52 template (e.g. an A2F-3D asset) via
    env var; this synthetic one only exercises the transfer + validity logic. It
    carries all 52 named blendshape delta fields and a labelled mouth region.
    """
    v, f = make_icosphere(subdiv)
    rng = np.random.default_rng(7)
    mouth = np.where((v[:, 2] > 0.55) & (v[:, 1] < -0.1))[0]  # front-lower band
    deltas: dict[str, np.ndarray] = {}
    for i, name in enumerate(ARKIT_52):
        # smooth, bounded per-vertex offsets (a stand-in for real shape keys)
        phase = rng.uniform(0, 2 * math.pi, size=3)
        d = 0.03 * np.sin(v * (1.0 + 0.1 * i) + phase)
        if "mouth" in name or "jaw" in name or name == "tongueOut":
            mask = np.zeros((v.shape[0], 1)); mask[mouth] = 1.0
            d = d * (0.4 + 0.6 * mask)
        deltas[name] = d.astype(np.float64)
    return {"verts": v, "faces": f, "deltas": deltas, "mouth_idx": mouth}


def wrap_mesh_to_arkit(src_verts: np.ndarray, src_faces: np.ndarray,
                       template: dict, max_delta_ratio: float = 0.5) -> dict:
    """Transfer the template's 52 blendshapes onto a source mesh; validate.

    Screening-grade: nearest-vertex correspondence after unit-normalization, then
    a finite + bounded check (a blowup/tear shows up as a delta larger than
    ``max_delta_ratio`` of the source bbox-diagonal). A non-manifold source
    auto-fails (matches real deformation-transfer NaN-out on torn geometry).
    """
    src_verts = np.asarray(src_verts, dtype=np.float64).reshape(-1, 3)
    res = {"ok": False, "reason": "", "n_targets": 0,
           "max_delta_ratio": 0.0, "mouth_ok": False, "manifold_ok": False}
    mr = manifold_report(src_verts, src_faces)
    res["manifold_ok"] = mr["manifold_ok"]
    if not mr["manifold_ok"]:
        res["reason"] = "source not manifold (non-manifold/degenerate/winding)"
        return res
    if src_verts.shape[0] < 4:
        res["reason"] = "source has too few vertices"
        return res
    tv, _, _ = _unit_normalize(np.asarray(template["verts"], dtype=np.float64))
    sv, _, s_scale = _unit_normalize(src_verts)
    corr = _nearest(sv, tv)                       # src vertex -> template vertex
    src_diag = bbox_diagonal(src_verts) or 1.0
    worst = 0.0
    finite = True
    for name in ARKIT_52:
        d_t = np.asarray(template["deltas"][name], dtype=np.float64)[corr]
        d_s = d_t * s_scale                       # template-normalized -> src scale
        if not np.all(np.isfinite(d_s)):
            finite = False
            break
        worst = max(worst, float(np.linalg.norm(d_s, axis=1).max()) / src_diag)
    res["n_targets"] = len(ARKIT_52)
    res["max_delta_ratio"] = worst
    if not finite:
        res["reason"] = "non-finite transferred delta (NaN/inf)"
        return res
    if worst > max_delta_ratio:
        res["reason"] = f"delta blowup {worst:.3f} > {max_delta_ratio} (tear)"
        return res
    # mouth interior: corresponding source verts must stay finite + bounded
    m_src = np.where(np.isin(corr, np.asarray(template["mouth_idx"])))[0]
    res["mouth_ok"] = bool(m_src.size > 0)
    res["ok"] = True
    res["reason"] = "ok"
    return res


def wrap_failure_rate(results: list[dict]) -> float:
    if not results:
        return 1.0
    failed = sum(1 for r in results if not r.get("ok"))
    return failed / len(results)


def keystone_gate(n_total: int, n_failed: int,
                  max_fail_fraction: float = 0.20) -> dict:
    """The B-S2 KEYSTONE gate. NO-GO if failures exceed 20% (5/25 == NO-GO).

    Strict boundary: pass iff failed/total < max_fail_fraction, so exactly 20%
    (e.g. 5 of 25) is a NO-GO, matching the sprint's cited example.
    """
    frac = (n_failed / n_total) if n_total else 1.0
    return {"n_total": n_total, "n_failed": n_failed, "fail_fraction": frac,
            "threshold": max_fail_fraction, "go": frac < max_fail_fraction}


# --------------------------------------------------------------------------- #
# A2F-3D onset variance (BUG-102): constant -> fixed VIDEO trim; variable -> NO-GO
# --------------------------------------------------------------------------- #
def classify_onset(onsets_frames, tol_frames: int = 1) -> dict:
    """Classify per-clip driver onset offsets.

    The FROZEN master audio is NEVER padded/trimmed (V-1); only the rendered
    VIDEO is trimmed/offset. An engine-CONSTANT onset can be absorbed by a fixed
    video trim; a VARIABLE onset is a v1 GO/NO-GO blocker (cannot fix-trim).
    """
    a = np.asarray(list(onsets_frames), dtype=np.float64)
    out = {"n": int(a.size), "min": 0.0, "max": 0.0, "mean": 0.0,
           "spread": 0.0, "constant": False, "recommendation": "",
           "video_trim_frames": 0}
    if a.size == 0:
        out["recommendation"] = "NO-GO: no onset measurements supplied"
        return out
    out["min"], out["max"] = float(a.min()), float(a.max())
    out["mean"], out["spread"] = float(a.mean()), float(a.max() - a.min())
    out["constant"] = out["spread"] <= tol_frames
    if out["constant"]:
        out["video_trim_frames"] = int(round(out["mean"]))
        out["recommendation"] = (
            f"GO: engine-constant onset -> fixed VIDEO trim of "
            f"{out['video_trim_frames']} frames (audio untouched)")
    else:
        out["recommendation"] = (
            f"NO-GO: variable onset spread {out['spread']:.1f} frames > "
            f"{tol_frames} -- v1 blocker (cannot fix-trim a moving onset)")
    return out


# --------------------------------------------------------------------------- #
# cu-toolkit / nvcc-on-PATH detection + sidecar env sanitization              #
# --------------------------------------------------------------------------- #
def parse_nvcc_version(version_text: str) -> Optional[str]:
    """Parse 'major.minor' from `nvcc --version` text, e.g. 'release 12.8'."""
    marker = "release "
    i = version_text.find(marker)
    if i < 0:
        return None
    num = ""
    for ch in version_text[i + len(marker):]:
        if ch.isdigit() or ch == ".":
            num += ch
        else:
            break
    parts = num.split(".")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return parts[0] + "." + parts[1]
    return None


def detect_nvcc_on_path(path_value: str, exists_fn: Callable[[str], bool]) -> list:
    """Return candidate nvcc paths found on a PATH string (injectable exists_fn)."""
    sep = ";" if os.name == "nt" else ":"
    names = ("nvcc.exe", "nvcc") if os.name == "nt" else ("nvcc",)
    found = []
    for d in path_value.split(sep):
        d = d.strip().strip('"')
        if not d:
            continue
        for nm in names:
            cand = os.path.join(d, nm)
            if exists_fn(cand):
                found.append(cand)
                break
    return found


def cu_toolkit_mismatch(target: str, found_versions: list) -> dict:
    """Flag ABI risk: a cu-toolkit on PATH whose version != the cu128 target.

    Linking a cu130 nvcc into a cu128 source build is the silent ABI-mismatch
    hazard (B-dep); the sidecar must isolate CUDA_HOME/PATH to the cu128 toolkit.
    """
    conflicting = [v for v in found_versions if v and v != target]
    major_conflict = [v for v in conflicting if v.split(".")[0] != target.split(".")[0]]
    return {"target": target, "found": list(found_versions),
            "conflicting": conflicting, "major_conflict": major_conflict,
            "risk": bool(conflicting)}


# Host env keys that must NOT leak into an isolated cu128 sidecar (torch.hub /
# HF caches / a stray device pin / host site-packages / pip redirection).
SIDECAR_LEAK_KEYS = (
    "PYTHONPATH", "PYTHONSTARTUP", "TORCH_HOME", "HF_HOME",
    "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME", "CUDA_VISIBLE_DEVICES", "PIP_TARGET", "PIP_PREFIX",
)

# VRAM ceilings (MB). The 3D sidecar asserts the STRICTER sub-ceiling, never the
# global machine ceiling.
SIDECAR_3D_BUDGET_MB = 14000


def build_sidecar_env(base_env: dict, cu_home: str, torch_ext_dir: str) -> dict:
    """Return a sanitized env for a cu128 3D sidecar (pure dict transform)."""
    env = dict(base_env)
    for k in SIDECAR_LEAK_KEYS:
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_HOME"] = cu_home
    env["CUDA_PATH"] = cu_home
    env["TORCH_EXTENSIONS_DIR"] = torch_ext_dir
    sep = ";" if os.name == "nt" else ":"
    cu_bin = os.path.join(cu_home, "bin")
    existing = env.get("PATH", "")
    env["PATH"] = cu_bin + (sep + existing if existing else "")
    return env


def path_leads_with(path_value: str, cu_home: str) -> bool:
    """True iff the cu128 toolkit bin is FIRST on PATH (so its nvcc wins)."""
    sep = ";" if os.name == "nt" else ":"
    cu_bin = os.path.join(cu_home, "bin").rstrip("\\/")
    first = path_value.split(sep)[0].strip().strip('"').rstrip("\\/") if path_value else ""
    return first.lower() == cu_bin.lower()


def sidecar_vram_ok(used_mb: int, budget_mb: int = SIDECAR_3D_BUDGET_MB) -> bool:
    """The 3D sidecar gate: machine-wide used VRAM must be <= the 14.0 GB sub-ceiling."""
    return int(used_mb) <= int(budget_mb)
