# Claude panelist review (written BEFORE reading the panel — independent)

Verdict up front: the pass00 framing hides the cheapest win in class F. Two of the six
classes need zero new dependencies because they ride engines OTR already ships. Rank:
F > A > B; C/D/E are NO-GOs for the first test case.

## Class-by-class

**F. Camera-control on engines we ALREADY run — RECOMMENDED FIRST TEST CASE.**
LTX-Video and Wan 2.2 i2v both have camera-motion conditioning (prompt-steerable and/or
published camera/orbit LoRAs). An "orbit_shot" preset on the EXISTING LTX or Wan adapter —
episode still in, slow orbit/dolly prompt + optional LoRA — reads as "3D" on screen with
ZERO new pip deps, zero new VRAM class, the existing seed plumbing, and the existing LOUD
fallback (falls back to still_kenburns like everything else). Integration: prompt-template
+ optional LoRA file + a registry row = 1–2 days. Risks: LoRA license must be checked
per-file; orbit fidelity is probabilistic (it is a diffusion prior, not geometry) — accept
for music/announcer slots, not a character close-up. This is "3D-feel", not a 3D asset;
label it honestly in the dropdown (e.g. `ltx_orbit`).

**A. Depth-parallax still ("still_parallax") — STRONG #2, the honest 2.5D workhorse.**
DepthAnythingV2-Small is Apache-2.0, pure-torch, wheel-only, <2 GB VRAM, deterministic.
CAUTION the panel should verify: the Base/Large DA-V2 checkpoints are CC-BY-NC — pin the
SMALL variant for license cleanliness. Warp/parallax camera path is ~200 lines of
grid_sample code in-repo (no new deps), extends the shipped kenburns cheap-family, CPU/GPU
cheap, slot-fits all three dropdowns as a floor-adjacent option. Integration 1–2 days.
It is 2.5D; sell it as the kenburns upgrade, not "the 3D engine".

**B. Native Hunyuan3D-2 (2mv) mesh + orbit render — the first REAL 3D ASSET, runner-up.**
ComfyUI core ships Hunyuan3D-2mv GEOMETRY natively (wheel-only, in-process) — but core has
no texture stage; the texture path lives in the community wrapper whose differentiable
rasterizer compiles CUDA (toolchain = NO-GO). So v1 = untextured mesh → vertex-color/matcap
"clay/bronze artifact" orbit. Two real build items the panel should not hand-wave:
(1) headless mesh→frames rendering is NOT a shipped ComfyUI path — Load 3D is an
interactive preview widget; we would write an offscreen renderer (pyrender/trimesh GL —
wheel-only but Windows-GL-context fiddly; or a tiny software rasterizer for a stylized
look); (2) the LOOK problem the 3D plan already flagged (gray clay) is mitigated by OWNING
it stylistically (bronze/wireframe/hologram matcap fits the sci-fi radio brand). VRAM
~6–10 GB transient, fits the ceiling with reclaim. License: Tencent community license —
flag for the operator (not OSI; usage-threshold terms). Integration 3–5 days. Best slot:
music_visual (rotating artifact), announcer second.

**C. SV3D / Zero123-class orbit video — NO-GO (license + age).** SV3D is research/
non-commercial-tier licensed; Zero123++ research-grade, 256–576px, dated. If the panel
knows a 2025–26 open-weight orbit-video model that is wheel-only + commercial-clean +
<=14.5 GB, that is the ONE thing worth surfacing here; otherwise skip the class.

**D. Splats / TRELLIS(.2) — NO-GO today.** TRELLIS deps (spconv/flash-attn class) and
splat rasterizers (diff-gaussian-rasterization; gsplat JIT) are compile-bound; no credible
sm_120 cu130 Windows wheels to my knowledge. The Trellis2 wrapper's "wheels" claim must be
verified against OUR stack before it counts — treat as verify-at-build, expect NO-GO.

**E. HyWorld / WorldMirror — NO-GO for the first test case.** Known stack-drift risk on
torch 2.10/cu130/sm_120; world-model outputs also do not map cleanly onto a per-beat
fixed-frame-count clip contract. Revisit post-toolkit.

## Recommendation

First test case: **F — `ltx_orbit` (or `wan_orbit`) camera-motion preset engine** (days,
zero deps, demoable immediately in all three dropdowns). Runner-up: **A `still_parallax`**
(honest 2.5D, license-pin DA-V2-Small). First REAL 3D asset when wanted: **B hy3d-2mv
clay-orbit** with an explicitly stylized matcap look + the two build items above costed.
Do NOT block any of this on the parked toolkit lane; none of it touches cu128.
