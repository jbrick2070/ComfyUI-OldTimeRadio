# Subproject B (3D character) de-risk spikes -- OPERATOR runs these on the 5080

These are the B-lane GO/NO-GO probes -- the B-analog of the shipped `_otr_a_s2_probes`.
A Cowork window writes them on CPU; **the operator runs them on the RTX 5080**
(the agent does NOT run the GPU ones -- results must not be faked). Each prints
`PASS` or `NO-GO: <reason>` and exits 0 (pass) / 1 (no-go). They are the gate
BEFORE the `character_3d` adapter (B1) may open; a NO-GO re-scopes B (HuMo-2D
stays the v1 character path -- Subproject A is unaffected).

The CPU-pure core lives in `_b_harness.py` (manifold checks, the ARKit WRAP
transfer, onset stats, cu-toolkit/env logic) and is verified by
`selftest_harness.py` (35 checks, no GPU, no pytest/conftest). The probes layer
the real GPU work on top.

## Keystone-first ordering (passPM reorder -- fail fast)
Run the cheap keystone screen FIRST so the abandon-decision comes before the
flash_attn/TRT toolchain is built:

1. **probe b** (manifold pre-screen) -> **probe c** (ARKit WRAP keystone). If c
   is a NO-GO (>20% wrap-failure), STOP -- character_3d does not ship.
2. only then **probe a** (cu128 CUDA-ext build) + **probe d** (A2F-3D onset) +
   **probe e** (render-spawn VRAM).

## Run

```
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
scripts\_otr_b_spikes\run_all.bat
```

The CUDA-ext probe (a) and the sidecar probe (e) must be run with the **cu128
sidecar venv** python (not the main cu130 venv). The self-test + probes b/c/d run
under any venv with numpy:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\_otr_b_spikes\selftest_harness.py
```

## The probes

| # | File | Sprint | Proves | PASS | Needs |
|---|------|--------|--------|------|-------|
| a | `probe_a_cudaext_compile_load_sm120.py` | B-dep | a CUDA extension compiles+loads+runs a kernel on sm_120 with the cu128 toolchain ISOLATED | ext runs on sm_120 AND the effective build nvcc is the cu128 target (no host cu13x leak) | GPU + cu128 toolkit (`OTR_CU128_HOME`) |
| b | `probe_b_manifold_prescreen.py` | B-S2 pre-screen | candidate meshes are manifold / consistently wound / non-degenerate (non-manifold mouth geo NaNs the wrap) | census produced; >=1 wrap-eligible mesh | CPU (real meshes via `OTR_B_MESH_DIR`) |
| c | `probe_c_arkit_wrap.py` | **B-S2 KEYSTONE** | mesh -> ARKit-52 topology WRAP is finite + bounded across the test set | keystone gate GO (wrap-failure **< 20%**; 5/25 == NO-GO) | CPU wrap (real meshes + `OTR_B_ARKIT_TEMPLATE_NPZ`) |
| d | `probe_d_a2f3d_onset_variance.py` | B-S1 | the A2F-3D driver onset is engine-CONSTANT (fix-trimmable; audio frozen) | spread <= tol frames | onsets from A2F-3D on the 5080 |
| e | `probe_e_sidecar_vram_free_3d.py` | B-dep/B-S1 | the RENDER spawn (TRT+mesh+raster) peak fits 14.0 GB AND is reclaimed after exit | peak <= 14000 MB AND reclaimed below floor; takes the AS-3 lease | GPU + cu128 sidecar (`OTR_B_SIDECAR_PYTHON`) |

All probes import the SHIPPED `nodes/_otr_shared/gpu_residency.py` (AS-3) where
they need machine-wide NVML / the cross-process lease, so the spike also
smoke-tests the lease on the real GPU. The 3D sidecar asserts the stricter
**14000 MB** sub-ceiling, never the 14.5 GB machine ceiling.

## Tunables (env vars)
- `OTR_B_MESH_DIR` (b, c) -- directory of generated meshes (OBJ native; others via trimesh).
- `OTR_B_ARKIT_TEMPLATE_NPZ` (c) -- real ARKit-52 template: `verts`, `faces`, `mouth_idx`, `delta_<name>` per coefficient. Synthetic stand-in if unset (logic check only).
- `OTR_B_ONSETS` / `OTR_B_ONSET_JSON` (d) -- A2F-3D per-clip onset frames; `OTR_B_ONSET_TOL` (default 1).
- `OTR_CU128_HOME` / `OTR_CU128_TARGET` (a) -- the cu128 toolkit root + target version (default 12.8).
- `OTR_B_RENDER_ALLOC_MB` (e, default 12000), `OTR_B_HOLD_S` (5), `OTR_SIDECAR_FLOOR_MB` (768), `OTR_B_SIDECAR_PYTHON` (cu128 sidecar python).

## After running
Paste the PASS/NO-GO lines back into the next window. Keystone GO (probe c) ->
B-dep/B-S1/B-S3 toolchain build. Keystone NO-GO -> record it; character_3d
re-scopes to v2 and HuMo-2D stays the v1 character path. These probes never touch
the frozen audio.
