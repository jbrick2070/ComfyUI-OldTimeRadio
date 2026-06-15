# Tier Capability Matrix — Multi-GPU Portability

- **Project:** ComfyUI-OldTimeRadio, branch `v2.0-alpha`, HEAD `74869cf`
- **Authored:** 2026-06-14 (doc-only; implementation GATED). Companion to `PROBLEM_STATEMENT.md` §3 and
  `RESEARCH_FINDINGS.md` (R1–R7).
- **Confidence:** STATIC ANALYSIS. No tier other than T0 has been booted. Cells are predictions backed by
  code citations, not test results.

-----

## 1. Tier overview (problem statement §3, annotated with findings)

| Tier | Arch | Compute | torch build | VRAM span | Verdict (predicted) |
|------|------|---------|-------------|-----------|---------------------|
| **T0** (ship) | Blackwell | sm_120 | cu130 | 16 GB | Reference; GREEN; must stay green. |
| **T1** | Ada (40xx) | sm_89 | cu12x | 8–24 GB | **Runs with MINOR changes** (torch build + ceiling params). Highest confidence. |
| **T2** | Ampere (30xx) | sm_86 | cu11x/cu12x | 8–24 GB | **Runs with MODERATE changes** (fp8→bf16/GGUF model swaps). |
| **T3a** | AMD | ROCm | rocm wheels | varies | **Runs, REDUCED set** (no fp8; non-bnb LLM path; GGUF audit). |
| **T3b** | Apple Silicon | MPS | mps-enabled | unified | **Runs, MOST reduced** + CPU fallback. Highest uncertainty. |

Root reasons (see `RESEARCH_FINDINGS.md`): only one hard torch pin (`requirements.video.txt:16`); no
compiled CUDA extensions in `nodes/`; heavy diffusion engines delegate device to ComfyUI core; the gating
constraints are fp8 weights (R3), bitsandbytes (R7), and a handful of 16 GB-tuned ceilings (R4).

-----

## 2. Per-engine × per-tier capability grid

Legend: **AS-IS** = expected to run unchanged once torch build matches · **SWAP** = needs a non-fp8
weight file / quant variant · **BLOCKED** = no acceptable backend path · **CPU** = pure CPU/numpy, runs
anywhere · **AUDIT** = depends on an unverified custom-node/backend support.

| Engine (registry.py) | Weights / dtype | T0 | T1 Ada | T2 Ampere | T3a ROCm | T3b MPS |
|----------------------|-----------------|----|--------|-----------|----------|---------|
| `abstract` | CPU, vram 0 (`registry.py:128`) | AS-IS | AS-IS | AS-IS | CPU | CPU |
| `still_kenburns` | CPU, vram 0 (`registry.py:130`) | AS-IS | AS-IS | AS-IS | CPU | CPU |
| `station_card` | CPU, vram 0 (`registry.py:132`) | AS-IS | AS-IS | AS-IS | CPU | CPU |
| `visualizer` | CPU, vram 0 (`registry.py:134`) | AS-IS | AS-IS | AS-IS | CPU | CPU |
| `still_parallax` | DA-V2-small + numpy, `cpu_ok` (`registry.py:161`) | AS-IS | AS-IS | AS-IS | AS-IS / CPU | AS-IS / CPU |
| `flux_still` | flux.1-dev fp8, heavy 12000 (`registry.py:136`) | AS-IS | AS-IS | SWAP (fp16/GGUF) | SWAP | AUDIT/SWAP |
| `ltx_video` | LTX 2B + t5xxl **fp16** (`eng_ltx_video.py:538`) | AS-IS | AS-IS | AS-IS | AS-IS (audit ROCm SDPA) | AUDIT (MPS op coverage) |
| `ltx_orbit` | LTX preset, same physics (`registry.py:155`) | AS-IS | AS-IS | AS-IS | AS-IS | AUDIT |
| `humo` (14B) | **fp8_e4m3** UNET (`eng_humo.py:72`) + umt5 fp8 | AS-IS | AS-IS | **SWAP/BLOCKED** (no Ampere fp8) | BLOCKED (no fp8) | BLOCKED |
| `humo_1.7B` | **fp16** UNET (`eng_humo.py:452`) | AS-IS | AS-IS | AS-IS | AS-IS (audit) | AUDIT |
| `wan_i2v` (14B) | **fp8** + umt5 fp8 (`eng_wan_i2v.py:190`), vram 14500 | AS-IS | AS-IS | SWAP (GGUF/bf16) | SWAP/BLOCKED | BLOCKED |
| `wan_ti2v` (5B) | **GGUF** Q5_K_M + umt5 fp8 (`eng_wan_ti2v.py:192,217`) | AS-IS | AS-IS | AUDIT (GGUF node) | AUDIT | AUDIT/BLOCKED |
| `latentsync` | sidecar venv (`registry.py:145`) | AS-IS | AS-IS (per-tier venv) | AS-IS (venv) | AUDIT (ROCm venv) | AUDIT |
| `mesh_stage` | hy3d core + Blender (`registry.py:172`) | AS-IS | AS-IS | AUDIT | AUDIT | AUDIT |
| `triposg_talk` | cu128 sidecar (`registry.py:201`) | dark | dark | dark | BLOCKED | BLOCKED |
| `hunyuan3d_talk` | cu128_toolkit (`registry.py:204`) | dark | dark | dark | BLOCKED | BLOCKED |
| `trellis_talk` | cu128_toolkit (`registry.py:207`) | dark | dark | dark | BLOCKED | BLOCKED |

Predicted minimum-viable **shippable set** per non-T0 tier:
- **T1:** full set (everything T0 runs).
- **T2:** drop `humo` 14B + `wan_i2v` 14B unless bf16/GGUF swaps land; keep `humo_1.7B`, `ltx_video`,
  `wan_ti2v`, `flux_still` (swap), all floors.
- **T3a:** `ltx_video`, `humo_1.7B`, `flux_still` (swap), floors; non-bnb LLM path required.
- **T3b:** `ltx_video` (audit), `still_*` / `abstract` / `visualizer` / `still_parallax` floors;
  non-bnb LLM path required; CPU fallback for MPS op gaps.

-----

## 3. LLM path × tier (the bitsandbytes question)

| Concern | file:line | T0 | T1 | T2 | T3a ROCm | T3b MPS |
|---------|-----------|----|----|----|----------|---------|
| default dtype `bfloat16` | `_otr_model_loader.py:282` | OK | OK | OK (bf16 native) | OK | fp16 (MPS) |
| auto NF4 via `bitsandbytes` | `_otr_model_loader.py:311-342`, `requirements.txt:19` | OK | OK (cu12x) | OK | PARTIAL (ROCm bnb) | NONE → non-bnb path |
| adaptive budget `total_vram - 2.5` | `_otr_model_loader.py:233-235` | OK | tier margin | tier margin | tier margin | unified-mem reinterpret |
| `>=14.5` 100%-GPU threshold | `_otr_model_loader.py:410` | OK | tier param | tier param | tier param | n/a |
| context cap 8192 | `_otr_model_loader.py:168-179` | OK | tier param | tier param | tier param | tier param |
| HTTP/Ollama local lane (zero VRAM) | `_otr_model_loader.py:780-794` | available | available | available | **recommended fallback** | **recommended fallback** |

-----

## 4. Ceiling constant inventory (the parameterization targets)

| file:line | Constant | Seam today |
|-----------|----------|------------|
| `motion_common.py:40` + `:43-55` | `VRAM_CEILING_MB = 14500` | **env `OTR_VRAM_CEILING_MB` — TEMPLATE** |
| `_otr_workflow_validator.py:333` | exports `OTR_VRAM_CEILING_MB` | seam producer |
| `_vram_log.py:45` | `VRAM_CEILING_GB = 14.5` | hardcoded |
| `_otr_model_catalog.py:1027` | `DEFAULT_VRAM_CEILING_GB = 14.5` | hardcoded |
| `_otr_lfc_watchdog.py:55` | `VRAM_DEFAULT_CEILING_GB = 14.0` | hardcoded |
| `_otr_freeze_cascade.py:577` | `vram_ceiling_gb = 14.0` | hardcoded default |
| `OTR_LedgerFreezeCascade.py:302` + `:214-226` | `14.0` default + "16 GB/14.5" tooltip | hardcoded |

-----

## 5. Invariant hold/break per tier

See `RESEARCH_FINDINGS.md` → "Invariants restated per tier." LOUD flags:
- `test_audio_byte_identical` becomes **T0-only** (Q1, operator decision).
- Non-CUDA tiers ship a **reduced model set** with fp8 engines **blocked-LOUD**, not silently floored
  (Q2; forced by dtype reality).
- T3b unified memory means the **ceiling concept must be reinterpreted**, not just re-numbered.

-----

## 6. What would move a cell from "predicted" to "confirmed"

1. Boot T1 on an Ada box with `cu12x` torch → run the suite + one real episode (validates the whole
   delegated-device thesis).
2. Verify `UnetLoaderGGUF` on ROCm/MPS (unblocks/blocks `wan_ti2v` on T3).
3. Probe MPS op coverage for the LTX/Wan core-node graphs.
4. Confirm current bitsandbytes ROCm support level.
5. Measure the real per-tier VRAM ceilings + the LLM `- 2.5` margin on 8 GB / 24 GB / unified memory.

All gated behind the operator lifting the §10 implementation gate.
