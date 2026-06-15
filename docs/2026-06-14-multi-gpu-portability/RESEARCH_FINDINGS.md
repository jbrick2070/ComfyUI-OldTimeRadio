# Research Findings — Multi-GPU Portability / Hardware-Tier Targets

- **Project:** ComfyUI-OldTimeRadio (OTR), branch `v2.0-alpha`
- **Authored:** 2026-06-14 (research window; doc-only, implementation GATED)
- **Companion:** `PROBLEM_STATEMENT.md` (operator seed), `TIER_MATRIX.md` (capability matrix)
- **Method:** read-only grep / inspect against the real code at HEAD `74869cf`. Every claim cites a
  `file:line`. No code changed. Coordinated read-only with a parallel session editing
  `wrapper_bridge.py` + `eng_wan_*.py` (cited here, never modified).

> SCOPE NOTE: This is *device/backend* portability research, not an implementation. The §10 gate in the
> problem statement is firmly closed. Nothing below is a change — it is a verdict + a prioritized plan to
> hand the operator. Q1/Q4 carry operator-decision flags.

-----

## Executive summary

1. **The codebase is far MORE portable than the appendix assumed.** The single hard CUDA/Blackwell pin
   lives in **one** file (`requirements.video.txt:16`, `torch==2.10.0+cu130`). The base
   `requirements.txt:2` explicitly says *"Do NOT pin torch — ComfyUI manages its own torch version."* So
   the torch build is not pinned by the node pack itself — it is inherited from the host ComfyUI install.
2. **No compiled CUDA extensions ship in the production path.** `load_inline` / `nvcc` / `cpp_extension`
   appear ONLY under `scripts/_otr_b_spikes/` (operator-dark 3D-toolchain probes), never in `nodes/`.
   OTR also explicitly **bans** `xformers` / `flash_attn` / `sageattention` as dependencies
   (`scripts/otr_video_dep_pilot.py:41`, `scripts/otr_image_dep_pilot.py:45`) — it rides on
   `torch.nn.functional.scaled_dot_product_attention` (SDPA), which is backend-portable.
3. **The heavy diffusion video engines do NOT pin a device.** LTX / Wan / HuMo build graphs of ComfyUI
   *core* node-class names (`CheckpointLoaderSimple`, `CLIPLoader`, `KSampler`, `VAEDecode`, …) and
   execute them through `wrapper_bridge.run_graph` (`eng_ltx_video.py:418-433`,
   `eng_wan_i2v.py:298`). Device placement is delegated to ComfyUI core
   (`comfy.model_management`), which is already CUDA/ROCm/MPS/CPU-aware. **There is no OTR-side
   `get_torch_device` call and no `.cuda()` in these engines.**
4. **Top blocker is dtype, not device.** The heavy models are pinned to specific quantized weight files:
   HuMo `fp8_e4m3fn` UNET (`eng_humo.py:72`), Wan-i2v 14B `fp8` + `umt5_xxl_fp8` (`eng_wan_i2v.py:190`),
   Wan-ti2v 5B GGUF + `umt5_xxl_fp8` (`eng_wan_ti2v.py:192`), LTX `t5xxl_fp16` (`eng_ltx_video.py:538`).
   **fp8 e4m3 has no native support on Ampere (sm_86) and is patchy/absent on MPS+ROCm.** These are
   env-overridable filenames, so the swap is a config/model-asset problem, not a code refactor.
5. **Second blocker is `bitsandbytes`** (`requirements.txt:19`), CUDA-only and effectively mandatory for
   the default LLM path: the loader auto-selects NF4 for almost every model via a broad `vram_safe_tags`
   match (`_otr_model_loader.py:156-157, 311-312`). ROCm has partial bnb support; MPS has none — the
   non-CUDA tiers need a non-bnb LLM path (fp16/bf16 full, or the already-built HTTP/Ollama lane).
6. **The 16 GB budget is a scatter of constants, but the seam already exists for video.**
   `motion_common.py:40` (`VRAM_CEILING_MB = 14500`) is read through `dynamic_vram_ceiling_mb()` which
   honors the `OTR_VRAM_CEILING_MB` env var, exported per-execution by the validator
   (`_otr_workflow_validator.py:333`). But four other ceiling constants are still hardcoded
   (`_vram_log.py:45`, `_otr_model_catalog.py:1027`, `_otr_lfc_watchdog.py:55`,
   `_otr_freeze_cascade.py:577`). The LLM loader's budget is already *adaptive* — it reads real
   `total_vram` (`_otr_model_loader.py:192-194, 233-235`).
7. **Determinism is already defined as within-render per-seed, not cross-run/cross-backend.**
   `_otr_determinism.py` keeps the process default non-strict *specifically so video does not crash on
   sm_120* (`_otr_determinism.py:11-13`); strict determinism is scoped to single audio forwards. This
   already matches the seed's recommended Q1 answer — minimal new work.
8. **Recommended next step (when the gate lifts):** do NOT refactor the device path first. The cheapest,
   highest-value tier is **T1 (Ada / sm_89)** — it shares fp8 support with Blackwell, so it likely "runs
   as-is" once torch is `cu12x` instead of `cu130`. Prove T1 first; it validates the
   "ComfyUI-core-delegated device path" hypothesis with near-zero code change. T2 (Ampere) needs the fp8
   model swaps; T3a/T3b (ROCm/MPS) need both the dtype path AND a non-bnb LLM path and should ship a
   reduced model set.
9. **Single biggest unknown:** none of T1–T3b has ever been booted. Every verdict below is a
   *static-analysis prediction*, not a tested result. The DoD's "runs with documented changes" cannot be
   asserted GREEN without hardware.

-----

## R1 — Inventory of Blackwell / CUDA / cu130 hard dependencies

**Verdict: ONE hard torch pin, in the video requirements only; NO compiled CUDA extensions in the
shipping node tree; attention-kernel deps are explicitly BANNED, not required.**

Evidence:

- **Torch pin (the only one):** `requirements.video.txt:16` — `torch==2.10.0+cu130`. Header
  (`requirements.video.txt:1-13`) says it is a *pin-to-installed lock* for reproducibility of the
  14-day video sprint, NOT a node-pack requirement, and *explicitly excludes* `flash_attn` ("NOT
  BUILDABLE on sm_120 — do not chase").
- **Base requirements do NOT pin torch:** `requirements.txt:2` — *"IMPORTANT: Do NOT pin torch — ComfyUI
  manages its own torch version."* The base deps (`transformers`, `soundfile`, `numpy`, `feedparser`,
  `tokenizers`, `sentencepiece`, `bitsandbytes`, `lm-format-enforcer`) are torch-agnostic except bnb (R7).
- **No compiled extensions in production:** a repo-wide grep for `cpp_extension` / `load_inline` / `nvcc`
  / `CUDAExtension` matches ONLY `scripts/_otr_b_spikes/` — e.g.
  `scripts/_otr_b_spikes/probe_a_cudaext_compile_load_sm120.py:92` (`load_inline`),
  `_b_harness.py:400` (`parse_nvcc_version`). These are the operator-blocked 3D-toolchain spike probes
  (cu128 isolation experiments), dark per the 3D plan. **`nodes/` contains zero compiled-extension code.**
- **Attention kernels are banned, not required:** `scripts/otr_video_dep_pilot.py:41` /
  `:50` and `scripts/otr_image_dep_pilot.py:45` list `("xformers", "flash_attn", "sageattention")` as
  `BANNED_DEPS` / `STARTUP_CONTAMINANTS`. SageAttention is treated as a *contaminant* to fail-closed
  against, not a dependency (R2). FA2 is probed-then-skipped with a clean SDPA fallback
  (`_otr_model_loader.py:289-305`).
- **"Blackwell" / "sm_120" string mentions are comments + guards, not pins:** e.g.
  `_otr_determinism.py:12` ("does not crash on sm_120"), `_otr_model_loader.py:300` (FA2-unavailable log),
  `rtx_upscale.py:133` (sm_120 incompat guard for an all-zero-tensor bug). None are version pins.
- **Sidecar venvs already isolate non-cu130 toolchains:** the audio engines that need older CUDA run in
  their own subprocess venvs — `eng_indextts2.py:4-5` (python 3.10 / torch 2.8 / cu128 sidecar),
  `eng_dia.py:6-7` (torch 2.8 nightly cu128), `eng_chatterbox.py:4`. This is a working precedent for the
  "incompatible-toolchain → isolated sidecar" portability pattern.

R1 conclusion: the only thing that *forces* cu130/Blackwell is the **model weight dtype choices** (R3)
and **bitsandbytes** (R7) — not torch pins or kernels. Swapping the host torch build to `cu12x`/ROCm/MPS
is not blocked by anything in this node pack's own dependency declarations.

-----

## R2 — GPU-touching stages & backend-sensitive ops

**Verdict: two distinct device regimes. The HEAVY diffusion engines are device-agnostic (delegated to
ComfyUI core); the LLM loader + the light/audio engines hardcode `device="cuda"` with inconsistent CPU
fallbacks. fp8 weight dtype is the single most backend-sensitive surface.**

Heavy engines — device delegated to ComfyUI core (portable):

- LTX builds core node classes and runs via the bridge — `eng_ltx_video.py:418-433` (`_node_candidates`:
  `CheckpointLoaderSimple`, `CLIPLoader`, `CLIPTextEncode`, `EmptyLTXVLatentVideo`, `LTXVConditioning`,
  `KSampler`, `VAEDecode`), executed at `eng_ltx_video.py:721` (`_wb.run_graph(... free_after_use=True)`).
- Wan-i2v / Wan-ti2v likewise — `eng_wan_i2v.py:298`, `eng_wan_ti2v.py:288`.
- HuMo likewise — `eng_humo.py:335`.
- **No `.cuda()` / `device="cuda"` in any of these.** They never place tensors themselves; ComfyUI's
  `comfy.model_management.get_torch_device()` (CUDA/ROCm/MPS/CPU-aware) governs placement. This is the
  key portability asset: the bulk of the GPU work already routes through a backend-agnostic layer.

Backend-sensitive ops the heavy engines DO assume:

- **fp8 e4m3 weights** (Ada/Blackwell-native; absent on Ampere, patchy on ROCm/MPS):
  `eng_humo.py:72` (`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`), `eng_humo.py:190`
  (`umt5_xxl_fp8_e4m3fn_scaled.safetensors`), `eng_wan_i2v.py:190`, `eng_wan_ti2v.py:192`.
- **fp16 text encoders:** `eng_ltx_video.py:538` (`t5xxl_fp16.safetensors`), `eng_humo.py:193`
  (`whisper_large_v3_fp16.safetensors`), `eng_humo.py:188` (lightx2v distill LoRA `bf16`).
- **GGUF UNET** (`UnetLoaderGGUF`) — Wan-ti2v 5B, `eng_wan_ti2v.py:217`. GGUF + the GGUF loader node is a
  custom-node dependency (R7) that may or may not have a non-CUDA path.
- **SageAttention contamination gate** — `motion_common.py:71-119`: LTX *fails closed* if SageAttention
  is patched (`assert_sage_not_patched`, `eng…:114`); Wan routes to a sidecar. This is portability-neutral
  (Sage is a CUDA-only KJ extra that OTR avoids), but the gate logic reads
  `comfy.model_management.sage_attention_enabled()` (`motion_common.py:96`) — fine on any backend.

LLM + light/audio engines — explicit device pins (these need a chokepoint):

- LLM loader default `device="cuda"` (`_otr_model_loader.py:105`); post-load `model.to(device)` only on
  the non-quantized path (`_otr_model_loader.py:526`); `torch_dtype=torch.bfloat16` hardcoded
  (`_otr_model_loader.py:282`).
- `nodes/_otr_bark_lib.py:127` — `device = "cuda" if torch.cuda.is_available() else "cpu"` (has fallback).
- `nodes/_otr_audio_engines/eng_musicgen.py:56,99` — same guarded pattern (has fallback).
- `nodes/_otr_audio_engines/eng_kokoro.py:106` — **hardcoded `device="cuda"` with NO CPU fallback** (a
  real T3 break point; flag LOUD).
- `nodes/_otr_audio_engines/eng_stable_audio.py:79`, `eng_stable_audio_3.py:187` — guarded.
- `nodes/_otr_video_engines/eng_still_parallax.py:250` — guarded `cuda`/`cpu`, `cpu_ok: True` in the
  registry (CPU-degradable by design).
- `nodes/rtx_upscale.py:307` — `.cuda()` with a sm_120 incompat guard already noted (`:133`).

VAE tiling: no OTR-side tiling logic found; tiling is whatever the ComfyUI core `VAEDecode` does
(backend-agnostic). No OTR-specific tiling constant to port.

-----

## R3 — Hardcoded video models & per-tier backend/dtype requirements

**Verdict: the model catalog is the `CAPABILITIES` table in `registry.py:127-210`. The "3-4 hardcoded
video models" the seed refers to are the real-render engines LTX, HuMo (14B + 1.7B), Wan-i2v, Wan-ti2v,
plus latentsync; the rest are CPU/light procedural floors. fp8 is the gating dtype for the heavy NVIDIA
models.**

Real-render motion/face engines (the seed's "3-4 hardcoded"):

| Engine        | Weights / dtype (file:line)                                   | Backend sensitivity |
|---------------|---------------------------------------------------------------|---------------------|
| `ltx_video`   | LTX 2B + `t5xxl_fp16` (`eng_ltx_video.py:538`)                 | fp16 broad; LTX runs on Ampere+. Lowest-risk heavy engine. |
| `humo` (14B)  | `…HuMo-14B_fp8_e4m3fn…` (`eng_humo.py:72`) + umt5 fp8 (`:190`) + whisper fp16 (`:193`) | fp8 → Ada/Blackwell only; blocked on Ampere/MPS/ROCm without a non-fp8 build. |
| `humo_1.7B`   | `humo_1.7B_fp16.safetensors` (`eng_humo.py:452`)              | fp16 → broad. The shipping default char tier; most portable HuMo. |
| `wan_i2v`     | Wan2.2 14B fp8 + umt5 fp8 (`eng_wan_i2v.py:190`), `vram 14500` (`registry.py:183`) | fp8 → Ada/Blackwell; needs GGUF/bf16 swap below. |
| `wan_ti2v`    | Wan2.2 5B GGUF Q5_K_M + umt5 fp8 (`eng_wan_ti2v.py:192,217`)  | GGUF UNET loader is a custom node (R7); 8 GB tier. |
| `latentsync`  | latentsync-1.5, `requires_sidecar: True` (`registry.py:145-147`) | sidecar venv; backend isolated. |

CPU / light floors (already portable — `cpu_ok: True` in `registry.py`):

- `abstract`, `still_kenburns`, `station_card`, `visualizer` — `vram_estimate_mb: 0`, `cpu_ok: True`
  (`registry.py:128-135`).
- `still_parallax` — DepthAnythingV2-small + numpy warp, `cpu_ok: True` (`registry.py:161-164`).
- `ltx_orbit` — prompt preset over LTX, same physics (`registry.py:155-157`).
- `mesh_stage` — hy3d core mesher + Blender (`registry.py:172-176`), `cpu_ok: False`.

3D / talk lanes (already toolchain-gated dark — out of this scope but relevant to R7):

- `triposg_talk` (cu128 prebuilt wheels, sidecar), `hunyuan3d_talk` / `trellis_talk`
  (`required_toolchain: "cu128_toolkit"`, sidecar) — `registry.py:201-209`.

Per-tier marking (full grid in `TIER_MATRIX.md`): the fp16/CPU engines (`ltx_video`, `humo_1.7B`,
`still_*`, floors) are the model set that survives onto T2/T3; the fp8 engines (`humo` 14B, `wan_i2v`)
need a quantized/bf16 swap or are blocked on the non-CUDA tiers.

-----

## R4 — Where the 14.5 GB / 16 GB budget is baked in

**Verdict: ONE constant already has a runtime env seam (video); FOUR more are hardcoded constants; the
LLM budget is already adaptive. A clean parameterization plan exists.**

Already parameterized (the model to copy):

- `nodes/_otr_video_engines/motion_common.py:40` — `VRAM_CEILING_MB = 14500`, read via
  `dynamic_vram_ceiling_mb()` (`:43-55`) which honors `OTR_VRAM_CEILING_MB`. Exported per-execution by
  `nodes/_otr_workflow_validator.py:333` (`os.environ["OTR_VRAM_CEILING_MB"] = str(ceiling)`).
- Headless launchers also set it: `scripts/_otr_soak_server_launch.cmd:19,21` (determinism env);
  `scripts/run_comfy_otr.ps1:7,9`.

Already adaptive (reads real hardware — no change needed):

- `_otr_model_loader.py:192-194` reads `torch.cuda.get_device_properties(0).total_memory`;
  `:233-235` sets `budget_gb = total_vram - 2.5`; `:410` `if total_vram >= 14.5: device_map={"":0}`.
  On a 24 GB Ada or an 8 GB card this already scales — though the `>= 14.5` "Flagship Sovereignty"
  threshold and the `- 2.5` margin are tuned for the 16 GB box and should become tier params.

Hardcoded ceiling constants (candidates for a single tier parameter):

- `nodes/_vram_log.py:45` — `VRAM_CEILING_GB: float = 14.5` (the `VRAM_CEILING_EXCEEDED` gate,
  `_vram_log.py:108-110`).
- `nodes/_otr_model_catalog.py:1027` — `DEFAULT_VRAM_CEILING_GB = 14.5` (drives `check_vram_fit`,
  `:1101-1166`).
- `nodes/_otr_lfc_watchdog.py:55` — `VRAM_DEFAULT_CEILING_GB: float = 14.0`.
- `nodes/_otr_freeze_cascade.py:577` + `nodes/OTR_LedgerFreezeCascade.py:302` — `vram_ceiling_gb: float
  = 14.0` defaults; the node UI default + tooltip hardcode "16 GB total, 0.5 GB margin under the 14.5 GB
  usable cap" (`OTR_LedgerFreezeCascade.py:214-226`).

Per-engine VRAM estimates (informational; would need per-tier review but not per-tier values):

- Video: `registry.py:127-210` (`vram_estimate_mb` per engine, e.g. `wan_i2v: 14500`, `humo: 14000`).
- Image: `nodes/_otr_image_engines/registry.py:108-129`.
- Audio: `nodes/_otr_audio_engines/registry.py:190-211`.

Context-window clamp (16 GB-specific): `_otr_model_loader.py:168-179` caps LLM context at 8192;
`_otr_model_catalog.py:921-1023` documents the 128k→8192 clamp "feeding 128k tokens on a 16 GB card OOMs
instantly."

Parameterization plan (for when the gate lifts): introduce ONE tier-config source (e.g.
`OTR_VRAM_CEILING_MB` already exists + a companion `tier` profile) and have ALL five ceiling sites read
it through a single helper, mirroring `dynamic_vram_ceiling_mb()`. The `motion_common` pattern is the
template; the four hardcoded constants are ~1-line reads each. The adaptive LLM budget needs its `14.5`
threshold + `- 2.5` margin turned into tier params. Low-to-moderate effort; the hard part is choosing
the per-tier numbers (operator + measurement), not the wiring.

-----

## R5 — Device-selection & dtype-selection chokepoint

**Verdict: there is NO single device chokepoint and NO single dtype chokepoint. There are TWO device
regimes (delegated vs hardcoded) and dtype is expressed as per-engine model FILENAMES, not a dtype
variable. This is the central refactor question and the answer to Q5.**

Device:

- Regime A (portable, ~all heavy GPU work): delegated to ComfyUI core via `wrapper_bridge.run_graph` —
  no OTR device code (R2). Nothing to refactor; it already follows whatever backend ComfyUI selects.
- Regime B (pinned): the LLM loader (`_otr_model_loader.py:105` default `"cuda"`) and the light/audio
  engines each independently compute `device = "cuda" if torch.cuda.is_available() else "cpu"`
  (`_otr_bark_lib.py:127`, `eng_musicgen.py:56`, `eng_stable_audio.py:79`, `eng_still_parallax.py:250`)
  — EXCEPT `eng_kokoro.py:106` which hardcodes `"cuda"` with no fallback. There is no shared
  `otr_get_device()` helper; the logic is duplicated ~6 places.

dtype:

- The LLM path hardcodes `torch_dtype=torch.bfloat16` (`_otr_model_loader.py:282`) and auto-NF4 via
  bitsandbytes (`:311-342`). bf16 is broad (Ampere+), so the dtype itself is fine on T1/T2; the bnb
  quantization is the T3 problem (R7).
- The diffusion engines do NOT have a dtype variable — they select a *weight file* (fp8/fp16/GGUF) via
  env-overridable names (`eng_humo.py:72,190`, `eng_wan_i2v.py:190`, `eng_ltx_video.py:534-538`). The
  "dtype chokepoint" for these is really "which model file is on disk + named in the env."

Prioritized change list (when gate lifts):
1. Add a single `otr_device.resolve()` helper that wraps `comfy.model_management.get_torch_device()`
   (so Regime B inherits ROCm/MPS) and replace the ~6 duplicated `device="cuda"` sites — **small**,
   mechanical, high-value for T3. Fixes the `eng_kokoro.py:106` no-fallback break.
2. Add a tier→dtype/model-file map so the fp8 engines can resolve a bf16/GGUF weight on non-fp8 tiers —
   **medium**, mostly config + asset fetch, gated on Q2 (reduced model set).
3. Replace the hardcoded `torch.bfloat16` + auto-NF4 with a tier-aware dtype/quant selector for the LLM
   loader (bnb only on CUDA/ROCm; fp16 or HTTP-lane elsewhere) — **medium**.

Effort estimate: device helper ~0.5 day; LLM dtype/quant selector ~1–2 days; per-engine model-file tier
map ~2–3 days plus the model-asset fetch/verify work and per-tier soak. The device path is NOT
"assumed throughout" in the scary sense — the heavy lifting is already delegated. **Q5 answer: a partial
chokepoint exists (ComfyUI core for diffusion); the gap is the LLM/audio/light Regime-B device pins and
the absence of a dtype/model-file tier map.**

-----

## R6 — Determinism audit & per-tier definition

**Verdict: determinism is ALREADY scoped as within-render per-seed, with the process default deliberately
non-strict so video does not crash on sm_120. This already matches the recommended Q1 answer; cross-tier
parity is already implicitly out of scope.**

Evidence:

- `nodes/_otr_determinism.py` is the determinism module. `REQUIRED_DETERMINISM_ENV`
  (`:27-32`): `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`, `NVIDIA_TF32_OVERRIDE=0`,
  `TOKENIZERS_PARALLELISM=false` — set BEFORE torch by the launcher
  (`scripts/run_comfy_otr.ps1:7-9`, `scripts/_otr_soak_server_launch.cmd:19-21`).
- `seed_all_rngs(seed)` (`:66-85`) seeds python/numpy/torch/cuda; `deterministic_inference(seed)`
  (`:116-183`) is a SCOPED context manager that enables `use_deterministic_algorithms(True)` + pins the
  SDPA MATH backend, then restores all flags — used ONLY around single audio forwards.
- Crucially, `:11-13` + `:87-96`: the PROCESS default stays non-strict
  (`apply_module_determinism_defaults` sets TF32 off + cuDNN deterministic but does NOT enable
  `use_deterministic_algorithms`) *"so the default video render does not crash on sm_120 (C-2/C-3)."*
- The CUDA branches all guard on `torch.cuda.is_available()` (`:83, 126, 155, 179`) and the SDPA-math
  pin has a CPU-only fallback (`:113`), so the module is already CPU-safe — and would be MPS/ROCm-safe
  (those report `cuda.is_available()` False / or ROCm masquerades as cuda).
- Video/image per-seed determinism flows through ComfyUI's own `KSampler` seed widget (in the graphs at
  `eng_ltx_video.py:431`), not OTR code.
- The frozen-audio invariant (`test_audio_byte_identical`) is byte-identical *on the reference tier* —
  GO_FORWARD §2 + the seed §4 already scope it to T0.

Proposed per-tier definition of "deterministic":
- **Within-tier, within-render, per-seed reproducibility** (same seed + same tier + same model files →
  same output), which is what the code already targets.
- **Cross-tier bit-identity is explicitly WAIVED** (different kernels/dtype across CUDA/ROCm/MPS make it
  unachievable — standard for these backends).
- The `test_audio_byte_identical` golden remains a **T0-only** assertion; on T1–T3 it should be replaced
  by a within-tier render-twice equality check (or marked tier-skipped, LOUD).

-----

## R7 — Dependency audit: which deps hard-require CUDA

**Verdict: `bitsandbytes` is the one base-requirements dep that hard-requires CUDA and is effectively
mandatory on the default LLM path. Everything else is either torch-build-inherited, host-ComfyUI-managed,
or already isolated in sidecar venvs.**

| Dep | Source | CUDA-hard? | Non-CUDA path |
|-----|--------|-----------|----------------|
| `torch==2.10.0+cu130` | `requirements.video.txt:16` | The build is, the pin isn't enforced by base reqs (`requirements.txt:2`) | Swap host torch to `cu12x` (T1/T2) / ROCm wheels (T3a) / MPS-enabled (T3b). ComfyUI manages it. |
| `bitsandbytes>=0.42.0` | `requirements.txt:19` | **Yes** — CUDA-only; auto-selected NF4 for ~all models (`_otr_model_loader.py:156-157, 311-342`) | ROCm: partial upstream bnb. MPS: none. Need fp16/bf16-full path or the existing HTTP/Ollama lane (`_otr_model_loader.py:780-794`). |
| `transformers`, `tokenizers`, `sentencepiece`, `accelerate`, `diffusers` | both reqs | No | Backend-agnostic (ride on torch). |
| `soundfile`, `numpy`, `feedparser`, `Pillow`, `huggingface_hub`, `lm-format-enforcer` | both reqs | No | Pure-python / C-lib, portable. |
| SageAttention / xformers / flash_attn | NOT deps — `BANNED` (`otr_video_dep_pilot.py:41`) | n/a | OTR uses SDPA; nothing to port. |
| `UnetLoaderGGUF` (Wan-ti2v) | ComfyUI custom node (`eng_wan_ti2v.py:217`) | Depends on the GGUF node pack | Verify ROCm/MPS support of city96 ComfyUI-GGUF; may block `wan_ti2v` on T3. |
| cu128 audio sidecars (indextts2, dia, chatterbox) | isolated subprocess venvs (`eng_indextts2.py:4-5`, `eng_dia.py:6-7`) | Their venvs are, the main process isn't | Already isolated; each sidecar would need its own per-tier venv (or fall back to bark/kokoro). |
| cu128 3D talk lanes (triposg/hunyuan3d/trellis) | toolchain-gated dark (`registry.py:201-209`) | Yes | Out of scope; already dark. |
| ComfyUI core | host install | No — it is the backend-abstraction layer | OTR depends on whatever backend the host ComfyUI supports (this is the portability lever). |

R7 conclusion: the ONLY new hard dependency work for T1/T2 is the torch build swap (host-managed) and
deciding whether bnb-NF4 stays (it works on cu12x). For T3a/T3b the bnb path must be replaced and the
GGUF + sidecar deps audited per backend.

-----

## Q1–Q5 answers (code-grounded)

**Q1 — Determinism scope (OPERATOR DECISION — recommended answer flagged).**
*Finding:* the code already implements within-render per-seed determinism with a non-strict process
default specifically to survive sm_120 (`_otr_determinism.py:11-13`); cross-run/cross-backend bit-identity
is not attempted. *Recommended:* **YES — within-tier reproducibility is sufficient; cross-tier parity is
explicitly waived, and `test_audio_byte_identical` becomes a T0-only golden** (replaced by within-tier
render-twice equality on T1–T3). This is the lowest-effort path and matches existing behavior.
**>>> Requires Jeffrey's confirmation (it relaxes the byte-identical invariant on non-T0 tiers).**

**Q2 — Reduced model set for T3a/T3b.** *Finding:* the fp8 engines (`humo` 14B `eng_humo.py:72`,
`wan_i2v` `eng_wan_i2v.py:190`) have no native non-CUDA path; the fp16/CPU engines (`ltx_video`,
`humo_1.7B`, `still_*`, floors) do. *Code-grounded recommendation:* a reduced model set for T3a/T3b is
**not just acceptable, it is forced by the dtype reality** — ship `ltx_video` + `humo_1.7B` + the CPU
floors on non-CUDA tiers, mark the fp8 engines blocked-LOUD. (Still ultimately an operator product call,
but the code leaves little choice.)

**Q3 — One build vs N builds.** *Finding:* the architecture already favors ONE build — the engine
registry is data-driven (`registry.py:127-210`), device is mostly delegated, and there's already a
runtime env seam (`OTR_VRAM_CEILING_MB`). *Recommendation:* **single tier-parameterized codebase** with
a `tier` profile gating model-file selection + ceilings + the bnb/non-bnb LLM path. N branches would
fork the already-converged registry and the frozen audio spine — avoid.

**Q4 — Floor VRAM target (OPERATOR DECISION — recommended answer flagged).**
*Finding:* the existing 8 GB tier is real and built — `wan_ti2v` is the explicit "8 GB tier"
(`registry.py:186-195`, `humo_1.7B` medium/7000 `registry.py:142`), and the LLM budget already adapts
down via `total_vram - 2.5` (`_otr_model_loader.py:233-235`) with NF4 quantization. *Recommended:*
**8 GB floor for T1/T2** (matches the existing `wan_ti2v` / `humo_1.7B` / NF4 design point), with the
caveat that 8 GB can only run the medium/light engines, not the 14B fp8 set. **>>> Requires Jeffrey's
confirmation — it sets the cap math and the minimum viable engine set per tier.**

**Q5 — Where the chokepoint lives.** *Finding (answered fully in R5):* a PARTIAL chokepoint exists —
ComfyUI core governs device for all heavy diffusion work — but there is no single OTR device or dtype
selector. Device is duplicated ~6× in Regime B (`_otr_model_loader.py:105`, `_otr_bark_lib.py:127`,
`eng_musicgen.py:56`, `eng_kokoro.py:106` [no-fallback], `eng_stable_audio.py:79`,
`eng_still_parallax.py:250`); dtype is per-engine model-filenames. *Effort:* device helper ~0.5 day;
LLM dtype/quant selector ~1–2 days; per-engine tier model-file map ~2–3 days + asset/soak. **Moderate,
not a rewrite** — the delegated diffusion path is the reason it's tractable.

-----

## Tier-by-tier verdict

(Static-analysis prediction; none booted. Full grid in `TIER_MATRIX.md`.)

- **T1 — Ada / sm_89 / cu12x — RUNS WITH MINOR DOCUMENTED CHANGES (highest confidence).** Ada has native
  fp8 e4m3, so HuMo-14B / Wan-i2v fp8 weights should load; bnb-NF4 works on cu12x; SDPA + the banned-Sage
  posture are unchanged. The only required change is the host torch build (`cu130`→`cu12x`) and turning
  the `14.5`-tuned ceilings/margins into tier params. **Prove this tier first.**
- **T2 — Ampere / sm_86 / cu11x-cu12x — RUNS WITH MODERATE DOCUMENTED CHANGES.** Ampere has NO native fp8
  → `humo` (14B) and `wan_i2v` need bf16/GGUF weight swaps (R3) or are dropped; `humo_1.7B` (fp16),
  `ltx_video` (fp16), `wan_ti2v` (GGUF, pending node audit), and all floors run. bnb-NF4 works. bf16 is
  native on Ampere.
- **T3a — AMD / ROCm — RUNS, REDUCED SET, NEEDS NON-BNB LLM PATH.** ComfyUI core handles ROCm device
  placement; SDPA works. Blockers: bitsandbytes (partial/absent → use fp16-full or HTTP/Ollama lane),
  fp8 weights (drop the 14B fp8 engines), GGUF-node ROCm support (audit), and the `eng_kokoro.py:106`
  no-fallback pin. Reduced set: `ltx_video`, `humo_1.7B`, floors.
- **T3b — Apple Silicon / MPS — RUNS, MOST REDUCED, CPU FALLBACK FOR GAPS.** MPS is fp16-centric with op
  gaps; no fp8, no bnb, no GGUF-CUDA. Realistic set: `ltx_video` (if MPS handles the LTX ops),
  `still_*`/`abstract`/`visualizer` floors (pure numpy/CPU), `still_parallax` (CPU-degradable). The LLM
  loader needs a non-bnb fp16 path or the HTTP lane. Unified memory means the ceiling constants must be
  reinterpreted (no discrete VRAM). Highest uncertainty.

-----

## Device/dtype chokepoint analysis (consolidated)

- **Single point?** NO. Partial delegation (ComfyUI core) for diffusion; duplicated `device="cuda"` for
  LLM/audio/light; dtype = per-engine model-file names.
- **Prioritized change list:** (1) `otr_device.resolve()` wrapper over
  `comfy.model_management.get_torch_device()` replacing the ~6 Regime-B pins (fixes kokoro no-fallback);
  (2) tier→dtype/model-file map for the fp8 engines; (3) tier-aware LLM dtype/quant selector
  (bnb-on-CUDA/ROCm, fp16/HTTP elsewhere).
- **Effort estimate:** device helper ~0.5 day; LLM selector ~1–2 days; engine tier map ~2–3 days + asset
  fetch + per-tier soak. Whole T1 bring-up plausibly < 1 week of code (the long pole is hardware access
  + soak, not code).

-----

## 16 GB budget enumeration (every site)

| File:line | Constant / use | Status |
|-----------|----------------|--------|
| `nodes/_otr_video_engines/motion_common.py:40` | `VRAM_CEILING_MB = 14500` | Already env-overridable via `dynamic_vram_ceiling_mb()` (`:43-55`) → `OTR_VRAM_CEILING_MB`. **Template.** |
| `nodes/_otr_workflow_validator.py:333` | exports `OTR_VRAM_CEILING_MB` | The seam producer. |
| `nodes/_vram_log.py:45` | `VRAM_CEILING_GB = 14.5` | Hardcoded; gate at `:108-110`. Parameterize. |
| `nodes/_otr_model_catalog.py:1027` | `DEFAULT_VRAM_CEILING_GB = 14.5` | Hardcoded; drives `check_vram_fit`. Parameterize. |
| `nodes/_otr_lfc_watchdog.py:55` | `VRAM_DEFAULT_CEILING_GB = 14.0` | Hardcoded. Parameterize. |
| `nodes/_otr_freeze_cascade.py:577` | `vram_ceiling_gb = 14.0` (default arg) | Hardcoded default. Parameterize. |
| `nodes/OTR_LedgerFreezeCascade.py:302` | `vram_ceiling_gb = 14.0` + UI default/tooltip `:214-226` | Node-widget default; "16 GB / 14.5 cap" tooltip. Parameterize + reword. |
| `nodes/_otr_model_loader.py:233-235` | `budget_gb = total_vram - 2.5` (adaptive) | Reads real VRAM; the `- 2.5` margin is 16 GB-tuned → tier param. |
| `nodes/_otr_model_loader.py:410` | `if total_vram >= 14.5: device_map={"":0}` | "Flagship Sovereignty" threshold; 16 GB-tuned → tier param. |
| `nodes/_otr_model_loader.py:168-179` | LLM context cap 8192 | 16 GB-tuned (`_otr_model_catalog.py:921-1023`); → tier param. |
| `nodes/_otr_video_engines/registry.py:127-210` | per-engine `vram_estimate_mb` | Policy estimates; per-tier review, not per-tier values. |
| `nodes/_otr_image_engines/registry.py:108-129` | per-engine `vram_estimate_mb` | Same. |
| `nodes/_otr_audio_engines/registry.py:190-211` | per-engine `vram_estimate_mb` | Same. |

Plan: one tier profile, one shared ceiling-reader (mirror `dynamic_vram_ceiling_mb`), all five hardcoded
constants read it; turn the LLM loader's `14.5` threshold + `- 2.5` margin + `8192` context cap into
tier fields. Wiring is low effort; choosing per-tier numbers needs measurement + operator input.

-----

## Invariants restated per tier (LOUD flags where they can't hold)

| Invariant (seed §4) | T0 (Blackwell) | T1 (Ada) | T2 (Ampere) | T3a (ROCm) | T3b (MPS) |
|---------------------|----------------|----------|-------------|------------|-----------|
| 100% local/offline | holds | holds | holds | holds | holds (HTTP-LLM lane is local Ollama only, or skip) |
| Determinism (seed-keyed) | holds (incl. byte-identical audio) | within-tier | within-tier | within-tier | within-tier |
| **Byte-identical audio (`test_audio_byte_identical`)** | holds | **LOUD: T0-only golden; T1+ = within-tier render-twice** | same | same | same |
| Every fallback LOUD | holds | holds | holds | **LOUD: fp8 engines blocked → must log, not silently floor** | **LOUD: same + MPS op-gap CPU fallbacks must be loud** |
| Dependency isolation (V-12) | holds | holds | holds | holds (per-tier sidecar venvs) | holds |
| UTF-8/SFW | holds | holds | holds | holds | holds |
| Single heavy engine ≤ ceiling | 14.5 GB | **tier param** (8–24 GB) | **tier param** | **tier param** | **LOUD: unified memory — ceiling concept must be reinterpreted** |
| Frozen audio spine untouched | holds | holds | holds | holds | holds |

-----

## Open risks / unknowns

- **No tier has been booted.** All verdicts are static predictions. "Runs with documented changes"
  cannot be GREEN without hardware. (Stop-condition-adjacent: nothing surfaced that *fundamentally*
  blocks a tier with no fallback — the fp8 engines degrade to the fp16/GGUF set, they don't hard-import
  an sm_120-only kernel.)
- **GGUF loader (`UnetLoaderGGUF`) backend support unknown** — gates `wan_ti2v` on T3 (`eng_wan_ti2v.py:217`).
- **MPS op coverage for the LTX/Wan graphs unknown** — could force more CPU fallback than predicted.
- **bitsandbytes ROCm maturity** moves quickly; the partial-support claim needs verification at build time.
- **ComfyUI core's own ROCm/MPS device routing** is assumed solid — OTR rides on it, so any core gap is
  inherited. Not verified against the current ComfyUI version.
- **The `- 2.5` GB margin + `14.5` thresholds** in the LLM loader are empirical 16 GB numbers; the right
  values for 8 GB / 24 GB / unified memory need measurement.
- The parallel session's `wrapper_bridge.py` flux-eviction/VRAM work may change the reclaim semantics
  cited here (`free_after_use`, `reclaim_idle_models`) — re-verify those line refs after that lands.

-----

## Recommended sequencing (only if/when the operator lifts the §10 gate)

1. **T1 proof-of-life first (highest ROI):** stand up a `cu12x` torch on an Ada box, run the existing
   suite + a single real episode. This validates the "ComfyUI-delegated device path" hypothesis with
   near-zero code. If T1 runs as-is, the whole portability thesis is de-risked.
2. **Ceiling parameterization:** wire the four hardcoded ceilings + the LLM thresholds to a single tier
   reader (copy `dynamic_vram_ceiling_mb`). Mechanical; unblocks 8 GB/24 GB cap math.
3. **`otr_device.resolve()` helper:** replace the ~6 `device="cuda"` Regime-B pins (fixes the
   `eng_kokoro.py:106` no-fallback). Unblocks T3 device routing.
4. **T2 (Ampere):** add the fp8→bf16/GGUF model-file tier map (Q2-gated); soak the fp16 set.
5. **LLM non-bnb path:** tier-aware dtype/quant selector for T3 (bnb-on-CUDA/ROCm, fp16/HTTP elsewhere).
6. **T3a (ROCm) then T3b (MPS):** reduced set, CPU fallback for gaps; audit GGUF + sidecars per backend.
7. **Determinism re-scope:** make `test_audio_byte_identical` T0-only; add within-tier render-twice
   checks (Q1-gated).

Each step is independently shippable and leaves T0 untouched. Do NOT start any of this until Jeffrey
lifts the gate.
