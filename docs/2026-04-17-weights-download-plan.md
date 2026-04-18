# Video Stack Weights Download Plan

**Date:** 2026-04-17
**Context:** Gemini's round-robin pick, Step 3 of 3.
Step 1 (repo hygiene `b89d650`) and Step 2 (WEDGE_PROBE `a690450`) are done.
Step 3 — FLUX FP8 smoke test — is **blocked** until the visual-stack weights
are on disk. Only LTX-2.3 22B distilled FP8 is currently present.

## What's missing

The weights audit at HEAD `a690450` shows these local paths empty:

| Asset | Expected path |
|---|---|
| FLUX.1-dev base | `C:\Users\jeffr\Documents\ComfyUI\models\diffusers\FLUX.1-dev` |
| PuLID-FLUX adapter | `C:\Users\jeffr\Documents\ComfyUI\models\pulid\pulid_flux.safetensors` |
| ControlNet Union Pro 2.0 | `C:\Users\jeffr\Documents\ComfyUI\models\controlnet\FLUX.1-dev-ControlNet-Union-Pro-2.0` |
| ControlNet Depth (fallback) | `C:\Users\jeffr\Documents\ComfyUI\models\controlnet\FLUX.1-dev-ControlNet-Depth` |
| DepthAnything V2 Large | HF transformers cache |
| Wan2.1 I2V 1.3B | `C:\Users\jeffr\Documents\ComfyUI\models\diffusers\Wan2.1-I2V-1.3B` |
| Florence-2 Large | `C:\Users\jeffr\Documents\ComfyUI\models\florence2\Florence-2-large` |
| SDXL Inpaint 1.0 | `C:\Users\jeffr\Documents\ComfyUI\models\diffusers\stable-diffusion-xl-1.0-inpainting-0.1` |

Env overrides are respected (`OTR_FLUX_MODEL`, `OTR_PULID_MODEL`,
`OTR_FLUX_CN_UNION`, `OTR_FLUX_CN_DEPTH`, `OTR_DEPTH_MODEL`,
`OTR_WAN_MODEL`, `OTR_FLORENCE_MODEL`, `OTR_SDXL_INPAINT_MODEL`), but the
download script targets the canonical defaults so backends work without
any env flags set.

## Canonical sources + sizes

| HF Repo | Target | Disk (approx) |
|---|---|---|
| `black-forest-labs/FLUX.1-dev` | `models\diffusers\FLUX.1-dev\` | **~23 GB (bf16)** |
| `guozinan/PuLID` (file: `pulid_flux.safetensors`) | `models\pulid\pulid_flux.safetensors` | ~1.2 GB |
| `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` | `models\controlnet\FLUX.1-dev-ControlNet-Union-Pro-2.0\` | ~6.6 GB |
| `Shakker-Labs/FLUX.1-dev-ControlNet-Depth` | `models\controlnet\FLUX.1-dev-ControlNet-Depth\` | ~1.4 GB |
| `depth-anything/Depth-Anything-V2-Large-hf` | HF cache (auto) | ~1.2 GB |
| `Wan-AI/Wan2.1-I2V-1.3B` | `models\diffusers\Wan2.1-I2V-1.3B\` | ~5.5 GB |
| `microsoft/Florence-2-large` | `models\florence2\Florence-2-large\` | ~1.5 GB |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | `models\diffusers\stable-diffusion-xl-1.0-inpainting-0.1\` | ~6.9 GB |

**Total download:** ~47 GB. Disk reservation: leave 60 GB free to cover
HF cache duplication during snapshot download.

## FP8 handling decision

The backends already cast FLUX / Wan2.1 / SDXL to
`torch.float8_e4m3fn` at load time (per C5 in CLAUDE.md). Downloading
the **bf16** canonical diffusers snapshots is the right move — the
runtime quantization happens in the sidecar, so we don't need a
pre-quantized HF repo. This keeps the weights swappable and avoids
committing to a particular community FP8 packaging.

If the `torch.float8_e4m3fn` path trips at load time, the backends
fall back to fp16 on Blackwell (per `_resolve_dtype()` in each backend),
so the same weights work either way.

## HF login

`black-forest-labs/FLUX.1-dev` is **gated** — user must accept the
license at https://huggingface.co/black-forest-labs/FLUX.1-dev and
then log in:

```powershell
# One-time: accept license on HF web, then
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
# Or use huggingface-cli login
& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\huggingface-cli.exe login
```

`HF_TOKEN` already lives in HKCU\Environment per CLAUDE.md. The script
reads `$env:HF_TOKEN` at the shell level. All other repos are public.

## How to run

Launch the script while away — idempotent, resumable, skips already-present
files (HF snapshot_download uses its cache).

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
powershell -ExecutionPolicy Bypass -File scripts\download_video_stack_weights.ps1
```

Expected wall-clock on a 200 Mbps connection: ~35-40 minutes.
Log file: `logs\weights_download.log` (gitignored).

After completion, verify:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\audit_video_stack_weights.ps1
```

Then Step 3 is unblocked: run the FLUX FP8 smoke test, which will be
the first end-to-end test that exercises the backends against real
weights rather than stubs.

## What Step 3's smoke test proves

Once weights are on disk, the smoke test (planned as a one-shot canary)
will exercise:

1. FLUX anchor render on a single reference shot at 1024x1024 FP8
2. WEDGE_PROBE NDJSON capture of the render span
3. LHM VRAM poll before / during / after (14.5 GB ceiling check)
4. Audio path untouched (C7 check — no ffmpeg invocation in FLUX code)

If any of those fail, the backends fall back to stub mode and the
episode renderer keeps producing audio-correct output with placeholder
visuals, so the smoke test is safe to run against a live workflow.

## References

- Backend source: `otr_v2/visual/backends/{flux_anchor,flux_keyframe,pulid_portrait,ltx_motion,wan21_loop,florence2_sdxl_comp}.py`
- WEDGE_PROBE: `otr_v2/visual/wedge_probe.py` (committed `a690450`)
- CLAUDE.md C5 (fp8 native), C7 (audio byte-identical), 14.5 GB VRAM ceiling
