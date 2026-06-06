# A-S2 de-risk spike probes (CW-3) -- OPERATOR runs these on the 5080

These are THROWAWAY de-risk probes for ticket A-S2 (the GO/NO-GO gate before the
M1 first-episode build). The CW-3 window writes them on CPU; **the operator runs
them on the RTX 5080** (the agent does NOT run them -- they need the real GPU and
results must not be faked). Each prints `PASS` or `NO-GO: <reason>` and exits 0
(pass) / 1 (no-go). A single NO-GO reshapes A-S3 -- do not proceed to the render
build until all five pass (or the NO-GO is consciously accepted + documented).

Run from the repo root with the ComfyUI venv:

```
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
scripts\_otr_a_s2_probes\run_all.bat
```

or individually, e.g.:

```
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\_otr_a_s2_probes\probe_e_sidecar_vram_free.py
```

## The five probes

| # | File | Proves | PASS criterion |
|---|------|--------|----------------|
| a | `probe_a_flux_load_teardown_leak.py` | a heavy load -> teardown -> reload leaves no resident VRAM | machine-wide NVML returns to within `OTR_LEAK_TOL_MB` (default 768) of baseline after teardown; reload shows no growth |
| b | `probe_b_coldimport_dep_pilot.py` | the cold-import dep-pilot catches a contaminating custom node | scans every `custom_nodes/*/__init__.py` for module-scope torch/sage/xformers/flash_attn; reports any that would fire at startup (KJNodes class) |
| c | `probe_c_silent_mux_audio_hash.py` | the silent mux is byte-identical (mux-LAST, `-c:a copy`, NO `-shortest`) | the audio stream extracted from the muxed output hashes EQUAL to the master; `-shortest` absent |
| d | `probe_d_vram_boundary_failclosed.py` | the VRAM boundary check FAILS CLOSED at the ceiling | `gpu_residency`-style floor probe refuses to proceed when used VRAM is over the ceiling, rather than OOMing mid-render |
| e | `probe_e_sidecar_vram_free.py` | a REAL subprocess sidecar's ~10 GB is fully reclaimed AFTER it exits (the actual hardest C->A boundary) | parent machine-wide NVML `used <= OTR_SIDECAR_FLOOR_MB` (default 768) after the child `sys.exit(0)` |

All probes import the SHIPPED `nodes/_otr_shared/gpu_residency.py` (AS-3) for the
machine-wide NVML probe + the cross-process lease, so the spike also smoke-tests
the lease on the real GPU.

## Tunables (env vars)
- `OTR_LEAK_TOL_MB` (probe a) -- post-teardown tolerance, default 768.
- `OTR_SIDECAR_ALLOC_MB` (probe e) -- child allocation target, default 10000.
- `OTR_SIDECAR_FLOOR_MB` (probe e/d) -- post-exit floor, default 768.
- `OTR_FLUX_CKPT` (probe a) -- path to a Flux checkpoint; if unset the probe
  uses a raw ~10 GB torch tensor as the heavy-residency proxy + says so.
- `OTR_MASTER_AUDIO` / `OTR_SILENT_CLIP` (probe c) -- paths; if unset the probe
  synthesizes a 3 s master + a silent clip with ffmpeg.

## After running
Paste the five PASS/NO-GO lines back into the next window. GO (all pass) ->
A-S3/CW-4 (the new render path). Any NO-GO -> record it + reshape A-S3.
