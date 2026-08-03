# IDEA: a hardware compatibility matrix (NVIDIA / AMD / Intel / Apple x VRAM)

**Operator request, 2026-08-02 late.** Captured now so it survives the night;
NOT scheduled, NOT scoped, NOT started.

> "I'd like at some point a matrix of major NVIDIA GPUs, AMD and Intel GPUs, and
> Mac silicon and VRAM levels, to see what is compatible or not -- maybe a
> problem statement for each, in a new menu."

## Why this is closer than it looks

`registry.CAPABILITIES` already declares, per engine: `device_backends`,
`requires_vendor`, `needs_fp8_te`, `needs_fp4_te`, `practical_without_gpu`,
`required_toolchain`, `requires_sidecar` and `model_requirements`. And
`nodes/_otr_shared/capability_profiles.py` already DERIVES per-profile
enable-sets from that table rather than hand-listing them.

So the data model exists. What is missing is the second axis (hardware) and,
much more importantly, the evidence.

## The one hard design rule, learned today

**Every cell must carry a RECEIPT, and DECLARED must never be rendered as
MEASURED.**

Today proved the current declarations cannot be trusted as capability claims:

* `humo_1.7B` declares `needs_fp8_te: False` / `needs_fp4_te: False` and is
  still marked `["cuda"]` with no stated reason -- it looks inherited from its
  14B sibling.
* `ltx_8gb` likewise: no fp8, no fp4, no GGUF, no SageAttention anywhere in its
  adapter, one incidental `cuda` mention in a comment -- and still CUDA-only.
* Meanwhile `humo`'s CUDA-only marking IS correct, for a real external reason
  found only by leaving the repo: Metal has no `Float8_e4m3fn`.

A matrix that renders all three the same colour would be worse than no matrix,
because it would look authoritative. Three cell states minimum:
**VERIFIED-WORKS / VERIFIED-BLOCKED(reason) / UNTESTED**, and untested must be
visually distinct rather than defaulting to either.

## Facts already established, with sources

**Apple Silicon**
* Metal does not support `Float8_e4m3fn`: `RuntimeError: Undefined type
  Float8_e4m3fn`. Blocks every fp8 engine outright. An unofficial ComfyUI patch
  routes fp8 ops through CPU (PR #12378, unmerged).
* ComfyUI + MPS for video is reported unusable in practice: Wan 2.2 GGUF took
  **82 minutes** for a 2-second 832x480 clip on an M1 Max 64 GB; LTX-2 GGUF
  produced collapsed motion; the official 2-stage LTX pipeline NaN'd on MPS.
* Draw Things (native Metal) and MLX ports ARE viable for LTX-2.3 including
  joint audio -- an M4 Max did a 5-second clip in ~152 s. **Different runtime
  from ComfyUI**, which is the whole point: ~100x apart on the same models.
* Practical Metal ceiling reported around 4 s per clip -- which OTR's 97-frame
  (3.88 s) architecture already respects.
* The `viz_*` and `still_*` lanes are pure numpy/PIL/pycairo/ffmpeg and need no
  GPU at all, so they are hardware-agnostic rather than Metal-dependent.

**NVIDIA** -- the only tier with real production evidence, all of it on one
RTX 5080 Laptop 16 GB. VRAM tiers below 16 GB (`ltx_8gb`, `fastwan_8gb`,
`wan_ti2v`) are DECLARED 8 GB-capable and, per `GO_FORWARD`, not yet proven on
physical 8 GB hardware.

**AMD / Intel** -- no evidence in this repo whatsoever. Not one line. Any cell
for them starts as UNTESTED, and saying so is the honest first version.

## Shape, if it is ever built

1. **Axes:** engine (19 local + 13 cloud) x hardware class. Hardware class is
   probably vendor + architecture + VRAM tier, not individual card names -- a
   card list ages badly and multiplies rows without adding truth.
2. **Cell:** state + blocker + what would unblock it. The "problem statement per
   cell" the operator asked for is the valuable half; the grid is just how it is
   read.
3. **Source of truth:** extend `CAPABILITIES` rather than starting a parallel
   table. A second table drifts from the first -- exactly how
   `otr_w45_campaign.py` came to run six engines while claiming nineteen.
4. **Menu surface:** a read-only report node, or a generated markdown doc like
   `ENGINE_MATRIX.md` already is. Generated, never hand-maintained.

## What it would cost, honestly

The grid is a day. The EVIDENCE is the project: every VERIFIED cell needs a real
run on real hardware this project does not own. The defensible first version is
therefore mostly UNTESTED cells with accurate blockers on the few that are
known -- which is still far more useful than nothing, and honest.
