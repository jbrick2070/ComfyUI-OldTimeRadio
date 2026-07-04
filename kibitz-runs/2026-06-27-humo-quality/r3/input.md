# HuMo quality + VRAM-fit -- R2 hardened coding plan (Claude + GPT-5.5 + Gemini-3.1-pro)

R2 panel: GPT-5.5 + Gemini-3.1-pro (both grounded; DeepSeek returned empty -- reasoning-
token exhaustion). Both panel reviews were truncated at the 2000-token cap; the captured,
grounded claims are folded below. Spend this pass ~$0.1227.

## Build order (each step has a KILL-GATE; do not chase a dead lever)
A. Allocator-cache probe (cheapest) -> B. GGUF feasibility gate (only if A still >13.5)
-> C. mouth ceiling leg (independent track) -> D. dep-probe script for a model swap.
Each measured across the FRAME MATRIX below before it counts as "viable".

## Concrete coding slices

### Step A -- allocator-cache probe (DO-NOW, ~1 hr, no engine/workflow change)
- `run_humo_bakeoff.boot_server` already builds an `env` dict; inject
  `PYTORCH_CUDA_ALLOC_CONF` from a new `OTR_BAKEOFF_ALLOC_CONF` knob and run leg ii A/B.
- **Gemini MUST-FIX (grounded):** nvidia-smi "used" reports PyTorch's RESERVED pool, not
  live demand -- `expandable_segments:True` mainly reduces fragmentation, so the NVML peak
  may not drop even if true demand is < 13.5 GB. So ALSO capture TRUE demand
  in-process: log `torch.cuda.max_memory_allocated()` / `memory_reserved()` from the
  server (a tiny always-dirty probe node on the latent edge, or a one-line log in the
  render path of the diagnostic graph) and report BOTH numbers. Decision uses
  max_memory_allocated (true need); NVML reserved is reported alongside.
- KILL-GATE: if true allocated peak < ~13.5 GB at all matrix points, the fp8 14B is
  promotable as-is and Step B is unnecessary.

### Step B -- GGUF 14B (only if A insufficient) -- distinct loader, NOT a kwarg swap
- **Gemini MUST-FIX (grounded):** `_build_graph` hardcodes `UNETLoader` kwargs
  (`unet_name`, `weight_dtype`); `UnetLoaderGGUF` takes DIFFERENT kwargs (`gguf_name`, no
  `weight_dtype`) -- a naive swap CRASHES. The bakeoff BUILDER must emit a separate GGUF
  loader node (its own class_type + widget set), NOT reuse the UNETLoader template, and
  must do it in the harness (NOT edit eng_humo).
- FIRST SLICE / feasibility gate, in order: (1) a HuMo-14B GGUF file exists on disk;
  (2) `UnetLoaderGGUF` is registered on `/object_info`; (3) builder emits a `gguf` leg
  swapping node 1; (4) ONE-FRAME smoke proves `WanHuMoImageToVideo` audio-cross-attn
  survives a GGUF-loaded model; THEN (5) the frame-matrix bakeoff. STOP if (1)/(2)/(4) fail.

### Step C -- mouth ceiling leg (independent; gate on the operator rubric, not a fake metric)
- Add per-leg `lora` + `steps` overrides to `build_leg_prompt` exactly like the `cfg`
  override already added (rewrite the node literal). Leg: `OTR_HUMO_LORA_NAME=none`,
  ~20-25 steps, to see if more compute fixes the mouth (expect higher VRAM/blue -- a
  CEILING probe, not a ship config).
- Mouth acceptance = fixed plosive/vowel clip(s) + the side-by-side montage (reuse the
  ffmpeg hstack already built) for operator eyeball. Do NOT invent an automated teeth
  metric (libs not installed).

### Step D -- model-swap dep probe ONLY (no adapter yet)
- Write a Windows/Blackwell sm_120/torch-2.10/offline dependency-probe script for the
  candidate lip-sync models; the adapter is a separate project gated on it passing.

## Frame matrix (GPT MUST-FIX -- concrete, replaces "representative beat")
Run every viability measurement at **frames = [49, 97, 177]** (49 = current; 97 ~3.9 s mid
beat; 177 = the engine max `_HUMO_MAX_FRAMES`, the worst case for production safety). Wire
it by generating `(leg x frames)` combinations in `run_humo_bakeoff.py`: include the frame
count in the leg `label`, the manifest, the frames/output paths, and the gates.
`build_leg_prompt` already accepts `frames`, so this is runner-side only.

## Per-idea promotion edit (GPT MUST-FIX -- name it now, wire LATER, operator-gated)
- Promote 14B (whatever variant wins): in `config/profiles/16gb_full.json` flip the
  other-beats/announcer video role engine `humo_1.7B_169 -> humo_14B_169` (+ for GGUF:
  set `OTR_HUMO_UNET_NAME` / a GGUF loader flag), AND re-express the winning graph through
  the in-process `wrapper_bridge.run_graph` path + `workflows/otr_scifi_16gb_full.json` in
  the SAME change (the harness is HTTP /prompt -- diagnostic only). Re-validate
  (OTR_WorkflowValidator + JSON round-trip), suite + Bug Bible + B7, commit v2.0-alpha.

## Carried from R1 (still binding)
Single resident <=14.5 GB (target <=13.5 w/ headroom); in-process always-silent path;
cold-import clean; 100% local; do-not-promote if audio present / peak >13.5 at any matrix
point / face crop regresses / mouth worse than current 14B; 1.7B is a measurement CONTROL,
not an acceptable final fallback (operator rejected it).

## Judgment (R2)
ACCEPTED (grounded): concrete frame matrix [49,97,177] + (leg x frames) runner combos;
"representative" pinned to 97f; per-idea profile/workflow edit named; GGUF needs a distinct
loader node (kwargs differ -> naive swap crashes); allocator probe must read in-process
torch max_memory_allocated, not just NVML reserved; mouth gate = operator rubric not a fake
metric. TRUNCATED (not available this pass; both panel reviews hit the 2000-token cap):
the tails of GPT MUST-FIX 3+ and most of Gemini's detail -- re-run with --max-tokens 4000
if a fuller R2 is wanted. NEXT: R3 wiring via LOCAL /kibitz (agents read the repo +
workflow JSON directly), per operator.
