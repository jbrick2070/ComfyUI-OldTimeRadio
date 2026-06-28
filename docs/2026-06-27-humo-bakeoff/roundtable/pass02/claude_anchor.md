# Claude anchor -- R2 (coding plan / implementability), written before the panel

VERDICT: the r1 idea set is implementable, but only idea 1 is "do-now"; ideas 2-4
need a stated FIRST coding slice + a kill-gate or they become open-ended. The
measurement harness needs two concrete extensions before any lever is trustworthy.

## Implementability per idea (grounded in the harness + engine)
1. **Allocator-cache probe -- DO-NOW, ~1 hr.** `run_humo_bakeoff.boot_server` builds an
   `env` dict already (it pops CUDA_VISIBLE_DEVICES etc.); inject
   `PYTORCH_CUDA_ALLOC_CONF` there from a new `OTR_BAKEOFF_ALLOC_CONF` knob and run leg
   ii A/B. No engine/workflow change. CONFIRMED feasible (boot_server env path exists).
   This is the cheapest test and gates idea 2.
2. **GGUF 14B -- needs a harness-side graph variant, NOT an eng_humo edit.** `_build_graph`
   hardcodes `UNETLoader` (unet_name + weight_dtype). A GGUF path needs a DIFFERENT node
   (`UnetLoaderGGUF`, gguf_name) -- so the bakeoff BUILDER must emit that loader variant
   itself (the diagnostic must not edit eng_humo). FIRST SLICE: (a) confirm a HuMo-14B
   GGUF file + that `UnetLoaderGGUF` is registered on `/object_info`; (b) builder emits a
   `gguf` leg swapping node 1's class+widget; (c) 1-frame smoke proves audio cross-attn
   survives; (d) frame-matrix bakeoff. KILL-GATE: if (a) or (c) fails, stop -- do not
   chase a GGUF port. CONFIRMED `_node_candidates` lacks a GGUF loader.
3. **Mouth probes -- partly wired already.** The engine reads `OTR_HUMO_LORA_NAME=none` +
   `OTR_HUMO_STEPS`; add per-leg `lora`/`steps` overrides to the builder exactly like the
   `cfg` override I just added (build_humo_bakeoff_workflow.py rewrites the node literal).
   So a "no-LoRA / 25-step" ceiling leg is ~30 min. Higher-res input still = a separate
   image-pipeline change, out of the bakeoff scope; treat as its own ticket.
4. **Lip-sync model swap -- NOT codeable yet.** It is a new engine adapter; blocked on a
   Windows/Blackwell/torch-2.10/offline dep probe. R2 = write the dep-probe script ONLY,
   not the adapter.

## Harness extensions REQUIRED before trusting any lever (from r1 gates)
- **Frame matrix:** `build_leg_prompt` already takes `frames`; add a `frames` field per
  leg + iterate {49, ~a representative beat, max-safe 4n+1 <=177}. Small, do first.
- **Mouth acceptance:** a true teeth/lip metric needs libs not installed; implement the
  rubric as fixed plosive/vowel clips + a side-by-side montage for the operator eyeball
  (reuse the ffmpeg hstack I built), NOT a fake automated score.
- **Promotion path note (not code yet):** any winner re-expresses through
  `wrapper_bridge.run_graph` + workflow JSON + 16gb_full profile -- R2 should not wire
  production; just record the exact mutation each surviving idea will need.

## Sequencing (R2 build order)
allocator probe -> (if still >13.5) GGUF feasibility gate -> mouth ceiling leg in
parallel (independent track) -> dep-probe script for the model swap. Each step has a
kill-gate so we don't open-endedly chase a dead lever.

CUT for R2: building the lip-sync adapter; any production wiring; an automated mouth
metric (operator eyeball is the gate).
