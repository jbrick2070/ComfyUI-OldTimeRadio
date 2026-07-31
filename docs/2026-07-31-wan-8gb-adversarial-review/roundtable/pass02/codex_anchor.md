# Round 2 Codex anchor: executable qualification plan

Written before reading the local Round 2 reviewer.

## Gate 0: make measurements meaningful

- Choose and declare the product canvas. The behavior-preserving candidate is
  832x480; 768x432 requires a separately proved grid-contract change.
- Route the value through `workflows/otr_canonical.json` and add a profile-drift
  test so the engine, profile, director, and request receipt agree.
- Add continuous timestamps/markers around text encode, latent encode, sampling,
  decode, transitions, and unloads. Record machine-wide baseline/peak/delta,
  PyTorch allocated/reserved where meaningful, host RAM/pagefile, wall time,
  loader logs, and artifact/OBS receipts.
- Add an audited qualification-only budget bypass/ceiling override. The current
  admission formula otherwise prevents measuring the rungs needed to replace
  it. It must be unreachable in ordinary production.
- Reset and fresh-boot every cold cell; pin exact ComfyUI/plugin commits and
  model hashes; use the real canonical workflow and replayable inputs.

## Gate 1: identify a viable execution mechanism

At one legal minimum request and with tiled decode enabled, compare at least:

- shipped Q5 GGUF UNet + Q5 GGUF text encoder on stock legacy patchers;
- official FP16 native UNet + scaled-FP8 native text encoder with Dynamic VRAM;
- native path with Dynamic VRAM disabled as a mechanism control, if it can run;
- optional explicit GGUF block-placement path only as a separate dependency
  candidate, never as an implied property of stock GGUF.

Hold request/seed/recipe constant and repeat each cell cold. Record actual
patcher classes, AIMDO state, async stream count, loaded/offloaded bytes, peak
envelope, host spill, time, and output validity. This is a product-candidate
comparison, not a pure quantization A/B.

## Gate 2: remove encoder ambiguity

For viable mechanisms, compare a real explicit CPU-only encoder path with loader
default, then compare cold precompute versus a cache-only conditioning source.
The cache hit path must have no model input, otherwise Comfy resolves the loader
before the hit. Reuse must be measured from real prompt traces.

## Gate 3: fit the render envelope

For the winning candidate(s), measure tiled/non-tiled decode over a structural
Wan frame ladder (for example 17, 33, 65, then extend toward the product maximum)
and at least two legal canvases if pixel scaling is to enter the model. Preserve
the actual production steps/scheduler for qualification. Use repeated cold runs,
randomized/interleaved order where practical, and retain continuous transition
peaks rather than assigning samples to mutually exclusive stages.

## Gate 4: stability and transferability

Run a multi-clip canonical soak to catch retained-model leaks or system-memory
spill. This is not an episode-length quality gate. A dev-card reserve clamp can
prequalify loader behavior, but the release label remains “physical 8 GB
unverified” until tested on that hardware.
