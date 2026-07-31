# Pass 01 judgment: high-level arc

## Panel status

The live OpenRouter call was attempted after the Codex anchor was frozen. The
sandboxed call could not connect; the required escalation was then rejected
because it would transmit repository-derived internal text to an external
service without payload-specific user approval. No repository text was sent,
no external review was received, and actual spend was **$0.00**. The failed
requests and zero-cost manifest remain in `pass01/` as the receipt.

The safe fallback was an in-workspace read-only panel: one repository/runtime
audit, one ComfyUI/GGUF source audit, and one Wan/models/licensing audit. Codex
grounded every material claim before accepting it.

## Claims accepted

- The current `7000 + 185*frames` seed makes a true 8 GiB free report
  inadmissible at every frame count. The guard, not a render, is the immediate
  refusal mechanism.
- The Wan engine has no authoritative static canvas declaration. A plain
  canonical process can therefore request 1472x832 instead of the profile's
  832x480.
- Stock ComfyUI-GGUF does not receive AIMDO/VBAR `ModelPatcherDynamic` behavior.
- The official ComfyUI 5B workflow is useful evidence that offload can make file
  size sum differ from VRAM peak, but it is not a published physical-8-GB
  benchmark or an OTR qualification.
- A format/patcher comparison is required before blaming or replacing GGUF.

## Claims corrected or rejected

- The July 23 production artifact was a 177-frame request on the 16 GB dev card,
  not a 17-frame run on physical 8 GB. It proved a dead launch-time ceiling
  channel. It does not prove 8 GB execution failure.
- The 33-frame arithmetic in the problem statement is wrong: the stated formula
  gives about **10,577 MiB**, not 10,647 MB. The code's “MB” values are MiB.
- An affine envelope is not inherently a co-residency model; it can conservatively
  bound a staged pipeline. One contaminated point cannot identify an intercept
  and slope. A max of affine stage curves is itself piecewise affine.
- `max(stages)+reserve` is not yet a safe replacement. The present graph shares
  a VAE across latent encode and decode, retained model patchers are not unloaded
  by Python reference release, and async transitions may overlap. “Pairwise max”
  is also only an assumption; the continuous lifetime envelope is the quantity
  to measure.
- GGUF does have legacy low-VRAM behavior: partial load/offload, possible pinning,
  and per-layer move/dequantization. The defensible claim is that it lacks
  AIMDO's allocator-pressure adaptation and async demand-loading path.
- The official 5B UNet is FP16, not FP8. The listed text encoder is scaled FP8.
- The profile canvas is consumed by the director/ledger, so “read by nothing” is
  too broad; it is simply not authoritative at per-clip render construction.
- 768x432 is not legal under OTR's current divisible-by-32 canvas declaration
  contract. A behavior-preserving declaration would be 832x480 unless a separate
  Wan grid/product decision is proved and wired into the canonical workflow.
- The proposed CPU/GPU text axis does not exist in the current Wan loader. The
  four cells are therefore not executable as stated.
- The suggested wrapper embedding cache is not native-graph compatible and its
  prompt-only key is unsafe across model/tokenizer/precision changes.

## Decision after Pass 01

Keep the Wan 5B profile **draft and unqualified**, not rejected. Correct request
authority and measurement first; compare execution paths; then derive a measured
admission envelope. Treat 14B+Lightning, FastWan 5B, and any alternative as
separate candidates rather than conclusions from anecdotes.
