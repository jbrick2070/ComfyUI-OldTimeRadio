# Round 3 Codex anchor: wiring and lifetime contract

Written before reading the local Round 3 reviewer.

## Required implementation shape

- The canonical workflow is the authority. Any new canvas, placement, cache, or
  qualification input must be wired into `workflows/otr_canonical.json` in the
  same change as its code and covered by the workflow/link/widget audit.
- Split the execution into explicit resource-lifetime phases rather than merely
  dropping Python values: conditioning precompute, latent encode, sample, and
  decode. A heavy patcher must be explicitly unloaded or isolated before its
  lifetime is considered ended.
- Preserve conditioning tensors in CPU memory or a validated artifact while the
  encoder is absent from the sampling graph. Split encode/decode VAE resources
  if the implementation otherwise pins the VAE through sampling, following Bug
  Bible rule 07.22.
- A cache-only native `CONDITIONING` source must not depend on CLIP/model loader
  inputs. Persistent keys must include exact prompt bytes, positive/negative
  role, encoder artifact digest/revision, tokenizer/config/template, truncation
  and max length, dtype/quantization, producer/schema version, and relevant
  extensions. Writes must be atomic and locked; reads validate metadata, shape,
  dtype, and digest and fail closed on corruption.
- Placement controls must bind a real loader behavior and be stamped in request
  receipts. Do not infer CPU/GPU placement from `--lowvram` under AIMDO.

## Admission model

The physical quantity is `max_t(memory_live_at_t)`, including prefetch and
transition scratch. Neither a clean stage max nor a fixed pairwise max is a safe
contract without a proven lifetime graph. Prefer a recipe-versioned empirical
upper envelope/table indexed by legal canvas and frame count, plus an explicit
machine reserve and measured baseline policy. Reject outside the qualified
domain with a named error; never extrapolate silently.
