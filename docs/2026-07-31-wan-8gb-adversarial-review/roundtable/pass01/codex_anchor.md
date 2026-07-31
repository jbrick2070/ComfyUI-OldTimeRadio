# Round 1 Codex anchor: high-level arc

Written before reading any panel output.

## Initial verdict

The problem statement has found two real defects but overstates their evidence
and proposes a sweep that cannot isolate the disputed mechanism. Claim 3 is
substantially established. Claim 1 correctly rejects the current affine
constant but prematurely replaces it with a clean stage maximum that the actual
graph does not implement. Claim 2 correctly identifies a different patcher path
but incorrectly equates legacy GGUF with no streaming/offload and misdescribes
the official 5B artifacts.

Most importantly, the cited production leg was a 177-frame request on the 16 GB
dev card. It proves the frame-ceiling configuration was not authoritative. It
does not prove a 17-frame request fails on physical 8 GB. The current formula
does prove a false refusal for every reported free value at or below 8192 MB,
but that is static arithmetic, not a successful execution benchmark.

## High-level correction

Treat this as three gates rather than one four-cell sweep:

1. Correct request authority and observability first: one declared legal canvas,
   real phase/lifetime markers, controllable placement, baseline/delta memory,
   host paging, wall time, artifact verification, and cold reset discipline.
2. Compare execution mechanisms at identical requests: official native
   FP16-UNet/FP8-text-encoder Dynamic VRAM versus the shipped Q5 GGUF legacy
   path, including the relevant dynamic/async-offload state. This comparison is
   format-plus-runtime, not a pure quantization experiment.
3. Only after a viable path exists, measure frame/pixel/decode/transition
   envelopes with repeated cold runs and a multi-clip soak, then derive an
   empirical per-recipe admission rule with an explicit reserve.

An 8 GiB reserve clamp on 16 GiB is prequalification, not certification. The
tier must remain draft/unqualified until a physical 8 GB run is obtained or the
product language explicitly says it was qualified only under a dev-card clamp.

## Leading risks the panel should try to overturn

- `max(stages)+reserve` is unsafe unless the implementation actually proves
  phase boundaries. The shared VAE, retained patchers, async prefetch, and
  allocator slack may create overlap. A fixed pairwise-max formula is also an
  assumption; instrument the transition envelope.
- The official “fits well” sentence lacks a published operating point. It is
  prior evidence, not the honest floor requested here.
- The proposed 768x432 canvas is currently illegal under OTR's divisible-by-32
  declaration validator. Canvas selection is a product-quality decision and a
  model-memory variable, not a free typo repair.
- CPU/GPU text placement is not an exposed independent variable in the current
  GGUF engine. Global launch flags would confound the experiment.
- Prompt-only persistent embedding cache keys are unsafe, and the suggested
  wrapper node returns an incompatible type for the native graph. Native
  episode-scoped conditioning reuse is the first candidate; persistent cache
  requires provenance-complete keys and measured hit rates.
- Released Wan Lightning support is A14B, while TI2V-5B is still a TODO. A 14B
  candidate deserves a separate quality/time/RAM experiment, not promotion on
  anecdotes. Motif's own published peak exceeds 8 GB; MobileWan fails current
  integration/licensing gates.

The desired outcome of Round 1 is a defensible decision arc and the smallest
set of experiments that could falsify it, not a fabricated frame ceiling.
