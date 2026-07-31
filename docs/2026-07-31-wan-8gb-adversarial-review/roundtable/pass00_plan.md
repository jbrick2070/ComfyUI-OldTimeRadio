# Wan 2.2 TI2V-5B 8 GB tier: adversarial review input

Date: 2026-07-31

The verbatim problem statement is tracked at
[`docs/2026-07-31-PROBLEM-STATEMENT-8gb-video.md`](../../2026-07-31-PROBLEM-STATEMENT-8gb-video.md).
This roundtable is asked to find where its three claims and proposed four-cell
sweep are wrong.

## Decision to make

Determine whether OTR can honestly ship an 8 GB Wan video tier, what the current
evidence actually proves, and what experiment would qualify it without silent
degradation or hardware/model/provider substitution.

## Claims under attack

1. The current affine `overhead + frames * cost` admission model has the wrong
   shape because low-VRAM execution is staged.
2. GGUF bypasses ComfyUI's Dynamic VRAM path and therefore bypasses the mechanism
   behind ComfyUI's official 8 GB support statement.
3. The Wan engine fails to declare a render canvas and therefore a plain
   canonical run requests the 1472x832 default instead of the profile's 832x480.
4. A CPU/GPU text-encoder by tiled/non-tiled-VAE four-cell sweep under an 8 GiB
   reserve clamp would settle the tier.

## Hard constraints

- No silent resize, clamp, substitution, null output, or fallback.
- Local and offline-first; distributable open-source licensing.
- The dev machine has a 16 GB RTX 5080; a physical 8 GB card is unavailable.
- Episode length is not a pass/fail criterion.
- Canonical headless evidence must use `workflows/otr_canonical.json`.

Review the exact repository and upstream facts in `grounding.md`. Separate what
is proved, what is a plausible mechanism, and what still requires measurement.
