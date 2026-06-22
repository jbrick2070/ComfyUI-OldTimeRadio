<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Open verify items + unconfirmed env propagation + _master_loudness default logic not shown in grounding make the plan unbuildable as-is.

MUST-FIX BEFORE BUILD:
1. [section 6] "Confirm `tests/_run_baseline.py` lets `OTR_SEGMENT_LOUDNORM` flow through at capture" is listed as open; replace with explicit code change (or prove the env already reaches `run_episode_and_save_wav` via the real assembler path) before Chunk 1 lands.
2. [section 5 + grounding_callsites.py:1122] The `_master_loudness` tweak that defaults `OTR_MASTER_MAKEUP_DB` to 0.0 only when `OTR_SEGMENT_LOUDNORM=rms` (and unset) is described but not shown; add the exact `os.environ` read + conditional before any rms path can be enabled.
3. [section 3 step 2] Re-baseline requires `OTR_SEGMENT_TARGET_RMS_DBFS=<measured>` but no code site that actually consumes this variable (or falls back) is referenced; supply the consumption site or the rms flip will be a no-op.

SHOULD-FIX:
1. [section 4] Calibration procedure relies on an uncommitted "throwaway numpy script"; either delete the step or commit the script under `tests/` so the measured value is reproducible.
2. [section 6] "Confirm module-level import os / import math" is a post-R2 assumption; if the actual scene_sequencer.py does not already contain them at module scope, the three call-site edits will fail at import time.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line guard in the byte-identical test that also skips when `OTR_SEGMENT_LOUDNORM != "peak"` so an accidental rms run cannot produce a false failure.

CUT THESE (over-engineering):
1. [section 3 step 5] "Validate perceived loudness across 2-3 episodes (manual release acceptance)" can be cut from the re-baseline procedure; it is release-process noise, not a build or wiring requirement.