<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan has plausible levers, but it is not build-ready because it does not isolate provider motion from downstream canonicalization, has no concrete pass/fail standard, and leaves the Seedance-only prompt conditioner’s insertion/rollout behavior underspecified.

MUST-FIX BEFORE BUILD:

1. [S3.5, S4.2, S4.5] Defect: canonicalization is named as a possible root cause, but the proposed A/B mostly tests prompt variants and says only to “record results” for eyeballing. That cannot distinguish “Seedance generated choppy motion” from “OTR conforming introduced cadence/stutter.” Grounding shows Seedance duration is integer-rounded and clamped to 4..15 seconds, then canonicalized downstream. Concrete fix: the A/B harness must save and compare both raw provider output and canonicalized output for each variant, with requested duration, provider/decoded duration, provider/decoded fps, raw frame count, canonicalized fps, canonicalized frame count, and target frame count. Do not merge prompt changes unless B improves or preserves smoothness after canonicalization, not just in the raw provider clip.

2. [S4.5] Defect: acceptance is