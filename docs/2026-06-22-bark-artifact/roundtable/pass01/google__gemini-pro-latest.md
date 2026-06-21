<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan fundamentally misunderstands Bark's EOS math and proposes a head-trim that will destroy an existing hallucination guard.

MUST-FIX BEFORE BUILD:
1. [ROUNDTABLE FOCUS] Raising `min_eos_p` causes MORE hallucinations, not fewer. `min_eos_p` is the probability threshold the EOS token must reach to *stop* generation. Raising it makes it harder for Bark to stop, leading to runaway hallucinated tails. The `_otr_bark_lib.py` docstring explicitly confirms `0.1` provides the "tightest + lowest variance". Fix: Do not raise `min_eos_p`; test lowering it or leaving it at 0.1.
2. [Candidate fixes #1] Head-trimming will destroy the `is_first_line` anchor. `_generate_single_line` explicitly prepends `[clears throat]` to the first line to prevent podcast-intro hallucinations. A "head silence/transient trim" will clip this intentional non-speech audio right off. Fix: Only apply the new trim to the tail, or explicitly disable head-trimming when `is_first_line=True`.
3. [Candidate fixes #3 & Constraints] Missing seed plumbing for deterministic rerolls. The plan requires a "seed-keyed (a reroll uses a derived seed, reproducible)" generation, but `_generate_single_line` does not accept a seed parameter, nor does it set `torch.manual_seed()`. Fix: Add a `seed=None` kwarg to `_generate_single_line` and set the PyTorch manual seed before calling `model.generate()`.

SHOULD-FIX:
1. [Candidate fixes #3] False positives on the high-band gate. A "sustained high-frequency band spike" scanner will likely trigger on valid Bark tokens like `[gasps]`, `[sighs]`, or heavy sibilance in female voices. Fix: Ensure the gate ignores the first/last 200ms (where valid breaths live) or restrict the FFT check to a very high shelf (e.g., >8kHz) with a strict duration threshold.

OPTIONAL / NICE-TO-HAVE:
- [Candidate fixes #2] The semantic temp cap for `is_first_line` is currently 0.5 (intl) / 0.6 (en). You can safely drop the short-line semantic temp to 0.4 without flattening the coarse/fine acoustic delivery, since `_stage_temps_for_line` isolates the stages.

CUT THESE (over-engineering):
1. [Candidate fixes #3] Deterministic high-band artifact GATE + reroll. Building a custom numpy FFT frequency scanner and retry loop for a single TTS model's edge-case artifact is heavy. If Candidate #1 (tail trim) and #2 (semantic temp drop) are implemented, the artifact rate will plummet. Cut the gate and let Kokoro (#4) handle the zero-tolerance broadcast paths.