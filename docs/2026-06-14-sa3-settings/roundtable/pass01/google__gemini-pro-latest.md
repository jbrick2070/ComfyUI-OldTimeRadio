<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The SDE sampler violates the strict determinism constraint, and the 30s context makes short 4s interstitials sound like aimless mid-song bridges.

MUST-FIX BEFORE BUILD:
1. [Section 4] SDE sampler breaks the strict determinism constraint. `dpmpp_3m_sde_gpu` injects per-step noise during sampling, which causes audio variations even with a fixed seed. Fix: Change `OTR_SA3_SAMPLER` to `dpmpp_2m` (an ODE solver) and `OTR_SA3_SCHEDULER` to `karras` for perfect seed determinism.
2. [Section 3 & 3b] Fixed 30s context makes 4s interstitials sparse and incoherent. Placing a 4s slice at `start=13.0s` of a 30s track yields a low-energy bridge, not a punchy cue. Fix: Set `OTR_SA3_CONTEXT_S = 12.0` (your max cue length). This forces the model to compose a tight 12s musical thought, making the 4s middle slice punchy (`start=4.0s`) and the 8s outro resolve quickly (`start=4.0s`), without requiring any changes to your Python logic.

SHOULD-FIX:
1. [Section 1] CFG 6.0 is slightly weak for SA3 small instrumental prompt adherence. Fix: Increase `OTR_SA3_CFG` to `7.0` (SA3's native default) to ensure the model strongly adheres to the "sci-fi / eerie" prompt without vocal bleed.
2. [Section 5] Negative prompt includes "harsh clipping, digital distortion" which inadvertently suppresses the requested "analog tape warmth" and "eerie" textures. Fix: Use final string: `vocals, singing, speech, spoken words, lyrics, voiceover, crowd noise, modern pristine mix, out of tune, low quality`.
3. [Section 6] The 1940/1950 genre anchor in `_SA3_PERIOD_GENRE` lacks the word "instrumental", risking vocal bleed despite the negative prompt. Fix: Insert "instrumental" into the anchors (e.g., `"vintage instrumental orchestral sci-fi score..."`).

OPTIONAL / NICE-TO-HAVE:
- [Section 6] [ASSUMPTION] SA3 responds well to BPM hints for pacing. If your 4s interstitials feel too slow, consider appending a tempo hint like `120 BPM` to the base prompt to force higher-energy generations.

CUT THESE (over-engineering):
1. [Section 2] `OTR_SA3_STEPS = 100`. SA3 converges mathematically well before 100 steps, especially when switching to the `dpmpp_2m` ODE solver. Safe to cut to `50` to halve your render time with zero perceived quality loss.