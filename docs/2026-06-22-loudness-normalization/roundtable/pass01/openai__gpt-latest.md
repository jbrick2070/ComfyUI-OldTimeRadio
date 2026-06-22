<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document still has unresolved design choices, a signature-level LUFS blocker, and a false “true peak” assumption; it is not build-ready.

MUST-FIX BEFORE BUILD:
1. [§4.1 / §2] LUFS cannot be implemented at the shown seam as written because `_normalize_clip(clip_np, target_peak=0.85)` has no sample-rate argument, while BS.1770/K-weighted LUFS requires sample rate. Concrete fix: either choose RMS for the existing `_normalize_clip` signature, or change the seam and every caller to pass `sample_rate` into the loudness function. Verify all `_normalize_clip` call sites before editing.

2. [§4 open questions / §2] The plan leaves the core algorithm undecided: LUFS vs RMS, target level, clamp values, noise gate threshold, short-clip fallback, and scope are all still open. That is not a build spec. Concrete fix: freeze explicit defaults before build, e.g. algorithm, target dB, max boost/cut dB, silence/noise threshold, minimum duration behavior, and which segment types are normalized.

3. [§2 / §5] The plan says `_master_loudness` remains the final “true peak” safety ceiling, but the grounded code does not compute true peak; it uses sample peak: `waveform.abs().max()`. Concrete fix: either change the wording/invariant to “sample peak ceiling” or implement actual true-peak/oversampled limiting and add tests. Do not rely on the current code for true-peak compliance.

4. [§2 / §4.2] “Reduce/retune the +4 dB master makeup” is underspecified and can double-gain the result. `_master_loudness` first peak-normalizes the entire episode to the ceiling, then applies makeup limiting, so any per-clip LUFS/RMS target will not directly predict final program loudness. Concrete fix: specify the new default `OTR_MASTER_MAKEUP_DB` behavior for this feature, preferably `0.0` or an explicitly calibrated value, and update tests/golden-baseline procedure accordingly.

5. [§2 / §4.3] Applying loudness gain alone can create very hot individual clips before final mastering. The final episode limiter may prevent final sample clipping, but it can also reshape/limit peaks and change relative clip dynamics if one segment is over-boosted. Concrete fix: after the loudness gain calculation, cap gain by both max-gain clamp and a per-clip peak headroom limit, e.g. `gain <= peak_ceiling / current_peak`, with a documented ceiling such as 0.85 or lower.

6. [§4.3 / §4.4] The silence/noise behavior is not specified enough to prevent hiss pumping. A peak guard like the current `peak < 1e-6` is insufficient for RMS/LUFS because a clip with leading/trailing silence or low room tone can measure quiet and get boosted. Concrete fix: define an active-speech measurement rule: trim/gate below a dBFS threshold, require a minimum active duration, and return unity gain for non-finite/too-quiet/too-short measurements.

7. [§4.4] Short clips are explicitly unresolved. Integrated LUFS can be unstable or invalid for one-word/very-short lines, depending on implementation. Concrete fix: define deterministic fallback behavior, e.g. if duration or active samples are below threshold, use RMS over active samples, use current peak normalization, or leave unchanged.

8. [§4.5 / §2] Scope is contradictory: the title says “per-segment,” §2 says “per dialogue clip,” and §4.5 asks whether announcer/SFX/music should be excluded. Changing `_normalize_clip` affects whatever callers feed into it, not necessarily only Bark dialogue. Concrete fix: enumerate the actual call sites and segment types; then explicitly normalize only intended speech/dialogue paths or add a `kind`/policy argument. [ASSUMPTION] Call-site behavior must be verified in the full file.

9. [§3 / §5] The golden re-baseline requirement is acknowledged but lacks release sequencing. Concrete fix: add an explicit build step: land algorithm behind an operator-gated flag or versioned normalization mode, generate the new fixture in the required headless/GPU environment, then update `test_audio_byte_identical` fixture in the same change or a clearly ordered two-step release.

SHOULD-FIX:
1. [§2] Define exact numeric stability behavior. The implementation should handle empty arrays, NaN/Inf samples, zero-length clips, integer input accidentally passed in, and non-float32 output. Concrete fix: sanitize measurement input, return original clip for invalid/empty/non-finite measurements, and always return `np.float32`.

2. [§2 / §5] Add deterministic unit tests beyond the golden byte test. Concrete fix: tests for silence unchanged, near-silence unchanged, quiet speech boosted only to clamp, loud speech attenuated, peak ceiling honored after gain, short clip fallback, repeated runs byte-identical, and no CUDA use. [ASSUMPTION] Existing tests may not cover these cases.

3. [§4.1 / §3] If adding `pyloudnorm` or another dependency, the plan needs a pinned version, license verdict, package-size/import-time verdict, and offline install/update path. Concrete fix: decide dependency explicitly or choose RMS zero-dependency.

4. [§2] Clarify mono/stereo handling. The grounded `_normalize_clip` docstring says “1-D float32 clip.” LUFS libraries usually accept mono or multichannel with defined channel weighting. Concrete fix: state whether normalization happens before mono conversion, after mono conversion, or per-channel; reject/reshape unexpected dimensions deterministically.

5. [§2 / §4.2] Define whether the target is intended pre-master or post-master. Concrete fix: document “per-clip target is pre-master only; final episode loudness may differ” or implement a final integrated loudness target instead of only peak/makeup limiting.

6. [§1] The current `_normalize_clip` docstring claim “matching overall loudness across clips” is inaccurate if this change is made and arguably inaccurate today because it is peak-only. Concrete fix: update the docstring to describe the actual algorithm and its limits.

7. [§3] “Changing `_normalize_clip` changes output bytes” is true for calls reaching it, but a change to signature or scope may alter more than dialogue. Concrete fix: include a small migration checklist: update callers, regenerate golden fixture, inspect representative output from each engine listed in §3.

OPTIONAL / NICE-TO-HAVE:
- Add debug logging or optional metadata for measured loudness/RMS, applied gain dB, clamp reason, and peak before/after for each normalized clip. Keep disabled or low-volume by default.
- Add an environment/config override for target level and clamps, but only after fixed safe defaults exist.
- Document recommended operator listening validation: before/after sequence with quiet line, loud line, short interjection, room tone.

CUT THESE (over-engineering):
1. [§4.1] Cut hand-rolled BS.1770/K-filter for the first build unless LUFS is mandated. It is easy to get subtly wrong and requires sample-rate/signaling changes. A gated active-RMS implementation can fit the current numpy seam and solve the immediate shot-to-shot speech imbalance.

2. [§4.5] Cut “unified per-segment pass” for music/SFX/themes in the first build. The stated problem is dialogue shot-to-shot perceived loudness. Applying the same target to music beds, SFX, and themes risks destroying mix intent. Normalize dialogue only first.

3. [§2] Cut configurable master-retune complexity until the per-clip algorithm is fixed. Set a single deterministic default, likely disabling or lowering makeup, then reintroduce operator tuning later if listening tests prove it is needed.