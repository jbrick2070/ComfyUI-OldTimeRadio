<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Open questions block wiring (Q1), trim/QA params unspecified (B2/FINAL QA), and multiple unverified invariants vs. grounded code.

MUST-FIX BEFORE BUILD:
1. [B1] `_clean_text_for_bark` and `_generate_single_line` have no `speech_only` param or first-line gate (see _otr_bark_lib.py:329-353 asterisk map + 353-355 whitelist + 412 is_first_line prepend); adding it will change every dialogue path. Fix: add `speech_only: bool = False` to both signatures, default False for backward compat, and wire the strip list (`[music]`,`[whistles]`,`[sneezes]`,`[gasps]`) only when True.
2. [WIRING + Q1] `eng_bark.py:generate_voice` (and `prepare_text`) has no knowledge of DIALOGUE vs. intro; it only tracks `_presets_started` for `is_first_line`. Fix: either add explicit `speech_only` + `is_intro` kwargs to the per_line call or document+implement the exact seam (speaker_role/beat) before any lib change.
3. [B2] `_chunk_text_for_bark` (see _otr_bark_lib.py:391) and `_trim_trailing_silence` (454) have no per-chunk head+tail logic or "protect plosives + min retained speech" rules. Fix: specify the exact split fallback + bounded trim window (ms + energy condition) or the change cannot be coded.
4. [B0 + Q4] Plan requires confirming artifact correlation and that `test_audio_byte_identical` is not a bark path, but neither fixture nor call site is in the grounding excerpts. Fix: run the isolation step and verify the fixture path before touching `_clean_text_for_bark` output.
5. [FINAL QA] "spectral centroid > 4-8 kHz" metric is a range, not a predicate; no implementation or test hook defined. Fix: replace with a concrete, deterministic check (threshold + window) or delete the QA requirement.

SHOULD-FIX:
1. [B1 + Q2] Whitelist in _otr_bark_lib.py:353-355 already keeps `[laughs]`,`[sighs]`; plan must explicitly state the HIGH-RISK strip list vs. kept tokens so the `speech_only` branch matches.
2. [B2] Plan says "per-CHUNK head+tail transient trim BEFORE the inter-chunk silence" but current `_generate_single_line` only trims the final concat (454). Fix: add the ordering note to the chunk loop so internal squeals are caught.
3. [B3] `BarkModel.generate` call (see _otr_bark_lib.py: _gen_kwargs) uses only `do_sample=True` + prefixed temps; no seed/generator path exists. Fix: either implement the minimal `seed` kwarg or drop B3 entirely.
4. [WIRING] `eng_bark.py` calls `_generate_single_line` without `semantic_min_eos_p` or any new flag; any lib change must also update this call site.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line regression test that a `[music]` input survives only when `speech_only=False`.

CUT THESE (over-engineering):
1. B3 reroll plumbing: plan itself states "B1+B2 should plummet the rate" and "defer the actual artifact-detect+reroll loop", so the seed/restore hook adds no value for this sprint and can be removed without affecting the goal.
2. Full "deterministic NON-SPEECH/high-band metric" addition to audio QA: a simple post-render energy check on first/last 150 ms would suffice; the spectral-centroid machinery is heavier than needed for a regression gate.

[ASSUMPTION] Plan's premise that non-speech tokens are the dominant cause rests on "pass01_judgment" which is outside the provided document/grounding.