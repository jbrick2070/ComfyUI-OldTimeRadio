<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Signature change + precompute + timing wiring contradict current __init__/render + caller in grounding_crt_code.py; ledger fields and music_open resolution are unverified assumptions.

MUST-FIX BEFORE BUILD:
1. [Signature] Plan requires new `__init__(w, h, title, volume, freqs, waves, fps, timing=None)` and `render(self, fi)` only, but grounding_crt_code.py:__init__ takes exactly (w,h,title) and render takes (fi,total,fps,vol,freq,wave). Fix: keep current signatures; move EMA precompute + arrays into a private _init_envelope call inside render_video after _analyze_audio, or accept arrays only via the existing closure.
2. [Timing extractor] Plan states "music_open = the first line whose speaker_role is a music-open role" and "resolve it the way otr_caption_burn does" from led["lines"], but grounding excerpts show only that led is passed to render_video and to _parse_hud_data; no speaker_role/start_s/dur_s schema or music_open logic exists in the provided code. Fix: add explicit fallback path that derives intro window solely from volume[0:first_dialogue] when start_s is absent/None, and document the exact ledger key names.
3. [S1 foundation] Plan says "precompute signal/loss/trig in __init__ from the full arrays" and "resolve the timing dict", but __init__ has no access to volume/freqs/waves or led today. Fix: perform the two EMAs (alpha 0.05/0.3) and first_dialogue_f calculation once in render_video after _analyze_audio, then pass the three precomputed arrays into the existing renderer or closure.
4. [No np.roll + center-band clip] Plan mandates "never draw scope primitives onto the base image" and "draw each scope onto a transparent layer sized to its gutter rect". Current render draws everything directly to the main ImageDraw. Fix: implement the two gutter layers + alpha_composite before section 8 vignette, or the scopes will overwrite the protected center band on portrait frames.

SHOULD-FIX:
1. [S2 gutter scopes] Plan retires sections 5+6 calls and thins section 3, but leaves the existing _waveform_mirror/_freq_bars_wide methods and the center-ring code in place. Fix: delete or guard the three retired blocks so they are never reached.
2. [Determinism] Plan specifies blake2s seed + np.random.default_rng for section-8 randint replacement, but current noise uses np.random.randint directly after the vignette. Fix: replace that single randint site only; do not touch any other RNG.
3. [Intervals] Plan says "half-open start <= fi < end; clamp to [0,total)". Current render has no such windows. Fix: add the clamp + disable check inside the new title-card state machine so an out-of-range music_close does not index past total_frames.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line comment in render noting that _brightness_ema is now read-only after S1.
- Expose the computed first_dialogue_f as a debug log at render_video start.

CUT THESE (over-engineering):
1. S5 "conditional music_close bookend" inside total_frames: the plan already says "render only for fi < total_frames; leave _hud_frames append unchanged", so the extra conditional adds no new behavior.
2. "2-beat gap-fill smoke" test item: it is implied by the existing timeline-aligned floor slice and does not require a separate build-time check.