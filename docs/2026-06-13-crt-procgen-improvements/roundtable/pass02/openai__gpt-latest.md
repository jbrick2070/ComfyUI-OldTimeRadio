<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still violates the portrait/landscape gutter constraint, relies on unresolved ledger/composite timing, and proposes unsafe full-frame rolls/dimming that can break the final overlay.

MUST-FIX BEFORE BUILD:
1. [OPEN DECISIONS 1] The default “fixed gutter rings clamped to the outer ~8-10%” violates the grounded hard constraint “GUTTERS ONLY ON PORTRAIT BEATS.” Landscape beats have no gutters, so fixed bright scopes at the edges will overlay landscape content. Concrete fix: plumb portrait/landscape beat intervals or safe-gutter intervals from `led` / clip manifest into `_CRTRenderer`, and gate or disable the gutter scopes on landscape/full-frame beats. If that plumbing is not available, the scopes must be suppressed during unknown beat types rather than drawn unconditionally.

2. [Cross-cutting envelope behaviors / Signal-loss sync-drift] “Clamp the roll so chrome never rolls into the center column” is not sufficient if implemented with full-frame `np.roll`: `np.roll` wraps pixels from one side to the other and will move all existing procgen content, including center/grid/title elements. Concrete fix: never roll the full rendered frame. Draw reactive chrome/scopes onto separate left/right gutter layers, roll only masked gutter crops, paste/alpha-composite them back, and leave the portrait-safe center band unchanged. Do not use wraparound into the center.

3. [S1 -- envelope + plumbing + determinism] “SEED the RNG from (`fi`, title hash)” is not deterministic if it uses Python’s built-in `hash()`, which is process-randomized, and it does not include the actual episode/render seed. It also must not use global `np.random` state. Concrete fix: create a local RNG per frame/effect using a stable hash such as `hashlib.blake2s`/`sha256` or `zlib.crc32` over an explicit render seed [ASSUMPTION: available upstream], episode title, `fi`, and an effect salt. Replace the current unseeded `np.random.randint` in render section 8 with that local RNG.

4. [#1 / S1 / VERIFY-AT-BUILD] The title-card timing depends on unresolved `led` fields: b000 start/end, first-dialogue frame, and music_close window. Current grounded code instantiates `_CRTRenderer(W, H, episode_title)` and calls `render(fi, total, fps, vol, freq, wave)`; no ledger timing reaches the renderer today. Concrete fix: define the exact timing payload before coding, e.g. `crt_timing = {"music_open_start_f": ..., "music_open_end_f": ..., "first_dialogue_f": ..., "music_close_start_f": ..., "music_close_end_f": ...}` in 24fps frame numbers, pass it into `_CRTRenderer`, clamp all values to `[0,total_frames)`, and make missing fields disable the corresponding effect rather than crash.

5. [VERIFY-AT-BUILD] The document acknowledges the title card could reappear if the procgen floor’s b000 frames are reused as inter-beat gap fill by `OTR_SilentComposite`. That is not a build-time detail; it is a release blocker for #1. Concrete fix: verify the composite never reuses frames from the b000 title-card window for later gaps. If it does, either change the slice source to exclude the b000 window or do not burn the title card into the reusable floor. [ASSUMPTION] This may require code outside `nodes/video_engine.py`, which conflicts with the “one file” constraint.

6. [Cross-cutting envelope behaviors / Audio-choked vignette] “Scale `self._vignette` intensity by `signal`” can dim the entire frame, including title/ident/text, repeating the exact failure noted in grounded code: adaptive brightness was disabled because it “dimmed the CRT text to unreadable levels.” It also contradicts “ident flickers LAST.” Concrete fix: keep the base vignette array immutable; compute a bounded loss-vignette multiplier with a readable floor, and apply hierarchy multipliers per layer before post-processing. Exempt or floor-clamp section 1/title text so it cannot fall below readable brightness.

7. [S2 / #2 / OPEN DECISIONS 1] “Center-column sanctity” is stated visually but not implemented as a checkable geometry rule. The grounded portrait-safe center is not “middle half” by definition; portrait beats are 480x832 scaled into 1920x1080 with ~647px side gutters. Concrete fix: compute the protected center band from the known portrait aspect/scale or from manifest safe-gutter bounds, then clip all bright scope geometry against that band. Do not rely on “outer ~8-10%” as the only rule.

SHOULD-FIX:
1. [S1 -- envelope + plumbing + determinism] The slow EMA starts at `0.5` in grounded code. That causes the first rendered frames to begin half-locked even if the opening is silent, and makes title-card lock behavior depend on an arbitrary initial value. Concrete fix: initialize slow/fast EMAs from the first frame’s `vol`, or explicitly reset them on `fi == 0` to a chosen documented value.

2. [#2 / S2] The proposed `_fft_history` and `_sweep_history` make rendering stateful. That is fine only if frames are rendered strictly sequentially. Concrete fix: reset histories on `fi == 0`, track `last_fi`, and either handle non-sequential calls by clearing/rebuilding history or document/assert sequential rendering. [ASSUMPTION] The shown `_frame_gen` appears sequential, but tests/retries may not be.

3. [#1 -- Title card] The big episode-title reveal at “2-3x `f_title`” with fake-bold overstrike can overflow for long titles. Concrete fix: measure with `ImageDraw.textbbox`, wrap or scale down to a maximum width, and clamp vertical placement before adding decode/cursor effects.

4. [#1 -- Title card] Fake-bold by drawing “3-5x at 1px offsets” can blur into unreadable blobs at small sizes or with scanlines. Concrete fix: use a minimal fixed offset set, e.g. `(0,0),(1,0),(0,1),(1,1)`, and test at the actual 1920x1080 render size.

5. [Cross-cutting envelope behaviors / Hierarchy clamp] “One multiplier used in sections 1/2/3/4/8” is underspecified and conflicts with the desired order “grid drops FIRST… ident LAST.” Concrete fix: define separate layer floors, e.g. grid floor lower than scopes, scopes lower than ident/title, and apply them before scanlines/vignette/noise.

6. [#2 -- Two gutter SCOPES] The plan says line weight 1-2px, but grounded section 2 currently uses `width=max(2, self.w // 400)`, which is 4px at 1920. Concrete fix: explicitly cap scope/spoke widths to 1-2px after moving them to gutter scopes.

7. [#2 -- Two gutter SCOPES] The idle behavior for FFT “collapses to a slow rotating radar sweep” needs a deterministic phase source. Concrete fix: derive sweep phase from `fi`/`fps` only, not from RNG or EMA history.

8. [#1 / S3] “Suppress/restore normal section-1” needs ordering specified because grounded section 1 is drawn first today. Concrete fix: decide title-card state before drawing section 1; if active, skip the existing ident/subtitle/timestamp block and draw the card instead.

9. [S5 -- outro bookend + regression] “music_close handoff to the HUD post-roll” needs to preserve existing frame count/audio mux behavior. Concrete fix: do not add or remove video frames for the outro; render it only inside existing `total_frames`, then let existing `_hud_frames` append unchanged.

OPTIONAL / NICE-TO-HAVE:
- Add a small debug-only overlay/log of computed `signal/loss/trig` and title-card window frame numbers during development, removed or disabled for release.
- Add a deterministic frame checksum test for a short render using fixed audio/title/timing.
- Add a visual safe-band test frame that asserts no bright scope pixels enter the protected center band.

CUT THESE (over-engineering):
1. [#2 -- LEFT FFT scope] Cut either the “peak-hold ghost ring” or the “noise-floor shadow ring” for the first build. The FFT scope already has spokes plus comet tails; those two extra layers add state and tuning without being necessary to move the ring into the gutter.

2. [#2 -- RIGHT circular oscilloscope] Cut the free-running/drifting “trigger seam” until the basic circular waveform and sweep dot are stable. It is another timing/state effect and is not required to replace the old bottom waveform.

3. [OPEN DECISIONS 2] Cut telemetry micro-text labels by default. The plan already has title text, bottom metadata, scopes, graticules, scanlines, and HUD post-roll; extra labels risk reading as a modern HUD and are not needed for the core upgrade.

4. [Cross-cutting envelope behaviors / Halation bloom] Cut halation from S4 if performance or readability is uncertain. It is purely decorative; the CRT look already has scanlines, vignette, dim duplicates can be added later, and green-only screen blend will compress it into brightness anyway.

5. [S5 -- outro bookend] Cut the music_close bookend from the first build if the exact close window is not available from `led`. The existing `_TelemetryHUDRenderer` post-roll already exists; the intro title card and gutter scopes are the primary functional changes.