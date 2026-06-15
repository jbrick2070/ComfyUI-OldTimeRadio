# BUG-410 Credits-Tail Fix — Problem Statement for Roundtable (2026-06-14)

**Goal:** restore the end-of-episode ROLLING CREDITS (the Telemetry HUD: `[ CLASSIFIED
TRANSCRIPT ]` + CAST & VOICES + metadata) so they fully SCROLL over a background at the end of
every episode — like the 6/5 reference render — WITHOUT breaking the frozen byte-identical audio
spine. The panel converges on the SAFEST wiring change given the mux invariant.

## Operator verbatim
- "end of current episode — no credits at all, no video/stills."
- "what we had on 6/5 … the still or video should loop while credits roll, now we have no credits."
- "it needs just enough seconds to scroll the video credits."

## Grounded facts (verified against the real code + a runtime probe of the failing render)
1. **The floor renders the FULL credits, but they live BEYOND the master audio.** `nodes/video_engine.py`
   (`SignalLostVideoRenderer`) appends the Telemetry HUD post-roll AFTER the audio-length main frames:
   `total_encode_frames = total_frames + _hud_frames` (L2001), and it also appends `_hud_frames`/fps of
   SILENT audio. The `_TelemetryHUDRenderer` (L1147) scrolls the transcript over `total_hud_frames`.
2. **Runtime probe (render `..._211154`):** floor `#12` video = **65.72 s**; master audio = **~45.7 s**;
   the composite `_silent.mp4`, the §4D blend `_silent_procgen_blended.mp4`, and the muxed
   `_..._final.mp4` are ALL **45.68 s v / 45.706 s a**. => the floor's ~20 s credits scroll (45.7→65.7 s)
   is CUT; only the first few static-header seconds survive. Operator sees the "SIGNAL LOST / title"
   card but no scroll.
3. **The cap is the MUX video≤audio gate (NOT the §4D `shortest=1`).** `nodes/otr_master_audio_mux.py`
   (invariant V-1) is the ONLY node that adds audio: it muxes the FROZEN master with `-c:a copy`
   (byte-identical), FORBIDS `-shortest`, and ASSERTS `composite_duration == master_audio` (within 1/fps)
   BEFORE the mux (a drift guard). So the silent composite is BUILT to the master-audio frame budget.
4. **The composite already restores credits up to the master.** `nodes/otr_silent_composite.py`
   `plan_timeline_segments` tail-fills `[last_beat, target_total_frames)` with the FLOOR's end-slice
   (the credits post-roll); and L420-440 extends `target_total` to the longest sibling `*_master.wav`.
   But that master is ~45.7 s, so there is no room for the ~20 s scroll.
5. **The §4D blend** (`nodes/otr_post_upscale_procgen_blend.py`) blends `[composite][floor][scopes]`
   with `shortest=1`; with composite/blend/final all at 45.68 s it is NOT the operative clamp now, but a
   fix that lengthens the deliverable must keep the three inputs length-consistent so `shortest=1` does
   not re-clamp.
6. **`test_audio_byte_identical`** re-runs the episode with fixed seeds and sha256-compares the FULL
   audio bytes to a golden — so ANY change to the master audio bytes fails it.

## The core tension
To let the credits SCROLL (~20 s), the deliverable VIDEO must run ~20 s past the master audio. But the
mux invariant locks video == master-audio and the master is FROZEN byte-identical. So we must choose how
to make room without corrupting the byte-identical master.

## Candidate fixes (panel: pick the safest + most correct, or propose better)
- **(A) Allow video > audio for a SILENT credits post-roll.** Extend the composite (and the §4D scopes
  input, padded) to the floor length (~65.7 s); relax the mux V-1 gate from `v == a` to `a <= v <= a +
  hud_tail` (an INTENTIONAL silent video tail), still `-c:a copy` (master audio byte-identical, just
  shorter than the video — the player shows silent scrolling credits at the end). Pros: master bytes
  unchanged → `test_audio_byte_identical` stays green; matches OTR (credits can roll in silence). Cons:
  touches the V-1 invariant + the drift assertion — must distinguish an intentional post-roll from real
  drift (e.g. assert `v - a == round(hud_frames/fps)` exactly, fail otherwise).
- **(B) Pad the master audio with ~20 s trailing silence** so v == a holds at 65.7 s. Cons: changes the
  master bytes → forces a `test_audio_byte_identical` golden RE-BASELINE (operator-gated); also the
  credits would roll in dead silence unless a closing bed is added.
- **(C) Extend the CLOSING THEME musically** (longer SA3/closing bed) so the master genuinely covers the
  credits. Cons: an audio-generation change (couples to the SA3 music work, BUG-408); bigger scope.
- **(D) Compress/​speed the credits scroll** to fit the available `[last_beat, master_end]` tail. Cons:
  may scroll too fast to read; doesn't match the 6/5 look; fragile to episode length.

## Questions for the panel
1. Which candidate is the safest correct fix that preserves the byte-identical master AND gives the
   credits enough seconds to scroll? (Lean: A — silent post-roll, audio copied unchanged.)
2. If A: exact change to the mux V-1 gate + drift assertion so an intentional `hud_tail` post-roll is
   allowed but real drift still fails LOUD. Does muxing a shorter `-c:a copy` audio onto a longer video
   (no `-shortest`) keep the audio byte-identical and play correctly (silent tail)?
3. Exact change to `otr_silent_composite.py` so `target_total` extends by the floor's `hud_tail` frames
   (how does the composite learn the hud-frame count — from the floor mp4 length vs the master, or a
   surfaced value?), and tail-fills the scroll from the floor's credits region.
4. Exact change to the §4D blend so the lengthened deliverable does not re-clamp via `shortest=1` (pad
   the scopes input to the floor length; the lighten-blend of a held/black scopes tail is fine).
5. Does the operator want the background to LOOP under the credits (6/5) — and if so, where does the
   loop live (composite tail-fill of the scene clip vs the floor's own credits background)? (Operator
   said "still/video should loop while credits roll.")
6. Risk check: anything that could corrupt the frozen master, the determinism, or the v≤a contract.

## Hard constraints
- Master audio BYTE-IDENTICAL; `-c:a copy`; no `-shortest`; `test_audio_byte_identical` green (unless the
  operator explicitly approves a re-baseline for option B/C).
- 100% local; determinism; UTF-8 no BOM; SFW; any JSON wiring change goes in
  `workflows/otr_scifi_16gb_full.json` + re-validate (workflow-source-of-truth).
- The credits content/renderer already EXISTS (`_TelemetryHUDRenderer`); this is a LENGTH/mux wiring fix,
  not a new renderer.

## Out of scope
Rewriting the HUD renderer; changing the master mix architecture; the SA3 music work (separate BUG-408).
