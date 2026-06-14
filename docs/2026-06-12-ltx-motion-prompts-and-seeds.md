# LTX radio-beat prompts + seeds — render "Dragon's Descent" (2026-06-12 16:04)

Episode: `signal_lost_dragons_descent_20260612_160416`
Config: 30-step KSampler euler **cfg 3.0** (the 6/5 dynamic-motion sampler) · LTX-I2V **strength 0.6** · canvas **1472×832** · length **169 frames** (~6.7s @ 25fps) · 2B v0.9 checkpoint.
Prompt source for every radio beat: **`motion_role`** (the restored 6/5 motion templates). Look comes from the FLUX scene still via the i2v anchor; the prompt's only job is motion.

The appended fragment this episode = **" tension mood."** (atmosphere top-1 from the brief).

---

## Per-beat listing (in playback order)

### 1. b000 — music open  ← your "first radio, not sharp"
- role: `music_visual` · template: **music_open** · 202 chars · sha8 `8694ed2a`
- scene still (the look): FLUX seed **1437746720** → `still_b000_music_open_5c28e5c7`
- **prompt (verbatim):**
  > Continuous shot, same console throughout. Dial whip-pans across frequencies. Tube filaments ignite from cold to white-hot. Speaker grille vibrates aggressively. Dynamic dolly push forward. tension mood.
- NOTE: this is the **most aggressive** template — "whip-pans", "vibrates aggressively", "dynamic dolly push". On the 2B model that much commanded motion + a hard cut can read as soft/smeary. This is the prime suspect for "not sharp." Levers: calmer verbs (swap to the `music_inter` template), or fewer simultaneous motions.

### 2. b001 — announcer  ← your "second, ok"
- role: `announcer_visual` · template: **announcer** · 193 chars · sha8 `b01a8c6d`
- scene still: FLUX seed **2966975686** → `still_b001_5ba0cc59`
- **prompt (verbatim):**
  > Continuous shot, same console throughout. Tuning dial needle sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille trembles. Dust motes drift. Slow handheld dolly forward. tension mood.

### 3. b003 — announcer  ← rendered with NO scene still
- role: `announcer_visual` · template: **announcer** (identical text to b001/b005) · sha8 `b01a8c6d`
- scene still: **MISSING** — log: "beat b003 has NO scene still in the ledger → falling back to the pre-spine text path." So this beat had **no i2v look anchor** (text-only motion). Worth fixing (the still spine dropped b003).

### 4. b005 — announcer (close)  ← your "third radio at end, ok"
- role: `announcer_visual` · template: **announcer** (identical text) · sha8 `b01a8c6d`
- scene still: FLUX seed **1647793901** → `still_b005_efb19db0`
- **prompt (verbatim):** same as b001 above.

---

## Observations for your comparison vs 6/5
- The current prompts **are** the verbatim 6/5 `_PROMPT_BY_ROLE` templates (restored @ `db14f9e`) — so text-wise you are back to the 6/5 baseline.
- Same-role beats share one template by design (b001/b003/b005 are byte-identical text, sha8 `b01a8c6d`); per-beat visual variety is supposed to come from the **different scene stills** — but b003 had no still, so its variety + sharpness suffered.
- The LTX **video sampler seed** is derived deterministically per beat from the request hash (not the same as the FLUX still seed above); it is not separately surfaced in the log. If you want it exposed in the trace, I can add a one-line stamp.

## The full 6/5 template set (for reference)
- **announcer:** Continuous shot, same console throughout. Tuning dial needle sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille trembles. Dust motes drift. Slow handheld dolly forward.
- **music_open:** Continuous shot, same console throughout. Dial whip-pans across frequencies. Tube filaments ignite from cold to white-hot. Speaker grille vibrates aggressively. Dynamic dolly push forward.
- **music_close:** Continuous shot, same console throughout. Dial settles. Tube filaments cool from white through deep amber. Smoke trails from cooling tubes. Slow dolly pull back.
- **music_inter:** Continuous shot, same console throughout. Dial steady, glowing. Oscilloscope dances to the rhythm. VU meters bounce. Tubes pulse with the bass. Slow orbit around the speaker.
- **sfx:** Continuous shot, same console throughout. Snap zoom on the dial as needle spikes hard. Tubes surge with electric arcs. Speaker grille rattles violently. Quick whip-pan to the dial.
