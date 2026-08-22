# Ghost cadence: the final plan (4 options + 1 free variant)

**Date:** 2026-08-22
**Status:** FINAL, driver-judged. The operator directed (2026-08-22): *"you be
the judge on the final plan, no need for a full kibitz."* This supersedes the
standing full-arc directive FOR THIS ITEM only.
**Provenance (stated exactly, per the substitution rule):** one full r1 round
ran before the halt -- Codex (gpt-5.6 chain) + Antigravity (Gemini 3.7 Flash
High) external reviews, a COLD Fable first-read, and the driver's code-grounded
anchor, all grounded against the real files in
`kibitz-runs/2026-08-22-ghost-cadence/r1/` (input, reviews, judgment).
r2-r4 were NOT run; this is not a full arc and is not reported as one.
**Input:** `docs/2026-08-22-GHOST-CADENCE-PROBLEM-STATEMENT.md` (commit
9e75d8b6) + three operator directives received 2026-08-22 (arms request,
audio question, audio elevation + judge handoff).

## What the operator ruled, in order

1. Stop-action is welcome; the current motion is "really wild" -- damp the
   energy, keep the character.
2. Give 2-3 more param options to try, research-grounded.
3. Audio should be used somewhere in the workflow if it makes it better --
   elevated (with pasted research, claims verified below) into motion
   strength that follows the beat's own audio.
4. Final plan may be 2-4 best options, driver judges.

## The diagnosis (r1-verified, every number checked against real code)

A beat is ONE AnimateDiff timeline spanning its whole audio budget
(`T = target_frame_count` at 25fps is the single duration authority). At
hold-2, a 12s beat generates U=150 fresh frames and traverses **13** fused
16-frame context windows (exact vs ADE `create_windows_static_standard`); a
1s beat traverses **1**. Motion energy per beat is an uncontrolled function
of dialogue length -- that is the "wild," and real episodes sit at 12s/beat.
LEADING HYPOTHESIS, not settled fact: the window count (vs seam jolt vs
stepping) is what the eye reads; the option set below separates these.

The audio claim is split (Codex r1): exact-T duration sync is a HARD GATE
that survives every option by construction; "it seems to match the beat" is
a perceptual outcome judged per option, never assumed.

---

## The four options

All are ADDITIVE PEER LANES over one new recipe seam. The golden lane
`animatediff15_video` stays byte-identical and keeps rendering dailies.
Working ids below; final ids at build.

| # | Option | Recipe delta vs golden | Hypothesis it isolates | Sampling cost |
|---|--------|------------------------|------------------------|---------------|
| 0 | `ghost_h2` (baseline) | none -- the golden lane on the frozen fixture | reference point | 100% |
| A | `ghost_h3` | hold-3 (8.33 fresh fps; 13 -> 8 windows on 12s) | motion-per-second is the dial; the sweet spot is above 5fps | ~68% |
| B | `ghost_h5` | hold-5 (5.0 fresh fps; 13 -> 5 windows) | the operator's own point: calm stop-action | ~40% |
| C | `ghost_h5_still` | hold-5 + CONSTANT `effect_multival` 0.55 | lower motion STRENGTH calms independent of cadence (control for D) | ~40% |
| D | `ghost_h5_pulse` | hold-5 + PER-FRAME `effect_multival` from the beat's own audio envelope | the video breathes with the soundtrack -- audio-reactive, not lip-sync | ~40% |

**Free fifth variant -- `ghost_h5_smooth`:** ffmpeg `minterpolate` over option
B's U UNIQUE frames presented as a 5fps timeline -> 25fps -> trim to exactly
T. Never interpolate the held 25fps stream (it sees 4 stills + a jump and
falsely convicts the lever). Shares B's latents bit-for-bit -- the set's one
TRUE pair -- and is the only variant that can DISPROVE stop-action as house
style. Costs one ffmpeg pass, no render slot. (Antigravity predicted 512x288
warping kills it; the variant IS that measurement, at $0.)

**Contingency (not rendered now) -- `ghost_ov8`:** hold-2 + context_overlap 8.
If wildness SURVIVES hold-5, the seams are implicated, and this becomes the
next leg. Costs ~138% of baseline (18 windows vs 13), which is why it waits.

D vs C is the experiment's heart: if D reads better than C, the PULSING
earned it; if they tie, lower average strength was the whole story. B vs A
settles the speed ladder; B vs smooth settles step-vs-flow.

### Option D's envelope, specifically

- Source: the beat's own conditioning WAV, already plumbed per-beat
  (`render_driver.py:324,342` builds `audio_ref = {"path": ...}`; engines
  gate on the `accepts_audio_ref` capability, line 1805 -- the pulse peer
  declares it, verify exact attribute form at build).
- Extraction: EXTEND `nodes/_otr_audio_motion.py` (C1, `amp-1`) with a
  per-source-frame RMS envelope -- U values, one per fresh frame's time
  window, numpy + soundfile only, read-only, deterministic. This is the
  C2 consumer that module explicitly deferred; the 8-field clip profile
  stays untouched, version bumps to `amp-2`.
- Mapping (operator's starting hypotheses, settled by eye): silence/quiet
  0.25-0.35, normal speech 0.45-0.60, accents and musical hits 0.80-0.90,
  EMA-smoothed so strength never steps harder than the footage. Mapping
  version + envelope hash go in the receipts.
- Wiring: ADE's multival sockets accept per-frame lists (VERIFIED:
  `nodes_multival.py` `MultivalDynamicNode.execute(float_val: Union[float,
  list[float]])`). The peer builds one extra node (`ADE_MultivalDynamic`)
  wired into `effect_multival` -- a socket the golden lane deliberately
  omits, so omission stays the golden contract. VERIFY AT BUILD with a
  2-frame probe: `effect_multival` (not `scale_multival`) is the knob that
  scales motion amount per-frame.

---

## Build items (in order; each lands with tests, commit+push per green chunk)

1. **The recipe seam.** `GhostSignalEngine` grows class attributes
   (`cadence_hold=2`, `context_length=16`, `context_overlap=4`,
   `motion_effect_mode=None`, `motion_effect_value=None`) and the render
   path READS THEM (today it calls `ghost_hold2_selector` and inlines
   `GHOST_CONTEXT_OVERLAP` directly -- lines 871 and 743 -- so a subclass
   override would be a SILENT NO-OP stamping a wrong receipt; all three r1
   voices convicted this independently). Generalize:
   `ghost_hold_selector(target, hold)`, `ghost_unique_source_count(target,
   hold)`, receipts emit `cadence_mode="hold_%d"`, plus new fields
   `cadence_hold_factor`, `cadence_window_count` (the campaign's own
   quantity: `1 if U <= L else 1 + ceil((U-L)/(L-overlap))`),
   `context_overlap`, `motion_effect_mode`. Bounds guards in
   `assert_usable`: `1 <= overlap < length`, `hold >= 1`. The old "ten
   lines per peer" claim is dead: the seam is the build; each peer after it
   is a ~10-line class.
2. **Golden identity proof.** With the seam in and defaults unchanged, the
   golden lane's selector output, receipts, graph inputs and recipe id are
   asserted BYTE-IDENTICAL by test before any peer lands. `cadence_mode`
   sits in the render-batch CACHE IDENTITY set
   (`otr_video_render_batch.py:69`), so distinct peers can never collide in
   cache -- and every peer MUST override its recipe id + cadence string or
   fail the identity test.
3. **The envelope + pulse peer** (option D, with C as the constant-mode
   degenerate case of the same seam). Pure extraction function first, CPU
   tests with synthetic WAVs (silence -> floor strength, a click -> a
   smoothed peak, determinism twice over).
4. **Profiles + the bakeoff harness.** One frozen fixture episode (one
   ledger, generated ONCE): mostly ~12s beats PLUS one <= 2s beat on
   purpose -- at hold-5 a 1s beat keeps only source frames 0..4 of a
   16-frame gesture (~31% of the composed arc), and the eye must see
   whether that truncation reads (Antigravity's catch). Per-option profile
   JSONs (`slot_overrides.video_render_engine` -> peer id -- the verified
   selection path; the canonical workflow JSON is NOT edited). Seeds pin
   for free: `_seed_from_hash(request_hash)` off the frozen creative hash,
   so every option renders the same beats with the same seeds. Each leg
   renders to a NEW directory (evidence cited by hash), and EVERY leg
   publishes to `otr/obs/` -- a leg that does not reach obs did not pass;
   >5 min with nothing in obs = read the leg log, not the clock.

## Eye protocol (the only instrument that settles look here)

Same beats, same order, one option at a time in obs. Judge: (1) wildness
down? (2) stop-action character kept? (3) does D visibly breathe with the
voice/music where C does not? (4) watch LINE ENDINGS -- ADE snaps the final
window back (14-frame tail overlap at U=150), so beat tails blend more
heavily than middles on every option; (5) the short beat -- does hold-5
gesture truncation read? Hard gates before judging: exact T, exact 512x288,
silent clip contract, obs publish. "Interesting but different" does not
win; an option must beat its control.

## Cut from this campaign (unanimous r1, operator-consistent)

- hold-4 (brackets nothing A/B do not; a later single leg if the eye lands
  between them).
- Beat-length-derived hold (cannot equalize -- a 12s beat needs hold >= 19
  for one window; degenerate on a 12s-beat fixture; texture consistency
  beats energy consistency). Phase-2 question at most.
- Upstream beat splitting (different subsystem, flirts with the T law).
- Context-length and source-floor sweeps, `framerate`/`tblend`, any
  RIFE/VFI install (minterpolate answers smoothing for $0).

## Verify at build (named, so nothing is assumed)

- `effect_multival` vs `scale_multival` semantics (2-frame probe render).
- Exact `accepts_audio_ref` capability spelling on the request-filter path.
- The multival value type accepted by `ADE_AnimateDiffLoaderGen1` when the
  graph is built through `wrapper_bridge` (list vs tensor).
- Frozen-ledger replay with a different profile: confirm the harness can
  re-render video against an existing ledger without re-running writing/TTS.
- minterpolate flag set for 5->25 (`mi_mode=mci` etc.) and its exact output
  frame count before trim.

## Constraints carried unchanged

T-based audio sync untouched at every option; canvas 512x288 -> 1920x1080
untouched; golden lane untouched and still the dailies engine; additive
peers only; obs publish per leg; no new dependency (numpy + soundfile +
ffmpeg all already in the toolchain).
