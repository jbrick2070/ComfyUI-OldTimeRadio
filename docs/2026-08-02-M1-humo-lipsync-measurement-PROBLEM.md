# M1 -- classifying the HuMo lip-sync error: the measurement will not close

**Status: TWO FAILED ATTEMPTS. Panel convened before a third.**
Branch `v2.0-alpha`, HEAD `06f7c0e7`. 2026-08-02.

## What M1 must decide, and why guessing is not allowed

`BUG_BIBLE.yaml:2343` records that HuMo audio leads the lips by 100-200 ms with
the face static for the first 3-6 frames. The prescribed remedy is a pre-roll
plus an equal trim. **That remedy cannot be built until the error is
classified**, because pre-roll plus an equal trim is algebraically a NO-OP when
the lag is constant rather than onset-only:

| classification | meaning | correct fix |
|---|---|---|
| EARLY-ONLY | large lag in the first window, ~0 later | pre-roll IS the fix |
| CONSTANT | same lag early, middle and late | advance the 25 Hz conditioning features; a pre-roll + equal trim cancels |
| GROWING | lag increases across the clip | a rate/timestamp bug (feature Hz vs fps), not a pad |

So the classification is the whole job. A wrong classification ships a fix that
provably does nothing.

## The instrument, and the proof that it works

`scripts/otr_measure_av_offset.py` builds two 1-D signals at the video frame
rate and cross-correlates them. Sign convention: **`lag_ms > 0` means the VIDEO
LAGS THE AUDIO** -- the direction the Bug Bible describes.

`--self-test` injects known offsets into synthetic signals:

    injected   +0.0 ms -> measured   +0.1 ms  (corr 0.999)
    injected  +40.0 ms -> measured  +39.8 ms  (corr 0.998)
    injected +120.0 ms -> measured +119.8 ms  (corr 0.996)
    injected +200.0 ms -> measured +199.7 ms  (corr 0.994)
    injected -120.0 ms -> measured -119.7 ms  (corr 0.996)

**The correlator and the sign convention are correct.** Whatever is wrong is
upstream of the maths.

## Attempt 1 -- FAILED its own control on real footage

Measured all five HuMo clips of
`otr/episodes/signal_lost_the_price_of_breath_20260802_003006` (rendered
2026-08-02 00:30, `delivered_engine=humo`, 480x832 portrait, CFR 25, clips 149-259
frames) against their ledger audio windows, each also re-measured with the audio
deliberately shifted +120 ms.

**A +120 ms injection must move every reading by exactly +120 ms.** Whole-clip
readings actually moved:

| clip | unshifted | shifted +120 ms | delta | verdict |
|---|---:|---:|---:|---|
| l001 | +27.7 ms | -1000.0 ms | **-1027.7** | FAIL |
| l002 | -117.1 ms | +9.6 ms | +126.7 | ok |
| l003 | -33.2 ms | -37.9 ms | **-4.7** | FAIL |
| l004 | -71.6 ms | +48.8 ms | +120.4 | ok |
| l005 | -114.4 ms | +3.7 ms | +118.1 | ok |

Two of five failed outright. Peak correlations were **0.11-0.40** against 0.99
on synthetic, so the correlation surface is nearly flat and its argmax wanders --
which is why windows returned physically absurd values (+705, +814, -803 ms) for
an effect specified as 100-200 ms. Two clips also fell back to a non-face ROI,
so those readings were taken on the wrong rectangle.

**The per-window verdicts this run printed (GROWING, EARLY-ONLY) are noise and
are hereby withdrawn.** They are recorded here only so nobody later quotes them.

## Attempt 2 -- the pairing itself does not hold up

Before rebuilding the instrument, I tested whether the clip/audio pairing is even
right. Four video signals (frame-difference motion energy, ROI intensity std,
dark-pixel fraction, dark-pixel delta) were crossed with two audio signals (RMS
envelope, onset strength) at the ledger window, then the best pairing was slid
across the entire 63.41 s master in 0.25 s steps.

    best pairing at the ledger window: motion_diff x onset, corr 0.114
    ledger window (9.50 s) ranks #137 of 223 candidate starts
    best window anywhere: 30.75 s, corr 0.284

**The ledger's own window is no better than chance**, and the best window
anywhere reaches only 0.28 -- which is roughly the maximum you expect from 223
draws of noise. There is no strong lip-sync correlation anywhere in this clip
against any window of this master, by any of the eight signal pairings tried.

## The three live hypotheses -- I cannot separate them, which is why the panel is here

1. **The video proxy does not capture articulation.** Frame-difference energy and
   ROI statistics may be dominated by head motion, background motion and encoder
   noise. A real mouth-aperture measurement (landmarks) might show what these
   proxies cannot. `mediapipe` is NOT installed; OpenCV 4.13, scipy, librosa are.
2. **The audio window is not the audio the engine was fed.** I assumed
   `lines[].start_s` / `dur_s` in `master_mix` space. But `GO_FORWARD_PLAN.md`
   records that the **per-segment AUDIO window** was changed on 2026-08-02 in the
   same commit that made one-segment coverage plans execute as coverage. Clip
   l001 is 198 frames -- above any single-segment HuMo cap -- so it is a
   CONCATENATION of coverage segments, each rendered against its own sub-window.
   If those sub-windows are not contiguous with the line's window, the whole
   comparison is invalid.
3. **These clips may simply not exhibit the defect** -- e.g. because a static or
   near-static face (the "face static for 3-6 frames" symptom, generalised)
   leaves nothing to correlate. If so, "no correlation" IS the finding, and the
   defect is far worse than a 100-200 ms offset.

These are not ranked. Distinguishing them is the ask.

## What the panel is asked to break

- **Is hypothesis 2 dispositive?** Before any more signal-processing work, should
  the engine's actual audio input window be extracted from the adapter rather
  than reconstructed from the ledger? Where does HuMo's audio conditioning window
  get computed, and is there a receipt of it on disk?
- **Is a frame-difference proxy defensible at all** for this, or does a
  trustworthy classification require real mouth landmarks -- and if so, what is
  available offline with no new paid dependency?
- **Is measuring an ALREADY-RENDERED production episode the wrong instrument
  entirely?** The plan calls for a purpose-built diagnostic: clear frontal lines
  with sharp plosive onsets, zero-based CFR-25 / 16 kHz. Existing footage was
  chosen to avoid spending GPU hours (the M3 lesson: evidence we had, never
  read). Was that the wrong economy here?
- **What is the cheapest experiment that DISCRIMINATES** between the three
  hypotheses, rather than one that would confirm any of them?

## NEW GROUND TRUTH found after the two attempts -- read before answering

### The Bug Bible entry contradicts itself, and that IS the question

`BUG_BIBLE.yaml:2343` lives in the SIBLING repo
(`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`), not this
one. It is entry **BUG-07.13**, and its three fields do not agree:

* **symptom:** "audio consistently leads the lips by 100-200 ms. **Constant
  offset every clip, every episode.**"
* **cause:** the model "hold[s] the reference portrait static for ~3-6 output
  frames before motion onset" -- an **ONSET** mechanism.
* **fix:** pre-pad leading silence, then "drop the leading pad_frames" -- a fix
  that only works for the ONSET reading.

So the Bible asserts CONSTANT in its symptom and prescribes an ONSET remedy. If
the symptom line is right, the prescribed fix is the algebraic no-op this whole
measurement exists to rule out. **Nobody has measured which it is** -- the
research report's own decision table says the public magnitude is "not
published" and warns "do not canonize 200 ms".

### The protocol was already specified and I had not read it

`docs/2026-08-02-HUMO-LTX-DEEP-RESEARCH-REPORT.md:451-458` already prescribes:
a diagnostic **zero-based CFR-25 / 16 kHz mux**, **SyncNet** offset plus
confidence in speech-active early / middle / late windows, **sign validated on a
deliberately shifted control first**, the **median across lines and seeds** (not
one attractive clip), and matched no-LoRA renders used **only as an attribution
control** -- never as a substitute for the correction sweep.

My two attempts used a hand-rolled frame-difference proxy on production footage
instead. That is the framing error.

### What the code actually does (verified, corrects two of my assumptions)

* `render_driver._slice_master_audio()` (`render_driver.py:440-514`) feeds the
  engine an ffmpeg slice of the frozen master over `[start_s, start_s+dur_s]` at
  44.1 kHz mono. **So the pairing I reconstructed is structurally correct** --
  which weakens hypothesis 2 and pushes weight onto hypothesis 1.
* **No leading pre-roll exists anywhere in the render path.** Padding is
  TAIL-only (`pad_tail_s`, ffmpeg `apad`); trim is TAIL-only
  (`fit_frames_to_target` does `arr[:target]`, keeping the FIRST N frames) and
  only runs on capped tiers. So there is no existing pad confounding a reading.
* `WanHuMoImageToVideo` (ComfyUI-core) exposes only `width/height/length/
  batch_size`. **There is no `frame0_idx` or feature-offset knob**, so a
  "advance the conditioning features" fix has no node-level surface -- it would
  have to be done by shifting the audio actually fed in.
* A purpose-built diagnostic is cheap: `scripts/_otr_single_engine_smoke.py`
  takes `--engine --frames --portrait --audio --timeout` and drives the real
  `render_driver.render_single()` dispatch, so `VramPeakProbe` fires exactly as
  in production.
* **The no-LoRA control needs more than the LoRA flag.** `OTR_HUMO_LORA_NAME=
  none|skip|off` drops the node, but `OTR_HUMO_STEPS` (default **6**) and
  `OTR_HUMO_CFG` (default **1.0**) are distill-tuned and do NOT auto-adjust.
  Running the control without raising them confounds "no LoRA" with "wrong
  sampler settings for a non-distilled model".

## Ground truth for reviewers -- verified, not remembered

- Instrument: `scripts/otr_measure_av_offset.py` (self-test passes, see above).
- Footage: `otr/episodes/signal_lost_the_price_of_breath_20260802_003006/clips/`,
  five `*_character_video_humo.mp4`, 480x832, 25 fps, 149-259 frames.
- Master: `audio/pending_20260802_001832_master.wav`, 48 kHz stereo, 63.41 s.
- Ledger: `meta.render_engines.per_clip` records `recipe=ia2v_canonical`,
  `use_lora=true`, `render_canvas`, and **`vram_peak_mb` per clip** -- the
  calibration identity the 2026-08-02 LTX measurement said was missing.
- 26 episodes on disk carry HuMo-delivered clips; this is the newest.
- ~~No per-line/per-beat wav survives on disk -- only music cues and the master.~~
  **WITHDRAWN -- see the corrections below. It survives, and that changes M1.**

## CORRECTIONS after grounding the r1 panel (2026-08-02, Codex lane)

Three panel claims were checked against the real files and all three hold. Two
of them falsify statements I made above and one falsifies a statement I made to
the operator.

### 1. HuMo has NO recorded VRAM peak. I read an LTX row. (CONFIRMED)

`meta.render_engines.per_clip` in the cited ledger has EIGHT rows. The three
`ltx_audio_in` rows carry `recipe=ia2v_canonical`, `quant=Q3_K_M`,
`use_lora=true`, `render_canvas=832x480`, `vram_peak_mb=13519/13401/13455`. **All
five HuMo rows are NULL in every one of those fields.** My earlier claim that
"`vram_peak_mb` per clip is the calibration identity M2 needs" was read off row
0, which is LTX. It is withdrawn.

This is consistent rather than surprising: HuMo's `VramPeakProbe` was added at
`eng_humo.py:811-820` on 2026-08-02 at ~18:00, and this episode rendered
00:40-01:53. **M2 therefore has no prior HuMo data at all** -- which is exactly
why `GO_FORWARD_PLAN` says the probe "finally produces data".

### 2. The exact conditioning WAV per SEGMENT survives on disk. (CONFIRMED)

`otr/episodes/_shared/tmp/audio_slices/` holds **12,475 `slice_*.wav`**, of which
**16 were written during the cited episode**. Matching by write time and
duration against the clips gives an exact, unambiguous pairing -- and it exposes
the real defect in both of my attempts:

| clip | clip duration | conditioning slices actually fed |
|---|---:|---|
| l001 | 7.92 s | **6.600 s + 1.320 s** |
| l002 | 5.96 s | 5.937 s |
| l003 | 6.12 s | 6.076 s |
| l004 | 7.08 s | 7.075 s |
| l005 | 10.36 s | **7.080 s + 3.400 s** |

**l001 and l005 are two-segment concatenations.** Every whole-clip measurement I
took on them cross-correlated ACROSS A COLD-START SEAM, where the second segment
begins from fresh noise with its own audio window. l001 and l003 were also the
two clips whose shifted control failed. That is not a coincidence; it is the
explanation.

So hypothesis 2 is not "the window was wrong" -- it is **the unit of analysis was
wrong**. Never classify an assembled beat.

**M1 no longer needs a GPU render to start.** The segments, their exact
conditioning audio, and their frame counts are all on disk.

### 3. A local SyncNet evaluator exists; its checkpoint does not. (CONFIRMED)

`latentsync/repo/eval/eval_sync_conf.py` and
`latentsync/repo/eval/syncnet/syncnet_eval.py` are PRESENT, and **`mediapipe` is
installed in `latentsync/.venv`** (it is absent from the ComfyUI venv, which is
what I checked before). `checkpoints/auxiliary/syncnet_v2.model` is ABSENT; the
only weights present are `latentsync_unet.pt` (5.07 GB) and `whisper/tiny.pt`.

So SyncNet is reachable in principle but not runnable today, while real mouth
landmarks via mediapipe ARE runnable today in a venv we already have.

### Incidental finding, recorded not chased

**1,310 of the 12,475 cached slices are ZERO BYTES (10.5%)**, and others are 13
or 16 bytes -- truncated WAV headers. Several zero-byte entries were written
during the cited episode. Whether a zero-byte conditioning WAV can ever reach a
render is unknown and is NOT part of M1; it is written down here so it is not
lost, per the standing rule that a number living only in a log is one cleanup
away from being gone.
