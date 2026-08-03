# M1 MEASUREMENT: the HuMo lip-sync premise does not survive contact with the data

**27 clips, 7 episodes, 20 usable segment measurements, ZERO GPU time.**
Branch `v2.0-alpha`, HEAD `1ccaddad`, 2026-08-02.
Instrument: `scripts/otr_measure_av_offset.py --batch-segments`.

## The headline

**BUG-07.13 asserts "audio consistently leads the lips by 100-200 ms, constant
offset every clip, every episode". Not one of twenty measured segments lands in
that band, at any confidence gate, and the sign is predominantly the OPPOSITE.**

Combined with `docs/2026-08-02-MEASUREMENT-humo-static-onset.md`, which found the
entry's stated CAUSE (a 3-6 frame leading freeze) absent from two-thirds of
clips, both halves of BUG-07.13 now fail against production output.

**Do not build the prescribed pre-roll fix.** It was specified to correct an
error whose magnitude, sign and mechanism are all unsupported.

## What was measured, and why this run is trustworthy where two earlier ones were not

Each measurement pairs ONE render segment with the EXACT conditioning WAV the
engine was fed, recovered from `otr/episodes/_shared/tmp/audio_slices/`. Video is
MediaPipe FaceMesh mouth aperture -- the normalised inner-lip gap, landmarks 13
and 14 over face height -- and audio is the RMS loudness envelope at the video
frame rate. Aperture against loudness is the physically direct pairing; the
rejected first instrument compared frame-difference energy against onset
strength, which is two derivatives away from the quantity of interest.

Three failures were fixed to get here, and all three mattered:

1. **The unit of analysis was wrong.** Beats above the tier cap are SPLIT into
   segments, each an independent render from fresh noise with its own audio
   window -- the operator's design, to keep VRAM low. Two of the first five
   clips measured were two-segment concatenations, so the earlier correlations
   crossed cold-start seams. Assembled beats are now refused, not measured.
2. **The pairing was reconstructed, not recovered.** Ledger `start_s`/`dur_s` is
   the right window in principle; the cached slice is the actual bytes.
3. **The proxy was too weak.** Frame-difference energy failed its own shifted
   control on 2 of 5 clips.

**The instrument is validated in both directions.** Synthetic injected offsets
recover to better than 0.3 ms (`--self-test`), and on real footage a deliberate
+120 ms audio shift moved the reading by exactly -120.0 ms.

## The result

Tightening the correlation gate is the test that separates signal from argmax
noise. If the offset were noise, tightening would shrink `n` without shrinking
dispersion.

| corr gate | n | median ms | mean ms | min | max | sd ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 20 | -27.5 | +163.0 | -895.8 | +967.9 | 497.1 |
| 0.25 | 14 | -40.4 | +32.8 | -895.8 | +900.5 | 430.3 |
| 0.30 | 11 | -16.1 | +142.3 | -517.8 | +900.5 | 375.6 |
| 0.35 | 7 | -38.8 | -36.9 | -517.8 | +435.3 | 255.6 |
| 0.40 | 6 | -27.5 | +43.3 | -67.9 | +435.3 | 176.7 |
| 0.50 | 2 | -60.1 | -60.1 | -67.9 | -52.2 | 7.9 |

**Dispersion falls monotonically -- 497 to 8 ms -- while the median stays inside
a -16 to -60 ms band the whole way.** That is a real quantity emerging from
noise, not noise alone.

Sign, positive meaning VIDEO LAGS AUDIO (the direction the Bible claims):

    gate 0.15: n=20  video lags 8   video leads 12
    gate 0.30: n=11  video lags 4   video leads 7
    gate 0.40: n= 6  video lags 1   video leads 5

**In BUG-07.13's predicted +100..+200 ms band: 0 of 20, 0 of 11, 0 of 6.**

## What this supports

* The premise is refuted. There is no constant +100-200 ms audio lead in this
  output.
* The residual offset is SMALL and NEGATIVE -- roughly **-30 to -60 ms**, about
  one frame at 25 fps, with the video very slightly ahead of the audio.
* At one frame, this is at the edge of what a 25 fps measurement resolves and is
  well inside normal lip-sync tolerance. **There is no defect here worth a fix**,
  and certainly not one worth a pre-roll that would move sync the WRONG WAY.

## What this does NOT support -- stated so nobody over-reads it

* **This is not SyncNet.** Aperture-against-loudness is a real but loose
  physical coupling; peak correlations top out near 0.7. A learned audio-visual
  embedding would resolve a small offset far better. LatentSync's evaluator is
  on disk at `latentsync/repo/eval/eval_sync_conf.py` but its
  `checkpoints/auxiliary/syncnet_v2.model` is ABSENT, so it could not be run.
* **n is small at the gates that matter** -- 6 at 0.40, 2 at 0.50.
* **This is production footage, not a purpose-built diagnostic.** The plan asked
  for plosive-rich frontal lines at a zero-based CFR-25/16 kHz mux. That
  experiment would be sharper. It is now hard to justify spending GPU hours on
  it to refine a number whose practical answer is already "about one frame".
* **A three-way classification was not reached, and forcing one would be
  dishonest.** Per-segment verdicts came out CONSTANT=1, EARLY-ONLY=2,
  GROWING=7, MIXED=9, INCONCLUSIVE=1 -- which is what a taxonomy looks like when
  applied to an effect near the noise floor. The five-way taxonomy exists for
  exactly this: the honest label here is that the premise failed, not that one of
  three remedies won.

## Coverage and exclusions -- counted, never silently dropped

    clips seen            27
    UNPAIRED               3   multi-segment: no single slice matches the duration
    NO-ARTICULATION        3   face tracked in 0%, 17% and 22% of frames
    usable measurements   20

**The three NO-ARTICULATION clips are their own finding.** Two are single-segment
`humo` renders where MediaPipe held a face for only 17% and 22% of frames, and
OpenCV's Haar detector independently fell back on the same two. A talking-head
render that does not hold a trackable face for most of its length is a quality
problem worth its own investigation; it is recorded here and not chased.

## Recommended disposition of BUG-07.13

Its symptom, cause and fix disagree with each other and now with the data.
Rewrite it to record what was measured: the leading freeze is absent from
two-thirds of clips, the offset is about one frame with the video slightly
ahead, and the pre-roll remedy is withdrawn. Keep the entry -- the reasoning is
worth preserving -- but it must stop prescribing a correction for an error that
was never demonstrated.
