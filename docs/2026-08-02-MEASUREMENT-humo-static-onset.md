# MEASUREMENT: HuMo leading-static-frame count -- BUG-07.13's cause field does not hold

**27 HuMo clips across 7 episodes, all already on disk, ZERO GPU time.**
Branch `v2.0-alpha`, HEAD `06f7c0e7`, 2026-08-02.
Instrument: `scripts/otr_measure_av_offset.py --static-onset`.

## What was tested and why it was the first thing tested

`BUG_BIBLE.yaml` entry **BUG-07.13** (sibling repo
`comfyui-custom-node-survival-guide`) states a symptom and a cause that do not
agree:

* **symptom:** audio leads the lips by 100-200 ms, "**Constant offset every
  clip, every episode**."
* **cause:** the model "hold[s] the reference portrait static for ~3-6 output
  frames before motion onset. The first articulated lip motion lands at frame
  3-6 (~120-240 ms at 25 fps)."
* **fix:** pre-pad leading silence, then drop the leading pad frames.

The cause is an ONSET mechanism and the prescribed fix only works for an onset
error -- but the symptom asserts a CONSTANT one, for which pre-roll plus an equal
trim is algebraically a no-op. M1 exists to settle which is true.

The cause field is the cheapest half to test, because it is a claim about VIDEO
ALONE: no audio, no cross-correlation, no SyncNet, no GPU. Count the leading
frames that do not move.

## Method

Per clip: decode to grayscale, locate the mouth ROI (OpenCV frontal Haar, median
box over sampled frames), take mean absolute difference between consecutive
frames inside the ROI, and count the leading frames below a motion threshold.

**The threshold is derived from each clip's own noise floor**, not fixed: at
CRF 18 a dark scene and a bright one have very different inter-frame jitter, and
one constant would silently mean different things on each. Floor = 10th
percentile of the clip's own consecutive-frame differences; the threshold sits
several times above it.

## Result: the leading freeze is absent from two-thirds of clips

| statistic | all 27 clips | 21 clips with a VALID face track |
|---|---:|---:|
| median static leading frames | **0** | **0** |
| mean | 10.93 | ~2.3 |
| clips with NO leading freeze | 17 (63%) | 14 (67%) |
| clips in BUG-07.13's claimed 3-6 frame band | **1 (4%)** | **1 (5%)** |

Six clips fell back to a non-face ROI and are reported separately above; the
right-hand column excludes them, and the conclusion does not depend on which
column you read.

**BUG-07.13's cause field is not supported by production output.** A mechanism
described as intrinsic and universal -- "constant offset every clip, every
episode" -- is absent from roughly two-thirds of the clips it should govern.

## The outliers are real and are NOT 3-6 frames

Two clips with valid face tracking show a leading freeze an order of magnitude
larger than the claim:

    shot_l001_character_video_humo.mp4          21 frames   840 ms
    shot_l001_character_video_humo_14B_169.mp4  18 frames   720 ms

Both are `l001` -- the FIRST dialogue line of their episodes. That is a pattern
worth chasing, and it is not the pattern the Bible describes. Two clips with a
fallback ROI showed 152 and 93 frames; those readings were taken on the wrong
rectangle and are not admitted as evidence either way.

## What this settles, and what it explicitly does not

**Settles:** the prescribed remedy's stated mechanism does not generalise. A
pre-roll works by giving the model silence to burn its motion-onset freeze on;
where there is no freeze, there is nothing for the pre-roll to burn, and the
"EARLY-ONLY -> pre-roll" branch loses the basis it was written on. **Do not build
the pre-roll fix on this cause field.**

**Does NOT settle:** whether an audio-to-lip OFFSET exists at all. This
measurement never looked at audio. A constant offset could be present with no
leading freeze whatsoever -- the two are independent, which is exactly why the
Bible's symptom and cause can disagree. Establishing the offset is M1 step 3,
per segment, against the exact cached conditioning WAV.

**Does NOT settle** the first-line pattern. Both large outliers being `l001` is
one observation on two clips, not a finding.

## Provenance

Every sample is a production clip written by the real render path, harvested
from `otr/episodes/*/clips/`. No bench graph, no synthetic fixture, no new
render. The per-clip rows are in the run's JSON output; the distribution above
is the durable record, per the standing rule that a number living only in a log
is one cleanup away from being the next `docs/2026-06-27-humo-bakeoff`.

The instrument's correlator is separately calibrated against synthetic injected
offsets (`--self-test`, error < 0.3 ms); the static-onset mode uses no
correlator at all.
