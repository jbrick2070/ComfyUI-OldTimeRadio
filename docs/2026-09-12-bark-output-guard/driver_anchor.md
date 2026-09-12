# Bark output guard -- the one design number, decided before code

Driver: Claude Fable 5.1 (Cowork, 5080). Written 2026-09-12 for the section-2
CODE row that the 2.4 voice/credits arc became after codex refuted its cut.
PBUG-20260902-03 (`docs/PROD_BUG_LOG.md:10233-10312`) is STATUS FIX-OPEN and
specifies the fix itself; it also says the threshold and retry count are
"a design choice ... so it gets the arc before code". This is that arc, one
round, one codex contrarian.

## 1. What the record already fixes

* **The defect** (`:10285-10288`): on the 4060, bark rendered a 9-word line
  as seven seconds of noise (dominant bin 0 Hz, spectral flatness 0.47-0.56)
  followed by two seconds of a steady tone at 2,524-2,679 Hz (flatness
  0.038). The server log showed nothing abnormal. Speech from the same
  speaker on the same line measured dominant 95-248 Hz in the voiced seconds
  at flatness 0.05-0.10. Bark's semantic stage can derail on any roll.
* **The fix, verbatim** (`:10298-10306`): score each generation for speech
  shape -- the fraction of one-second windows whose dominant frequency sits
  in 70-400 Hz with spectral flatness under 0.2 -- re-roll with `seed + 1`
  up to two more times when it fails, log every re-roll at WARNING with the
  score, keep the best-scoring take if all three fail; the ledger field is
  always filled. *"A legitimate guard by the standing rule -- the
  alternative is a silent wrong render."*
* **Bible verify condition** (`:10311-10312`): a stub engine that returns a
  pure tone on the first call and speech on the second must produce speech
  from the guarded path in one retry.
* **Where it sits** (Sonnet reader, grounded): inside
  `BarkEngine.generate_voice` (`nodes/_otr_audio_engines/eng_bark.py:156-235`)
  around the single `_generate_single_line(..., seed=seed)` call at
  `:195-204`; the seed is a plain argument (`_otr_bark_lib.py:792-804` seeds
  torch once per call), so a re-roll is the same call with `seed + 1`. NOT
  in `_otr_voice_node_common.py:1451-1470`, which wraps every engine. The
  existing `BarkSilentOutputError` gate (`:208-234`) stays: true silence is
  still a hard failure; this guard is for audio that is loud and wrong.
* **Replay** is untouched: `engine_seed` is derived, not stored
  (`_otr_voice_node_common.py:673-676`), `_generate_single_line` is
  byte-deterministic per seed (`tests/test_bark_seed_determinism.py`), and
  the scorer is a pure function of the waveform, so the ladder
  `seed, seed+1, seed+2, best-of-three` replays identically with no new
  ledger field.

## 2. The scorer that shipped, and the draft it replaced

**THE PBUG'S OWN CRITERION WAS MEASURED AND REJECTED.** The record proposed
"the fraction of one-second windows whose dominant frequency sits in
70-400 Hz with spectral flatness under 0.2". Calibrated on 32 real bark
takes (section 6), that scored real speech from 0.00 to 1.00: a voice whose
formants carry the whole-second peak reads as "not speech", so it would have
re-rolled good takes, and on bark a re-roll costs tens of seconds. What
separates speech from both documented artifacts is PITCH, frame by frame.

`speech_shape_score(audio, sample_rate)` in `nodes/_otr_bark_lib.py`,
numpy only, beside `high_band_edge_ratio`:

* **Level is not its business.** The clip is peak-normalised first, so a
  take that clears the engine's 1e-4 peak gate but is soft is judged on
  shape rather than scored 0 and re-rolled twice for being quiet (codex,
  finished-diff review). A window under 2% of the clip's own peak in RMS is
  a PAUSE and leaves the denominator: a breath is not "not speech".
* **Per non-silent one-second window**, 40 ms frames at 20 ms hop: a frame
  is PITCHED when the FIRST local maximum of its normalised autocorrelation
  after the correlation first falls away is stronger than 0.45 AND sits at
  a lag in the 70-400 Hz range. First LOCAL maximum, not the strongest one:
  a voice with a loud second harmonic correlates better at twice its pitch
  period, and taking the global maximum reads it an octave down. This rule
  is also exactly why a 2.6 kHz tone fails -- its first peak sits at 0.38
  ms, far below the 2.5 ms floor.
* A window is SPEECH-SHAPED when at least 30% of its frames are pitched and
  its 20 Hz-8 kHz spectral flatness is under 0.5 (a noise floor measured
  0.47-0.56 on the record). Score = speech-shaped / non-silent windows.
* **Cost:** a 30-second clip scores in 0.13 s.

**The one shape it turns away, measured across 95-350 Hz:** a voice ABOVE
200 Hz whose second harmonic is several times louder than its fundamental.
Below 200 Hz the half-period is still in range, so every measured bark
preset passes on any balance. Widening this by accepting integer multiples
of the first peak would also admit the 2.6 kHz tone (9-sample period times
seven lands in the speaking range), so the edge stays and is pinned by a
test: such a line costs up to three takes and never costs the take.

## 3. The design choice: pass at 0.30 of the voiced windows

**Decision: a take PASSES at 0.30; re-roll up to twice; keep the best.**

Measured on the 16 saved takes: the three normal presets score 0.60-1.00
(median 1.00) and every artifact -- pure tone, white noise, noise then tone,
the record's own 7-seconds-then-2 shape -- scores 0.00. The line sits in
that gap with margin on both sides. `en_speaker_5` scores 0.00-0.50, which
is discussed in section 6.

Why not higher: a threshold near the normal-preset minimum would re-roll
good takes the moment a line is short or breathy, and each re-roll is a
full bark generation. Why not lower: a take that is half tone is still a
wrong render. Why two re-rolls: three takes bound a failing line; the
record's own odds ("two of six lines in one episode") make three
consecutive failures rare.

What a false positive costs: one extra generation and a WARNING, never a
lost take. What a false negative costs: the tone that ships today.

## 4. The ordered contract, and the tests

One ordered contract per take, folded verbatim from codex's r1:

1. An unusable take -- empty, non-finite, or peak under 1e-4 -- is NOT a
   candidate and is not scored.
2. A usable take scoring at or above the pass line returns immediately.
3. A usable take below it is kept as a candidate and the line is re-rolled.
4. When the ladder is spent, the best-scoring usable take ships. The ledger
   field is always filled; a hole is never the answer.
5. Only when NO take was usable does `BarkSilentOutputError` raise, with
   the same message contract it had before the guard existed.

The seed ladder is `seed + k * 0x9E3779B97F4A7C15` masked to 63 bits, not
`seed + 1`: engine seeds are truncated SHA-256 reductions and adjacent
integers are not reserved (codex r1). It is a pure function of the line
seed, so a replay walks it to the same winner and no new ledger field is
needed.

`tests/test_bark_output_guard.py`: the scorer against the record's exact
artifact shapes, a pause, a quiet-but-usable take, the harmonic sweep and
the documented edge; the adapter against the Bible condition (tone then
speech -> speech in one retry), silent-then-speech, all-fail-ships-best,
all-silent-still-raises, and the ladder's bound.

## 5. What this is not

Not content filtering (no words are read); not a gate that refuses a
model; not a change to kokoro, the shipped default, nor to any other
engine. It is bounded to bark's adapter, which is forced only in the
`otr_4060_floor`, `otr_rot_tts_bark` and `otr_bark_announcer_acceptance`
variants and otherwise selectable by hand.

## 6. The contrarian round, and what the calibration then showed

**codex, r1: REFUTED, seven items.** Grounded and folded:

| item | disposition |
|---|---|
| 0.5 rests on two male probes; real speech may put the whole-second peak above 400 Hz | **grounded, and it was worse than codex said** -- see the calibration below; the criterion itself was replaced |
| "0 Hz dominant" cannot occur inside a 20 Hz-8 kHz search; the noise fails on flatness | grounded; section 2's reasoning was wrong, the outcome was right |
| the silence gate and the re-roll loop need one ordered contract | grounded; folded verbatim (unusable takes are not candidates; best usable take ships; the error raises only when nothing was usable) |
| cached bark audio bypasses the adapter | **refuted**: `profile.use_cache` is False for every local profile (`_otr_voice_node_common.py:1121-1128`), only two Google TTS profiles set it, and no `audio_cache` directory exists on this box |
| `seed + 1` is not domain-separated | grounded; retry seeds are `seed + k * 0x9E3779B97F4A7C15` masked to 63 bits |
| three takes on the 4060 are unmeasured | grounded and stated: unmeasured; the loop reuses the loaded model; the cost is bounded at three takes for a FAILING line only |
| the listed tests were not the scorer's | grounded; `tests/test_bark_output_guard.py` |

**Operator, mid-round:** *"as long as it's bounded, not going to increase all
episodes 2x for re-rolls or an infinite loop"*, *"bark takes ages ... but
let's try it and see"*, *"bark is an alternate lane anyway"*. So: bounded,
and tuned so a GOOD take is never re-rolled.

**The calibration** (design-fork bench, not a leg; 32 real bark takes on
the 5080 through the pack's own adapter, four presets x four deliveries x
two seeds; the second pass saved every take under
`output/otr/obs/bark_calibration/` with per-window stats). The PBUG's own
criterion -- fraction of one-second windows whose dominant 20 Hz-8 kHz bin
lies in 70-400 Hz -- scored the 16 saved takes from **0.00 to 1.00, median
0.60**; `en_speaker_5` scored 0.00-0.40 on every line because its whole-
second peak sits at 400-1,500 Hz (formants, flatness 0.02-0.15), and a
threshold of 0.5 would have re-rolled half the set. What separates speech
from both documented artifacts is PITCH, frame by frame: an autocorrelation
peak in the 70-400 Hz lag range on 40 ms frames -- with one rule that the
first draft of that detector lacked and a pure tone exposed: take the FIRST
peak after the first dip, because a 2.6 kHz tone's autocorrelation returns
to 1.0 at every multiple of its 0.38 ms period and would otherwise read as
pitched in the speech range.

| take (seed 4243) | peak-bin score (PBUG draft) | pitch score (shipped) |
|---|---|---|
| synthetic 180 Hz buzz | 1.00 | 1.00 |
| synthetic 2.6 kHz tone | 0.00 | 0.00 |
| white noise | 0.00 | 0.00 |
| 7 s noise then 2 s tone (the record's shape) | 0.00 | 0.00 |
| en_speaker_3, four lines | 0.62-0.86 | 0.86-1.00 |
| en_speaker_7, four lines | 0.38-0.80 | 1.00 x4 |
| en_speaker_9, four lines | 0.40-1.00 | 0.60-0.80 |
| en_speaker_5, four lines | 0.00-0.40 | 0.00-0.50 |

(The shipped column is the FINAL scorer, after the finished-diff round fixed
the first-peak search and made the measure level-independent. The normal
presets rose to a 0.60 minimum and a 1.00 median under those two fixes.)

**Decision, revised: pass at 0.30 on the pitch score; re-roll at most twice;
keep the best.** Every take of the three normal presets is at or above 0.60,
every artifact at 0.00; 0.30 sits in that gap with margin on both sides.

**`en_speaker_5` IS IN THE SHIPPED CAST POOL** (`config/cast_pools.py:301`,
"male, warm, weary, wry, 50s, 60s, doctor, scientist", and it is one of the
three `recommended_speakers` in `config/audio_engine_profiles.yaml:45`). It
scores 0.00, 0.00, 0.20 and 0.50 across four deliveries, on both scorers and
both seeds -- so under this guard every `en_speaker_5` line on a bark route
will spend three takes and ship the best one. That is the honest cost and it
is stated rather than hidden by lowering the line. Its takes are in
`output/otr/obs/bark_calibration/` for the operator's ear: if they sound
like speech, the fix is to retire or re-weight the preset in the pool; if
they sound like the defect, the guard is doing exactly its job on the one
preset that needed it. **His ear decides; nothing here does.**

## 7. Cost, bounded, stated

A passing take costs one `speech_shape_score` call: numpy, a few
milliseconds. A failing line costs up to three bark generations (measured
here 15-47 s each on the 5080, the first including the model load; the model
stays loaded across attempts). No loop: `range(1 + BARK_REROLLS_MAX)`.
Not measured on the 4060 (testing closed); the 4060 floor is the one shipped
profile that forces bark, so its drill is where the first real re-roll rate
will be read.
