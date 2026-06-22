# Per-segment loudness normalization -- operator notes

**RMS leveling is ON by default** (baked in 2026-06-22). Every dialogue line is leveled toward a target
RMS so the Commander doesn't whisper while the Pilot screams -- no switch to remember. The episode master
is unchanged (makeup 4.0), and the target sits at ~ the level the old peak-normalization produced, so
overall loudness is preserved -- only the per-line balance evens out.

## Knobs (all optional; read at server boot, per render)
| Env var | Default | Meaning |
|---|---|---|
| `OTR_SEGMENT_LOUDNORM` | `rms` | `peak` = escape hatch to the legacy sample-peak path |
| `OTR_SEGMENT_TARGET_RMS_DBFS` | `-16.0` | target dialogue loudness (clamped -60..0) |
| `OTR_SEGMENT_MAX_BOOST_DB` | `9.0` | max upward gain on a quiet line |
| `OTR_SEGMENT_MAX_CUT_DB` | `-12.0` | max downward gain on a loud line |
| `OTR_SEGMENT_GATE_DBFS` | `-50.0` | below this = room tone, left untouched |
| `OTR_SEGMENT_PEAK_CEILING` | `0.95` | linear peak cap after gain (clip-safety) |

SFX, music, and themes are NOT touched (dialogue-only by construction).

## Calibrate the target (optional, recommended)
Run on a real dialogue stem to confirm the -16.0 default fits your voices:
```
python tools/measure_dialogue_rms.py output\otr\episodes\<ep>\audio\<dialogue>.wav
```
It prints the mean active RMS + a suggested target. Set `OTR_SEGMENT_TARGET_RMS_DBFS` to that if you want
to match the current loudness exactly. (Panel note: Bark sits ~ -16..-18 dBFS, so -16 is close already.)

## Re-baseline the audio golden (one-time, since the default output changed)
The `test_audio_byte_identical` golden was captured under the old peak path. Because RMS is now the
default, regenerate it once on the GPU box:
```
# restart ComfyUI server with the desired target (and OTR_CAST_SEED=42 / OTR_STYLE_SEED=42), then:
python tests/test_audio_byte_identical.py --capture-baseline
```
Commit the new `tests/fixtures/baseline_v1.5.wav` + `.sha256`. (The normal suite skips the byte-compare
unless `OTR_REGRESSION_RUNTIME=1`, so day-to-day tests stay green meanwhile.)
