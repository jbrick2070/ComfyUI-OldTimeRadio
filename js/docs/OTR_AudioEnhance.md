<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Spatial Audio Enhance

Masters the dialogue track: a common sample rate, stereo, and optional room and vintage-radio treatments.

## What it does

Masters dialogue audio to 48 kHz stereo with optional spatial and vintage analog effects. Resamples input waveforms, converts mono signals to stereo, and applies optional bass warmth, low-pass chirp cleanup, Haas widening, and tape emulation. Adjust spatial width or tape emulation intensity for retro character, or leave defaults for clean pass-through mastering.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**`audio`** -- a connection from another node (`AUDIO`).

Input audio to resample and enhance (mono or stereo, any sample rate).

### Optional

**`target_sample_rate`** -- a whole number from 24000 to 96000; starts as `48000`.

Target sample rate in Hz (48000 = broadcast standard)

**`spatial_width`** -- a number from 0.0 to 1.0; starts as `0.0`.

How much stereo width to add. 0 leaves the stereo image exactly as it arrived, 0.3 is natural, 1.0 is extreme.

**`haas_delay_ms`** -- a number from 0.0 to 2.0; starts as `0.0`.

Haas effect delay ms (0=off, default; 0.2-0.8 = natural widening)

**`bass_warmth`** -- a number from 0.0 to 0.5; starts as `0.0`.

Low-freq warmth shelf (0=off, default; 0.1=subtle)

**`lpf_cutoff_hz`** -- a number from 0.0 to 24000.0; starts as `0.0`.

Optional low-pass cutoff Hz for a voice engine that leaves high-frequency chirp artifacts (0=off, default; 16000 is a gentle cleanup). Applies only below half the target sample rate: a cutoff at or above it is skipped silently, not clamped.

**`tape_emulation`** -- a choice from the dropdown; starts as `off`.

Deterministic analog tape emulation. Each step up adds more period character (saturation, wow/flutter; no synthetic hiss) and costs more clarity -- 'heavy' audibly softens dialogue. 'off' is the clean master; this is an audible episode-facing choice, not a technical knob.

## Outputs

- `enhanced_audio` (`AUDIO`)

## Read more

- [VOICES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VOICES.md)

Class `OTR_AudioEnhance`, category `OldTimeRadio`.
