# ComfyUI-OldTimeRadio v1.0 — Complete Pipeline Briefing

**Date:** 2026-04-04
**Status:** v1.0 Release
**Author:** Jeffrey Brick

---

## Executive Summary

**SIGNAL LOST** is a fully automated end-to-end AI radio drama engine. The pipeline fetches real science news headlines via RSS feed, uses Gemma 4 LLM to write multi-act scripts with 12 distinct dramatic story arcs, procedurally generates unique character profiles with international accents, voices every line via Bark TTS, adds procedural sound effects, masters the mix to broadcast-quality stereo, and renders procedural CRT-aesthetic MP4 video with audio-reactive visualizations.

**Pipeline:** RSS Feed → Gemma 4 (ScriptWriter) → Gemma 4 (Director) → BatchBark (TTS) → SceneSequencer (Assembly) → AudioEnhance (Mastering) → EpisodeAssembler (Intro/Outro) → SignalLostVideo (Rendering) → MP4 + Treatment File

**Output:** 1 complete episode per run (5–15+ mins) | Zero manual intervention | Zero API keys

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ ScriptWriter (Gemma4ScriptWriter)                              │
│ • Fetches real RSS science headlines                           │
│ • Generates multi-act script with LEMMY easter egg (11% roll)  │
│ • Canonical 1.0 JSON format with [VOICE:], [SFX:], [ENV:]     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ script_json, episode_seed
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Director (Gemma4Director)                                      │
│ • Scans script for characters                                  │
│ • Generates production plan with voice assignments             │
│ • Procedural character profile override (seeded, deterministic)│
│ • LEMMY always locked to en_speaker_8 + signature SFX          │
│ • ANNOUNCER random balanced gender preset per episode          │
│ • International accents (de, fr, es, hi, it, ja, ko, ru, pt)  │
│ • Character traits: gender, age, demeanor, accent              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ production_plan_json
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ BatchBarkGenerator                                              │
│ • Pre-computes ALL dialogue TTS before scene assembly          │
│ • Groups lines by voice preset (minimizes GPU context switches)│
│ • Hallucination guards: [clears throat] on first line          │
│ • Temperature cap (0.55 intl, 0.5 first line)                  │
│ • ASCII sanitizer prevents language drift                      │
│ • Outputs batched AUDIO tensor in script order                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ tts_audio_clips
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ SFX Generator                                                   │
│ • Procedural synthesis (theremin, radio tuning, etc.)          │
│ • No external models required                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ sfx_audio_clips
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ SceneSequencer                                                  │
│ • Assembles TTS clips + SFX + room tone bed                    │
│ • Deterministic beat pauses (beat, pause, breath)              │
│ • Room tone generation based on [ENV:] descriptors             │
│ • Resampling (GPU torchaudio or CPU scipy)                     │
│ • Outputs scene_audio (single long clip)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ scene_audio
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ AudioEnhance                                                    │
│ • Resample to target rate (24kHz → 48kHz standard)             │
│ • Mono → Stereo conversion                                     │
│ • Bass warmth (biquad low-shelf filter)                        │
│ • Haas effect spatial widening (0.2–0.8ms delay)               │
│ • Mid-side stereo decorrelation                                │
│ • Low-pass filter (16kHz cutoff) kills Bark chirp artifacts    │
│ • Peak normalization to -1dBFS broadcast standard              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ enhanced_audio (stereo 48kHz)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ EpisodeAssembler                                                │
│ • Prepends intro theme music                                   │
│ • Appends outro theme music                                    │
│ • Final stereo master ready for video sync                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ master_audio
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ SignalLostVideoRenderer                                         │
│ • Procedural CRT frame rendering (PIL/ImageDraw)               │
│ • Audio-reactive analysis (per-frame RMS + FFT spectrum)       │
│ • Frame-by-frame math-driven generative art                    │
│ • NVIDIA h264_nvenc hardware encoding (RTX fallback to CPU)    │
│ • Post-roll Telemetry HUD (cast, voices, stats, script)        │
│ • Outputs MP4 + _treatment.txt companion file                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Node Details

### 1. Gemma4ScriptWriter

**What It Does:**
Fetches real science headlines via RSS, seeds Gemma 4 LLM with news content, and generates a full multi-act radio drama script in Canonical 1.0 format.

**Key Inputs:**
- `news_sources` (STRING, optional): Comma-separated RSS URLs. Defaults to curated science feed list.
- `target_minutes` (INT, 1–30): Episode duration target
- `genre_flavor` (COMBO): "sci_fi", "mystery", "horror", "drama"
- `temperature` (FLOAT, 0.4–1.2): LLM sampling temperature
- `model_id` (COMBO): "gemma-4-e4b-it" (3GB, lite mode) or "gemma-4-12b-it" (24GB, high quality)

**Key Configurable Parameters:**
- **Story Arc Template:** Random selection from 12 proven dramatic structures (Shakespeare tragedies/comedies, Larry David spirals, Marvel escalation, Twilight Zone twists, etc.)
- **Dialogue Accessibility:** 30% elementary, 30% high school, 20% college, 10% graduate-level
- **Lemmy Easter Egg:** 11% probability each run. If triggered:
  - Character named LEMMY is REQUIRED in the script
  - Must have ≥3 lines of dialogue
  - Signature SFX: `[SFX: heavy wrench strike on metal pipe, single resonant clank]` (plays ONCE before first line)
  - Uses en_speaker_8 (gravelly male, 50s, British accent)

**Output Format:**

```json
[
  {
    "type": "scene_header",
    "text": "=== SCENE 1 ===",
    "title": "Opening"
  },
  {
    "type": "dialogue",
    "character_name": "ANNOUNCER",
    "voice_traits": "male, 50s, authoritative",
    "line": "[VOICE: ANNOUNCER, male, 50s] This is the late-night science broadcast...",
    "emotion": "neutral"
  },
  {
    "type": "sfx",
    "description": "radio tuning static",
    "duration_ms": 2000
  },
  {
    "type": "env",
    "description": "night city street, distant traffic",
    "duration_ms": 3000
  }
]
```

**Notable Algorithms:**
- **NewsFeeds Fallback Chain:** Tries multiple RSS sources in sequence; on total failure, falls back to hardcoded sci-fi premises (never fails the generation)
- **Content Safety Filter:** Blacklist for curse words; replacements with "***" during output
- **Act-by-Act Generation:** For >5min episodes, generates outline first, then each act sequentially to avoid token truncation
- **Citation Hallucination Guard:** Strips fake citations that Gemma 4 may hallucinate
- **Script Parser Streaming:** Emits per-line stats to logging infrastructure for live monitoring

---

### 2. Gemma4Director

**What It Does:**
Scans the generated script, identifies all characters, and builds a production plan with voice assignments. Procedurally overrides character traits deterministically so every cast is unique per episode.

**Key Inputs:**
- `script_json` (STRING): Output from ScriptWriter
- `episode_seed` (STRING, optional): Seed for deterministic character generation. Auto-generated from script hash if not provided.

**Procedural Character Generation:**

**_VOICE_PROFILES** (gender + accent + language code):
- 10 English-native presets (en_speaker_0–9): authoritative/deep, calm/measured, energetic/sharp, warm/weary, intense/dry, gravelly/anxious
- ~30 international presets: German (de_0, de_4), Spanish (es_0, es_9), French (fr_0, fr_4), Indian (hi_0, hi_3), Italian (it_0, it_4), Japanese (ja_0, ja_3), Korean (ko_0, ko_3), Russian (ru_0, ru_3), Brazilian/Portuguese (pt_0, pt_4), Polish (pl_0, pl_3)

**_ACCENTS** (weighted pool):
```python
neutral:  60%  (en_speaker_*)
German:   6%   (de_speaker_*)
Spanish:  6%   (es_speaker_*)
French:   5%   (fr_speaker_*)
Russian:  4%   (ru_speaker_*)
Indian:   4%   (hi_speaker_*)
Italian:  4%   (it_speaker_*)
Japanese: 3%   (ja_speaker_*)
Brazilian: 3%  (pt_speaker_*)
Korean:   3%   (ko_speaker_*)
Polish:   2%   (pl_speaker_*)
```

**_ANNOUNCER_PRESETS** (balanced gender pool):
- 6 presets split between male (en_0, en_1, en_5) and female (en_2, en_4, en_9)
- Random per-episode assignment (seeded)
- Always neutral English accent

**_LEMMY_PROFILE** (immutable):
```python
{
  "name": "LEMMY",
  "gender": "male",
  "age": "50s",
  "demeanor": "gravelly",
  "accent": "British",
  "voice_preset": "v2/de_speaker_0",  # German preset = British accent illusion
  "notes": "Male, gravelly/raspy, iconic"
}
```

**_generate_character_profile()** Algorithm:
1. Deterministic RNG seeded by (episode_seed, character_index)
2. Pick gender: 50/50 random or weighted from traits
3. Pick age: random from [20s, 30s, 40s, 50s, 60s]
4. Pick demeanor: random from 12 archetypes (calm, intense, warm, sharp, dry, energetic, measured, wry, stoic, anxious, confident, weary)
5. Pick accent: weighted random (60% neutral, 40% international)
6. Filter voice profiles: gender + accent language code
7. Score candidates by trait overlap (age/demeanor tag matches)
8. Best match wins; ties broken by RNG shuffle
9. If no exact match, fall back to same-gender English
10. Return (name, gender, age, demeanor, accent, voice_preset, notes)

**Character Name Randomization** (_randomize_character_names):
- RNG seeded per episode ensures reproducible casts
- First name + last name pools (deterministic selection)
- Names converted to ALL CAPS for canonical parsing
- LEMMY always stays LEMMY
- ANNOUNCER always stays ANNOUNCER

**Output:** production_plan_json
```json
{
  "voice_assignments": {
    "LEMMY": {
      "voice_preset": "v2/de_speaker_0",
      "notes": "Male, gravelly/raspy, 50s, British accent, iconic"
    },
    "DR_VOSS": {
      "voice_preset": "v2/es_speaker_0",
      "accent": "Spanish",
      "gender": "female",
      "age": "40s",
      "demeanor": "intense"
    }
  },
  "pacing": {
    "breath_pause_ms": 400,
    "beat_pause_ms": 1500,
    "pause_ms": 2000,
    "scene_transition_ms": 2500,
    "act_break_ms": 5000
  }
}
```

---

### 3. BatchBarkGenerator

**What It Does:**
Pre-computes all dialogue TTS audio before scene assembly. Groups lines by voice preset to keep the GPU hot and minimize context switching. Output is a batched AUDIO tensor in original script order, so SceneSequencer consumes pre-rendered clips without inline TTS.

**Key Inputs:**
- `script_json` (STRING): From ScriptWriter
- `production_plan_json` (STRING): From Director
- `temperature` (FLOAT, 0.1–1.5, default 0.7): Bark sampling temperature

**Key Algorithms:**

**Voice Preset Resolution** (_voice_preset_for_character):
1. Check character cache (same character always gets same voice across episode)
2. Check Director's voice_assignments (explicit mapping)
3. Fuzzy match (uppercase, underscored, partial substring)
4. Gender-aware hash fallback:
   - Extract gender from voice_traits ("female", "male", "unknown")
   - Deterministic RNG seeded by character name
   - ~85% chance English-native preset, ~15% international accent preset
   - Return best-fit preset

**ASCII Sanitizer** (_clean_text_for_bark):
1. Strip structural tags: [VOICE:], [ENV:], [SFX:], [MUSIC:], scene headers
2. Convert parenthetical stage directions to Bark tokens:
   - "(laughs)" → "[laughs]"
   - "(sigh)" → "[sighs]"
   - "(whisper)" → "" (Bark can't whisper; drop direction)
3. Convert asterisk actions: "*laughs*" → "[laughs]"
4. Whitelist Bark tokens: [laughter], [laughs], [sighs], [music], [gasps], [clears throat], [coughs], [pants], [sobs], [grunts], [groans], [whistles], [sneezes], ♪ music ♪
5. Strip remaining unrecognized brackets
6. NFKD Unicode normalization + ASCII-only preservation (prevents language drift on intl presets)
7. Collapse whitespace

**Hallucination Guard** (_generate_single_line):
- **First line per preset:** Prepend "[clears throat]" to reset model away from "podcast intro" mode
- **Temperature floor (international presets):** Cap to 0.55 to keep model committed to English
- **Temperature floor (first line):** Cap to 0.6 (0.5 if intl) to commit text vs. hallucinate continuations

**Text Chunking** (_chunk_text_for_bark):
- Split at sentence boundaries if > 180 characters
- Preserve whole sentences (avoid mid-word splits)
- Ensures Bark generates coherent chunks

**Batch Grouping Strategy:**
1. Extract all dialogue lines with script index, character name, voice preset
2. Group by preset (OrderedDict, preserves order):
   ```
   preset "v2/en_speaker_0": [line1, line3, line5]
   preset "v2/es_speaker_0": [line2, line4]
   ```
3. Load Bark once, generate all lines per preset in sequence
4. Track which presets have started (first line gets hallucination guard)
5. Results dict: {script_idx → (audio_np, sample_rate)}

**GPU Optimization:**
- Torch tensor/arange monkey-patching to default to CUDA (prevents CPU/CUDA device mismatches in Bark sub-models)
- Recursive device movement (_move_to_device) handles nested dicts (history_prompt with semantic/coarse/fine numpy arrays)
- Vectorized padding via torch.nn.functional.pad_sequence on GPU
- Move to CPU only at final assembly

**Output Format:** (AUDIO, batch_log)
```python
{
  "waveform": torch.Tensor([B, 1, max_T]),  # [batch, channels, time]
  "sample_rate": 24000  # Bark native rate
}
# Plus detailed batch_log of generation timings
```

---

### 4. SFX Generator

**What It Does:**
Generates procedural sound effects via additive/subtractive synthesis. No external model (optional AudioLDM 2 path exists but not required). Covers common radio drama SFX.

**SFX Types (SFX_GENERATORS dict):**

| Type | Algorithm | Use Case |
|------|-----------|----------|
| **radio_tuning** | Frequency sweep (400–1200Hz) + random static bursts | Sci-fi opening, tuning in station |
| **sci_fi_beep** | Layered tones (880Hz, 1320Hz, 440Hz) with pulsing envelope | Computer beep, alert sequence |
| **theremin** | Frequency vibrato + portamento + fade envelope | 1950s sci-fi hallmark, eerie |
| **explosion** | White noise + exponential decay + low-pass rumble | Distant blast, impact |
| **footsteps** | Impulse clicks (0.5s interval) with resonant decay | Movement, approach |
| **heartbeat** | Low-frequency sine thumps (40Hz, 75 BPM, lub-dub pattern) | Tension, medical |
| **door_knock** | Three sharp noise bursts with exponential decay | Knock, arrival |
| **wind** | Filtered noise with slow amplitude modulation (0.15Hz) | Ambience, storm |
| **siren** | Frequency sweep (400–700Hz) at 0.5Hz LFO | Air raid, alert |
| **ticking_clock** | Sharp clicks (1/sec) with exponential decay tail | Time pressure, mechanical |
| **white_noise** | Raw Gaussian noise × 0.3 | General ambience |
| **pink_noise** | Cumulative filtered white noise × 0.3 | Softer ambience |

**Key Parameters:**
- `sfx_type` (COMBO): Type name
- `duration_sec` (FLOAT, 0.1–30.0): Duration
- `sample_rate` (INT, 22050–96000, default 48000)
- `volume_db` (FLOAT, -30 to +6): Gain adjustment

**Notable Techniques:**
- **ADSR Envelopes:** Attack 5%, Decay 10%, Sustain 70%, Release 20%
- **Colored Noise:** White → Pink (cumsum HP) → Brown (cumsum / sqrt)
- **Low-pass Filters:** Simple alpha filter (α=0.05–0.1) for smooth texture
- **Decay Curves:** exp(-t*factor) for natural tail-off

**Output:** (AUDIO, generation_info)
```python
{
  "waveform": torch.Tensor([1, 1, samples]),
  "sample_rate": 48000
}
```

---

### 5. SceneSequencer

**What It Does:**
Assembles pre-rendered TTS clips + SFX + room tone into a complete scene. Handles deterministic beat pauses, resampling, and continuous background ambience based on [ENV:] descriptors.

**Key Inputs:**
- `script_json` (STRING): From ScriptWriter
- `production_plan_json` (STRING): From Director
- `tts_audio_clips` (AUDIO, optional): Pre-rendered from BatchBark
- `sfx_audio_clips` (AUDIO, optional): From SFX Generator
- `start_line`, `end_line` (INT): Scene range for partial renders
- `default_tts` (COMBO): "bark" or "parler" if inline TTS needed

**Pacing Configuration** (from production_plan_json):
```python
{
  "breath_pause_ms": 400,        # Between every dialogue line (breath)
  "beat_pause_ms": 1500,         # [BEAT] tag (dramatic beat)
  "pause_ms": 2000,              # [PAUSE] tag (longer silence)
  "scene_transition_ms": 2500,   # Between scenes
  "act_break_ms": 5000           # Between acts
}
```

**Room Tone Generation** (_generate_room_tone):

Continuous background bed based on [ENV:] descriptors. Builds layered noise texture:

1. **Hiss (white noise base):** × 0.3 amplitude
2. **Hum (60Hz + harmonics):** Fundamental + 2nd + 3rd at 1.0/0.5/0.25 ratio
3. **Environmental Texture:**
   - "wind" / "storm" → add low-frequency rumble (HPF 50Hz)
   - "city" / "traffic" → add mid-range tone (~500Hz)
   - "rain" → add high-frequency crackle (HPF 2kHz)
4. **Crackle:** Sparse random pops with decay envelope
5. **GPU Path (long audio >5s):** Vectorized torch ops; CPU path (numpy)

**Resampling** (_resample_audio):

Path selection (RTX 5080 optimized):
- **GPU (CUDA + clip >5s):** torchaudio.functional.resample (sinc interpolation, 8–12x faster)
- **CPU Polyphase:** scipy.signal.resample_poly (high-quality anti-aliased)
- **Fallback:** np.interp linear (if scipy unavailable)

**Assembly Logic:**

1. Extract clips from batched AUDIO inputs (_extract_clips_from_audio)
2. Parse script JSON sequentially
3. For each line:
   - Load pre-rendered TTS clip (or inline generate if not batched)
   - Normalize to target peak (-1dBFS)
   - Resample to target_sr (48000Hz standard)
   - Append breath pause (400ms)
4. For [BEAT] tags: insert 1500ms silence
5. For [PAUSE] tags: insert 2000ms silence
6. For [ENV:] tags: generate room tone bed over duration
7. For [SFX:] tags: insert SFX audio at position
8. Trim trailing silence from final scene
9. Output single concatenated AUDIO tensor

**Output:** (AUDIO, scene_log, script_excerpt)

---

### 6. AudioEnhance

**What It Does:**
Master broadcast-quality stereo from mono TTS. Full DSP pipeline: resample, mono→stereo, bass warmth, LPF chirp cleanup, Haas spatial widening, mid-side decorrelation, peak normalize.

**All DSP is fully vectorized (no Python for-loops over samples).**

**Key Inputs:**
- `audio` (AUDIO): From SceneSequencer
- `target_sample_rate` (INT, 24–96kHz, default 48000): Target Hz
- `spatial_width` (FLOAT, 0.0–1.0, default 0.3): Stereo width (0=mono, 0.3=natural, 1.0=extreme)
- `haas_delay_ms` (FLOAT, 0.0–2.0, default 0.4): Haas effect delay (0.2–0.8 natural for speech)
- `bass_warmth` (FLOAT, 0.0–0.5, default 0.1): Low-freq boost (0.1 = ~3dB, 0.3 = ~9dB)
- `lpf_cutoff_hz` (FLOAT, 8–24kHz, default 16000): LPF cutoff for Bark chirp cleanup
- `normalize_dbfs` (FLOAT, -12 to 0dB, default -1.0): Peak norm target

**Processing Chain:**

1. **Resample** (_resample): torchaudio sinc interpolation (zero aliasing, mathematical analog reconstruction)
2. **Mono → Stereo** (_mono_to_stereo): Duplicate channel if mono
3. **Bass Warmth** (_apply_bass_warmth):
   - Biquad low-shelf filter (industry standard for smooth warmth)
   - Gain formula: gain_db = warmth × 30dB
   - Q=0.707 (no comb ripples)
   - CPU bounce for IIR (faster on Blackwell than GPU IIR)
4. **Low-Pass Cleanup** (_lowpass_16k):
   - Windowed-sinc FIR filter (Hann window, kernel ~101 taps @ 48kHz)
   - Eliminates Bark's high-frequency chirping (>16kHz)
   - Grouped conv1d (same kernel per channel)
5. **Haas Effect** (_haas_delay):
   - Delay right channel by 0.2–0.8ms
   - Creates spatial width without loudness change
   - Psychoacoustic: delays <20ms perceived as spatial image, not echo
6. **Mid-Side Decorrelation** (_stereo_decorrelate):
   - Mid = (L+R)/2, Side = (L-R)/2
   - Boost Side: Side' = Side × (1 + amount)
   - Reconstruct: L' = Mid + Side', R' = Mid − Side'
   - amount=0.15 = subtle, 0.5 = very wide
7. **Peak Normalize** (_normalize):
   - Scale to target_linear = 10^(target_dbfs/20)
   - Result: peak amplitude = target_linear (~0.891 at -1dBFS)

**Output:** (AUDIO,)
```python
{
  "waveform": torch.Tensor([B, 2, T]),  # Stereo
  "sample_rate": 48000  # Broadcast standard
}
```

---

### 7. EpisodeAssembler

**What It Does:**
Sandwiches the scene audio with intro/outro theme music. Produces final master ready for video sync.

**Key Inputs:**
- `scene_audio` (AUDIO): From AudioEnhance
- `intro_music` (AUDIO, optional): Pre-recorded or procedural intro
- `outro_music` (AUDIO, optional): Pre-recorded or procedural outro

**Output:** (AUDIO, assembly_log, treatment_excerpt)

---

### 8. SignalLostVideoRenderer

**What It Does:**
Renders procedural CRT-aesthetic MP4 from finished audio master. Pure math-driven generative art (no AI video, no script text on screen). Hardware NVIDIA h264_nvenc encoding with CPU fallback. Post-roll Telemetry HUD with cast, voices, and production stats.

**Key Inputs:**
- `master_audio` (AUDIO): From EpisodeAssembler
- `episode_title` (STRING): Episode name
- `script_json` (STRING): For HUD metadata
- `production_plan_json` (STRING): Voice assignments
- `width`, `height` (INT): Frame dimensions (1920×1080 standard)
- `fps` (INT): Frame rate (24–60, default 30)

**CRT Renderer** (_CRTRenderer):

**Centre Art Rendering:**
1. **Audio Analysis (per frame):**
   - RMS volume envelope (normalization, dynamic range)
   - 32-bin FFT spectrum (frequency domain visualization)
   - Waveform sample (raw time-domain)

2. **Procedural Elements:**
   - **Concentric rings:** Rotate based on time, scale by volume
   - **Spectrum bars:** 32 bars, height = frequency bin magnitude
   - **Waveform trace:** Oscilloscope-style wave display
   - **Scanline effect:** Horizontal lines simulating CRT phosphor
   - **Vignette:** Dark edges for authenticity

3. **Color Palette:**
   - CRT_FG (green): #00FF00 — scanlines, spectrum
   - CRT_BG (black): #000000 — background
   - CRT_DARK: #001100 — darkened areas

**Telemetry HUD Renderer** (_TelemetryHUDRenderer):

**Left Panel (Static):**
- Episode title
- Runtime statistics (duration, generation time, tokens)
- Voice assignments (character → preset)
- Scene count, dialogue line count

**Right Panel (Scrolling):**
- Full cast list (name, gender, age, demeanor, accent, voice_preset)
- Scene markers
- Production metadata

**Post-Roll Duration:** Dynamic (20–90s) based on content length.

**Video Encoding** (_encode_mp4):

1. **NVIDIA h264_nvenc (RTX 5080):**
   - Hardware encoder: quality VBR, ~8Mbps target
   - Preset: slow (quality over speed)
   - RC: VBR (variable bitrate)
   - ~10–50x faster than CPU

2. **CPU Fallback (libx264):**
   - Software: CRF 20 (visually lossless)
   - Preset: medium
   - Always available, slower (~1min per min video)

**Treatment File Output** (_treatment.txt):

Companion metadata saved alongside MP4:
```
SIGNAL LOST — Episode Treatment
================================
Title: [episode_title]
Duration: [minutes]
Generated: [timestamp]

CAST & VOICES
=============
LEMMY — v2/de_speaker_0 (gravelly, male, 50s, British)
DR_VOSS — v2/es_speaker_0 (intense, female, 40s, Spanish)
...

FULL SCRIPT
===========
=== SCENE 1 ===
[VOICE: ANNOUNCER] This is the late-night science broadcast...
[SFX: radio tuning static]
[VOICE: LEMMY] What do we know?
...

PRODUCTION STATS
================
Scenes: 5
Dialogue lines: 47
Total tokens: 3,250
Generation time: 8m 23s
Unique characters: 12
```

---

## System Requirements

### GPU Requirements

| Setup | GPU | VRAM | Mode |
|-------|-----|------|------|
| **Recommended** | RTX 5080 / 4090 | 16GB+ | Full pipeline (15+ min episodes) |
| **Minimum** | RTX 4070 / 3060 | 8GB | Lite mode (gemma-4-e4b, 5 min episodes) |

### Dependency Pipeline

- **Python 3.9+**
- **PyTorch** (ComfyUI manages; do NOT pin)
- **transformers ≥4.40, <6.0** (Gemma 4, Bark)
- **torchaudio** (resampling, audio DSP)
- **scipy** (signal processing — optional but recommended)
- **numpy** (array ops)
- **feedparser ≥6.0** (RSS parsing)
- **soundfile ≥0.12** (WAV I/O)
- **Pillow** (image rendering for CRT)
- **ffmpeg** (video encoding)

---

## Notable Design Patterns

### 1. Canonical 1.0 Token Format

**Script tokens:**
- `[VOICE: NAME, traits]` dialogue — character name + voice traits
- `[SFX: description]` — sound effect cue
- `[ENV: description]` — environmental ambience
- `[MUSIC: description]` — music cue
- `(beat)` — 0.8s deterministic pause
- `=== SCENE X ===` — scene header

**Bark tokens:**
- `[laughter]` `[laughs]` `[sighs]` `[music]` `[gasps]` `[clears throat]` `[coughs]` `[pants]` `[sobs]` `[grunts]` `[groans]` `[whistles]` `[sneezes]`
- `♪ text ♪` — sung/hummed

### 2. Procedural Determinism

- **Episode seed:** Hash of script text (reproducible across reruns)
- **Character profiles:** RNG(seed, char_idx) → deterministic cast
- **Voice assignments:** Seeded RNG ensures same character always gets same voice
- **Accent selection:** Weighted cumulative distribution with deterministic RNG
- **LEMMY:** Always locked to en_speaker_8 + signature SFX

### 3. GPU Memory Management

- **Strict Device Sentry:** gc.collect() + VRAM flush between model lifetimes
- **Gemma4 Unload:** Freed before BatchBark TTS to prevent CUDA OOM
- **Recursive Device Movement:** Handles nested dicts (history_prompt)
- **Vectorized DSP:** No Python loops over samples; torch/scipy/torchaudio only
- **Tactical CPU Bounce:** IIR filters move to CPU (faster on Blackwell)

### 4. Safety Rails

**Language Drift Prevention (International Presets):**
- ASCII sanitizer: NFKD normalization → ASCII-only
- Temperature cap: 0.55 max for intl presets (keeps model committed to English)
- First-line guard: [clears throat] prefix resets model context

**Hallucination Prevention (Bark):**
- Torch tensor/arange monkey-patching (defaults to CUDA)
- [clears throat] anchor on first line per preset
- Content filter: blacklist curse words

**RSS Feed Fallback:**
- Multiple feed sources in priority order
- Total failure → hardcoded sci-fi premises (never fails)

---

## Performance Notes

### Typical Render Times (RTX 5080, 48kHz stereo)

- **ScriptWriter:** 1–3 min (depends on model, episode length, headline fetching)
- **Director:** 10–30 sec
- **BatchBark:** 3–8 min (TTS generation, depends on character count)
- **SceneSequencer:** 30–60 sec
- **AudioEnhance:** 10–20 sec
- **EpisodeAssembler:** 5–10 sec
- **SignalLostVideo:** 30–90 sec (encoding; h264_nvenc ~2–5x faster than CPU)

**Total:** ~8–15 min per 15-min episode (GPU-bound, parallelizable)

---

## Logging & Observability

### Runtime Log Infrastructure

Every node emits to `otr_runtime.log` via `_runtime_log()`:
- Line-by-line ScriptWriter stats (dialogue count, scene progress)
- BatchBark generation metrics (tokens/sec, lines generated, progress %)
- SceneSequencer assembly progress
- Video rendering frame-by-frame updates

### Live Monitor Dashboard

```bash
cd custom_nodes/ComfyUI-OldTimeRadio
python otr_monitor.py
```

Displays real-time stats in terminal (60Hz refresh).

---

## Known Limitations & Workarounds

### CUDA OOM During BatchBark
**Cause:** Gemma4 didn't fully un-pin VRAM (PyTorch fragmentation)
**Fix:** Restart ComfyUI, re-queue. Lite workflow (gemma-4-e4b) safer.

### ffmpeg Not Found
**Cause:** No ffmpeg on PATH
**Fix:** `winget install ffmpeg` (Windows) or add to PATH

### RSS Feed Fails
**Cause:** Network timeout or feed server down
**Fix:** Automatic fallback to hardcoded sci-fi premises; generation succeeds

### International Preset Language Drift
**Cause:** Model biased toward native language
**Fix:** ASCII sanitizer + temperature cap (0.55) + first-line guard

---

## File Locations

| File | Purpose |
|------|---------|
| `/nodes/gemma4_orchestrator.py` | ScriptWriter, Director, story arc templates, character generator |
| `/nodes/batch_bark_generator.py` | BatchBark TTS batching, voice preset resolution, ASCII sanitizer |
| `/nodes/scene_sequencer.py` | SceneSequencer, room tone, beat pacing, resampling |
| `/nodes/audio_enhance.py` | AudioEnhance DSP pipeline, bass warmth, spatial widening |
| `/nodes/sfx_generator.py` | SFX synthesis (theremin, radio tuning, etc.) |
| `/nodes/vintage_radio_filter.py` | Vintage radio effects (bandpass, tube, hum, crackle) |
| `/nodes/video_engine.py` | CRT renderer, telemetry HUD, video encoding |
| `/workflows/old_time_radio_scifi_lite.json` | Lite workflow (5 min episodes) |
| `/workflows/old_time_radio_scifi_full.json` | Full workflow (15+ min episodes) |
| `README.md` | User-facing documentation |

---

## Summary: v1.0 Feature Checklist

✅ **RSS-driven script generation** — Real science headlines → Gemma 4 → multi-act script
✅ **12 dramatic story arcs** — Shakespeare, Twilight Zone, Marvel, Larry David, etc.
✅ **Procedural character generation** — Deterministic, seeded, reproducible casts
✅ **International accents** — 11 accent types via Bark foreign presets + ASCII sanitizer
✅ **Lemmy easter egg** — 11% chance; signature wrench SFX + gravelly voice
✅ **Batch TTS pre-computation** — GPU-efficient grouped generation
✅ **Procedural SFX synthesis** — Theremin, radio tuning, explosions, etc.
✅ **Room tone generation** — Dynamic ENV-based ambient beds
✅ **Deterministic beat pauses** — BEAT / PAUSE / BREATH tags
✅ **Broadcast-quality mastering** — 48kHz stereo, spatial widening, bass warmth
✅ **Procedural CRT video** — Audio-reactive generative art rendering
✅ **NVIDIA h264_nvenc** — Hardware encoding + CPU fallback
✅ **Telemetry HUD** — Post-roll cast list, scripts, stats
✅ **Treatment file output** — Metadata companion to MP4
✅ **Zero API keys** — Fully local, no external services
✅ **Automatic model download** — Handles first-run setup
✅ **Live render monitor** — Real-time dashboard

---

**End of Briefing**
