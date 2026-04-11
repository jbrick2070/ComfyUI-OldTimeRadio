r"""
SFX Generator - Sound Effects via Procedural Synthesis
========================================================

Generates common radio drama sound effects without requiring an external model
like AudioLDM. Uses additive/subtractive synthesis and noise shaping.

If AudioLDM 2 is available, an optional path uses it for complex SFX.
Otherwise, the built-in synthesizer covers the essentials:

  - Door sounds (open, close, knock)
  - Footsteps (various surfaces)
  - Thunder / storms
  - Explosions
  - Electronic beeps / sci-fi tones
  - Radio tuning static
  - Heartbeat
  - Ticking clock
  - Sirens
  - Wind / ambience

For the sci-fi anthology, the electronic/sci-fi tones and radio tuning
effects are especially useful.

v1.0  2026-04-04  Jeffrey Brick
"""

import json
import logging
import math
import torch
import numpy as np

log = logging.getLogger("OTR")


def _generate_tone(freq, duration_sec, sample_rate, amplitude=0.5, envelope="adsr"):
    """Generate a sine tone with optional ADSR envelope."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    signal = np.sin(2 * np.pi * freq * t) * amplitude

    if envelope == "adsr":
        n = len(t)
        env = np.ones(n, dtype=np.float32)
        a, d, s_level, r = int(n * 0.05), int(n * 0.1), 0.7, int(n * 0.2)
        env[:a] = np.linspace(0, 1, a)
        env[a:a + d] = np.linspace(1, s_level, d)
        env[a + d:-r] = s_level
        env[-r:] = np.linspace(s_level, 0, r)
        signal *= env

    return signal


def _generate_noise(duration_sec, sample_rate, color="white"):
    """Generate colored noise."""
    n = int(sample_rate * duration_sec)
    white = np.random.randn(n).astype(np.float32)

    if color == "white":
        return white * 0.3
    elif color == "pink":
        # Simple pink noise via cumulative sum + high-pass
        pink = np.cumsum(white)
        pink -= np.mean(pink)
        pink /= max(np.abs(pink).max(), 1e-8)
        return pink * 0.3
    elif color == "brown":
        brown = np.cumsum(white) / np.sqrt(n)
        brown -= np.mean(brown)
        brown /= max(np.abs(brown).max(), 1e-8)
        return brown * 0.3

    return white * 0.3


# --- SFX LIBRARY -------------------------------------------------------------

def _sfx_radio_tuning(duration_sec, sample_rate):
    """Radio dial tuning - sweeping frequencies with static bursts."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    freq_sweep = 400 + 800 * np.sin(2 * np.pi * 0.5 * t)
    signal = np.sin(2 * np.pi * freq_sweep * t / sample_rate * np.cumsum(np.ones_like(t))) * 0.3
    static = np.random.randn(len(t)).astype(np.float32) * 0.2
    # Modulate static with random bursts
    burst = (np.random.rand(len(t)) > 0.7).astype(np.float32)
    return signal + static * burst


def _sfx_sci_fi_beep(duration_sec, sample_rate):
    """Sci-fi computer beep sequence - layered tones with digital character."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    n = len(t)

    # Multiple tones at different frequencies
    beep1 = np.sin(2 * np.pi * 880 * t) * 0.3
    beep2 = np.sin(2 * np.pi * 1320 * t) * 0.15
    beep3 = np.sin(2 * np.pi * 440 * t) * 0.1

    # Pulsing envelope
    pulse = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float32)
    return (beep1 + beep2 + beep3) * pulse


def _sfx_theremin(duration_sec, sample_rate):
    """Theremin-like sci-fi warble - the sound of 1950s sci-fi."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    # Vibrato + portamento
    freq = 440 + 200 * np.sin(2 * np.pi * 5 * t) + 100 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    signal = np.sin(phase) * 0.4
    # Fade in/out
    n = len(t)
    fade = int(n * 0.1)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    return signal


def _sfx_explosion(duration_sec, sample_rate):
    """Distant explosion - noise burst with exponential decay."""
    n = int(sample_rate * duration_sec)
    noise = np.random.randn(n).astype(np.float32)
    # Exponential decay
    decay = np.exp(-np.linspace(0, 8, n))
    # Sharp attack
    attack = np.ones(n, dtype=np.float32)
    attack_len = int(n * 0.02)
    attack[:attack_len] = np.linspace(0, 1, attack_len)
    # Low-pass to make it rumbly
    signal = noise * decay * attack
    # Simple LPF
    alpha = 0.1
    for i in range(1, n):
        signal[i] = signal[i - 1] + alpha * (signal[i] - signal[i - 1])
    return signal * 0.8


def _sfx_footsteps(duration_sec, sample_rate):
    """Footsteps - regular impulse clicks with resonance."""
    n = int(sample_rate * duration_sec)
    signal = np.zeros(n, dtype=np.float32)
    step_interval = int(sample_rate * 0.5)  # ~2 steps/sec

    for pos in range(0, n, step_interval):
        # Short noise burst for each step
        burst_len = int(sample_rate * 0.05)
        end = min(pos + burst_len, n)
        burst = np.random.randn(end - pos).astype(np.float32) * 0.5
        burst *= np.exp(-np.linspace(0, 10, len(burst)))
        signal[pos:end] += burst

    return signal


def _sfx_heartbeat(duration_sec, sample_rate):
    """Heartbeat - low-frequency thumps."""
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    signal = np.zeros(n, dtype=np.float32)

    beat_interval = 0.8  # 75 BPM
    for beat_time in np.arange(0, duration_sec, beat_interval):
        pos = int(beat_time * sample_rate)
        # Double thump (lub-dub)
        for offset in [0, int(0.15 * sample_rate)]:
            start = pos + offset
            burst_len = int(sample_rate * 0.08)
            end = min(start + burst_len, n)
            if start < n:
                t_burst = np.linspace(0, 0.08, end - start, dtype=np.float32)
                burst = np.sin(2 * np.pi * 40 * t_burst) * np.exp(-t_burst * 30) * 0.6
                signal[start:end] += burst

    return signal


def _sfx_door_knock(duration_sec, sample_rate):
    """Three sharp knocks on a wooden door."""
    n = int(sample_rate * duration_sec)
    signal = np.zeros(n, dtype=np.float32)

    for i in range(3):
        pos = int(sample_rate * (0.1 + i * 0.3))
        burst_len = int(sample_rate * 0.04)
        end = min(pos + burst_len, n)
        if pos < n:
            burst = np.random.randn(end - pos).astype(np.float32)
            burst *= np.exp(-np.linspace(0, 15, len(burst)))
            signal[pos:end] += burst * 0.7

    return signal


def _sfx_wind(duration_sec, sample_rate):
    """Wind ambience - filtered noise with slow modulation."""
    n = int(sample_rate * duration_sec)
    noise = np.random.randn(n).astype(np.float32)

    # Slow amplitude modulation
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t)

    # Low-pass filter
    signal = noise * mod * 0.3
    alpha = 0.05
    for i in range(1, n):
        signal[i] = signal[i - 1] + alpha * (signal[i] - signal[i - 1])

    return signal


def _sfx_siren(duration_sec, sample_rate):
    """Air raid siren - sweeping frequency."""
    n = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    freq = 400 + 300 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    return np.sin(phase).astype(np.float32) * 0.5


def _sfx_ticking_clock(duration_sec, sample_rate):
    """Clock ticking - sharp clicks at regular intervals."""
    n = int(sample_rate * duration_sec)
    signal = np.zeros(n, dtype=np.float32)
    tick_interval = int(sample_rate * 1.0)  # 1 tick/sec

    for pos in range(0, n, tick_interval):
        click_len = int(sample_rate * 0.005)
        end = min(pos + click_len, n)
        signal[pos:end] = 0.8 * np.exp(-np.linspace(0, 20, end - pos))

    return signal


SFX_GENERATORS = {
    "radio_tuning":   _sfx_radio_tuning,
    "sci_fi_beep":    _sfx_sci_fi_beep,
    "theremin":       _sfx_theremin,
    "explosion":      _sfx_explosion,
    "footsteps":      _sfx_footsteps,
    "heartbeat":      _sfx_heartbeat,
    "door_knock":     _sfx_door_knock,
    "wind":           _sfx_wind,
    "siren":          _sfx_siren,
    "ticking_clock":  _sfx_ticking_clock,
    "white_noise":    lambda d, sr: _generate_noise(d, sr, "white"),
    "pink_noise":     lambda d, sr: _generate_noise(d, sr, "pink"),
}


class SFXGenerator:
    """Generate radio drama sound effects via procedural synthesis."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("sfx_audio", "generation_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sfx_type": (list(SFX_GENERATORS.keys()), {
                    "default": "radio_tuning",
                    "tooltip": "Type of sound effect to generate"
                }),
                "duration_sec": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 30.0, "step": 0.1,
                    "tooltip": "Duration of the sound effect"
                }),
            },
            "optional": {
                "sample_rate": ("INT", {
                    "default": 48000, "min": 22050, "max": 96000, "step": 1000,
                }),
                "volume_db": ("FLOAT", {
                    "default": 0.0, "min": -30.0, "max": 6.0, "step": 1.0,
                    "tooltip": "Volume adjustment in dB"
                }),
            },
        }

    def generate(self, sfx_type, duration_sec, sample_rate=48000, volume_db=0.0):
        log.info(f"[SFXGenerator] Generating '{sfx_type}' ({duration_sec:.1f}s @ {sample_rate}Hz)")

        generator = SFX_GENERATORS.get(sfx_type)
        if generator is None:
            log.error(f"[SFXGenerator] Unknown SFX type: {sfx_type}")
            # Return silence
            silence = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)
            waveform = torch.from_numpy(silence).unsqueeze(0).unsqueeze(0)
            return ({"waveform": waveform, "sample_rate": sample_rate},
                    json.dumps({"error": f"Unknown SFX: {sfx_type}"}))

        # Generate audio
        audio_np = generator(duration_sec, sample_rate)

        # Apply volume adjustment
        if volume_db != 0.0:
            gain = 10.0 ** (volume_db / 20.0)
            audio_np = audio_np * gain

        # Normalize
        peak = np.abs(audio_np).max()
        if peak > 0.95:
            audio_np = audio_np * (0.95 / peak)

        # To ComfyUI AUDIO format
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0)
        audio_out = {"waveform": waveform, "sample_rate": sample_rate}

        info = json.dumps({
            "sfx_type": sfx_type,
            "duration_sec": round(duration_sec, 2),
            "sample_rate": sample_rate,
            "volume_db": volume_db,
            "samples": len(audio_np),
        })

        log.info(f"[SFXGenerator] Complete: Generated {len(audio_np)/sample_rate:.1f}s of audio")
        return (audio_out, info)
