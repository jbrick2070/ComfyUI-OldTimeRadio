r"""
Vintage Radio Filter — Authentic Old-Time Radio Audio Degradation
==================================================================

Applies a chain of audio effects to make clean TTS audio sound like it's
coming through a 1940s AM radio broadcast:

  1. Bandpass filter (300Hz - 6000Hz) — AM radio frequency response
  2. Tube saturation — soft-clip warmth from vacuum tube amplifiers
  3. 60Hz AC hum — the ever-present electrical hum of vintage gear
  4. Vinyl crackle — random pops and surface noise
  5. Radio static — broadband white noise with AM characteristics
  6. Compression — heavy broadcast limiting (loud and punchy)

Each effect is individually controllable. The Gemma 4 Director node
outputs recommended settings per-episode.

VECTORIZED FILTERS:
  Uses scipy.signal.sosfilt (vectorized C) with a pure-torch
  FFT fallback if scipy is unavailable. Bandpass filtering for all
  audio effects is fully vectorized for optimal performance.

v1.0  2026-04-04  Jeffrey Brick
"""

import logging
import math
import torch
import numpy as np

log = logging.getLogger("OTR")


# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED FILTERS — scipy.signal primary, FFT fallback
# ─────────────────────────────────────────────────────────────────────────────

def _iir_filter_scipy(waveform_np, sample_rate, low_hz, high_hz, order=4):
    """Butterworth bandpass via scipy.signal.sosfilt (vectorized C code).

    This is ~1000x faster than the Python for-loop version.
    Returns numpy array same shape as input.
    """
    from scipy.signal import butter, sosfilt

    nyquist = sample_rate / 2.0
    low = max(low_hz / nyquist, 0.001)
    high = min(high_hz / nyquist, 0.999)

    sos = butter(order, [low, high], btype="band", output="sos")

    # sosfilt operates along last axis — perfect for (B, C, N) shape
    return sosfilt(sos, waveform_np, axis=-1).astype(np.float32)


def _iir_filter_fft(waveform, sample_rate, low_hz, high_hz):
    """FFT-based bandpass fallback when scipy is not available.

    Uses torch.fft for GPU-accelerated frequency domain filtering.
    Not as clean as Butterworth IIR, but still vectorized and fast.
    """
    N = waveform.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0 / sample_rate, device=waveform.device)

    # Smooth bandpass mask (raised cosine / half-Hann edges to reduce ringing).
    # 200 Hz rolloff gives ~40dB suppression with minimal pre-ringing on transients.
    transition_width = 200.0  # Hz — wider than brick-wall, gentler on transients
    mask = torch.ones_like(freqs)

    # High-pass rolloff (half-Hann window rising from 0 → 1)
    hp_start = max(low_hz - transition_width, 0)
    hp_width = low_hz - hp_start  # actual width after clamping
    hp_mask = (freqs >= low_hz).float()
    if hp_width > 0:
        transition = (freqs >= hp_start) & (freqs < low_hz)
        hp_mask[transition] = 0.5 * (1 - torch.cos(
            math.pi * (freqs[transition] - hp_start) / hp_width
        ))
    mask *= hp_mask

    # Low-pass rolloff (half-Hann window falling from 1 → 0)
    nyquist = sample_rate / 2
    lp_end = min(high_hz + transition_width, nyquist)
    lp_width = lp_end - high_hz  # actual width after clamping
    lp_mask = (freqs <= high_hz).float()
    if lp_width > 0:
        transition = (freqs > high_hz) & (freqs <= lp_end)
        lp_mask[transition] = 0.5 * (1 + torch.cos(
            math.pi * (freqs[transition] - high_hz) / lp_width
        ))
    mask *= lp_mask

    # Apply in frequency domain
    spectrum = torch.fft.rfft(waveform, dim=-1)
    filtered = spectrum * mask.unsqueeze(0).unsqueeze(0)
    return torch.fft.irfft(filtered, n=N, dim=-1)


def _bandpass_filter(waveform, sample_rate, low_hz=300, high_hz=6000):
    """Bandpass filter — tries scipy (fast), falls back to FFT (still fast).

    AM radio bandwidth: ~300-5000Hz. We use 300-6000Hz for intelligibility.
    """
    try:
        # scipy path — ~50ms for 30 min @ 48kHz
        waveform_np = waveform.cpu().numpy()
        filtered_np = _iir_filter_scipy(waveform_np, sample_rate, low_hz, high_hz)
        result = torch.from_numpy(filtered_np).to(device=waveform.device, dtype=waveform.dtype)
        log.info("[VintageRadio] Bandpass: scipy.signal.sosfilt (vectorized)")
        return result
    except ImportError:
        # FFT fallback — ~200ms for 30 min @ 48kHz, fully on GPU
        log.info("[VintageRadio] Bandpass: FFT fallback (scipy unavailable)")
        return _iir_filter_fft(waveform, sample_rate, low_hz, high_hz)


def _one_pole_lpf_vectorized(signal, sample_rate, cutoff_hz):
    """Vectorized one-pole low-pass filter using FFT.

    Replaces the O(n) Python for-loop version used in vinyl crackle
    and radio static generation.
    """
    N = signal.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0 / sample_rate, device=signal.device)
    # Simple RC filter response: 1 / (1 + j*f/fc)
    response = 1.0 / (1.0 + (freqs / cutoff_hz) ** 2).sqrt()
    spectrum = torch.fft.rfft(signal, dim=-1)
    filtered = spectrum * response.unsqueeze(0).unsqueeze(0)
    return torch.fft.irfft(filtered, n=N, dim=-1)


def _one_pole_hpf_vectorized(signal, sample_rate, cutoff_hz):
    """Vectorized one-pole high-pass filter using FFT."""
    N = signal.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0 / sample_rate, device=signal.device)
    response = (freqs / cutoff_hz) / (1.0 + (freqs / cutoff_hz) ** 2).sqrt()
    # Avoid DC boost
    response[0] = 0.0
    spectrum = torch.fft.rfft(signal, dim=-1)
    filtered = spectrum * response.unsqueeze(0).unsqueeze(0)
    return torch.fft.irfft(filtered, n=N, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO EFFECTS — all fully vectorized
# ─────────────────────────────────────────────────────────────────────────────

def _tube_saturation(waveform, warmth=0.7):
    """Soft-clip saturation emulating vacuum tube amplifiers.

    Uses tanh waveshaping — the classic tube saturation curve.
    warmth: 0.0 = clean, 0.5 = subtle, 1.0 = heavy drive
    """
    if warmth <= 0.0:
        return waveform
    drive = 1.0 + warmth * 3.0  # 1x - 4x drive
    return torch.tanh(waveform * drive) / torch.tanh(torch.tensor(drive))


def _add_hum(waveform, sample_rate, hum_amount=0.05, freq=60.0):
    """Add 60Hz AC mains hum (or 50Hz for British vintage).

    Fully vectorized — pure torch tensor ops.
    """
    if hum_amount <= 0.0:
        return waveform

    B, C, N = waveform.shape
    t = torch.linspace(0, N / sample_rate, N, device=waveform.device)

    # Fundamental + harmonics (60, 120, 180 Hz)
    hum = (
        torch.sin(2 * math.pi * freq * t) * 1.0 +
        torch.sin(2 * math.pi * freq * 2 * t) * 0.5 +
        torch.sin(2 * math.pi * freq * 3 * t) * 0.25
    )
    hum = hum / hum.abs().max()
    hum = hum.unsqueeze(0).unsqueeze(0).expand(B, C, -1) * hum_amount

    return waveform + hum


def _add_vinyl_crackle(waveform, sample_rate, amount=0.1):
    """Add vinyl record surface noise — random pops and crackle.

    Fully vectorized FFT HPF — no Python for-loops over samples.
    """
    if amount <= 0.0:
        return waveform

    B, C, N = waveform.shape

    # Impulse pops — sparse, sharp transients (already vectorized)
    pop_density = amount * 0.002
    pops = torch.zeros(B, C, N, device=waveform.device)
    mask = torch.rand(B, C, N, device=waveform.device) < pop_density
    pops[mask] = (torch.rand(mask.sum(), device=waveform.device) * 2 - 1) * amount * 2

    # Surface noise — high-pass filtered white noise (vectorized)
    surface = torch.randn(B, C, N, device=waveform.device) * amount * 0.3
    surface = _one_pole_hpf_vectorized(surface, sample_rate, cutoff_hz=800)

    return waveform + pops + surface


def _add_radio_static(waveform, sample_rate, amount=0.15):
    """Add AM radio static — band-limited white noise.

    Fully vectorized FFT LPF — no Python for-loops over samples.
    """
    if amount <= 0.0:
        return waveform

    B, C, N = waveform.shape
    static = torch.randn(B, C, N, device=waveform.device) * amount * 0.4

    # Low-pass to remove extreme highs (vectorized)
    static = _one_pole_lpf_vectorized(static, sample_rate, cutoff_hz=6000)

    return waveform + static


def _broadcast_compress(waveform, threshold_db=-12.0, ratio=4.0):
    """Heavy broadcast-style dynamic range compression.

    1940s radio used extremely heavy compression to maintain consistent
    levels over AM broadcast. Fully vectorized — pure tensor ops.
    """
    threshold_linear = 10.0 ** (threshold_db / 20.0)

    envelope = waveform.abs()

    # Vectorized compression
    gain = torch.ones_like(envelope)
    above = envelope > threshold_linear
    gain[above] = threshold_linear + (envelope[above] - threshold_linear) / ratio

    compressed = torch.where(above, gain * torch.sign(waveform), waveform)

    # Makeup gain
    makeup = 10.0 ** (abs(threshold_db) / (2 * ratio * 20.0))
    compressed = compressed * makeup

    # Limiter
    peak = compressed.abs().max()
    if peak > 0.95:
        compressed = compressed * (0.95 / peak)

    return compressed


# ─────────────────────────────────────────────────────────────────────────────
# COMFYUI NODE
# ─────────────────────────────────────────────────────────────────────────────

class VintageRadioFilter:
    """Apply authentic old-time radio audio degradation effects."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "filter"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("vintage_audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "vintage_preset": (["off", "subtle", "authentic", "heavy_am",
                                    "war_era", "crystal_radio"], {
                    "default": "authentic",
                    "tooltip": "Quick preset — or use manual controls below"
                }),
                "bandpass_low_hz": ("INT", {
                    "default": 300, "min": 100, "max": 1000, "step": 50,
                    "tooltip": "High-pass cutoff (AM radio: ~300Hz)"
                }),
                "bandpass_high_hz": ("INT", {
                    "default": 6000, "min": 2000, "max": 12000, "step": 500,
                    "tooltip": "Low-pass cutoff (AM radio: ~5000Hz)"
                }),
                "tube_warmth": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Vacuum tube saturation warmth"
                }),
                "hum_60hz": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.3, "step": 0.01,
                    "tooltip": "60Hz AC mains hum level"
                }),
                "vinyl_crackle": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Vinyl surface noise and pops"
                }),
                "radio_static": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01,
                    "tooltip": "AM radio static/hiss level"
                }),
                "broadcast_compression": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply heavy broadcast-style compression"
                }),
                "hum_frequency": ("FLOAT", {
                    "default": 60.0, "min": 50.0, "max": 60.0, "step": 10.0,
                    "tooltip": "50Hz (European) or 60Hz (American) AC hum"
                }),
            },
        }

    # Preset parameter maps
    PRESETS = {
        "off":           {"tube_warmth": 0.0, "hum_60hz": 0.0, "vinyl_crackle": 0.0, "radio_static": 0.0, "bandpass_low_hz": 20, "bandpass_high_hz": 20000, "broadcast_compression": False},
        "subtle":        {"tube_warmth": 0.3, "hum_60hz": 0.02, "vinyl_crackle": 0.03, "radio_static": 0.05, "bandpass_low_hz": 200, "bandpass_high_hz": 8000, "broadcast_compression": False},
        "authentic":     {"tube_warmth": 0.7, "hum_60hz": 0.05, "vinyl_crackle": 0.10, "radio_static": 0.15, "bandpass_low_hz": 300, "bandpass_high_hz": 6000, "broadcast_compression": True},
        "heavy_am":      {"tube_warmth": 0.9, "hum_60hz": 0.08, "vinyl_crackle": 0.20, "radio_static": 0.25, "bandpass_low_hz": 400, "bandpass_high_hz": 4500, "broadcast_compression": True},
        "war_era":       {"tube_warmth": 1.0, "hum_60hz": 0.10, "vinyl_crackle": 0.25, "radio_static": 0.30, "bandpass_low_hz": 500, "bandpass_high_hz": 4000, "broadcast_compression": True},
        "crystal_radio": {"tube_warmth": 0.2, "hum_60hz": 0.15, "vinyl_crackle": 0.0, "radio_static": 0.40, "bandpass_low_hz": 600, "bandpass_high_hz": 3000, "broadcast_compression": False},
    }

    def filter(self, audio, vintage_preset="authentic",
               bandpass_low_hz=300, bandpass_high_hz=6000,
               tube_warmth=0.7, hum_60hz=0.05, vinyl_crackle=0.1,
               radio_static=0.15, broadcast_compression=True,
               hum_frequency=60.0):

        # Extract waveform
        if isinstance(audio, tuple):
            audio = audio[0]
        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            sample_rate = int(audio.get("sample_rate", 24000))
        else:
            waveform = audio
            sample_rate = 24000

        if waveform is None:
            log.warning("[VintageRadio] No waveform — passing through")
            return (audio,)

        # Ensure 3D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        # Apply preset overrides if not "off"
        if vintage_preset != "off" and vintage_preset in self.PRESETS:
            p = self.PRESETS[vintage_preset]
            tube_warmth = p["tube_warmth"]
            hum_60hz = p["hum_60hz"]
            vinyl_crackle = p["vinyl_crackle"]
            radio_static = p["radio_static"]
            bandpass_low_hz = p["bandpass_low_hz"]
            bandpass_high_hz = p["bandpass_high_hz"]
            broadcast_compression = p["broadcast_compression"]

        log.info(f"[VintageRadio] Applying '{vintage_preset}' filter "
                 f"(warmth={tube_warmth}, static={radio_static}, "
                 f"crackle={vinyl_crackle}, band={bandpass_low_hz}-{bandpass_high_hz}Hz)")

        # Apply processing chain in order — all vectorized
        waveform = _bandpass_filter(waveform, sample_rate, bandpass_low_hz, bandpass_high_hz)
        waveform = _tube_saturation(waveform, tube_warmth)
        waveform = _add_hum(waveform, sample_rate, hum_60hz, hum_frequency)
        waveform = _add_vinyl_crackle(waveform, sample_rate, vinyl_crackle)
        waveform = _add_radio_static(waveform, sample_rate, radio_static)

        if broadcast_compression:
            waveform = _broadcast_compress(waveform)

        # Final normalization
        peak = waveform.abs().max()
        if peak > 1e-8:
            waveform = waveform * (0.95 / peak)

        log.info(f"[VintageRadio] Done: {waveform.shape[-1]/sample_rate:.1f}s processed")
        return ({"waveform": waveform, "sample_rate": sample_rate},)
