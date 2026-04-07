"""
OTR_AudioEnhance — Broadcast-quality spatial audio enhancement.

Upscales mono TTS audio (typically 24kHz from Bark) to stereo at a target
sample rate with faux-spatial widening, bass warmth, and peak normalization.

Pipeline position:  SceneSequencer → AudioEnhance → EpisodeAssembler

Processing chain:
  1. Resample to target rate (sinc-interpolated via torchaudio — zero aliasing)
  2. Mono → Stereo duplication
  3. Low-frequency warmth (bass biquad shelf filter — no comb ripples)
  4. High-frequency cleanup — gentle LPF at 16kHz kills Bark "chirp" artifacts
  5. Haas-effect spatial widening (delay one channel 0.2–0.8 ms)
  6. Mid-side stereo decorrelation for image width
  7. Peak-normalize to target dBFS

All DSP is fully vectorized (no Python for-loops over samples).

Input:   AUDIO  (mono or stereo, any sample rate)
Output:  AUDIO  (stereo, target sample rate, spatially widened)

v1.0  2026-04-04  Jeffrey Brick
"""

import logging
import math
import torch

log = logging.getLogger("OTR.AudioEnhance")


# ── DSP building blocks (all vectorized) ─────────────────────────────────────

def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample waveform using sinc interpolation (torchaudio.transforms.Resample).

    Sinc interpolation mathematically reconstructs the analog waveform before
    resampling — zero foldover aliasing, unlike linear interpolation which
    draws straight lines between samples and creates high-frequency distortion.

    Expects input shape (B, C, N), returns (B, C, new_N).
    """
    if orig_sr == target_sr:
        return waveform

    import torchaudio

    # torchaudio.transforms.Resample uses a polyphase sinc filter internally.
    # We move the resampler to the waveform's device for GPU acceleration.
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr,
    ).to(waveform.device)

    # Resample expects (*, time) — our (B, C, N) shape works directly.
    return resampler(waveform.float())


def _mono_to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Convert mono (B,1,N) to stereo (B,2,N) by duplicating the channel."""
    if waveform.shape[1] >= 2:
        return waveform[:, :2, :]  # clip to stereo if multichannel
    return torch.cat([waveform, waveform], dim=1)


def _haas_delay(waveform: torch.Tensor, sample_rate: int,
                delay_ms: float = 0.4) -> torch.Tensor:
    """Apply Haas effect: delay the right channel by delay_ms milliseconds.

    Creates spatial width perception without loudness change.
    0.2–0.8 ms = natural spatial image for speech.
    """
    delay_samples = int(sample_rate * delay_ms / 1000.0)
    if delay_samples < 1 or waveform.shape[1] < 2:
        return waveform

    B, C, N = waveform.shape
    # Pad right channel at front, trim to original length
    right = waveform[:, 1:2, :]
    right_delayed = torch.nn.functional.pad(right, (delay_samples, 0))[:, :, :N]

    return torch.cat([waveform[:, 0:1, :], right_delayed], dim=1)


def _stereo_decorrelate(waveform: torch.Tensor, amount: float = 0.15) -> torch.Tensor:
    """Mid-side stereo width enhancement.

    Boosts the Side (L-R) component relative to Mid (L+R).
    amount: 0.0 = mono, 0.15 = subtle, 0.5 = very wide.
    """
    if waveform.shape[1] < 2:
        return waveform

    left = waveform[:, 0:1, :]
    right = waveform[:, 1:2, :]

    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    side = side * (1.0 + amount)

    new_left = mid + side
    new_right = mid - side
    return torch.cat([new_left, new_right], dim=1)


def _normalize(waveform: torch.Tensor, target_dbfs: float = -1.0) -> torch.Tensor:
    """Peak-normalize to target_dbfs.  -1.0 dBFS = tiny headroom."""
    peak = waveform.abs().max()
    if peak < 1e-8:
        return waveform
    target_linear = 10.0 ** (target_dbfs / 20.0)
    return waveform * (target_linear / peak)


def _apply_bass_warmth(waveform: torch.Tensor, sample_rate: int,
                       warmth: float = 0.1) -> torch.Tensor:
    """Add low-frequency warmth via biquad low-shelf filter.

    A biquad low-shelf is the industry-standard approach: smooth roll-on
    below cutoff, flat above, no phase ringing or comb artifacts.

    warmth: 0.0 = none, 0.1 = subtle broadcast tone (~3dB), 0.3 = noticeable (~9dB)

    NOTE (2026): torchaudio now supports CUDA for biquad/lfilter, but IIR filters
    are inherently sequential — GPU execution is slower than CPU due to poor
    parallelism on Tensor Cores. CPU bounce is the tactical win: minimal overhead
    (~microseconds per clip) while keeping the rest of DSP on GPU (RTX 5080).
    """
    if warmth <= 0.0:
        return waveform

    import torchaudio

    gain_db = warmth * 30.0
    orig_device = waveform.device
    orig_dtype = waveform.dtype

    # Tactical bounce to CPU for the IIR step (faster than GPU IIR on Blackwell)
    waveform_cpu = waveform.cpu()

    result = torchaudio.functional.bass_biquad(
        waveform_cpu, sample_rate,
        gain=gain_db,
        central_freq=200.0,
        Q=0.707,
    )

    # Back to original device & dtype (non_blocking for Blackwell pipelining)
    return result.to(orig_device, dtype=orig_dtype, non_blocking=True)


def _lowpass_16k(waveform: torch.Tensor, sample_rate: int,
                 cutoff_hz: float = 16000.0) -> torch.Tensor:
    """Gentle low-pass filter at cutoff_hz to kill Bark TTS "chirp" artifacts.

    Bark sometimes produces high-frequency ringing above 16kHz that gets
    amplified during 24k→48k upsampling. A smooth rolloff above 16kHz
    removes these digital chirps without affecting vocal clarity.

    Uses a windowed-sinc FIR filter (Hann window) for clean phase response.
    Kernel length adapts to sample rate for consistent rolloff steepness.
    """
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return waveform  # nothing to filter

    B, C, N = waveform.shape

    # Design a windowed-sinc LPF kernel
    # Kernel length: ~101 taps at 48kHz gives a gentle rolloff
    kernel_len = max(31, min(201, int(sample_rate / 500)))
    if kernel_len % 2 == 0:
        kernel_len += 1  # keep odd for symmetric

    half = kernel_len // 2
    dev = waveform.device
    n = torch.arange(-half, half + 1, dtype=torch.float32, device=dev)

    # Normalized cutoff frequency
    fc = cutoff_hz / sample_rate

    # Sinc function: sin(2*pi*fc*n) / (pi*n)
    # Handle n=0 case
    sinc = torch.where(
        n == 0,
        torch.tensor(2.0 * fc, device=dev),
        torch.sin(2.0 * math.pi * fc * n) / (math.pi * n)
    )

    # Hann window for smooth rolloff (no Gibbs ringing)
    window = 0.5 * (1.0 - torch.cos(2.0 * math.pi * torch.arange(kernel_len, dtype=torch.float32, device=dev) / (kernel_len - 1)))

    kernel = sinc * window
    kernel = kernel / kernel.sum()  # normalize to unity gain

    # Apply via conv1d (grouped convolution — same kernel per channel)
    kernel = kernel.view(1, 1, -1).repeat(C, 1, 1)  # already on waveform.device

    pad = half
    padded = torch.nn.functional.pad(waveform, (pad, pad), mode="reflect")
    # Reshape for grouped conv: (B*1, C, N) with groups=C
    filtered = torch.nn.functional.conv1d(padded.reshape(B, C, -1), kernel, groups=C)

    # Trim to exact original length
    if filtered.shape[-1] > N:
        filtered = filtered[:, :, :N]
    elif filtered.shape[-1] < N:
        filtered = torch.nn.functional.pad(filtered, (0, N - filtered.shape[-1]))

    return filtered


# ── ComfyUI Node ──────────────────────────────────────────────────────────────

class AudioEnhance:
    """Broadcast-quality spatial audio enhancement for TTS output."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "enhance"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("enhanced_audio",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "target_sample_rate": ("INT", {
                    "default": 48000, "min": 24000, "max": 96000, "step": 8000,
                    "tooltip": "Target sample rate in Hz (48000 = broadcast standard)"
                }),
                "spatial_width": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stereo width: 0=mono, 0.3=natural, 1.0=extreme"
                }),
                "haas_delay_ms": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Haas effect delay ms (0.2-0.8 = natural, 0=off)"
                }),
                "bass_warmth": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Low-freq warmth for broadcast tone (0=off, 0.1=subtle)"
                }),
                "lpf_cutoff_hz": ("FLOAT", {
                    "default": 16000.0, "min": 8000.0, "max": 24000.0, "step": 1000.0,
                    "tooltip": "Low-pass filter cutoff Hz — kills Bark chirp artifacts (0=off, 16000=default)"
                }),
                "normalize_dbfs": ("FLOAT", {
                    "default": -1.0, "min": -12.0, "max": 0.0, "step": 0.5,
                    "tooltip": "Peak normalization target dBFS (-1.0 = broadcast)"
                }),
            },
        }

    def enhance(self, audio, target_sample_rate=48000, spatial_width=0.3,
                haas_delay_ms=0.4, bass_warmth=0.1, lpf_cutoff_hz=16000.0,
                normalize_dbfs=-1.0):

        # ── Extract waveform & sample rate ──
        if isinstance(audio, tuple):
            audio = audio[0]

        if isinstance(audio, dict):
            waveform = audio.get("waveform")
            orig_sr = int(audio.get("sample_rate", 24000))
        else:
            waveform = audio
            orig_sr = 24000

        if waveform is None:
            log.warning("[OTR_AudioEnhance] No waveform data — passing through")
            return (audio,)

        # Ensure 3D: (batch, channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        orig_channels = waveform.shape[1]
        orig_samples = waveform.shape[-1]
        orig_duration = orig_samples / orig_sr

        log.info("[OTR_AudioEnhance] Input: %dHz %dch %d samples (%.1fs)",
                 orig_sr, orig_channels, orig_samples, orig_duration)

        # ── Move to GPU for DSP acceleration ──
        # All torchaudio DSP ops (resample, biquad, conv1d) run on CUDA when
        # tensors are on GPU. VRAM impact is negligible (<50 MB for long clips).
        _use_cuda = torch.cuda.is_available()
        if _use_cuda:
            waveform = waveform.to("cuda", non_blocking=True)
            log.info("[OTR_AudioEnhance] DSP on GPU (CUDA)")

        # ── Step 1: Resample ──
        waveform = _resample(waveform, orig_sr, target_sample_rate)
        log.info("[OTR_AudioEnhance] Resampled %dHz -> %dHz (%d samples)",
                 orig_sr, target_sample_rate, waveform.shape[-1])

        # ── Step 2: Mono → Stereo ──
        waveform = _mono_to_stereo(waveform)
        log.info("[OTR_AudioEnhance] Channels: %d -> %d",
                 orig_channels, waveform.shape[1])

        # ── Step 3: Bass warmth (before spatial — centered) ──
        if bass_warmth > 0:
            waveform = _apply_bass_warmth(waveform, target_sample_rate, bass_warmth)
            log.info("[OTR_AudioEnhance] Bass warmth %.2f applied", bass_warmth)

        # ── Step 4: High-frequency cleanup (kill Bark chirps) ──
        if lpf_cutoff_hz > 0 and lpf_cutoff_hz < target_sample_rate / 2:
            waveform = _lowpass_16k(waveform, target_sample_rate, lpf_cutoff_hz)
            log.info("[OTR_AudioEnhance] LPF at %.0fHz applied (chirp cleanup)",
                     lpf_cutoff_hz)

        # ── Step 5: Haas spatial delay ──
        if haas_delay_ms > 0:
            waveform = _haas_delay(waveform, target_sample_rate, haas_delay_ms)
            log.info("[OTR_AudioEnhance] Haas delay %.1fms applied", haas_delay_ms)

        # ── Step 6: Mid-side stereo widening ──
        if spatial_width > 0:
            waveform = _stereo_decorrelate(waveform, spatial_width)
            log.info("[OTR_AudioEnhance] Stereo width %.2f applied", spatial_width)

        # ── Step 7: Peak normalize skipped — moved to EpisodeAssembler ──
        # Normalizing here (before crossfades) caused clipping during segment
        # overlaps. Final -1.0 dBFS pass now runs post-crossfade in Assembler.
        log.info("[OTR_AudioEnhance] Normalize deferred to EpisodeAssembler (post-crossfade)")

        # ── Move back to CPU for ComfyUI pipeline ──
        if _use_cuda:
            waveform = waveform.cpu()

        # ── Verify stereo output ──
        assert waveform.shape[1] == 2, (
            f"[OTR_AudioEnhance] BUG: expected 2 channels, got {waveform.shape[1]}"
        )

        enhanced = {"waveform": waveform, "sample_rate": target_sample_rate}

        final_samples = waveform.shape[-1]
        final_duration = final_samples / target_sample_rate
        log.info("[OTR_AudioEnhance] Output: %dHz %dch %d samples (%.1fs) "
                 "spatial=%.2f Haas=%.1fms warmth=%.2f",
                 target_sample_rate, waveform.shape[1], final_samples,
                 final_duration, spatial_width, haas_delay_ms, bass_warmth)

        return (enhanced,)
