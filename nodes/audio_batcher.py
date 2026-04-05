r"""
Audio Batcher — Merge Multiple AUDIO Outputs Into a Single Batched Tensor
==========================================================================

ComfyUI only allows one cable per input socket. When the pipeline has
multiple TTS nodes (e.g., Bark for emotional characters, Parler for
the announcer), their individual AUDIO outputs need to be merged into
a single batched AUDIO dict before feeding into SceneSequencer.

This utility node accepts up to 8 AUDIO inputs, pads shorter clips to
the length of the longest, and stacks them along the batch dimension
(B in [B, C, T]).

v1.0  2026-04-04  Jeffrey Brick
"""

import logging
import numpy as np

log = logging.getLogger("OTR")


class AudioBatcher:
    """Merge 2–8 individual AUDIO clips into a single batched AUDIO dict.

    Each input is one rendered TTS line or SFX clip. The output is a
    single AUDIO dict whose waveform tensor has shape (N, C, T_max)
    where N = number of non-None inputs and T_max = longest clip.
    Shorter clips are zero-padded on the right.

    SceneSequencer's _extract_clips_from_audio() + _trim_trailing_silence()
    will strip the padding back off at consumption time.
    """

    CATEGORY = "OldTimeRadio/Utils"
    FUNCTION = "batch"
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("batched_audio", "clip_count")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "audio_7": ("AUDIO",),
                "audio_8": ("AUDIO",),
            },
        }

    def batch(self, audio_1=None, audio_2=None, audio_3=None, audio_4=None,
              audio_5=None, audio_6=None, audio_7=None, audio_8=None):
        import torch

        inputs = [audio_1, audio_2, audio_3, audio_4,
                  audio_5, audio_6, audio_7, audio_8]

        # Collect non-None AUDIO dicts
        clips = []
        sample_rate = 48000
        for inp in inputs:
            if inp is None:
                continue
            if isinstance(inp, dict):
                wf = inp.get("waveform")
                sr = inp.get("sample_rate", 48000)
            else:
                wf = inp
                sr = 48000

            if wf is None:
                continue

            # Normalize to 3D (B, C, T)
            if wf.dim() == 1:
                wf = wf.unsqueeze(0).unsqueeze(0)
            elif wf.dim() == 2:
                wf = wf.unsqueeze(0)

            # If input is already a batch, add each element separately
            for b in range(wf.shape[0]):
                clips.append(wf[b:b+1])  # keep as (1, C, T)

            sample_rate = sr  # use last seen sample rate

        if not clips:
            # Return 1 second of silence if no inputs
            silence = torch.zeros(1, 1, sample_rate, dtype=torch.float32)
            log.warning("[AudioBatcher] No audio inputs provided, returning silence")
            return ({"waveform": silence, "sample_rate": sample_rate}, 0)

        # Ensure all clips have the same number of channels
        max_channels = max(c.shape[1] for c in clips)
        normalized = []
        for c in clips:
            if c.shape[1] < max_channels:
                # Repeat mono to match channel count
                c = c.repeat(1, max_channels, 1)
            normalized.append(c)

        # Zero-pad to max length
        max_len = max(c.shape[2] for c in normalized)
        padded = []
        for c in normalized:
            if c.shape[2] < max_len:
                pad = torch.zeros(1, c.shape[1], max_len - c.shape[2],
                                  dtype=c.dtype, device=c.device)
                c = torch.cat([c, pad], dim=2)
            padded.append(c)

        # Stack along batch dimension: (N, C, T_max)
        batched = torch.cat(padded, dim=0)

        log.info(f"[AudioBatcher] Batched {len(padded)} clips → "
                 f"shape {list(batched.shape)}, sr={sample_rate}")

        return ({"waveform": batched, "sample_rate": sample_rate}, len(padded))
