"""Shared PCM-WAV admission contract for portable cloning references.

IndexTTS2 resamples a reference to 16 kHz and runs a 400-sample Kaldi feature
window. A syntactically valid millisecond WAV therefore fails after provisioning.
OTR requires one second of complete PCM speech: deliberately above that kernel
floor, below every qualified local reference (the shortest is 5.55 seconds),
and still easy for a stranger to provide.
"""
from __future__ import annotations

import wave


MIN_REFERENCE_SECONDS = 1.0


def wav_problem(path: str) -> str:
    try:
        with wave.open(path, "rb") as handle:
            if handle.getcomptype() != "NONE":
                return "must be uncompressed PCM WAV"
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            if channels <= 0 or sample_width <= 0:
                return "has invalid PCM channel/sample width"
            if sample_rate <= 0 or frames <= 0:
                return "has no playable audio frames"
            expected = frames * channels * sample_width
            if len(handle.readframes(frames)) != expected:
                return "has a truncated PCM payload"
            duration = frames / float(sample_rate)
            if duration < MIN_REFERENCE_SECONDS:
                return ("is too short for IndexTTS2: %.3f s < %.3f s minimum" %
                        (duration, MIN_REFERENCE_SECONDS))
    except (OSError, EOFError, wave.Error) as exc:
        return "is not a readable WAV (%s)" % exc
    return ""


def require_usable_wav(path: str) -> None:
    problem = wav_problem(path)
    if problem:
        raise ValueError(problem)
