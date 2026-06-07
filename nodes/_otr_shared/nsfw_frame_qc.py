"""Offline NSFW frame-QC sampler (A-S7) -- DEFAULT-OFF, warn-only.

SFW is an OTR invariant, so this is a belt-and-suspenders safety sampler, NOT a
gate that can ever stop a render. It is:

* DEFAULT-OFF -- it does nothing unless ``OTR_NSFW_FRAME_QC=1`` is set; when off,
  :func:`run_frame_qc` returns a disabled verdict and never flags.
* WARN-ONLY -- a flagged frame maps (via the shared retry taxonomy) to a WARN
  ``BlockClass`` decision: it appends a warning to the clip's ``qc`` record and
  may trigger an operator-enabled reseed/fallback, but it NEVER aborts the
  episode, NEVER discards rendered output, and NEVER touches the frozen audio.
* OFFLINE -- no network and no bundled model. The actual classifier is a
  PLUGGABLE local ``detector`` callable injected by the render node at the GPU
  smoke; the default :func:`abstain_detector` scores every frame 0.0 (no
  detector wired). The sampler OWNS the deterministic frame-sampling stride, the
  threshold compare, the verdict, and the warn-only routing -- all CPU-testable.
* ROBUST -- the QC must never break a render: a detector exception on a frame is
  caught and treated as an abstain (score 0.0), never propagated.

Cold-import clean (V-12): stdlib + the dep-free
:mod:`nodes._otr_shared.retry_taxonomy` only; no torch / model framework at
module scope. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from . import retry_taxonomy as _rt

#: The env flag that turns the sampler ON (default OFF -- SFW is an invariant so
#: the sampler is opt-in belt-and-suspenders, never a default gate).
QC_ENABLE_FLAG = "OTR_NSFW_FRAME_QC"

#: Default score (in [0, 1]) at/above which a sampled frame is flagged.
DEFAULT_NSFW_THRESHOLD = 0.85

#: Default number of frames sampled per clip (evenly strided).
DEFAULT_MAX_SAMPLES = 8


def qc_enabled(env=None) -> bool:
    """True iff the operator opted the sampler IN (``OTR_NSFW_FRAME_QC=1``)."""
    environ = os.environ if env is None else env
    return environ.get(QC_ENABLE_FLAG, "0") == "1"


def abstain_detector(frame) -> float:
    """The default local detector: no model wired -> score every frame 0.0.

    The render node injects a real LOCAL classifier at the GPU smoke; until then
    the sampler abstains (it can only ever WARN, never block, so abstaining is
    safe). Pure + offline.
    """
    return 0.0


def sample_frame_indices(frame_count: int,
                         max_samples: int = DEFAULT_MAX_SAMPLES) -> Tuple[int, ...]:
    """Deterministic, evenly-strided sample indices over ``[0, frame_count)``.

    Returns all indices when the clip is shorter than ``max_samples``; otherwise
    ``max_samples`` evenly spaced indices spanning the first to the last frame.
    Pure + deterministic (render-twice stable). Empty for a zero-length clip.
    """
    n = int(frame_count)
    if n <= 0:
        return ()
    m = max(1, int(max_samples))
    if n <= m:
        return tuple(range(n))
    step = (n - 1) / (m - 1)                  # evenly spaced, first..last inclusive
    return tuple(int(round(i * step)) for i in range(m))


@dataclass(frozen=True)
class QCVerdict:
    """The outcome of one clip's NSFW frame-QC pass.

    ``ran`` is False when the sampler is disabled. ``flagged`` (only possible
    when ``ran`` and a real detector scored a frame >= ``threshold``) is
    advisory: it WARNs, never blocks. ``detector_errors`` counts frames whose
    detector raised (treated as abstains) so a broken detector is visible
    without ever breaking the render.
    """

    ran: bool
    flagged: bool
    max_score: float
    frames_sampled: int
    threshold: float
    sample_indices: Tuple[int, ...] = ()
    detector_errors: int = 0
    reason: str = ""


def run_frame_qc(
    *,
    frame_count: int,
    get_frame: Optional[Callable[[int], object]] = None,
    detector: Optional[Callable[[object], float]] = None,
    threshold: float = DEFAULT_NSFW_THRESHOLD,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    env=None,
) -> QCVerdict:
    """Sample frames and score them with a LOCAL detector; warn-only.

    Returns a disabled verdict (``ran=False``) unless ``OTR_NSFW_FRAME_QC=1``.
    When enabled, samples ``max_samples`` evenly-strided frames, scores each with
    ``detector`` (default :func:`abstain_detector`), and flags the clip when the
    max score reaches ``threshold``. NEVER raises on a flag and NEVER lets a
    detector error escape (a raising detector counts as an abstain) -- the QC is
    advisory and must not break a render.
    """
    if not qc_enabled(env):
        return QCVerdict(ran=False, flagged=False, max_score=0.0,
                         frames_sampled=0, threshold=float(threshold),
                         reason="disabled (%s!=1)" % QC_ENABLE_FLAG)
    det = detector or abstain_detector
    indices = sample_frame_indices(frame_count, max_samples)
    max_score = 0.0
    errors = 0
    for idx in indices:
        try:
            frame = get_frame(idx) if get_frame is not None else idx
            score = float(det(frame))
        except Exception:                     # noqa: BLE001 - QC must not break a render
            errors += 1
            continue
        if score > max_score:
            max_score = score
    flagged = bool(indices) and max_score >= float(threshold)
    reason = "flagged" if flagged else "clear"
    if errors:
        reason += " (%d detector error(s) abstained)" % errors
    return QCVerdict(ran=True, flagged=flagged, max_score=max_score,
                     frames_sampled=len(indices), threshold=float(threshold),
                     sample_indices=indices, detector_errors=errors,
                     reason=reason)


def qc_decision(verdict: QCVerdict):
    """Map a flagged verdict to its WARN ``RetryDecision``, else ``None``.

    A flag classifies as :data:`FailureKind.NSFW` -> a WARN decision (warn-only,
    keeps output, never aborts / discards / touches audio). A clear or disabled
    verdict yields ``None`` (no action). The decision can never block a render --
    that is guaranteed by the taxonomy invariants.
    """
    if not verdict.flagged:
        return None
    return _rt.classify(_rt.FailureKind.NSFW)


def build_qc_warning(verdict: QCVerdict, *, shot_id: str, beat_id: str) -> dict:
    """Build the warn-only QC record appended to the clip's ``qc.warnings``.

    Records the score / threshold + the WARN block class so the audit trail
    shows the sampler fired without implying any blocking action was taken.
    """
    return {
        "kind": "nsfw_frame_qc",
        "shot_id": shot_id,
        "beat_id": beat_id,
        "flagged": bool(verdict.flagged),
        "max_score": float(verdict.max_score),
        "threshold": float(verdict.threshold),
        "frames_sampled": int(verdict.frames_sampled),
        "block_class": _rt.BlockClass.WARN.value,
        "action": "warn_only",
    }


def format_qc_warn_log(verdict: QCVerdict, shot_id: str) -> str:
    """A LOUD-but-non-blocking warn line for a flagged clip (episode + frozen
    audio untouched)."""
    return (
        "[OTR video] NSFW frame-QC WARN: shot=%s max_score=%.3f >= "
        "threshold=%.3f over %d frame(s) (warn-only; episode + frozen audio "
        "untouched)" % (shot_id, verdict.max_score, verdict.threshold,
                        verdict.frames_sampled)
    )


__all__ = [
    "QC_ENABLE_FLAG",
    "DEFAULT_NSFW_THRESHOLD",
    "DEFAULT_MAX_SAMPLES",
    "qc_enabled",
    "abstain_detector",
    "sample_frame_indices",
    "QCVerdict",
    "run_frame_qc",
    "qc_decision",
    "build_qc_warning",
    "format_qc_warn_log",
]
