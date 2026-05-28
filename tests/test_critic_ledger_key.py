"""Build 1 (2026-05-28) -- pin the canonical whole-episode critic ledger key.

Decision (Build 1, measurement integrity): there is exactly one
whole-episode critic and it is stamped under ``meta.stage7_shadow_critic``.
The key ``meta.whole_episode_critic`` is never written anywhere -- it was
only a stale docstring label on ``CriticResult.to_dict``. The 3.70 baseline
mean came from ``meta.stage7_shadow_critic.mean_score``, so that key stays
canonical (no behavioral rename) and the stale docstring was corrected to
match. These tests pin that decision so a future rename or doc-drift is
caught by the regression rather than silently re-splitting the score basis.

Import-free on purpose: scanning source keeps this deterministic and avoids
pulling the critic module's heavy runtime imports into the test session.
"""

from __future__ import annotations

from pathlib import Path

_NODES = Path(__file__).resolve().parent.parent / "nodes"


def _read(name: str) -> str:
    return (_NODES / name).read_text(encoding="utf-8")


def test_freeze_cascade_stamps_canonical_key():
    """The live ledger stamp uses stage7_shadow_critic, never whole_episode_critic."""
    src = _read("_otr_freeze_cascade.py")
    assert 'meta["stage7_shadow_critic"]' in src
    assert 'meta["whole_episode_critic"]' not in src


def test_critic_to_dict_docstring_matches_live_key():
    """CriticResult.to_dict docstring names the real ledger key."""
    src = _read("_otr_whole_episode_critic.py")
    assert "meta.stage7_shadow_critic ledger stamp" in src
    assert "meta.whole_episode_critic ledger stamp" not in src
