"""BUG-LOCAL-269: the cast RNG is decoupled from the `seed` widget.

The cast must be truly random every episode, not pinned by a fixed
seed. A fixed `seed` reproduced ONE cast forever (seed 42 always rolled
HAYES VANCE / GULLIVER REEVES / JIMBO BLACK). These tests pin the
contract of `_resolve_cast_rng_seed`: OS entropy by default, with the
OTR_CAST_SEED env var as the C7 audio-regression reproducibility path.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nodes.OTR_LedgerScriptWriter import _resolve_cast_rng_seed


def test_env_override_pins_the_seed(monkeypatch):
    """OTR_CAST_SEED forces a fixed seed -- the C7 reproducibility path."""
    monkeypatch.setenv("OTR_CAST_SEED", "12345")
    seed, source = _resolve_cast_rng_seed()
    assert seed == 12345
    assert source == "OTR_CAST_SEED override"


def test_no_env_uses_os_entropy(monkeypatch):
    """With no override, the cast seed is drawn from OS entropy."""
    monkeypatch.delenv("OTR_CAST_SEED", raising=False)
    seed, source = _resolve_cast_rng_seed()
    assert isinstance(seed, int)
    assert 0 <= seed < 2 ** 32
    assert source == "OS entropy"


def test_os_entropy_varies_across_calls(monkeypatch):
    """True randomization: separate episodes get different cast seeds.

    Two 32-bit OS-entropy draws colliding is ~1 in 4 billion -- a
    repeat across 20 calls means the cast is not actually randomized
    (the BUG-LOCAL-269 regression).
    """
    monkeypatch.delenv("OTR_CAST_SEED", raising=False)
    seeds = {_resolve_cast_rng_seed()[0] for _ in range(20)}
    assert len(seeds) == 20, (
        "cast RNG seed repeated across calls -- cast is not truly random"
    )


def test_blank_env_falls_back_to_os_entropy(monkeypatch):
    """A blank / whitespace OTR_CAST_SEED is treated as unset."""
    monkeypatch.setenv("OTR_CAST_SEED", "   ")
    seed, source = _resolve_cast_rng_seed()
    assert source == "OS entropy"
    assert isinstance(seed, int)
