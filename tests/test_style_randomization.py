"""BUG-LOCAL-270: the style-picker RNG is decoupled from the `seed` widget.

Twin of BUG-LOCAL-269 (cast). A fixed `seed` made the style picker
sample the identical 5 inspiration "seed flavors" every episode. These
tests pin the contract of `_resolve_style_rng_seed`: OS entropy by
default, with the OTR_STYLE_SEED env var as the C7 audio-regression
reproducibility path.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nodes.OTR_LedgerScriptWriter import _resolve_style_rng_seed


def test_env_override_pins_the_seed(monkeypatch):
    """OTR_STYLE_SEED forces a fixed seed -- the C7 reproducibility path."""
    monkeypatch.setenv("OTR_STYLE_SEED", "777")
    seed, source = _resolve_style_rng_seed()
    assert seed == 777
    assert source == "OTR_STYLE_SEED override"


def test_no_env_uses_os_entropy(monkeypatch):
    """With no override, the style seed is drawn from OS entropy."""
    monkeypatch.delenv("OTR_STYLE_SEED", raising=False)
    seed, source = _resolve_style_rng_seed()
    assert isinstance(seed, int)
    assert 0 <= seed < 2 ** 32
    assert source == "OS entropy"


def test_os_entropy_varies_across_calls(monkeypatch):
    """True randomization: separate episodes sample different flavors.

    A repeat across 20 calls means the seed-flavor sampling is not
    actually randomized (the BUG-LOCAL-270 regression).
    """
    monkeypatch.delenv("OTR_STYLE_SEED", raising=False)
    seeds = {_resolve_style_rng_seed()[0] for _ in range(20)}
    assert len(seeds) == 20, (
        "style RNG seed repeated -- seed-flavor sampling is not random"
    )


def test_blank_env_falls_back_to_os_entropy(monkeypatch):
    """A blank / whitespace OTR_STYLE_SEED is treated as unset."""
    monkeypatch.setenv("OTR_STYLE_SEED", "   ")
    seed, source = _resolve_style_rng_seed()
    assert source == "OS entropy"
    assert isinstance(seed, int)
