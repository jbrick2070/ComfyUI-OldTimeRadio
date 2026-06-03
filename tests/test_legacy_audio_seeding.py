"""R0a (d): the legacy batch audio nodes (MusicGen / AudioGen) seed every RNG
from the FROZEN script_json (no parse) before their forward, so a render-twice
is reproducible (I-2). Bark (1a) and Kokoro (1b) dropped here in the audio
clean-break -- they are now per_line registry engines seeded via
deterministic_inference, not batch nodes. Source-proxy guard (the real
bit-identity check is operator GPU, step f), plus a functional check that
seed_all_rngs is deterministic."""
import random
from pathlib import Path

import pytest
import torch

from nodes._otr_determinism import seed_all_rngs

REPO = Path(__file__).resolve().parent.parent

CASES = {
    "musicgen_theme.py": "musicgen_legacy_v1",
    "batch_audiogen_generator.py": "audiogen_legacy_v1",
}


@pytest.mark.parametrize("fname,tag", list(CASES.items()))
def test_legacy_node_seeds_from_script_json(fname, tag):
    src = (REPO / "nodes" / fname).read_text(encoding="utf-8")
    assert "from ._otr_determinism import seed_all_rngs" in src, fname
    assert "from ._otr_resolved_request import _seed_to_int64" in src, fname
    assert f'seed_all_rngs(_seed_to_int64("{tag}", script_json))' in src, fname


def test_seed_all_rngs_deterministic():
    seed_all_rngs(123)
    a = (random.random(), torch.rand(3))
    seed_all_rngs(123)
    b = (random.random(), torch.rand(3))
    assert a[0] == b[0]
    assert torch.equal(a[1], b[1])


def test_seed_all_rngs_distinct_seeds_differ():
    seed_all_rngs(1)
    a = torch.rand(3)
    seed_all_rngs(2)
    b = torch.rand(3)
    assert not torch.equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
