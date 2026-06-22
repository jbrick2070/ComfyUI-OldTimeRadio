"""Story-Quality R2 C3 -- contrasting speech signatures + hard register clause.

diversify_speech_signatures makes every cast row's register distinct (keeps a
good LLM signature; replaces empty/default/duplicate from a deterministic
contrasting pool). The composer CRAFT prompt promotes the register to a hard
constraint. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_casting import diversify_speech_signatures, _norm_sig  # noqa: E402

_NODES = Path(__file__).resolve().parents[1] / "nodes"


def _sigs(cast):
    return [_norm_sig(r["speech_signature"]) for r in cast]


class TestDiversify:
    def test_duplicate_defaults_become_distinct(self):
        cast = [
            {"char_id": "c01", "speech_signature": "plain spoken"},
            {"char_id": "c02", "speech_signature": "plain spoken"},
            {"char_id": "c03", "speech_signature": ""},
        ]
        diversify_speech_signatures(cast, seed=0)
        sigs = _sigs(cast)
        assert len(set(sigs)) == 3            # all distinct
        assert all(s for s in sigs)           # none empty

    def test_keeps_distinct_llm_signatures(self):
        cast = [
            {"char_id": "c01", "speech_signature": "clipped, military"},
            {"char_id": "c02", "speech_signature": "warm, folksy"},
        ]
        diversify_speech_signatures(cast, seed=0)
        assert _norm_sig(cast[0]["speech_signature"]) == "clipped, military"
        assert _norm_sig(cast[1]["speech_signature"]) == "warm, folksy"

    def test_collision_with_llm_sig_reassigned(self):
        cast = [
            {"char_id": "c01", "speech_signature": "gruff, few words"},
            {"char_id": "c02", "speech_signature": "gruff, few words"},
        ]
        diversify_speech_signatures(cast, seed=0)
        assert _sigs(cast)[0] != _sigs(cast)[1]

    def test_deterministic_by_seed(self):
        def _mk():
            return [{"char_id": f"c{i}", "speech_signature": "plain spoken"}
                    for i in range(3)]
        a, b = _mk(), _mk()
        diversify_speech_signatures(a, seed=42)
        diversify_speech_signatures(b, seed=42)
        assert _sigs(a) == _sigs(b)

    def test_never_raises(self):
        assert diversify_speech_signatures(None) is None or True
        diversify_speech_signatures([{"char_id": "c01"}], seed=1)  # no key


class TestComposerRegisterConstraint:
    def test_craft_prompt_has_register_clause(self):
        src = " ".join((_NODES / "_otr_line_composer.py").read_text(
            encoding="utf-8").split())
        assert "Match the speaker's stated speech register" in src


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
