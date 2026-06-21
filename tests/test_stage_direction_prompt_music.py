"""Chunk 5 -- composer prompt hardening + the S1 music re-stamp patch.

Source-level pins (no heavy node import): the writer's NON_VOICED branch must
take ONLY a genuine sfx_cue (the intent re-stamp that defeated S1 is removed),
and the composer system prompt must forbid a leading action prefix. UTF-8 no BOM.
"""
from __future__ import annotations

from pathlib import Path

_NODES = Path(__file__).resolve().parents[1] / "nodes"


def _read(name):
    return (_NODES / name).read_text(encoding="utf-8")


class TestMusicPatch:
    def test_non_voiced_takes_only_sfx_cue(self):
        src = _read("OTR_LedgerScriptWriter.py")
        # the S1-defeating intent fallback is gone
        assert '(beat.sfx_cue or beat.intent or "")' not in src
        # non-voiced text is the cue only
        assert '(beat.sfx_cue or "").strip()' in src


class TestPromptHardening:
    def test_prompt_forbids_leading_action(self):
        # collapse source whitespace so the wrapped prompt lines match
        src = " ".join(_read("_otr_line_composer.py").split())
        assert "NEVER prefix the line with an action description" in src
        assert "Start the output directly with the first spoken word" in src
