"""Bark artifact B1 -- speech-only dialogue mode + first-line anchor gate.

Prevents the high-pitched non-speech squeal at the SOURCE:
  * _clean_text_for_bark(text, speech_only=True) drops the high-risk tokens
    [music]/[whistles]/[sneezes]/[gasps] (and the asterisk forms that create
    them); keeps [laughs]/[sighs].
  * _generate_single_line(..., inject_first_line_anchor=False) skips the
    first-line [clears throat] prepend.
  * eng_bark.generate_voice (char_voice = dialogue) resolves both from env
    (OTR_BARK_SPEECH_ONLY default 1, OTR_BARK_DISABLE_THROAT_CLEAR default 1)
    and passes them as explicit kwargs.

Pure / source-level; no torch model, no GPU. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nodes._otr_bark_lib as bl  # noqa: E402
from nodes._otr_bark_lib import (  # noqa: E402
    _clean_text_for_bark,
    _env_flag,
    _resolve_bark_inject_anchor,
    _resolve_bark_speech_only,
)
from nodes._otr_audio_engines.eng_bark import BarkEngine  # noqa: E402

_PREFIX = "[VOICE: HAYES, male, 40s, calm, low] "
_RISK = ["[music]", "[whistles]", "[sneezes]", "[gasps]"]
_KEEP = ["[laughs]", "[sighs]"]


# ---------------------------------------------------------------------------
# _clean_text_for_bark speech_only behavior
# ---------------------------------------------------------------------------


class TestCleanTextSpeechOnly:
    def test_risk_tokens_stripped_under_speech_only(self):
        for tok in _RISK:
            out = _clean_text_for_bark(f"{_PREFIX}{tok} Speech here.",
                                       speech_only=True)
            assert tok not in out, f"{tok} should be dropped under speech_only"
            assert "Speech here." in out

    def test_risk_tokens_survive_when_not_speech_only(self):
        # Default (speech_only=False) keeps the legacy whitelist intact.
        for tok in _RISK:
            out = _clean_text_for_bark(f"{_PREFIX}{tok} Speech here.")
            assert tok in out, f"{tok} must survive when speech_only is False"

    def test_emotive_tokens_survive_under_speech_only(self):
        for tok in _KEEP:
            out = _clean_text_for_bark(f"{_PREFIX}{tok} Speech here.",
                                       speech_only=True)
            assert tok in out, f"{tok} must be kept under speech_only"

    def test_asterisk_gasps_dropped_but_laughs_kept_under_speech_only(self):
        # *gasps* -> [gasps] in step 3, then stripped by the shrunk whitelist;
        # *laughs* -> [laughs] survives.
        out = _clean_text_for_bark(f"{_PREFIX}*gasps* Wait *laughs* now.",
                                   speech_only=True)
        assert "[gasps]" not in out
        assert "[laughs]" in out

    def test_default_keeps_all_thirteen(self):
        # Parity guard: default behavior is unchanged (scene_sequencer copy +
        # test_core both call with no kwarg).
        all_tokens = [
            "[laughter]", "[laughs]", "[sighs]", "[music]", "[gasps]",
            "[clears throat]", "[coughs]", "[pants]", "[sobs]", "[grunts]",
            "[groans]", "[whistles]", "[sneezes]",
        ]
        for tok in all_tokens:
            assert tok in _clean_text_for_bark(f"{_PREFIX}{tok} Speech.")

    def test_clears_throat_literal_survives_under_speech_only(self):
        # The throat-clear is gated by the anchor flag, NOT the token list;
        # a literal one in the text is still a valid token.
        out = _clean_text_for_bark(f"{_PREFIX}[clears throat] Hello.",
                                   speech_only=True)
        assert "[clears throat]" in out


# ---------------------------------------------------------------------------
# env resolvers
# ---------------------------------------------------------------------------


class TestEnvResolvers:
    def test_env_flag_parsing(self, monkeypatch):
        monkeypatch.delenv("OTR_TEST_FLAG", raising=False)
        assert _env_flag("OTR_TEST_FLAG", True) is True
        assert _env_flag("OTR_TEST_FLAG", False) is False
        for v in ("1", "true", "YES", "On"):
            monkeypatch.setenv("OTR_TEST_FLAG", v)
            assert _env_flag("OTR_TEST_FLAG", False) is True
        for v in ("0", "false", "no", "off", "garbage"):
            monkeypatch.setenv("OTR_TEST_FLAG", v)
            assert _env_flag("OTR_TEST_FLAG", True) is False

    def test_speech_only_default_on(self, monkeypatch):
        monkeypatch.delenv("OTR_BARK_SPEECH_ONLY", raising=False)
        assert _resolve_bark_speech_only() is True
        monkeypatch.setenv("OTR_BARK_SPEECH_ONLY", "0")
        assert _resolve_bark_speech_only() is False

    def test_inject_anchor_default_off(self, monkeypatch):
        # default: throat-clear DISABLED -> anchor OFF
        monkeypatch.delenv("OTR_BARK_DISABLE_THROAT_CLEAR", raising=False)
        assert _resolve_bark_inject_anchor() is False
        # operator can restore the anchor
        monkeypatch.setenv("OTR_BARK_DISABLE_THROAT_CLEAR", "0")
        assert _resolve_bark_inject_anchor() is True
        monkeypatch.setenv("OTR_BARK_DISABLE_THROAT_CLEAR", "1")
        assert _resolve_bark_inject_anchor() is False


# ---------------------------------------------------------------------------
# _generate_single_line plumbing (signature + source contract)
# ---------------------------------------------------------------------------


class TestGenerateSingleLineGating:
    def test_new_params_are_keyword_only_with_safe_defaults(self):
        sig = inspect.signature(bl._generate_single_line)
        for name, default in (("inject_first_line_anchor", True),
                              ("speech_only", False)):
            p = sig.parameters[name]
            assert p.kind == inspect.Parameter.KEYWORD_ONLY
            assert p.default is default

    def test_speech_only_threaded_into_cleaner(self):
        src = inspect.getsource(bl._generate_single_line)
        assert "_clean_text_for_bark(text, speech_only=speech_only)" in src

    def test_anchor_prepend_is_gated(self):
        src = inspect.getsource(bl._generate_single_line)
        assert "if is_first_line and inject_first_line_anchor:" in src


# ---------------------------------------------------------------------------
# eng_bark wiring (source contract -- mirrors test_bark_voice_stage_temps)
# ---------------------------------------------------------------------------


class TestEngBarkWiring:
    def test_generate_voice_resolves_and_passes_flags(self):
        src = inspect.getsource(BarkEngine.generate_voice)
        assert "_resolve_bark_speech_only()" in src
        assert "_resolve_bark_inject_anchor()" in src
        assert "inject_first_line_anchor=inject_first_line_anchor" in src
        assert "speech_only=speech_only" in src


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
