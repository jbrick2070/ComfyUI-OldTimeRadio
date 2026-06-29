"""KILL 2 -- C3: the dynamic NEWS CODA (compose_news_coda).

The LLM writes ONLY a short bridge clause from the premise + the safe intro tone
(never the outcome / the news fact); the real news_close_brief is APPENDED
deterministically, so the weak local model can never write the fact wrong. A
sha256(cast_seed) rotating-pool floor guarantees a valid close even when every
bridge attempt is rejected. compose_announcer_outro is left UNTOUCHED, so the
off / no-news-brief path is byte-identical.

Pure Python; no GPU, no LLM, no network.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_line_composer as LC  # noqa: E402


def _make_fn(reply, capture):
    def _fn(messages, **kw):
        if capture is not None:
            capture.append(messages)
        return reply
    return _fn


_FACT = "NASA confirmed the Mars probe landed safely on June 1"
_PREMISE = "A keeper waits for a ship while a strange signal pulses offshore"
_INTRO = "The lamp turns over black water as the keeper waits."


class TestComposeNewsCoda:
    def test_bridge_path_appends_fact_and_starves_outcome(self):
        cap = []
        fn = _make_fn("A tale of a lighthouse keeper and a signal in the dark", cap)
        res = LC.compose_news_coda(
            creative_fn=fn, news_close_brief=_FACT, premise=_PREMISE,
            intro_text=_INTRO, cast_seed=42,
        )
        assert res.compose_flags == ("news_coda_bridge",)
        # the bridge clause turns with a colon, then the real fact is appended
        assert res.text.startswith(
            "A tale of a lighthouse keeper and a signal in the dark:"
        )
        assert _FACT in res.text
        # the LLM saw NEITHER the fact NOR the outcome -- only the premise + intro
        sysmsg, usermsg = cap[0][0]["content"], cap[0][1]["content"]
        assert "NASA" not in usermsg and "Mars" not in usermsg
        assert "landed safely" not in usermsg.lower()
        assert "A keeper waits for a ship" in usermsg          # premise passed
        assert "lamp turns over black water" in usermsg.lower()  # safe intro tone
        # story-quality v2 (baked in): the sent system message is the base
        # _NEWS_CODA_SYSTEM contract with the few-shot _NEWS_CODA_SYSTEM_V2_EXAMPLES
        # appended (the base constant is never mutated).
        assert sysmsg.startswith(LC._NEWS_CODA_SYSTEM)
        assert "Examples (tonight's tale -> your bridge clause):" in sysmsg

    def test_generic_opener_retries_then_floor(self):
        cap = []
        fn = _make_fn("And now, the real world calls out to us", cap)
        res = LC.compose_news_coda(
            creative_fn=fn, news_close_brief="The dam held through the night",
            premise=_PREMISE, cast_seed=7,
        )
        assert res.compose_flags == ("news_coda_fallback", "news_coda_bridge_invalid")
        assert len(cap) == 2  # bridge + one reroll, both rejected
        assert any(res.text.startswith(p) for p in LC.NEWS_CODA_POOL)
        assert "The dam held through the night" in res.text

    def test_fallback_floor_deterministic_by_cast_seed(self):
        a = LC.compose_news_coda(
            creative_fn=_make_fn("[bad bracket]", None),
            news_close_brief="A real fact here tonight", premise="p", cast_seed=99,
        )
        b = LC.compose_news_coda(
            creative_fn=_make_fn("[bad bracket]", None),
            news_close_brief="A real fact here tonight", premise="p", cast_seed=99,
        )
        assert a.text == b.text
        assert a.text != ""
        assert a.compose_flags == ("news_coda_fallback", "news_coda_bridge_invalid")

    def test_fallback_floor_varies_by_seed(self):
        prefixes = set()
        for seed in range(30):
            r = LC.compose_news_coda(
                creative_fn=_make_fn("[x]", None),
                news_close_brief="A fact", premise="p", cast_seed=seed,
            )
            prefixes.add(next(p for p in LC.NEWS_CODA_POOL if r.text.startswith(p)))
        assert len(prefixes) > 1  # the rotating pool actually rotates

    def test_empty_brief_returns_no_brief(self):
        res = LC.compose_news_coda(
            creative_fn=_make_fn("anything", None),
            news_close_brief="", premise="p", cast_seed=1,
        )
        assert res.text == ""
        assert res.compose_flags == ("news_coda_no_brief",)

    def test_reroll_prompt_differs_from_first(self):
        cap = []
        LC.compose_news_coda(
            creative_fn=_make_fn("meanwhile back at the ranch tonight we see", cap),
            news_close_brief="A fact", premise=_PREMISE, cast_seed=3,
        )
        assert "Attempt 2" in cap[1][1]["content"]
        assert "Attempt 2" not in cap[0][1]["content"]


class TestValidateNewsCodaBridge:
    def test_accepts_trailing_colon(self):
        ok, c = LC.validate_news_coda_bridge("A turn toward the harbor tonight:")
        assert ok and c.endswith(":")

    def test_rejects_leading_speaker_label(self):
        assert not LC.validate_news_coda_bridge("ANNOUNCER: the real tale")[0]

    @pytest.mark.parametrize("g", [
        "And now we turn", "Meanwhile across town", "In reality the dam",
        "The real story is", "But in the real world tonight",
    ])
    def test_rejects_generic_openers(self, g):
        assert not LC.validate_news_coda_bridge(g)[0]

    def test_rejects_too_long(self):
        assert not LC.validate_news_coda_bridge("x " * 60)[0]

    def test_rejects_brackets_and_multiline_and_empty(self):
        assert not LC.validate_news_coda_bridge("a [cue] turn")[0]
        assert not LC.validate_news_coda_bridge("line one\nline two")[0]
        assert not LC.validate_news_coda_bridge("")[0]


class TestOutroUntouched:
    def test_compose_announcer_outro_signature_unchanged(self):
        params = set(inspect.signature(LC.compose_announcer_outro).parameters)
        assert "story_scaffold" not in params
        assert "coda_lead_in" not in params
        assert "climax_character_line" not in params
        assert {
            "creative_fn", "script_brief", "news_close_brief", "intro_text",
            "ending_change", "final_character_line",
        } <= params


# ---------------------------------------------------------------------------
# Writer wiring (source-level; run() needs the full model stack).
# ---------------------------------------------------------------------------
class TestWriterC3Wiring:
    SRC = (
        Path(__file__).resolve().parents[1] / "nodes" / "OTR_LedgerScriptWriter.py"
    ).read_text(encoding="utf-8")

    def test_early_coda_branch_then_else_outro_in_order(self):
        src = self.SRC
        i_branch = src.index("if _style_grammar_on and nc_brief.strip():")
        i_coda = src.index("_OTRLC.compose_news_coda(", i_branch)
        i_else = src.index("\n            else:", i_coda)
        i_outro = src.index("_OTRLC.compose_announcer_outro(", i_else)
        assert i_branch < i_coda < i_else < i_outro

    def test_coda_passes_premise_not_script_brief(self):
        i_coda = self.SRC.index("_OTRLC.compose_news_coda(")
        seg = self.SRC[i_coda:i_coda + 400]
        assert 'premise=str(getattr(outline, "premise"' in seg
        assert "script_brief" not in seg          # the bridge never sees the brief

    def test_no_brief_marker_via_frozen_replace(self):
        assert "news_coda_no_brief" in self.SRC
        assert "_dc.replace(" in self.SRC


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
