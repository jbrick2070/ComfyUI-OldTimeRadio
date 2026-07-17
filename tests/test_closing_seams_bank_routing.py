"""2026-07-09 source-route QA F1 regression -- the closing layer is bank-routed.

Before this fix the four closing seams (coda_system, announcer_intro_system,
announcer_intro_safe_system, announcer_outro_system) were authored in every
bank's pack, REQUIRED by banks.json, validated present by the registry sweep --
and never routed: every bank's episode closed with the science-lane constants
("the real news report is added AFTER your clause"), and the title pass told
the LLM it was titling "a sci-fi radio drama" for every bank.

Pins:
  1. router: the four closing phases resolve to each bank's own pack seam
     text, and the science values byte-match the legacy constants/assembly.
  2. composer: compose_news_coda / compose_announcer_intro /
     compose_announcer_outro actually SEND the bank's pack text as the
     system message (captured via a stub creative_fn).
  3. title: _generate_title_from_script frames per the bank's
     banks.json title_form_label (first live consumer of that field).
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes._otr_creative_prompt_router import (  # noqa: E402
    resolve_creative_system_prompt,
)
from nodes._otr_story_pack import get_pack_prompt  # noqa: E402
from nodes._otr_story_routing import resolve_story_pack  # noqa: E402
from nodes import _otr_line_composer as LC  # noqa: E402


RUNNABLE_BANKS = (
    "media_archive",
    "public_domain_story_v3",
    "shakespeare_v3",
)
CLOSING_PHASES = (
    "coda_system",
    "announcer_intro_system",
    "announcer_intro_safe_system",
    "announcer_outro_system",
)

# Valid stub outputs (must clear the composer validators).
_VALID_BRIDGE = "Beyond tonight's flickering reel"
_VALID_INTRO = (
    "Tonight, from a shelf of forgotten reels, a quiet archive hums "
    "with one question that will not rest."
)
_VALID_OUTRO = (
    "The reel is shelved once more, its label true at last, and the "
    "archive light burns a little steadier tonight."
)


def _capturing_fn(reply: str):
    """A stub creative_fn that records every messages array it is
    called with and returns `reply`."""
    calls: list = []

    def fn(messages, **kwargs):
        calls.append(messages)
        return reply

    return fn, calls


def _sysmsg(calls) -> str:
    assert calls, "creative_fn was never called"
    return calls[0][0]["content"]


# ---------------------------------------------------------------------------
# 1. Router resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bank", RUNNABLE_BANKS)
@pytest.mark.parametrize("phase", CLOSING_PHASES)
def test_closing_phase_resolves_the_banks_own_pack_seam(bank, phase):
    """Each closing phase resolves to that bank's own pack seam text."""
    resolved = resolve_creative_system_prompt(
        None, phase=phase, source_bank_id=bank
    )
    expected = get_pack_prompt(resolve_story_pack(bank), phase)
    assert resolved == expected


@pytest.mark.parametrize("bank", RUNNABLE_BANKS)
def test_legacy_coda_seam_is_not_a_news_report(bank):
    """The retired science lane's 'real news report' coda framing must not
    survive in any legacy lane's coda seam (the exact failure QA F1 found:
    'real news report' framing leaking onto a source note)."""
    other = resolve_creative_system_prompt(
        None, phase="coda_system", source_bank_id=bank
    )
    assert "real news report" not in other


def test_non_science_coda_seams_keep_the_bridge_contract():
    """The re-authored PD/Shakespeare coda seams (and media's) must keep
    the KILL-2 bridge mechanics the composer enforces: a SHORT pivot
    clause; the real note is appended by the producer."""
    for bank in ("media_archive", "public_domain_story_v3", "shakespeare_v3"):
        seam = resolve_creative_system_prompt(
            None, phase="coda_system", source_bank_id=bank
        )
        assert "bridge clause" in seam, bank
        assert "AFTER your clause" in seam, bank
        assert "ending with a colon" in seam, bank


# ---------------------------------------------------------------------------
# 2. Composer sends the routed text
# ---------------------------------------------------------------------------


def test_compose_news_coda_sends_bank_coda_system():
    fn, calls = _capturing_fn(_VALID_BRIDGE)
    res = LC.compose_news_coda(
        creative_fn=fn,
        news_close_brief=(
            "Adapted from a Folger Shakespeare Library scene under "
            "noncommercial terms."
        ),
        premise="Three strangers on a storm-lit heath make a promise.",
        source_bank_id="shakespeare_v3",
    )
    expected = get_pack_prompt(
        resolve_story_pack("shakespeare_v3"), "coda_system"
    )
    assert _sysmsg(calls) == expected
    assert res.text.startswith(_VALID_BRIDGE)


def test_compose_announcer_intro_sends_bank_intro_system():
    fn, calls = _capturing_fn(_VALID_INTRO)
    LC.compose_announcer_intro(
        creative_fn=fn,
        script_brief=(
            "A projectionist races to save a brittle newsreel before "
            "its final show."
        ),
        source_bank_id="media_archive",
    )
    expected = get_pack_prompt(
        resolve_story_pack("media_archive"), "announcer_intro_system"
    )
    assert _sysmsg(calls) == expected


def test_compose_announcer_intro_safe_sends_bank_safe_system():
    fn, calls = _capturing_fn(_VALID_INTRO)
    brief = types.SimpleNamespace(
        cast=("MARA",),
        era="1947",
        time_of_day="night",
        setting="a county records archive",
        opening_status_quo="A mislabeled reel waits on the top shelf.",
    )
    LC.compose_announcer_intro(
        creative_fn=fn,
        script_brief="",
        story_scaffold=True,
        safe_open_brief=brief,
        source_bank_id="public_domain_story_v3",
    )
    expected = get_pack_prompt(
        resolve_story_pack("public_domain_story_v3"),
        "announcer_intro_safe_system",
    )
    assert _sysmsg(calls) == expected


def test_compose_announcer_outro_sends_bank_outro_system():
    fn, calls = _capturing_fn(_VALID_OUTRO)
    LC.compose_announcer_outro(
        creative_fn=fn,
        script_brief=(
            "A traveler returns from a machine-made distance with one "
            "warning."
        ),
        news_close_brief=(
            "Adapted from a public-domain novel; characters and ending "
            "preserved."
        ),
        intro_text="",
        source_bank_id="public_domain_story_v3",
    )
    expected = get_pack_prompt(
        resolve_story_pack("public_domain_story_v3"),
        "announcer_outro_system",
    )
    assert _sysmsg(calls) == expected


# ---------------------------------------------------------------------------
# 3. Bank-aware title framing
# ---------------------------------------------------------------------------


_TITLE_REPLY = (
    "DETAILS:\n- a brittle reel\n- a penciled card\n- a dark booth\n"
    "CANDIDATES:\nThe Brittle Reel\nLamp and Ladder\nThe Penciled Card\n"
    "TITLE: The Brittle Reel"
)

_SCRIPT = (
    "ANNOUNCER: Tonight, a reel that would not stay shelved.\n"
    "MARA: The label says 1931. The film says otherwise.\n"
    "TOM: Then one of them is lying.\n"
    "MARA: Films don't lie. People file them wrong.\n"
    "ANNOUNCER: The archive keeps its own hours.\n"
)


def _capturing_title_fn():
    calls: list = []

    def fn(messages, **kwargs):
        calls.append(messages)
        return _TITLE_REPLY

    return fn, calls


def test_title_prompt_uses_bank_title_form_label():
    from nodes.OTR_LedgerScriptWriter import _generate_title_from_script

    fn, calls = _capturing_title_fn()
    out = _generate_title_from_script(
        fn, _SCRIPT, title_form_label="Shakespeare radio adaptation"
    )
    assert out == "The Brittle Reel"
    sysmsg = calls[0][0]["content"]
    assert "Shakespeare radio adaptation" in sysmsg
    assert "sci-fi radio drama" not in sysmsg


def test_title_prompt_default_stays_scifi():
    from nodes.OTR_LedgerScriptWriter import _generate_title_from_script

    fn, calls = _capturing_title_fn()
    _generate_title_from_script(fn, _SCRIPT)
    assert "sci-fi radio drama" in calls[0][0]["content"]


def test_every_runnable_bank_declares_a_title_form_label():
    """The writer threads banks.json defaults.title_form_label into the
    title pass -- every runnable bank must declare one."""
    import json

    banks = json.loads(
        (REPO_ROOT / "nodes" / "story_packs" / "banks.json").read_text(
            encoding="utf-8"
        )
    )
    for row in banks["banks"]:
        if not row.get("runnable"):
            continue
        label = (row.get("defaults") or {}).get("title_form_label")
        assert isinstance(label, str) and label.strip(), (
            f"{row['source_bank_id']} lacks defaults.title_form_label"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
