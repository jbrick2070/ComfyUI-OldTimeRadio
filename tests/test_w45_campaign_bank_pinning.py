"""The sweep can pin its source bank, so a writer defect cannot eat a video leg.

WHY. The 45-word campaign's gate is *"a render of every VISUAL path"* -- the
engine is the variable under test. But every leg rolled its source bank, which
put a whole second variable on top of the measurement, and on 2026-08-12 it cost
two legs outright:

* `ltx_8gb` rolled `scifi_news_pro`, died in `OTR_LedgerScriptWriter` after four
  markup attempts, and **the LTX video lane was never exercised at all** -- the
  leg was recorded as a lane failure when no lane had run;
* `mesh_stage` rolled `shakespeare` and died on a cast-coverage gap in the
  freeze cascade, again before any video work.

Rolling is not wrong -- it is how both of those writer defects were FOUND -- so
the sentinel stays the default and pinning is opt-in. The rule is: pin when the
question is "does this engine render", roll when the question is "does the
pipeline hold up across banks".
"""
from __future__ import annotations

import importlib.util
import inspect
import pathlib

import pytest


REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def campaign():
    spec = importlib.util.spec_from_file_location(
        "w45_campaign_under_test", REPO / "scripts" / "otr_w45_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rolling_is_still_the_DEFAULT(campaign):
    """Pinning must not silently narrow the campaign. Rolling is how the two
    writer defects were found in the first place."""
    assert campaign.BANK_SENTINEL == "roll (any eligible bank)"
    sig = inspect.signature(campaign.run_leg)
    assert sig.parameters["source_bank"].default == campaign.BANK_SENTINEL


def test_the_sentinel_comes_from_the_WRITERS_OWN_MODULE(campaign):
    """A hand-typed copy of this string would drift from the value the node
    actually accepts, and the failure would be a silent no-op roll."""
    from nodes._otr_rolls import BANK_SENTINEL as canonical

    assert campaign.BANK_SENTINEL == canonical


def test_run_leg_passes_the_bank_it_was_GIVEN(campaign, monkeypatch):
    """The behaviour, not the flag: whatever bank is chosen must reach the
    runner's argv."""
    seen = {}

    class _Result:
        returncode = 0

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return _Result()

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    monkeypatch.setattr(campaign, "verify_asset",
                        lambda started_at: {"ok": True, "coverage": "stub"})
    campaign.run_leg("still_flat", "otr_w45_still_flat", 45,
                     source_bank="shakespeare")

    cmd = seen["cmd"]
    assert "--source-bank" in cmd
    assert cmd[cmd.index("--source-bank") + 1] == "shakespeare"


def test_run_leg_rolls_when_not_told_otherwise(campaign, monkeypatch):
    seen = {}

    class _Result:
        returncode = 0

    monkeypatch.setattr(campaign.subprocess, "run",
                        lambda cmd, **kw: (seen.__setitem__("cmd", list(cmd)),
                                           _Result())[1])
    monkeypatch.setattr(campaign, "verify_asset",
                        lambda started_at: {"ok": True, "coverage": "stub"})
    campaign.run_leg("still_flat", "otr_w45_still_flat", 45)

    cmd = seen["cmd"]
    assert cmd[cmd.index("--source-bank") + 1] == campaign.BANK_SENTINEL


def test_the_flag_is_documented_as_a_MEASUREMENT_choice(campaign):
    """The help text has to say WHEN to pin, not just that you can. A knob
    without a rule gets used at random, which is the problem it fixes."""
    src = inspect.getsource(campaign.main)
    assert "--source-bank" in src
    assert "variable under test" in src
