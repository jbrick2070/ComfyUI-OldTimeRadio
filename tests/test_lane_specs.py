"""The ONE story-lane authority: `nodes/_otr_lane_specs.py`.

Two entry points with deliberately different contracts -- `assert_supported`
preserves the lane's NATIVE error identity, `is_roll_compatible` answers a
bool and swallows ONLY declared compat errors. Everything here pins that
split, plus the names-only/lazy-import contract that keeps runner modules
out of ComfyUI startup.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lane_specs as LANES
from nodes import _otr_scifi_codex as CODEX
from nodes import _otr_scifi_fable2 as FABLE2


def _bank(pipeline_id: str):
    """The only field the lane authority reads off a bank row."""
    return SimpleNamespace(default_story_pipeline=pipeline_id)


# ---------------------------------------------------------------------------
# names-only / lazy-import contract
# ---------------------------------------------------------------------------

def test_lane_specs_hold_names_never_callables_or_classes():
    """Importing objects to BUILD the table would drag every runner into
    ComfyUI startup, which is exactly what the lazy wrappers existed to
    prevent."""
    for pipeline_id, spec in LANES.LANE_SPECS.items():
        assert isinstance(spec.module, str) and spec.module, pipeline_id
        assert isinstance(spec.runner_attr, str) and spec.runner_attr
        assert isinstance(spec.compat_attr, str)   # "" = declared unconstrained
        assert isinstance(spec.compat_error_attrs, tuple)
        for name in spec.compat_error_attrs:
            assert isinstance(name, str) and name


def test_every_declared_name_actually_resolves():
    """A typo in a NAMES-ONLY table is invisible until dispatch -- so resolve
    every one of them here."""
    for pipeline_id, spec in LANES.LANE_SPECS.items():
        module = LANES._load(spec.module)
        assert callable(getattr(module, spec.runner_attr)), pipeline_id
        if spec.compat_attr:
            assert callable(getattr(module, spec.compat_attr)), pipeline_id
        for name in spec.compat_error_attrs:
            error_cls = getattr(module, name)
            assert isinstance(error_cls, type) and issubclass(
                error_cls, BaseException), f"{pipeline_id}:{name}"


def test_a_lane_with_declared_compat_errors_also_declares_a_hook():
    """Declaring errors but no predicate would silently never catch them."""
    for pipeline_id, spec in LANES.LANE_SPECS.items():
        if spec.compat_error_attrs:
            assert spec.compat_attr, pipeline_id


# ---------------------------------------------------------------------------
# runner_for
# ---------------------------------------------------------------------------

def test_runner_for_resolves_the_dispatched_lanes():
    assert LANES.runner_for("scifi_news_circuit") is (
        CODEX.run_scifi_codex_episode)
    assert LANES.runner_for("scifi_news_pro_multipass") is (
        FABLE2.run_scifi_fable2_episode)


def test_runner_for_returns_none_for_a_known_inline_lane():
    """None is a real answer -- 'the writer's own body runs this'."""
    for pipeline_id in LANES.INLINE_PIPELINES:
        assert LANES.runner_for(pipeline_id) is None


def test_runner_for_raises_on_an_unregistered_pipeline():
    """No fallback: an unwired runnable bank is a wiring bug, and quietly
    running the legacy inline branch under its name would hide that."""
    with pytest.raises(LANES.UnknownLanePipelineError) as caught:
        LANES.runner_for("not_a_real_pipeline")
    assert "not_a_real_pipeline" in str(caught.value)
    assert "no fallback" in str(caught.value)


def test_is_dispatched_matches_the_table():
    assert LANES.is_dispatched("scifi_news_circuit") is True
    assert LANES.is_dispatched("legacy_many_pass") is False
    assert LANES.is_dispatched("not_a_real_pipeline") is False


# ---------------------------------------------------------------------------
# assert_supported -- NATIVE error identity, never wrapped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_words", [29, 901, 5000])
def test_assert_supported_raises_the_lanes_own_error_type(target_words):
    with pytest.raises(CODEX.CodexTargetRangeError):
        LANES.assert_supported(
            _bank("scifi_news_circuit"),
            LANES.RollRequest(target_words=target_words),
        )


@pytest.mark.parametrize("target_words", [30, 320, 720, 900])
def test_assert_supported_passes_inside_the_band(target_words):
    LANES.assert_supported(
        _bank("scifi_news_circuit"),
        LANES.RollRequest(target_words=target_words),
    )


def test_assert_supported_is_a_noop_for_an_unconstrained_lane():
    """fable2 declares itself unconstrained -- an extreme target must not
    raise here, because the lane genuinely accepts it."""
    for target_words in (30, 5000):
        LANES.assert_supported(
            _bank("scifi_news_pro_multipass"),
            LANES.RollRequest(target_words=target_words),
        )


def test_assert_supported_is_a_noop_for_inline_lanes():
    for pipeline_id in LANES.INLINE_PIPELINES:
        LANES.assert_supported(
            _bank(pipeline_id), LANES.RollRequest(target_words=5000))


def test_assert_supported_raises_on_an_unregistered_pipeline():
    with pytest.raises(LANES.UnknownLanePipelineError):
        LANES.assert_supported(
            _bank("not_a_real_pipeline"), LANES.RollRequest(target_words=320))


# ---------------------------------------------------------------------------
# is_roll_compatible -- a BOOL, and only declared errors are swallowed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_words,expected", [
    (29, False), (30, True), (320, True), (900, True), (901, False),
])
def test_is_roll_compatible_tracks_the_band(target_words, expected):
    assert LANES.is_roll_compatible(
        _bank("scifi_news_circuit"),
        LANES.RollRequest(target_words=target_words),
    ) is expected


def test_is_roll_compatible_is_true_for_unconstrained_and_inline_lanes():
    req = LANES.RollRequest(target_words=5000)
    assert LANES.is_roll_compatible(_bank("scifi_news_pro_multipass"), req)
    for pipeline_id in LANES.INLINE_PIPELINES:
        assert LANES.is_roll_compatible(_bank(pipeline_id), req)


def test_is_roll_compatible_is_false_for_an_unregistered_pipeline():
    """Ineligible, not eligible-by-default: an unwired pipeline must never
    win a roll. `assert_supported` still RAISES on it -- the roll simply
    declines to gamble on a wiring bug."""
    assert LANES.is_roll_compatible(
        _bank("not_a_real_pipeline"),
        LANES.RollRequest(target_words=320),
    ) is False


def test_is_roll_compatible_does_not_swallow_undeclared_errors(monkeypatch):
    """A bare `except` would turn any runner defect into 'ineligible' and
    hide the breakage behind a smaller pool."""
    def _explode(_target_words):
        raise RuntimeError("preflight is broken")

    monkeypatch.setattr(
        CODEX, "assert_supported_target_words", _explode, raising=True)
    with pytest.raises(RuntimeError, match="preflight is broken"):
        LANES.is_roll_compatible(
            _bank("scifi_news_circuit"),
            LANES.RollRequest(target_words=320),
        )


def test_is_roll_compatible_does_not_swallow_an_import_error(monkeypatch):
    def _no_module(_module_name):
        raise ImportError("runner module is missing")

    monkeypatch.setattr(LANES, "_load", _no_module, raising=True)
    with pytest.raises(ImportError, match="runner module is missing"):
        LANES.is_roll_compatible(
            _bank("scifi_news_circuit"),
            LANES.RollRequest(target_words=320),
        )


# ---------------------------------------------------------------------------
# the hoisted codex preflight speaks for the runner, not beside it
# ---------------------------------------------------------------------------

def test_codex_preflight_shares_one_source_of_truth_with_the_runner():
    """The hoisted predicate is a reachable spelling of WordSteerV4, not a
    second copy of the numbers -- so the band can never drift apart."""
    field = CODEX.WordSteerV4.model_fields["requested_words"]
    bounds = {type(m).__name__: getattr(m, "ge", getattr(m, "le", None))
              for m in field.metadata}
    assert bounds.get("Ge") == 30 and bounds.get("Le") == 900
    CODEX.assert_supported_target_words(30)
    CODEX.assert_supported_target_words(900)
    for bad in (29, 901, "320", None, True):
        with pytest.raises(CODEX.CodexTargetRangeError):
            CODEX.assert_supported_target_words(bad)
