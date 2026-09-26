"""The generated asset index follows real fetch lanes, never source regexes."""
from __future__ import annotations

import importlib.util
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asset_index_reads_exactly_the_real_lane_mapping():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_test")
    fetcher = _load("scripts/otr_fetch_lane_weights.py", "_otr_fetcher_index_test")

    assert set(index._fetcher_lanes()) == set(fetcher.LANES)


def test_humo_row_names_the_14b_lane_and_keeps_1_7b_manual():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_humo_test")
    rendered = index.render()

    row = next(line for line in rendered.splitlines()
               if line.startswith("| `humo` |"))
    assert "14B: `otr_fetch_lane_weights.py humo`" in row
    assert "1.7B: [exact manual download](RUNPOD_INSTALL.md)" in row


def test_bundle_and_unresolved_names_never_become_fake_commands():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_commands_test")
    fetcher = _load("scripts/otr_fetch_lane_weights.py", "_otr_fetcher_commands_test")
    rendered = index.render()

    def names_a_command(name):
        # A whole lane name, not a prefix: the `minimax_h3` bundle must not
        # match the real `minimax_h3_video` lane's command.
        return re.search(r"python scripts/otr_fetch_lane_weights\.py %s(?!\w)"
                         % re.escape(name), rendered) is not None

    for name in fetcher.BUNDLES:
        if name not in fetcher.LANES:
            assert not names_a_command(name), name
    for name in getattr(fetcher, "UNRESOLVED", {}):
        assert not names_a_command(name), name


def test_h3_row_says_auto_at_queue_time_and_names_both_lanes():
    """H3 downloads its own weights at queue time (operator 2026-09-25), so
    its row reads like LTX 2.5's and its two per-DiT lanes are public
    commands like every other lane."""
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_h3_test")
    rendered = index.render()

    row = next(line for line in rendered.splitlines()
               if line.startswith("| `minimax_h3` |"))
    assert "auto at queue time" in row
    assert "`otr_fetch_lane_weights.py minimax_h3_video`" in row
    assert "`minimax_h3_audio_in`" in row
    for lane in ("minimax_h3_video", "minimax_h3_audio_in"):
        assert "python scripts/otr_fetch_lane_weights.py %s\n" % lane in rendered
    assert "operator-local" not in rendered


def test_profile_usage_counts_resolve_public_video_ids_to_internal_owners():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_alias_test")
    profiles = index.collect_profiles()

    # otr_16gb_video saves the public id `ltx25_high_video`.
    assert profiles["ltx25_video"] == ["otr_16gb_video"]
    assert "ltx25_high_video" not in profiles
    # otr_8gb_video names the internal id; otr_mac16_video saves the public
    # `ltx098_low_video`. Both are counted against the one owner.
    assert profiles["ltx_8gb"] == ["otr_8gb_video", "otr_mac16_video"]
    assert "ltx098_low_video" not in profiles


def test_committed_asset_index_has_no_generator_drift():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_drift_test")
    committed = (ROOT / "apple" / "MODEL_ASSET_INDEX.md").read_text("utf-8")

    assert index.render() == committed
