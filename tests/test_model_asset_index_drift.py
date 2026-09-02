"""The generated asset index follows real fetch lanes, never source regexes."""
from __future__ import annotations

import importlib.util
import pathlib


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
    assert "1.7B: [exact manual tier](RUNPOD_INSTALL.md)" in row


def test_bundle_and_unresolved_names_never_become_fake_commands():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_commands_test")
    fetcher = _load("scripts/otr_fetch_lane_weights.py", "_otr_fetcher_commands_test")
    rendered = index.render()

    for name in fetcher.BUNDLES:
        if name not in fetcher.LANES:
            assert "python scripts/otr_fetch_lane_weights.py %s" % name not in rendered
    for name in getattr(fetcher, "UNRESOLVED", {}):
        assert "python scripts/otr_fetch_lane_weights.py %s" % name not in rendered


def test_h3_command_is_segregated_as_explicit_operator_local():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_h3_test")
    rendered = index.render()

    assert "The complete H3 manifest is deliberately explicit and operator-local" in rendered
    assert "python scripts/otr_fetch_lane_weights.py minimax_h3" in rendered
    public_block = rendered.split("The complete H3 manifest", 1)[0]
    assert "python scripts/otr_fetch_lane_weights.py minimax_h3" not in public_block


def test_committed_asset_index_has_no_generator_drift():
    index = _load("scripts/otr_asset_index.py", "_otr_asset_index_drift_test")
    committed = (ROOT / "docs" / "MODEL_ASSET_INDEX.md").read_text("utf-8")

    assert index.render() == committed
