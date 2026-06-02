"""R0a (c): FreezeCascade episode_seed + v2_ledger_json ports appended at
indices 5,6 (never inserted). Outputs 0-4 and the out[1] raw-delegation fan-out
must stay byte-identical; the workflow JSON instance must match the surface."""
import json
from pathlib import Path

from nodes.OTR_LedgerFreezeCascade import (
    OTR_LedgerFreezeCascade,
    _episode_seed_from_ledger,
)

REPO = Path(__file__).resolve().parent.parent
WF = json.loads((REPO / "workflows" / "otr_scifi_16gb_full.json").read_text(encoding="utf-8"))

_BASE5_NAMES = ("script_text", "script_json", "news_used",
                "estimated_minutes", "freeze_verdict")
_BASE5_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")


# --- class surface: append-only -------------------------------------------

def test_return_surface_is_seven_append_only():
    assert OTR_LedgerFreezeCascade.RETURN_NAMES == _BASE5_NAMES + (
        "episode_seed", "v2_ledger_json")
    assert OTR_LedgerFreezeCascade.RETURN_TYPES == _BASE5_TYPES + ("INT", "STRING")


def test_indices_0_to_4_unchanged():
    # The legacy raw-delegation surface (out[1] script_json especially) is frozen.
    assert OTR_LedgerFreezeCascade.RETURN_NAMES[:5] == _BASE5_NAMES
    assert OTR_LedgerFreezeCascade.RETURN_TYPES[:5] == _BASE5_TYPES
    assert OTR_LedgerFreezeCascade.RETURN_NAMES[1] == "script_json"
    assert OTR_LedgerFreezeCascade.RETURN_NAMES[5] == "episode_seed"
    assert OTR_LedgerFreezeCascade.RETURN_TYPES[5] == "INT"
    assert OTR_LedgerFreezeCascade.RETURN_NAMES[6] == "v2_ledger_json"
    assert OTR_LedgerFreezeCascade.RETURN_TYPES[6] == "STRING"


# --- episode_seed: deterministic, read-only -------------------------------

def test_episode_seed_is_positive_int64():
    s = _episode_seed_from_ledger('{"meta": {"a": 1}}')
    assert isinstance(s, int) and 0 <= s < (1 << 63)


def test_episode_seed_deterministic():
    j = '{"lines": [], "meta": {"freeze_verdict": "frozen_clean"}}'
    assert _episode_seed_from_ledger(j) == _episode_seed_from_ledger(j)


def test_episode_seed_changes_with_ledger():
    a = _episode_seed_from_ledger('{"meta": {"freeze_verdict": "frozen_clean"}}')
    b = _episode_seed_from_ledger('{"meta": {"freeze_verdict": "frozen_with_warns"}}')
    assert a != b


def test_episode_seed_empty_ledger_stable():
    assert _episode_seed_from_ledger("") == _episode_seed_from_ledger("{}")


# --- workflow JSON instance matches the surface ---------------------------

def _node62():
    n = [n for n in WF["nodes"] if n.get("type") == "OTR_LedgerFreezeCascade"]
    assert len(n) == 1
    return n[0]


def test_workflow_node62_has_seven_outputs():
    outs = _node62()["outputs"]
    assert [o["name"] for o in outs] == list(_BASE5_NAMES) + [
        "episode_seed", "v2_ledger_json"]
    assert outs[5]["type"] == "INT" and outs[6]["type"] == "STRING"


def test_workflow_new_ports_have_no_links():
    outs = _node62()["outputs"]
    assert outs[5]["links"] == [] and outs[6]["links"] == []


def test_workflow_out1_fanout_preserved():
    # out[1] script_json keeps its 13-consumer raw-delegation fan-out (link 110
    # on out[2] news_used is unrelated; appending 5,6 shifts nothing).
    outs = _node62()["outputs"]
    assert outs[1]["name"] == "script_json"
    assert len(outs[1]["links"]) == 13


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
