"""The listener bank is complete and, since 2026-09-13, rolls like any other."""
import copy
import json
from pathlib import Path

import pytest

from nodes import _otr_story_routing as RT, _otr_rolls as ROLLS
from nodes import _otr_lane_specs as LS


def bank_row():
    rows = json.loads((Path(RT.__file__).parent / "story_packs" / "banks.json").read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows["banks"]
    return copy.deepcopy(next(row for row in rows if row["source_bank_id"] == "my_story"))


def test_listener_bank_resolves_and_rolls_like_any_other():
    """OPERATOR DECISION 2026-09-13: my_story is roll-eligible on equal footing
    with every other shipped bank. It was excluded at birth because a blank
    automatic run had nothing to write from; `_otr_story_input.DEFAULT_IDEA` is
    now that floor, so the exclusion no longer has a reason."""
    bank = RT.require_runnable_bank("my_story")
    assert RT.story_input_mode(bank) == "user_fields_v1"
    assert RT.effective_auto_select(bank)
    assert "my_story" in ROLLS.eligible_bank_ids()
    assert RT.resolve_story_pack("my_story")
    spec = LS.LANE_SPECS["my_story_multipass"]
    assert spec.module == "_otr_my_story" and spec.runner_attr == "run_my_story_episode"


def test_absent_policy_defaults_preserve_existing_banks():
    assert RT.find_bank("roll (any eligible bank)") is None
    assert RT.story_input_mode(None) == "legacy"
    assert RT.effective_auto_select(None) is True


# ("auto_select", True) was here until 2026-09-13, when the parse-time guard
# that refused user_fields_v1 + auto_select=true was removed on purpose. The
# TYPE check below still stands -- a string "false" is still a malformed row.
@pytest.mark.parametrize("key,value", [("auto_select", "false"),
                                       ("story_input_mode", "unknown")])
def test_invalid_user_input_policy_is_rejected(key, value):
    row = bank_row()
    row["defaults"][key] = value
    with pytest.raises(RT.RegistryValidationError):
        RT._parse_bank(row, "test")
