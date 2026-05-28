"""QA item C regression -- OTR_StoryRoomCommit must FAIL LOUD on an
uncommittable extraction when commit=True, instead of silently
passing the writer's legacy script_json through (false-green where
the graph renders but the Story Room dialogue never landed).

Hermetic: the commit=True hard-fail branch raises BEFORE any ledger
load / lazy import, so these tests need no in-flight ledger, no
torch, and no comfy.
"""
from __future__ import annotations

import pytest

from nodes.OTR_StoryRoomCommit import OTR_StoryRoomCommit, StoryRoomCommitError


def test_commit_false_passthrough_even_on_failed_status():
    """commit=False keeps the legacy PD1 pass-through (no raise)."""
    node = OTR_StoryRoomCommit()
    out = node.run(
        script_json='{"legacy": true}',
        story_room_extraction='{"status": "failed"}',
        commit=False,
    )
    assert out == ('{"legacy": true}',)


@pytest.mark.parametrize(
    "payload,reason",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("{not valid json", "invalid_json"),
        ("[1, 2, 3]", "non_object_root"),
        ('{"status": "dormant"}', "status_dormant"),
        ('{"status": "failed"}', "status_failed"),
        ('{"status": "partial"}', "status_partial"),
        ('{"status": "ok", "dialogue": []}', "zero_dialogue_rows"),
        ('{"status": "ok"}', "zero_dialogue_rows"),
    ],
)
def test_commit_true_raises_on_uncommittable(payload, reason):
    """commit=True + any uncommittable extraction -> StoryRoomCommitError,
    and the message carries the precise reason code."""
    node = OTR_StoryRoomCommit()
    with pytest.raises(StoryRoomCommitError) as exc:
        node.run(
            script_json='{"legacy": true}',
            story_room_extraction=payload,
            commit=True,
        )
    assert reason in str(exc.value)


def test_reason_classifier_matches_parse_reject_ladder():
    """_extraction_block_reason mirrors _parse_extraction's reject set:
    every payload the classifier calls committable must also parse to a
    non-None dict, and vice-versa."""
    node = OTR_StoryRoomCommit()
    committable = '{"status": "ok", "dialogue": [{"dialogue_slot_id": "d001", "text": "hi"}]}'
    assert node._parse_extraction(committable) is not None
    assert node._extraction_block_reason(committable) == "uncommittable"
    # "uncommittable" is only returned when parse would have SUCCEEDED,
    # so it is never the reason on the real raise path.
