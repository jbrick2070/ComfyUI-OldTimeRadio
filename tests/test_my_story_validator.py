"""Exercise the real queue gate before downloads and durable draft admission."""
import json

import pytest

from nodes import _otr_story_input as SI, _otr_story_drafts as DR
from nodes import _otr_visual_assets as VA
from nodes._otr_workflow_validator import WorkflowValidator, _DEFAULT_WORKFLOW_PATH


def prompt(**overrides):
    inputs = dict(source_bank="my_story", custom_premise="The fog bell rings.",
                  num_characters=2, act_count="1", include_act_breaks=True,
                  visual_style="roll (any style)", gate_in=["94", 0])
    inputs.update(overrides)
    return {"1": {"class_type": "OTR_LedgerScriptWriter", "inputs": inputs}}


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("OTR_SOURCE_SNAPSHOT_MANIFEST", raising=False)
    downloads = []
    monkeypatch.setattr(VA, "ensure_prompt_visual_assets", lambda *a: downloads.append(a))
    monkeypatch.setattr(DR, "_executing_context",
                        lambda: DR.SubmissionContext(prompt_id="prompt-123", node_id="94"))

    def run(queued, enabled=True):
        return WorkflowValidator().validate(str(_DEFAULT_WORKFLOW_PATH), enabled, False,
                                            prompt=queued, unique_id="94")
    return run, downloads


@pytest.mark.parametrize("enabled", [False, True])
def test_literal_admission_persists_before_assets_with_real_prompt_identity(gate, enabled, monkeypatch):
    run, downloads = gate
    def assets(*args):
        files = list(DR.drafts_root().glob("*/input.json"))
        assert len(files) == 1
        saved = json.loads(files[0].read_text(encoding="utf-8"))
        assert saved["submitted_by"] == {"prompt_id": "prompt-123", "node_id": "1", "caller": ""}
        assert saved["request"]["visual_style_requested"] == "roll (any style)"
        downloads.append(True)
    monkeypatch.setattr(VA, "ensure_prompt_visual_assets", assets)
    run(prompt(story_author="A. Listener"), enabled)
    assert downloads == [True]


# {"custom_premise": " "} and {"custom_premise": "", "story_author": [...]}
# lived here until 2026-09-13. A My Story submission with nothing creative in
# it is no longer refused -- it writes `SI.DEFAULT_IDEA`, which is what lets
# the bank sit in the roll pool at all. See the floor test below. Every row
# that is still genuinely unrunnable stays exactly where it was.
@pytest.mark.parametrize("values", [
    {"source_bank": "original", "story_plot": "A bell rings"},
    {"custom_premise": ["8", 0], "source_ref": "forbidden"},
    {"custom_premise": ["8", 0], "replay_from": "forbidden"},
    {"custom_premise": [True, {}]},
])
def test_invalid_literal_values_refuse_before_download(gate, values):
    run, downloads = gate
    with pytest.raises(SI.StoryInputError):
        run(prompt(**values))
    assert downloads == []
    assert not list(DR.drafts_root().glob("*/input.json"))


@pytest.mark.parametrize("values", [
    {"custom_premise": ["8", 0]}, {"story_plot": ["8", 0]},
    {"story_author": ["8", 0]}, {"num_characters": ["8", 0]},
    {"visual_style": ["8", 0]},
])
def test_partial_linked_input_defers_persistence(gate, values):
    run, downloads = gate
    run(prompt(**values))
    assert len(downloads) == 1
    assert not list(DR.drafts_root().glob("*/input.json"))


def test_snapshot_conflict_is_known_even_with_linked_creative_input(gate, monkeypatch):
    run, downloads = gate
    monkeypatch.setenv("OTR_SOURCE_SNAPSHOT_MANIFEST", "configured.json")
    with pytest.raises(SI.StoryInputError, match="snapshot"):
        run(prompt(custom_premise=["8", 0]))
    assert downloads == []


def test_unrelated_writers_are_ignored_and_all_dependent_writers_checked(gate):
    """A writer behind a DIFFERENT gate is not this validator's business; a
    second writer behind THIS one is.

    The refusal vehicle is `source_ref` rather than a blank premise: since
    2026-09-13 a blank My Story submission is floored, not refused, so it can
    no longer prove that the second writer was reached at all.
    """
    run, downloads = gate
    run(prompt(source_ref="forbidden", gate_in=["93", 0]))
    queued = prompt()
    queued["2"] = prompt(source_ref="forbidden")["1"]
    with pytest.raises(SI.StoryInputError):
        run(queued)
    assert len(downloads) == 1


@pytest.mark.parametrize("blank", ["", " ", "\n  "])
def test_a_blank_my_story_submission_writes_the_standing_premise(gate, blank):
    """THE FLOOR (2026-09-13). Nothing typed is no longer a refusal: the run
    proceeds and the persisted draft carries `SI.DEFAULT_IDEA`, which is the
    whole reason my_story may be drawn by a blank automatic roll.
    """
    run, downloads = gate
    run(prompt(custom_premise=blank))
    assert len(downloads) == 1
    files = list(DR.drafts_root().glob("*/input.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["fields"]["idea"] == SI.DEFAULT_IDEA


def test_the_writers_own_preroll_check_floors_too_not_just_the_validator():
    """THE GAP A QA PASS FOUND ON 2026-09-13, and the reason it existed.

    The writer checks admission TWICE: once as the first statement of run(),
    BEFORE the bank roll, and once after the bank row is bound. The floor was
    applied only at the second site. A MANUAL `my_story` pick is a plain combo
    value, not the roll sentinel, so the first check saw it, saw every creative
    field blank, and raised -- after the VALIDATOR had already admitted the
    same submission by flooring it. The gate passed a request the writer then
    refused, three lines above a comment claiming the two agree "by
    construction".

    Nothing caught it because every my_story test either went through the
    validator (which floored) or supplied a premise. This asserts the writer's
    OWN pre-roll policy path, with no premise, admits.
    """
    from nodes import _otr_story_routing as RT

    raw = SI.capture_raw(idea="", characters="", plot="", setting="",
                         author="")
    row = RT.find_bank("my_story")
    policy = SI.StoryInputPolicy(
        mode=RT.story_input_mode(row),
        bank_id=getattr(row, "source_bank_id", "") or "my_story",
    )
    floored = SI.with_default_idea(raw, policy)
    SI.check_selection(floored, policy)          # must not raise
    assert floored.idea == SI.DEFAULT_IDEA

    # ...and the roll sentinel is still NOT a user-fields bank here, so the
    # floor must leave it exactly alone (same object, not merely equal).
    sentinel_row = RT.find_bank("roll (any eligible bank)")
    sentinel_policy = SI.StoryInputPolicy(
        mode=RT.story_input_mode(sentinel_row),
        bank_id="roll (any eligible bank)",
    )
    assert SI.with_default_idea(raw, sentinel_policy) is raw
    assert SI.would_apply_default_idea(raw, policy) is True
    assert SI.would_apply_default_idea(raw, sentinel_policy) is False
    typed = SI.capture_raw(idea="a story about my mother")
    assert SI.would_apply_default_idea(typed, policy) is False


def test_a_typed_my_story_submission_is_never_overwritten_by_the_floor(gate):
    """The floor is a floor, not a default: one typed word beats it."""
    run, downloads = gate
    run(prompt(custom_premise="a story about my mother"))
    files = list(DR.drafts_root().glob("*/input.json"))
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["fields"]["idea"] == "a story about my mother"


def test_storage_failure_prevents_assets(gate, monkeypatch):
    run, downloads = gate
    def fail(*a, **k):
        raise DR.StoryDraftError("disk full")
    monkeypatch.setattr(DR, "ensure_draft", fail)
    with pytest.raises(DR.StoryDraftError):
        run(prompt())
    assert downloads == []
