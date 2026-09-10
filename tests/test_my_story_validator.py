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


@pytest.mark.parametrize("values", [
    {"custom_premise": " "},
    {"source_bank": "original", "story_plot": "A bell rings"},
    {"custom_premise": "", "story_author": ["8", 0]},
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
    run, downloads = gate
    run(prompt(custom_premise="", gate_in=["93", 0]))
    queued = prompt()
    queued["2"] = prompt(custom_premise="")["1"]
    with pytest.raises(SI.StoryInputError):
        run(queued)
    assert len(downloads) == 1


def test_storage_failure_prevents_assets(gate, monkeypatch):
    run, downloads = gate
    def fail(*a, **k):
        raise DR.StoryDraftError("disk full")
    monkeypatch.setattr(DR, "ensure_draft", fail)
    with pytest.raises(DR.StoryDraftError):
        run(prompt())
    assert downloads == []
