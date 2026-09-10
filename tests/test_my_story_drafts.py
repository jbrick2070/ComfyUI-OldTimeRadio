"""My Story drafts: the promise that a submission survives the run.

The contract under test is narrow and load-bearing. "We saved your input" has
to be true at the moment generation starts, has to stay true when the run
fails, and must never be claimed when the write did not happen.

Pure / CPU, writes only under a tmp output root. UTF-8 no BOM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_drafts as DR  # noqa: E402
from nodes import _otr_story_input as SI  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_output(tmp_path, monkeypatch):
    """Every draft in this module lands under a throwaway output root."""
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    yield


def _bundle(**fields):
    fields.setdefault("idea", "a keeper hears a voice in the fog")
    return SI.build_bundle(
        SI.capture_raw(**fields),
        SI.StoryRequest(num_characters=2, act_count="1",
                        include_act_breaks=True,
                        source_bank_requested="my_story",
                        visual_style_requested="viz_camera"),
    )


CALLER = "tests.test_my_story_drafts"


def test_a_draft_lands_under_the_shared_state_tier_not_a_new_top_level():
    """The output contract admits `episodes` and `obs` at the top level.

    A third top-level directory would be a change to that contract, which is
    why drafts live inside the per-machine state tier instead.
    """
    receipt = DR.ensure_draft(_bundle(), caller=CALLER)
    path = Path(receipt.path)
    assert path.is_file()
    parts = path.parts
    assert "episodes" in parts and "_shared" in parts and "state" in parts
    assert DR.DRAFTS_DIRNAME in parts
    assert "otr" in parts


def test_the_submission_is_on_disk_verbatim():
    bundle = _bundle(idea="  spaced  ", characters="Ada\nTom", author="A. Name")
    receipt = DR.ensure_draft(bundle, caller=CALLER)
    stored = json.loads(Path(receipt.path).read_text(encoding="utf-8"))
    assert stored["fields"] == bundle.fields.as_dict()
    assert stored["fields"]["idea"] == "  spaced  ", "raw text, not normalized"
    assert stored["digest"] == bundle.digest


def test_saving_the_same_submission_twice_verifies_instead_of_duplicating():
    bundle = _bundle()
    first = DR.ensure_draft(bundle, caller=CALLER)
    second = DR.ensure_draft(bundle, caller=CALLER)
    assert first.status == DR.STATUS_CREATED
    assert second.status == DR.STATUS_VERIFIED
    assert first.path == second.path
    assert len(list(DR.drafts_root().iterdir())) == 1


def test_two_different_submissions_get_two_drafts():
    a = DR.ensure_draft(_bundle(idea="one"), caller=CALLER)
    b = DR.ensure_draft(_bundle(idea="two"), caller=CALLER)
    assert a.digest != b.digest
    assert Path(a.path).is_file() and Path(b.path).is_file()


def test_the_timestamp_and_submitter_ride_outside_the_identity():
    """The same words submitted from a different node are the same submission.

    If either were digested, the validator and the writer would file one run
    under two identities and the writer's verification would fail every time.
    """
    bundle = _bundle()
    receipt = DR.ensure_draft(
        bundle, context=DR.SubmissionContext(prompt_id="p1", node_id="1"),
        now="2026-09-10T00:00:00Z")
    stored = json.loads(Path(receipt.path).read_text(encoding="utf-8"))
    assert stored["created_at"] == "2026-09-10T00:00:00Z"
    assert stored["submitted_by"]["node_id"] == "1"
    assert SI.compute_digest(bundle.fields, bundle.request) == bundle.digest
    # A second submission of the same words from elsewhere verifies the same file.
    again = DR.ensure_draft(
        bundle, context=DR.SubmissionContext(prompt_id="p2", node_id="9"),
        now="2026-09-11T00:00:00Z")
    assert again.status == DR.STATUS_VERIFIED
    assert again.path == receipt.path


def test_a_tampered_draft_is_refused_rather_than_reused():
    """A file that no longer hashes to the identity it is filed under cannot
    answer for that submission, and silently trusting it would let an edited
    draft stand in for what the person actually typed."""
    bundle = _bundle()
    receipt = DR.ensure_draft(bundle, caller=CALLER)
    path = Path(receipt.path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["fields"]["idea"] = "something else entirely"
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(DR.StoryDraftError) as caught:
        DR.ensure_draft(bundle, caller=CALLER)
    assert "hashes to" in str(caught.value)


def test_an_unreadable_draft_is_refused_with_the_path():
    bundle = _bundle()
    path = DR.draft_path(bundle.digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DR.StoryDraftError) as caught:
        DR.ensure_draft(bundle, caller=CALLER)
    assert str(path) in str(caught.value)


def test_a_storage_failure_refuses_rather_than_claiming_a_save(monkeypatch):
    """A storage error must PREVENT generation, not be reported beside it.

    Returning a receipt here would make "your input is saved" false at the
    exact moment the person is relying on it.
    """
    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(DR.StoryDraftError) as caught:
        DR.ensure_draft(_bundle(), caller=CALLER)
    assert "Nothing was generated" in str(caught.value)


def test_no_temp_file_is_left_behind_on_a_failed_write(monkeypatch):
    bundle = _bundle()
    real = Path.write_text

    def fail_once(self, *a, **kw):
        if self.name.endswith(".tmp"):
            real(self, *a, **kw)
            raise OSError("interrupted")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "write_text", fail_once)
    with pytest.raises(DR.StoryDraftError):
        DR.ensure_draft(bundle, caller=CALLER)
    monkeypatch.undo()
    leftovers = list(DR.draft_dir(bundle.digest).glob("*.tmp"))
    assert leftovers == [], leftovers


# ---------------------------------------------------------------------------
# who submitted it
# ---------------------------------------------------------------------------

def test_an_explicit_context_wins():
    ctx = DR.SubmissionContext(prompt_id="abc", node_id="7")
    assert DR.resolve_context(ctx, "ignored") is ctx


def test_a_caller_is_accepted_when_there_is_no_execution_context():
    resolved = DR.resolve_context(None, "a_script")
    assert resolved.caller == "a_script"
    assert not resolved.prompt_id and not resolved.node_id


def test_no_identity_at_all_is_refused_rather_than_invented():
    """A shared fallback identity would make two independent writers look like
    one submitter, which is exactly the collision this refusal prevents."""
    with pytest.raises(DR.StoryDraftError) as caught:
        DR.resolve_context(None, "")
    assert "explicit context" in str(caught.value)


def test_an_empty_context_object_is_not_an_identity():
    with pytest.raises(DR.StoryDraftError):
        DR.resolve_context(DR.SubmissionContext(), "")


def test_the_module_imports_without_comfyui_on_the_path():
    """The execution-context import is lazy, so a standalone test can use this
    module without ComfyUI installed -- proven by this suite running at all."""
    import importlib

    assert importlib.import_module("nodes._otr_story_drafts") is DR


@pytest.mark.parametrize("corruption", ["array", "count", "normalized", "digest", "version"])
def test_corrupted_draft_is_always_a_draft_error(corruption):
    bundle = _bundle()
    path = Path(DR.ensure_draft(bundle, caller=CALLER).path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "array":
        data = []
    elif corruption == "count":
        data["request"]["num_characters"] = "not an integer"
    elif corruption == "normalized":
        data["normalized"]["idea"] = "replaced"
    else:
        data["digest" if corruption == "digest" else "schema_version"] = "corrupt"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(DR.StoryDraftError):
        DR.ensure_draft(bundle, caller=CALLER)


def test_default_timestamp_is_utc_and_directory_failure_is_wrapped(monkeypatch):
    from datetime import datetime
    path = Path(DR.ensure_draft(_bundle(), caller=CALLER).path)
    assert datetime.fromisoformat(json.loads(path.read_text())["created_at"]).utcoffset().total_seconds() == 0
    def fail(*a, **kw):
        raise OSError("directory denied")
    monkeypatch.setattr(Path, "mkdir", fail)
    with pytest.raises(DR.StoryDraftError, match="directory denied"):
        DR.ensure_draft(_bundle(idea="another idea"), caller=CALLER)
