"""PBUG-20260817-05 -- the soak harness must not name the episodes it renders.

THE DEFECT, from the pixels. Published episodes shipped with hero title cards
reading ``CHUNKB ACCEPT forced lemmy scifi_news_pro`` and ``SOAK05
scifi_news_pro visual_storybased``. The soak passed its own run label as
``--title``, which lands in the writer's ``episode_title`` widget, and the
writer reads any non-empty value there as a person naming their episode
("User typed a value; respect it verbatim") -- so the label became the title
card, the filenames, the episode folder, the ledger name, the treatment, the
canon, the credits and the published obs artifact. Nothing malfunctioned; the
contract simply never distinguished a person's title from a harness label.

WHAT THESE TESTS PIN:
  * the leg command carries NO ``--title`` -- titling belongs to the
    canonical workflow, which is what "mimic the entire workflow" means;
  * the receipt key is ``leg_label``, never ``title``. A key named ``title``
    holding a run label is the one-field-two-meanings shape of Bible
    ``12.110`` / ``11.61``, and it is why the drift stayed invisible;
  * the receipt records what the WORKFLOW titled the episode, read back from
    the ledger the run wrote;
  * a headless run reporting ``title_source == "user"`` is flagged -- it is a
    contradiction on its face, since nobody typed anything;
  * the read-back REFUSES TO GUESS. If two episodes land in one leg's window
    it reports ambiguity rather than picking the newest, because silently
    recording another episode's title is the same false-title defect this
    receipt exists to catch;
  * nothing in the read-back can end a campaign. A torn ledger is valid JSON
    of the wrong SHAPE as often as it is invalid JSON, and the module's own
    contract is that a bad leg is logged and skipped;
  * and the flag REPORTS without failing the leg. An audit may never fail an
    episode (operator law 2026-07-22); ``ok`` keeps meaning "the render
    succeeded" and nothing else.
"""
from __future__ import annotations

import datetime
import importlib.util
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SOAK_SRC = REPO_ROOT / "scripts" / "otr_gpu_soak_matrix.py"


def _load_soak():
    spec = importlib.util.spec_from_file_location(
        "otr_gpu_soak_matrix", SOAK_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOAK = _load_soak()


def _write_ledger(episodes_root, episode_id, *, title, source):
    audio = episodes_root / episode_id / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    led = audio / (episode_id + "_ledger.json")
    led.write_text(
        json.dumps({"episode_id": episode_id,
                    "meta": {"episode_title": title, "title_source": source}}),
        encoding="utf-8")
    return led


@pytest.fixture()
def episodes_root(tmp_path, monkeypatch):
    """An isolated output tree, resolved through the one OTR path authority."""
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    root = tmp_path / "otr" / "episodes"
    root.mkdir(parents=True)
    return root


class _Proc:
    """A finished canonical-api run, as `leg()` reads one."""

    def __init__(self, stdout="[canonical-api] RESULT SUCCESS", returncode=0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


# --------------------------------------------------------------------------- #
# the leg command: the harness stops naming episodes
# --------------------------------------------------------------------------- #
def test_leg_command_passes_no_title(monkeypatch, episodes_root):
    """The regression pin. `--title` reaching the writer IS the defect."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _Proc()

    monkeypatch.setattr(SOAK.subprocess, "run", fake_run)
    SOAK.leg(5, "shakespeare", "anime", "otr_soak_still_flat_flux_gen1", 60)

    assert "--title" not in captured["cmd"], (
        "the soak named its own episode again -- that label becomes the "
        "on-screen title card (PBUG-20260817-05)")
    # The rest of the leg's creative surface is untouched.
    for flag in ("--source-bank", "--visual-style", "--profile", "--act-count"):
        assert flag in captured["cmd"]


def test_leg_label_is_not_a_title_key(monkeypatch, episodes_root):
    """The receipt names the label a label. One field, one meaning."""
    monkeypatch.setattr(SOAK.subprocess, "run", lambda cmd, **kw: _Proc())
    row = SOAK.leg(7, "original", "cartoon", "otr_soak_still_pan_ideo", 60)

    assert row["leg_label"] == "SOAK07 original cartoon still_pan_ideo"
    assert "title" not in row, (
        "a receipt key named `title` holding a run label is the "
        "one-field-two-meanings shape of Bible 12.110 / 11.61")
    for key in ("episode_title", "title_source", "title_guard"):
        assert key in row


# --------------------------------------------------------------------------- #
# the guard verdict
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source, verdict", [
    ("user", "VIOLATION_headless_title_source_user"),
    ("llm_post_composition", "ok"),
    ("outline_fallback", "ok"),
    ("scifi_news_pro_script_title", "ok"),
    (None, "unknown"),
    ("", "unknown"),
])
def test_title_guard_verdict(source, verdict):
    assert SOAK.title_guard_verdict(source) == verdict


def test_headless_forbidden_source_is_the_writers_own_branch_name():
    """Pinned to the writer's branch name, not to a paraphrase of it."""
    assert SOAK.HEADLESS_FORBIDDEN_TITLE_SOURCE == "user"


# --------------------------------------------------------------------------- #
# reading the title back from the ledger the run wrote
# --------------------------------------------------------------------------- #
def _window(before=30, after=30):
    """A leg window wide enough to contain a ledger written right now."""
    now = datetime.datetime.now()
    return (now - datetime.timedelta(seconds=before),
            now + datetime.timedelta(seconds=after))


def test_title_receipt_reports_the_workflow_title(episodes_root):
    started, finished = _window()
    _write_ledger(episodes_root, "signal_lost_the_blackwood_enigma_1",
                  title="The Blackwood Enigma", source="llm_post_composition")

    assert SOAK.title_receipt(started, finished, "original", "anime") == {
        "episode_title": "The Blackwood Enigma",
        "title_source": "llm_post_composition",
        "title_guard": "ok",
    }


def test_title_receipt_flags_a_headless_user_title(episodes_root):
    started, finished = _window()
    _write_ledger(episodes_root, "signal_lost_soak05_scifi_news_pro_1",
                  title="SOAK05 scifi_news_pro visual_storybased",
                  source="user")

    got = SOAK.title_receipt(started, finished, "scifi_news_pro",
                             "visual_storybased")
    assert got["title_guard"] == "VIOLATION_headless_title_source_user"
    assert got["episode_title"] == "SOAK05 scifi_news_pro visual_storybased"


def test_title_receipt_degrades_when_no_episode_was_written(episodes_root):
    """A failed leg renders nothing. The receipt says so; it never raises."""
    started, finished = _window()
    assert SOAK.title_receipt(started, finished, "original",
                              "anime")["title_guard"] == "no_ledger"


@pytest.mark.parametrize("payload", [
    "{not json",          # not JSON at all
    "null",               # valid JSON, no top-level mapping
    '{"meta": "torn"}',   # valid JSON, `meta` is not a mapping
    '["meta"]',           # valid JSON, top level is a list
])
def test_title_receipt_degrades_on_a_torn_ledger(episodes_root, payload):
    """A torn ledger is valid JSON of the WRONG SHAPE as often as it is
    invalid JSON, and both raise -- `AttributeError`, not `ValueError`. An
    escape here would end an overnight campaign over a receipt field."""
    started, finished = _window()
    audio = episodes_root / "signal_lost_torn_1" / "audio"
    audio.mkdir(parents=True)
    (audio / "signal_lost_torn_1_ledger.json").write_text(
        payload, encoding="utf-8")

    assert SOAK.title_receipt(started, finished, "original",
                              "anime")["title_guard"] == "unreadable_ledger"


# --------------------------------------------------------------------------- #
# which ledger belongs to this leg -- and refusing to guess
# --------------------------------------------------------------------------- #
def test_lookup_ignores_episodes_outside_the_leg_window(episodes_root):
    _write_ledger(episodes_root, "signal_lost_yesterday_1",
                  title="Yesterday", source="llm_post_composition")
    now = datetime.datetime.now()
    started = now + datetime.timedelta(seconds=5)
    finished = now + datetime.timedelta(seconds=30)

    assert SOAK.ledgers_in_window(started, finished) == []


def test_the_legs_own_ledger_survives_the_window_edge(episodes_root):
    """`os.stat` is finer-grained than `datetime.now()`.

    A ledger written immediately before `finished` reports an mtime GREATER
    than it by a fraction of a microsecond, so a hard upper edge drops the
    leg's OWN episode whenever the ledger is the last write before the runner
    exits -- measured, not hypothesised. `READBACK_GRACE_S` is what keeps it.
    """
    started = datetime.datetime.now()
    _write_ledger(episodes_root, "signal_lost_right_on_the_edge_1",
                  title="Right On The Edge", source="llm_post_composition")
    finished = datetime.datetime.now()

    got = SOAK.title_receipt(started, finished, "original", "anime")
    assert got["episode_title"] == "Right On The Edge"


def test_lookup_skips_the_reserved_system_tier(episodes_root):
    """`_shared` is never an episode -- `_otr_paths` owns that rule."""
    started, finished = _window()
    _write_ledger(episodes_root, "_shared",
                  title="not an episode", source="user")

    assert SOAK.ledgers_in_window(started, finished) == []


def test_two_episodes_in_one_window_refuse_to_guess(episodes_root):
    """A concurrent writer must not have its title recorded as this leg's.

    Picking "the newest" would silently attribute another episode's title --
    the same false-title defect this receipt exists to catch.
    """
    started, finished = _window()
    _write_ledger(episodes_root, "signal_lost_this_legs_episode_1",
                  title="This Leg's Episode", source="llm_post_composition")
    _write_ledger(episodes_root, "signal_lost_someone_elses_1",
                  title="Someone Else's Episode", source="outline_fallback")

    got = SOAK.title_receipt(started, finished, "original", "anime")
    assert got["title_guard"] == "ambiguous_2_episodes"
    assert got["episode_title"] is None


def test_the_legs_own_parameters_break_the_tie(episodes_root):
    """When both candidates record their lane, the leg's own bank+style pick."""
    started, finished = _window()
    mine = _write_ledger(episodes_root, "signal_lost_this_legs_episode_1",
                         title="This Leg's Episode",
                         source="llm_post_composition")
    theirs = _write_ledger(episodes_root, "signal_lost_someone_elses_1",
                           title="Someone Else's Episode",
                           source="outline_fallback")
    for path, bank, style in ((mine, "original", "anime"),
                              (theirs, "shakespeare", "cartoon")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["meta"].update(source_bank=bank, visual_style=style)
        path.write_text(json.dumps(data), encoding="utf-8")

    got = SOAK.title_receipt(started, finished, "original", "anime")
    assert got["episode_title"] == "This Leg's Episode"


def test_a_readback_failure_cannot_end_the_campaign(monkeypatch,
                                                    episodes_root):
    """The module contract: a leg is logged and skipped, never fatal."""
    monkeypatch.setattr(SOAK.subprocess, "run", lambda cmd, **kw: _Proc())

    def boom(*args, **kwargs):
        raise RuntimeError("the tree moved under us")

    monkeypatch.setattr(SOAK, "title_receipt", boom)
    row = SOAK.leg(3, "original", "anime", "otr_soak_still_pan_ideo", 60)

    assert row["ok"] is True
    assert row["title_guard"] == "readback_error_RuntimeError"


# --------------------------------------------------------------------------- #
# THE LAW: the guard reports, it never fails an episode
# --------------------------------------------------------------------------- #
def test_a_title_violation_does_not_fail_the_render(monkeypatch,
                                                    episodes_root):
    """`ok` means the render succeeded, and only that."""

    def fake_run(cmd, **kwargs):
        _write_ledger(episodes_root, "signal_lost_planted_label_1",
                      title="SOAK01 planted label", source="user")
        return _Proc()

    monkeypatch.setattr(SOAK.subprocess, "run", fake_run)
    row = SOAK.leg(1, "shakespeare", "anime",
                   "otr_soak_still_flat_z_image_turbo", 60)

    assert row["ok"] is True
    assert row["title_guard"] == "VIOLATION_headless_title_source_user"


def test_the_campaign_summary_names_the_violating_legs(monkeypatch, capsys,
                                                       tmp_path):
    """A guard nobody can read is not a guard. The summary is the surface."""
    monkeypatch.setattr(SOAK, "REPO", tmp_path)

    def fake_leg(index, bank, style, profile, timeout):
        return {"leg": index, "bank": bank, "style": style, "profile": profile,
                "ok": True, "rc": 0, "minutes": 1.0,
                "leg_label": "SOAK%02d planted" % index,
                "episode_title": "SOAK%02d planted" % index,
                "title_source": "user",
                "title_guard": "VIOLATION_headless_title_source_user"}

    monkeypatch.setattr(SOAK, "leg", fake_leg)
    assert SOAK.main(["--legs", "2", "--seed", "1"]) == 0

    out = capsys.readouterr().out
    assert "TITLE GUARD: 2 leg(s)" in out
    assert "SOAK01 planted" in out and "SOAK02 planted" in out
    # Reporting a violation is not failing the campaign (operator law).
    assert "2/2 passed" in out
