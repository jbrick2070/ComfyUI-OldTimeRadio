"""The environment stamp (nodes/_otr_shared/env_fingerprint.py), its place in
every ledger, and the drift report (scripts/otr_env_drift.py). PLAN rows 2.1 and
2.2, planning/creatoros-concepts/PLAN.md.

The rules under test: versions and commits only, never a path, user or machine
name; a value that cannot be read says why; stamped once and kept; a replay
records today's machine, never its source's; the drift report reads and never
writes, except the known-good snapshot it is asked for.
"""
from __future__ import annotations

import getpass
import json
import os
import platform
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from nodes import production_ledger as PL  # noqa: E402
from nodes._otr_shared import env_fingerprint as EF  # noqa: E402
import otr_canonical_api_run as canonical  # noqa: E402
import otr_env_drift as drift  # noqa: E402

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture(autouse=True)
def _fresh_cache():
    EF._reset_cache_for_tests()
    yield
    EF._reset_cache_for_tests()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- git state read from files ------------------------------------------------

def test_a_detached_head_is_read_directly(tmp_path):
    _write(tmp_path / ".git" / "HEAD", SHA_A + "\n")
    assert EF.git_commit(str(tmp_path)) == SHA_A[:12]


def test_a_branch_head_is_resolved_through_a_loose_ref(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(tmp_path / ".git" / "refs" / "heads" / "main", SHA_B + "\n")
    assert EF.git_commit(str(tmp_path)) == SHA_B[:12]


def test_a_branch_head_is_resolved_through_packed_refs(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(tmp_path / ".git" / "packed-refs",
           "# pack-refs with: peeled fully-peeled sorted\n%s refs/heads/main\n" % SHA_B)
    assert EF.git_commit(str(tmp_path)) == SHA_B[:12]


def test_a_worktree_dot_git_file_and_its_common_dir_are_followed(tmp_path):
    common = tmp_path / "main_repo" / ".git"
    wt_git = common / "worktrees" / "wt"
    _write(wt_git / "HEAD", "ref: refs/heads/feature\n")
    _write(wt_git / "commondir", "../..\n")
    _write(common / "refs" / "heads" / "feature", SHA_A + "\n")
    work = tmp_path / "wt"
    _write(work / ".git", "gitdir: %s\n" % wt_git)
    assert EF.git_commit(str(work)) == SHA_A[:12]


def test_an_unresolvable_head_is_not_a_commit(tmp_path):
    _write(tmp_path / ".git" / "HEAD", "ref: refs/heads/gone\n")
    assert EF.git_commit(str(tmp_path)) is None
    assert EF.pack_state(str(tmp_path)) == "unknown: a git checkout whose HEAD cannot be read"
    assert EF.git_commit(str(tmp_path / "nothing")) is None


def test_pack_state_prefers_git_then_the_pyproject_version(tmp_path):
    git_pack = tmp_path / "git_pack"
    _write(git_pack / ".git" / "HEAD", SHA_A)
    assert EF.pack_state(str(git_pack)) == "git " + SHA_A[:12]
    versioned = tmp_path / "versioned"
    _write(versioned / "pyproject.toml", '[project]\nname = "x"\nversion = "1.4.2"\n')
    assert EF.pack_state(str(versioned)) == "version 1.4.2"
    bare = tmp_path / "bare"
    bare.mkdir()
    assert EF.pack_state(str(bare)) == "unknown: no .git and no pyproject.toml version"


def test_packs_are_named_by_folder_and_the_first_directory_wins(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _write(first / "PackA" / ".git" / "HEAD", SHA_A)
    _write(second / "PackA" / ".git" / "HEAD", SHA_B)
    _write(second / "PackB" / "pyproject.toml", 'version = "2.0"\n')
    (second / "__pycache__").mkdir()
    (second / ".hidden").mkdir()
    (second / "Off.disabled").mkdir()
    _write(second / "single_file_node.py", "")
    packs = EF.packs_state([str(first), str(second)])
    assert packs == {"PackA": "git " + SHA_A[:12], "PackB": "version 2.0"}


# --- ComfyUI, torch, the driver ---------------------------------------------

def test_comfyui_outside_comfyui_says_why():
    state = EF.comfyui_state(None)
    assert state["version"].startswith("unknown: not running inside ComfyUI")
    assert state["commit"] == state["version"]


def test_comfyui_version_and_commit_come_from_its_folder(tmp_path):
    _write(tmp_path / "comfyui_version.py", '__version__ = "0.42.1"\n')
    _write(tmp_path / ".git" / "HEAD", SHA_A)
    assert EF.comfyui_state(str(tmp_path)) == {"version": "0.42.1", "commit": SHA_A[:12]}
    (tmp_path / ".git" / "HEAD").unlink()
    (tmp_path / ".git").rmdir()
    assert EF.comfyui_state(str(tmp_path))["commit"] == (
        "unknown: the core folder is not a git checkout")


def test_torch_is_never_imported_and_its_absence_is_named(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    state = EF.torch_state()
    assert state == {k: "unknown: torch is not loaded in this process"
                     for k in ("torch", "cuda", "gpu")}
    assert "torch" not in sys.modules


def test_a_loaded_torch_reports_its_build_and_gpu(monkeypatch):
    fake = types.SimpleNamespace(
        __version__="2.10.0+cu130",
        version=types.SimpleNamespace(cuda="13.0", hip=None),
        cuda=types.SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                                   get_device_name=lambda idx: "NVIDIA GeForce RTX 5080 Laptop GPU"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert EF.torch_state() == {"torch": "2.10.0+cu130", "cuda": "13.0",
                                "gpu": "NVIDIA GeForce RTX 5080 Laptop GPU"}


def _completed(stdout, rc=0):
    return types.SimpleNamespace(stdout=stdout, returncode=rc)


def test_driver_from_one_bounded_nvidia_smi_call(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return _completed("595.71\n")

    monkeypatch.setattr(EF.otr_proc, "run", run)
    assert EF.driver_state() == "595.71"
    ((argv, kwargs),) = calls
    assert argv[0] == "nvidia-smi" and kwargs["timeout"] == 5


@pytest.mark.parametrize("raise_it,expected", [
    (FileNotFoundError(), "unknown: no nvidia-smi"),
    (EF.otr_proc.TimeoutExpired("nvidia-smi", 5), "unknown: nvidia-smi gave no answer within 5 s"),
])
def test_driver_unknowns_say_why(monkeypatch, raise_it, expected):
    def run(argv, **kwargs):
        raise raise_it

    monkeypatch.setattr(EF.otr_proc, "run", run)
    assert EF.driver_state() == expected


def test_a_failing_nvidia_smi_is_not_a_driver_version(monkeypatch):
    monkeypatch.setattr(EF.otr_proc, "run", lambda argv, **kw: _completed("", rc=9))
    assert EF.driver_state() == "unknown: nvidia-smi exited 9 with no driver version"


# --- the fingerprint as a whole ---------------------------------------------

def test_the_fingerprint_is_computed_once_and_handed_out_as_a_copy(monkeypatch):
    calls = []
    monkeypatch.setattr(EF, "compute", lambda: calls.append(1) or {"packs": {"A": "git x"}, "python": "3"})
    first = EF.fingerprint()
    first["packs"]["A"] = "edited"
    first["python"] = "edited"
    assert EF.fingerprint() == {"packs": {"A": "git x"}, "python": "3"}
    assert calls == [1]


def test_a_reader_that_raises_is_named_by_type_only(monkeypatch):
    def boom():
        raise OSError(r"C:\Users\someone\secret\path")

    monkeypatch.setattr(EF, "os_state", boom)
    monkeypatch.setattr(EF, "driver_state", lambda: "1.0")
    env = EF.compute()
    assert env["os"] == "unknown: OSError while reading it"


def test_the_real_fingerprint_carries_no_path_user_or_machine_name(monkeypatch, tmp_path):
    """Read on this machine, with this pack's own custom_nodes folder."""
    monkeypatch.setattr(EF, "_custom_node_dirs", lambda: [str(_REPO.parent)])
    env = EF.compute()
    assert set(env) == {"comfyui", "packs", "python", "os", "driver", "torch", "cuda", "gpu"}
    assert env["python"] == platform.python_version()
    assert _REPO.name in env["packs"]
    text = json.dumps(env)
    for private in (getpass.getuser(), platform.node(), str(_REPO), str(Path.home())):
        if private:
            assert private.lower() not in text.lower(), private
    assert "\\\\" not in text and ":/" not in text.replace("://", "")


def test_stamp_fills_an_empty_meta_and_keeps_an_existing_one(monkeypatch):
    monkeypatch.setattr(EF, "compute", lambda: {"python": "now"})
    meta = {}
    assert EF.stamp(meta) is True
    assert meta == {"env": {"python": "now"}, "env_version": EF.ENV_VERSION}
    kept = {"env": {"python": "then"}, "env_version": 1}
    assert EF.stamp(kept) is False
    assert kept["env"] == {"python": "then"}


# --- in the ledger -----------------------------------------------------------

@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(EF, "compute", lambda: {"python": "first-process"})
    out_dir = str(tmp_path / "episodes" / "signal_lost_env_20261024_000000" / "audio")
    led = PL.Ledger(episode_id="signal_lost_env_20261024_000000", out_dir=out_dir)
    led.test_out_dir = out_dir
    return led


def test_the_first_save_stamps_meta_env(ledger):
    path = ledger.save()
    on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
    assert on_disk["meta"]["env"] == {"python": "first-process"}
    assert on_disk["meta"]["env_version"] == EF.ENV_VERSION


def test_a_later_save_in_a_new_process_keeps_the_first_stamp(ledger, monkeypatch):
    path = ledger.save()
    EF._reset_cache_for_tests()
    monkeypatch.setattr(EF, "compute", lambda: {"python": "second-process"})
    # A fresh in-memory ledger for the same episode (no meta.env in memory):
    # the disk merge carries the stamp forward instead of re-stamping.
    again = PL.Ledger(episode_id="signal_lost_env_20261024_000000", out_dir=ledger.test_out_dir)
    again.save()
    on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
    assert on_disk["meta"]["env"] == {"python": "first-process"}


def test_env_is_run_volatile_for_a_replay():
    assert {"env", "env_version"} <= set(PL._REPLAY_RUN_VOLATILE_META)


def test_the_stamp_is_wired_into_save_and_the_replay_import():
    """The helper proves itself above; this proves something calls it."""
    source = (_REPO / "nodes" / "production_ledger.py").read_text(encoding="utf-8")
    save_body = source[source.index("    def save(self)"):source.index("def _merge_with_disk")]
    assert "otr_env_fingerprint.stamp(" in save_body
    imp = source[source.index("def import_replay_bundle"):]
    imp = imp[:imp.index("led.save()")]
    assert imp.index("for key in _REPLAY_RUN_VOLATILE_META") < imp.index("otr_env_fingerprint.stamp(meta)")


# --- the drift report --------------------------------------------------------

OLD_ENV = {
    "python": "3.12.10", "torch": "2.10.0+cu130", "cuda": "13.0", "driver": "590.10",
    "gpu": "RTX 5080", "os": "Windows 11 10.0.26200",
    "comfyui": {"version": "0.42.0", "commit": "aaaaaaaaaaaa"},
    "packs": {"ComfyUI-KJNodes": "git 111111111111", "Gone": "version 1.0",
              "Same": "git 222222222222"},
}
NEW_ENV = dict(OLD_ENV, driver="595.71", comfyui={"version": "0.42.1", "commit": "bbbbbbbbbbbb"},
               packs={"ComfyUI-KJNodes": "git 333333333333", "Same": "git 222222222222",
                      "Added": "version 0.1"})


def test_the_diff_reads_the_stack_first_then_packs():
    assert drift.diff_env(OLD_ENV, NEW_ENV) == [
        "driver: 590.10 -> 595.71",
        "comfyui version: 0.42.0 -> 0.42.1",
        "comfyui commit: aaaaaaaaaaaa -> bbbbbbbbbbbb",
        "pack changed ComfyUI-KJNodes: git 111111111111 -> git 333333333333",
        "pack added Added: version 0.1",
        "pack removed Gone: version 1.0",
    ]
    assert drift.diff_env(OLD_ENV, OLD_ENV) == []


def _episode(root: Path, ep_id: str, env: dict | None) -> Path:
    meta = {"env": env, "env_version": 1} if env is not None else {}
    return _write(root / ep_id / "audio" / (ep_id + "_ledger.json"),
                  json.dumps({"episode_id": ep_id, "meta": meta}))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    episodes = tmp_path / "otr" / "episodes"
    state = episodes / "_shared" / "state"
    fake_paths = types.SimpleNamespace(
        otr_episodes_root=lambda: episodes,
        otr_state_dir=lambda: state,
        is_reserved_episode_entry=lambda name: str(name).startswith("_"),
    )
    monkeypatch.setattr(drift, "_paths", lambda: fake_paths)
    return episodes


def test_two_episodes_name_what_moved(tree, capsys):
    _episode(tree, "ep_a", OLD_ENV)
    _episode(tree, "ep_b", NEW_ENV)
    assert drift.main(["ep_a", "ep_b"]) == 1
    out = capsys.readouterr().out
    assert "[env-drift] ep_a -> ep_b" in out
    assert "  driver: 590.10 -> 595.71" in out
    assert drift.main(["ep_a", str(tree / "ep_a")]) == 0
    assert "no difference" in capsys.readouterr().out


def test_an_episode_from_before_the_stamp_is_named_not_guessed(tree, capsys):
    _episode(tree, "ep_old", None)
    _episode(tree, "ep_b", NEW_ENV)
    assert drift.main(["ep_old", "ep_b"]) == 2
    assert "has no meta.env" in capsys.readouterr().out
    assert drift.main(["nope", "ep_b"]) == 2


def test_no_arguments_compares_the_newest_stamped_episode_with_known_good(tree, capsys):
    a = _episode(tree, "ep_a", OLD_ENV)
    b = _episode(tree, "ep_b", NEW_ENV)
    newest_unstamped = _episode(tree, "ep_c", None)
    os.utime(a, (1, 1))
    os.utime(b, (2, 2))
    os.utime(newest_unstamped, (3, 3))
    assert drift.main([]) == 2
    assert "no known-good snapshot" in capsys.readouterr().out
    assert drift.main(["--freeze", "ep_a"]) == 0
    snap = json.loads((tree / "_shared" / "state" / drift.KNOWN_GOOD_NAME).read_text(encoding="utf-8"))
    assert snap["episode"] == "ep_a" and snap["env"] == OLD_ENV and snap["how"] == "--freeze"
    capsys.readouterr()
    assert drift.main([]) == 1
    out = capsys.readouterr().out
    assert "known-good ep_a" in out and "-> ep_b" in out
    assert "driver: 590.10 -> 595.71" in out


def test_record_known_good_writes_the_snapshot_and_never_raises(tree):
    _episode(tree, "ep_ok", NEW_ENV)
    assert drift.record_known_good("ep_ok") == (
        "known-good recorded: ep_ok -> " + drift.KNOWN_GOOD_NAME)
    snap = json.loads((tree / "_shared" / "state" / drift.KNOWN_GOOD_NAME).read_text(encoding="utf-8"))
    assert snap["env"] == NEW_ENV and snap["how"] == "runner SUCCESS"
    assert drift.record_known_good(None) == "known-good not recorded: the episode id was not read"
    assert drift.record_known_good("missing").startswith("known-good not recorded: no episode")
    _episode(tree, "ep_unstamped", None)
    assert "has no meta.env" in drift.record_known_good("ep_unstamped")
    assert [p.name for p in (tree / "_shared" / "state").iterdir()] == [drift.KNOWN_GOOD_NAME]


# --- the runner's known-good step -------------------------------------------

def test_the_runner_records_known_good_only_for_a_confirmed_obs_file(monkeypatch):
    recorded = []
    monkeypatch.setattr(canonical.otr_env_drift, "record_known_good",
                        lambda ep: recorded.append(ep) or "known-good recorded: %s" % ep)
    monkeypatch.setattr(canonical, "published_obs_file", lambda *a: (None, "the obs file is zero bytes"))
    assert canonical.record_known_good_after_success({}, "p", "ep") == (
        "known-good not recorded: the obs file is zero bytes")
    assert canonical.record_known_good_after_success({}, "p", None) == (
        "known-good not recorded: the episode id was not read")
    assert recorded == []
    monkeypatch.setattr(canonical, "published_obs_file", lambda *a: ("x_final.mp4", ""))
    assert canonical.record_known_good_after_success({}, "p", "ep") == "known-good recorded: ep"
    monkeypatch.setattr(canonical, "published_obs_file",
                        lambda *a: (_ for _ in ()).throw(RuntimeError("history broke")))
    assert canonical.record_known_good_after_success({}, "p", "ep") == (
        "known-good not recorded: RuntimeError: history broke")


def test_main_prints_the_known_good_line_after_success(monkeypatch, tmp_path, capsys):
    class _Live:
        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def set_prompt_id(self, pid):
            pass

        def wrap(self, on_tick):
            return on_tick

        def stop(self):
            pass

        def summary(self):
            return "text_frames=0"

    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(canonical, "LiveView", _Live)
    monkeypatch.setattr(canonical, "submit_prompt", lambda p, client_id=None: "pid")
    monkeypatch.setattr(canonical, "poll_history", lambda *a, **k: ("SUCCESS", ""))
    monkeypatch.setattr(canonical, "episode_of_prompt", lambda pid: "ep")
    monkeypatch.setattr(canonical, "record_known_good_after_success",
                        lambda prompt, pid, ep: "known-good recorded: %s" % ep)
    assert canonical.main(["--dump-prompt", str(tmp_path / "p.json")]) == 0
    out = capsys.readouterr().out
    assert out.index("RESULT SUCCESS") < out.index("[canonical-api] known-good recorded: ep")
