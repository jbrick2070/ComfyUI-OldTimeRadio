"""The A/B harness must never report a number from a render that is not the
arm's own.

WHY THIS FILE EXISTS. `scripts/otr_music_ab.py` is a measuring instrument, and
on the night it was written it produced two wrong measurements in a row for the
same reason: it decided which episode an arm had rendered by looking for the
newest directory. The first time, the newest directory was an empty ``pending_``
stage folder and the arm measured nothing; the second time the settings never
reached the server at all (``OTR_SA3_*`` is read inside the ComfyUI process, not
the runner) and the arm cheerfully reported the shipped defaults as if they were
the experiment.

A harness that attributes somebody else's music is worse than one that reports
nothing, because the number looks like evidence. Every binding below therefore
FAILS CLOSED: the episode must be named in this arm's own runner log, its cues
must have been written after the arm started, and its receipt must carry the
settings the arm asked for (codex, 2026-09-12).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "otr_music_ab.py"


def _load():
    spec = importlib.util.spec_from_file_location("otr_music_ab_under_test", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AB = _load()


def _episode(root: Path, name: str, *, cues=("opening",), receipt=None,
             age_s: float = 0.0) -> Path:
    """A minimal episode on disk: music cues, and the ledger receipt beside them."""
    episode = root / name
    audio = episode / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    for cue in cues:
        wav = audio / ("music_cue_%s.wav" % cue)
        wav.write_bytes(b"RIFF placeholder")
        if age_s:
            stamp = time.time() - age_s
            import os
            os.utime(wav, (stamp, stamp))
    if receipt is not None:
        (audio / "episode_ledger.json").write_text(json.dumps({
            "music": [{"cue_id": cue, "render_receipt": {"palette_key": "house",
                                                         "engine_prompt": "strings",
                                                         "params": receipt}}
                      for cue in cues]}), encoding="utf-8")
    return episode


def test_an_arm_refuses_an_episode_whose_receipt_is_not_its_request(tmp_path):
    """THE INERT-ARM CASE, and it really happened: the arm asked for the lcm
    sampler, the server had never heard of it, and the episode that came back
    was the shipped recipe wearing the arm's name."""
    episode = _episode(tmp_path, "ep_20260912_010101",
                       receipt={"sampler": "dpmpp_3m_sde_gpu", "steps": 100, "cfg": 7.0})
    assert not AB.receipt_matches(episode, {"OTR_SA3_SAMPLER": "lcm"})
    assert not AB.receipt_matches(episode, {"OTR_SA3_STEPS": "8"})
    # the same settings, spelled as the environment spells them, do match
    assert AB.receipt_matches(episode, {"OTR_SA3_STEPS": "100", "OTR_SA3_CFG": "7.0"})


def test_an_arm_that_asks_for_nothing_has_nothing_to_prove(tmp_path):
    """The shipped-defaults arm sets no engine variable, so there is no receipt
    field to compare -- it must not be refused for that."""
    episode = _episode(tmp_path, "ep_20260912_010102", receipt={"sampler": "anything"})
    assert AB.receipt_matches(episode, {})


def test_no_receipt_at_all_fails_closed(tmp_path):
    """An episode that never wrote a music receipt cannot be shown to be this
    arm's, so it is not measured."""
    episode = _episode(tmp_path, "ep_20260912_010103", receipt=None)
    assert not AB.receipt_matches(episode, {"OTR_SA3_SAMPLER": "lcm"})


def test_the_context_ratio_is_compared_as_an_equality_not_a_floor(tmp_path):
    """CURSOR R3'S COUNTEREXAMPLE, and it is the sharpest finding of the three
    reviews: the first cut checked `seconds_total >= duration_s * ratio`, which
    a DEFAULT 3x episode satisfies for a 1.0 control arm (36 >= 12). The one
    knob this harness exists to A/B was the one it could not tell apart from the
    thing it controls for. The engine records the ratio it used, and the
    comparison is equality."""
    control = _episode(tmp_path, "ep_20260912_010104",
                       receipt={"duration_s": 12.0, "seconds_total": 12.0,
                                "context_ratio": 1.0})
    assert AB.receipt_matches(control, {"OTR_SA3_CONTEXT_RATIO": "1.0"})
    assert not AB.receipt_matches(control, {"OTR_SA3_CONTEXT_RATIO": "3.0"})
    shipped = _episode(tmp_path, "ep_20260912_010104b",
                       receipt={"duration_s": 12.0, "seconds_total": 36.0,
                                "context_ratio": 3.0})
    assert not AB.receipt_matches(shipped, {"OTR_SA3_CONTEXT_RATIO": "1.0"}), (
        "a shipped 3x episode is not a 1.0 control arm")
    # a receipt that never recorded the ratio proves nothing
    bare = _episode(tmp_path, "ep_20260912_010104c", receipt={"sampler": "lcm"})
    assert not AB.receipt_matches(bare, {"OTR_SA3_CONTEXT_RATIO": "3.0"})


def test_an_empty_pending_directory_is_not_an_episode(tmp_path):
    """The first wrong measurement: a render leaves ``pending_`` stage folders
    that are NEWER than the finished episode, and the harness measured one."""
    (tmp_path / "pending_video" / "audio").mkdir(parents=True)
    assert not AB.cues_written_after(tmp_path / "pending_video", time.time() - 60)


def test_music_written_before_the_arm_started_is_not_the_arm_s(tmp_path):
    episode = _episode(tmp_path, "ep_20260912_010105", age_s=600)
    assert AB.cues_written_after(episode, time.time() - 3600)
    assert not AB.cues_written_after(episode, time.time() - 60)


def test_there_is_no_newest_directory_fallback_left(tmp_path, monkeypatch):
    """CUT ENTIRELY (cursor r3). The runner now prints the episode it produced,
    so a timestamp heuristic has nothing left to do -- and while it existed, an
    arm that rendered NOTHING (a timeout, a crash, a leg still running when the
    watcher gave up) was handed whichever foreign episode happened to finish in
    its window."""
    assert not hasattr(AB, "newest_episode")
    monkeypatch.setattr(AB, "EPISODES", tmp_path)
    _episode(tmp_path, "ep_20260912_010106")
    log = tmp_path / "silent.log"
    log.write_text("[canonical-api] RESULT SUCCESS prompt_id=abc", encoding="utf-8")
    assert AB.episode_from_log(log, time.time() - 60) is None


def test_the_runner_line_the_harness_reads_is_the_one_the_runner_prints():
    """THE BINDER WAS A NO-OP (cursor r3). `episode_from_log` matched
    `episodes/<id>` and `scripts/otr_canonical_api_run.py` printed QUEUED,
    RESULT and prompt_id -- never a path -- so the preferred binding never fired
    once and every arm silently fell through to the heuristic. A synthetic log
    in a test proves the regex, never the wiring, so this reads the runner."""
    runner = (ROOT / "scripts" / "otr_canonical_api_run.py").read_text(encoding="utf-8")
    assert 'print(f"[canonical-api] EPISODE episodes/{episode}"' in runner
    assert "def episode_of_prompt(" in runner


def test_the_log_named_episode_still_has_to_be_fresh(tmp_path, monkeypatch):
    """The runner log is this arm's, but a name in it is not proof the music is:
    an episode it merely mentions may predate the arm entirely."""
    monkeypatch.setattr(AB, "EPISODES", tmp_path)
    _episode(tmp_path, "stale_20260912_010108", age_s=600)
    log = tmp_path / "arm.log"
    log.write_text(r"wrote C:\out\otr\episodes\stale_20260912_010108\audio\x.wav",
                   encoding="utf-8")
    assert AB.episode_from_log(log, time.time() - 3600) is not None
    assert AB.episode_from_log(log, time.time() - 60) is None


# --------------------------------------------------------------------------- #
# codex r2: four holes in the bindings above, each of which let an episode pass
# --------------------------------------------------------------------------- #
def test_a_setting_the_receipt_does_not_record_fails_closed(tmp_path):
    """"No field" used to read as "no objection": a ledger with empty params
    passed every check an arm made."""
    episode = _episode(tmp_path, "ep_20260912_020101", receipt={})
    assert not AB.receipt_matches(episode, {"OTR_SA3_CFG": "1.0"})
    episode = _episode(tmp_path, "ep_20260912_020102",
                       receipt={"sampler": "lcm"})  # right sampler, no cfg
    assert not AB.receipt_matches(episode, {"OTR_SA3_SAMPLER": "lcm",
                                            "OTR_SA3_CFG": "1.0"})


def test_an_arm_the_receipt_cannot_prove_is_refused_outright(tmp_path):
    """A server-side variable the receipt never records cannot be attributed to
    any episode, so the arm is refused rather than measured. The message names
    the fix: record it first."""
    episode = _episode(tmp_path, "ep_20260912_020103",
                       receipt={"sampler": "lcm"})
    assert not AB.receipt_matches(episode, {"OTR_MASTER_TARGET_LUFS": "-16"})
    # and the ones it DOES record are not refused
    assert AB.receipt_matches(episode, {"OTR_SA3_SAMPLER": "lcm"})


def test_the_engine_negative_prompt_is_what_gets_compared(tmp_path):
    """The durable receipt carries the COMPOSER'S negative beside the engine's
    own; `OTR_SA3_NEG_PROMPT` overrides the composer inside the engine, so the
    binding has to read the engine's."""
    episode = _episode(tmp_path, "ep_20260912_020104",
                       receipt={"negative_prompt": "no drums"})
    assert AB.receipt_matches(episode, {"OTR_SA3_NEG_PROMPT": "no drums"})
    assert not AB.receipt_matches(episode, {"OTR_SA3_NEG_PROMPT": "no loops"})


def test_a_cue_with_no_receipt_row_of_its_own_fails_closed(tmp_path):
    """The measurement is per wav and the check was per receipt row, so an
    episode with two cues and one row passed on the strength of half its
    music."""
    episode = _episode(tmp_path, "ep_20260912_020105", cues=("opening",),
                       receipt={"sampler": "lcm"})
    (episode / "audio" / "music_cue_closing.wav").write_bytes(b"RIFF placeholder")
    assert not AB.receipt_matches(episode, {"OTR_SA3_SAMPLER": "lcm"})


def test_the_freshness_proof_has_no_grace_window(tmp_path):
    """A proof with a tolerance is not a proof: a second of slack is wide enough
    to admit a neighbouring render, and an arm's music is written minutes into
    its own leg."""
    import os
    episode = _episode(tmp_path, "ep_20260912_020106")
    started = time.time()
    stamp = started - 0.5
    for cue in (episode / "audio").glob("music_cue_*.wav"):
        os.utime(cue, (stamp, stamp))
    assert not AB.cues_written_after(episode, started)


def test_an_arm_can_express_a_value_that_contains_commas():
    """cursor r3: every real negative prompt is a comma list, so the anti-loop
    negative -- the single biggest lever measured in this campaign -- could not
    be written as an arm at all. A semicolon separates the pairs when one is
    there, and commas inside a value are then just commas."""
    with pytest.raises(SystemExit):
        AB.parse_arm("neg:OTR_SA3_NEG_PROMPT=loop, ostinato, drum machine")
    name, env = AB.parse_arm("neg:OTR_SA3_NEG_PROMPT=loop, ostinato, drum machine;OTR_SA3_CFG=7")
    assert name == "neg"
    assert env == {"OTR_SA3_NEG_PROMPT": "loop, ostinato, drum machine",
                   "OTR_SA3_CFG": "7"}


def test_a_defaults_arm_after_a_recipe_arm_boots_a_clean_server(monkeypatch):
    """cursor r3, and it is the quietest of the three: only an arm with
    server-side settings booted a server, so `--arms lcm shipped` measured the
    LCM server twice and called the second one the baseline. Argv order was
    load-bearing and nothing said so."""
    booted = []
    monkeypatch.setattr(AB, "boot_server", lambda env, log: booted.append(dict(env)) or True)
    monkeypatch.setattr(AB, "server_is_up", lambda: True)
    monkeypatch.setattr(AB, "episode_from_log", lambda log, after: None)

    class _Args:
        no_boot = False
        profile = "p"
        acts = 1
        bank = None
    monkeypatch.setattr(AB.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0})())

    AB.run_arm("shipped", {}, _Args(), server_is_dirty=False)
    assert booted == [], "the first defaults arm inherits a clean resident server"
    AB.run_arm("shipped", {}, _Args(), server_is_dirty=True)
    assert booted == [{}], "after a recipe arm it boots its own, with no overrides"
