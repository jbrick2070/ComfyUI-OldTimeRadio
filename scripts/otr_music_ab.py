"""Music A/B on the CANONICAL path -- one arm per env recipe, measured on the
cue stems the real workflow actually rendered.

WHY THIS EXISTS, and why it is not a bench (operator, 2026-09-12: *"you need
to create a new harness from canonical to be safe"*). The music questions that
came up that night -- is the cue looping, is the sampler recipe right, is the
level too hot -- were first explored on an in-process bench that called
``generate_clip`` directly. That bench was fast and it was WRONG in a way worth
recording: reproducing a shipped cue from its own ledger receipt (same prompt,
same seed, same sampler settings) came back at correlation 0.92 and nine
decibels hotter, because a bare process does not reproduce the server's model
load. A lab number that cannot reproduce a shipped artifact cannot qualify a
change.

So this harness has exactly one way to render: ``otr_canonical_api_run.py``,
which loads ``workflows/otr_canonical.json`` (CLAUDE.md section 0, absolute
since the bench carve-out was struck on 2026-08-23). Every arm is a real
episode, published to ``otr/obs/`` like any other.

WHAT IT MEASURES, per music cue wav in each episode:

* ``loopiness`` -- the strongest autocorrelation of the 10 ms loudness envelope
  between 0.25 s and 6 s. This is the number that matches the complaint: a cue
  that repeats a short figure scores high, a cue that develops scores low.
  Measured on the shipped 2026-09-11 episodes: opening 0.485, closing 0.713.
* ``peak_dbfs`` / ``clipped`` -- the writer casts to int16 and hard-clips, and
  nothing on the music path normalises, so a hot render ships as square waves.
  Measured across 1,984 cues on the dev box: 14% carry more than four clipped
  samples, worst 4,710.
* ``rms_dbfs`` -- the level the cue actually sits at under the dialogue.

IT QUALIFIES NOTHING. It reports numbers for a listening decision; the verdict
on music is the operator's ear (docs/OTR_STANDING_RULINGS.md, "judge it as
radio drama"). A difference here is a reason to listen, never a reason to ship.

USAGE (a server must already be listening on :8000 -- reset and boot per
CLAUDE.md sections 4 and 5):

    python scripts/otr_music_ab.py --arms shipped
    python scripts/otr_music_ab.py --banks shakespeare \\
        --arms "shipped" "lcm:OTR_SA3_SAMPLER=lcm,OTR_SA3_SCHEDULER=simple,OTR_SA3_STEPS=8,OTR_SA3_CFG=1.0"
    python scripts/otr_music_ab.py --measure-only   # re-measure the newest episodes

An arm is ``name`` or ``name:KEY=VALUE,KEY=VALUE``; the bare name ``shipped``
sets nothing and renders the current defaults.

AN ARM THAT SETS AN ENGINE VARIABLE GETS ITS OWN SERVER. ``OTR_SA3_*`` and
friends are read inside the ComfyUI process, and the runner only POSTs a
prompt to whatever server is already listening -- so setting them on the
runner does nothing at all. Such an arm therefore resets and boots its own
server first (CLAUDE.md sections 4 and 5), which costs about a minute.
``--no-boot`` refuses those arms loudly rather than measuring nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "otr_canonical_api_run.py"
LAUNCHER = REPO_ROOT / "scripts" / "_otr_soak_server_launch.cmd"
BOOT_SCRIPT = REPO_ROOT / "scripts" / "_otr_music_ab_boot.ps1"
CANONICAL = REPO_ROOT / "workflows" / "otr_canonical.json"
#: Where episodes land. The operator's own output tree, never a test rig
#: (memory: "obs is always the local output folder").
EPISODES = (Path(os.environ.get("OTR_OUTPUT_DIR")
                 or r"C:\Users\jeffr\Documents\ComfyUI\output")
            / "otr" / "episodes")
DEFAULT_PROFILE = "otr_w45_still_flat"
LEG_TIMEOUT_S = 3600
BOOT_TIMEOUT_S = 300
#: Engine settings that only exist inside the SERVER process. An arm that
#: sets one of these needs its own server, because the runner merely POSTs a
#: prompt to a server that was booted with whatever environment it was booted
#: with. Setting them on the runner is a silent no-op -- the first cut of
#: this harness did exactly that and would have reported "no difference".
SERVER_SIDE_PREFIXES = ("OTR_SA3_", "OTR_MUSIC_", "OTR_MASTER_", "OTR_SEGMENT_")


def _say(message: str) -> None:
    print("[music-ab] %s  %s" % (dt.datetime.now().strftime("%H:%M:%S"), message),
          flush=True)


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _dbfs(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else float("-inf")


def loopiness(mono, sample_rate: int):
    """``(lag_s, strength)`` of the strongest repeat in the loudness envelope
    between 0.25 s and 6 s, or ``None``.

    A 10 ms RMS envelope, mean-removed, autocorrelated and normalised at zero
    lag. Music that develops scores low; music that repeats a bar scores high.
    Deliberately crude and deliberately ONE estimator: the point is to compare
    two arms of the same pipeline, not to characterise audio in general.
    """
    import numpy as np

    hop = max(1, int(0.01 * sample_rate))
    frames = len(mono) // hop
    if frames < 64:
        return None
    envelope = np.sqrt(np.mean(mono[:frames * hop].reshape(frames, hop) ** 2, axis=1))
    envelope = envelope - envelope.mean()
    if not np.any(envelope):
        return None
    correlation = np.correlate(envelope, envelope, mode="full")[frames - 1:]
    zero = correlation[0]
    if zero <= 0:
        return None
    correlation = correlation / zero
    low, high = int(0.25 / 0.01), min(int(6.0 / 0.01), len(correlation) - 1)
    if high <= low:
        return None
    lag = low + int(np.argmax(correlation[low:high]))
    return round(lag * 0.01, 2), round(float(correlation[lag]), 3)


def measure_episode(episode_dir: Path) -> list[dict]:
    """One row per music cue wav in ``episode_dir``."""
    import numpy as np
    import soundfile as sf

    rows = []
    for path in sorted(glob.glob(str(episode_dir / "audio" / "music_cue_*.wav"))):
        try:
            data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
        except Exception as exc:  # noqa: BLE001 -- a measurement never fails a run
            rows.append({"cue": Path(path).stem, "error": str(exc)[:120]})
            continue
        if not len(data):
            rows.append({"cue": Path(path).stem, "error": "empty"})
            continue
        mono = data.mean(axis=1)
        peak = float(np.abs(data).max())
        found = loopiness(mono, int(sample_rate))
        rows.append({
            "cue": Path(path).stem.replace("music_cue_", ""),
            "loop_lag_s": found[0] if found else None,
            "loopiness": found[1] if found else None,
            "peak_dbfs": round(_dbfs(peak), 2),
            "rms_dbfs": round(_dbfs(float(np.sqrt(np.mean(data ** 2)))), 2),
            "clipped": int(np.count_nonzero(np.abs(data) >= 0.9999)),
            "seconds": round(len(data) / sample_rate, 2),
        })
    return rows


def receipt_of(episode_dir: Path) -> dict:
    """What the ledger says the engine was asked for, when it recorded it."""
    # THE EPISODE'S OWN LEDGER FIRST. A stray `pending_*_ledger.json` can
    # sit beside it and would otherwise win on glob order (cursor r3).
    own = episode_dir / "audio" / ("%s_ledger.json" % episode_dir.name)
    ordered = ([str(own)] if own.is_file() else []) + [
        p for p in sorted(glob.glob(str(episode_dir / "audio" / "*_ledger.json")))
        if Path(p) != own]
    for path in ordered:
        try:
            ledger = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        out = {}
        for row in ledger.get("music") or []:
            receipt = row.get("render_receipt") or {}
            if receipt:
                out[str(row.get("cue_id"))] = {
                    "palette": receipt.get("palette_key"),
                    "params": receipt.get("params") or {},
                    "engine_prompt": (receipt.get("engine_prompt") or "")[:240],
                    # The COMPOSER'S negative, kept for reading. The one the
                    # binding compares is the ENGINE'S, inside `params`,
                    # because OTR_SA3_NEG_PROMPT overrides the composer.
                    "negative_prompt": receipt.get("negative_prompt") or "",
                }
        return out
    return {}


# --------------------------------------------------------------------------- #
# rendering -- the ONE path
# --------------------------------------------------------------------------- #
def needs_its_own_server(env_overrides: dict) -> bool:
    """True when a setting is read inside the server rather than the runner."""
    return any(key.startswith(SERVER_SIDE_PREFIXES) for key in env_overrides)


def server_is_up() -> bool:
    import socket
    with socket.socket() as probe:
        probe.settimeout(2.0)
        return probe.connect_ex(("127.0.0.1", 8000)) == 0


def boot_server(env_overrides: dict, log_path: Path) -> bool:
    """Reset and boot ONE server with ``env_overrides`` in its environment.

    Delegates to ``scripts/_otr_music_ab_boot.ps1``, and the reason is measured
    rather than stylistic -- BOTH Python-side attempts failed SILENTLY on
    2026-09-12:

    * a ``wmic``-based selective kill does nothing at all, because wmic is gone
      on Windows 11; the old server kept :8000 and the new one could not bind;
    * ``subprocess.Popen`` on the launcher ``.cmd`` with ``DETACHED_PROCESS``
      returns a pid and never runs the batch -- no log, no server, no error.

    Each failure cost five minutes of boot timeout per arm and would have
    reported "the server did not come up" without saying why. PowerShell's
    ``Get-CimInstance`` + ``Start-Process -FilePath`` is the combination
    CLAUDE.md sections 4 and 5 already prescribe and the only one proven here.

    The arm's settings travel by ENVIRONMENT: this process -> powershell ->
    Start-Process -> the launcher -> ComfyUI, which is the only place the
    engine reads them.
    """
    env = dict(os.environ, PYTHONUTF8="1", **env_overrides)
    _say("booting a server for this arm with %s" % (env_overrides or "no overrides"))
    done = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(BOOT_SCRIPT),
         "-Launcher", str(LAUNCHER),
         "-LogPath", str(log_path),
         "-TimeoutSeconds", str(BOOT_TIMEOUT_S)],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        timeout=BOOT_TIMEOUT_S + 120)
    for line in (done.stdout or "").splitlines():
        if line.strip():
            _say(line.strip())
    if done.returncode != 0:
        _say("boot failed (exit %s); see %s" % (done.returncode, log_path))
        return False
    return True


def parse_arm(text: str) -> tuple[str, dict]:
    """``name`` or ``name:KEY=VALUE,KEY=VALUE`` -> ``(name, env)``.

    A SEMICOLON SEPARATES PAIRS WHEN ONE IS PRESENT, and then commas are
    ordinary characters inside a value (cursor r3, 2026-09-12). Every real
    negative prompt is a comma list, so the anti-loop negative -- the single
    biggest lever measured in this campaign -- could not be expressed as an
    arm at all: it either truncated at the first comma or exited.
    """
    name, _, rest = text.partition(":")
    env = {}
    separator = ";" if ";" in rest else ","
    for pair in rest.split(separator) if rest else []:
        if not pair.strip():
            continue
        key, _, value = pair.partition("=")
        if not key.strip() or not value:
            raise SystemExit("bad arm setting %r in %r (want KEY=VALUE)" % (pair, text))
        env[key.strip()] = value.strip()
    return name.strip() or "arm", env


def episode_from_log(log_path: Path, after: float) -> Path | None:
    """The episode THIS arm rendered, taken from the runner's own output.

    Preferred over "the newest directory" because the episode root is shared:
    a second window, an overnight batch or a leg already in flight can finish
    into it while an arm runs, and a timestamp heuristic would then attribute
    somebody else's music to this arm with no error at all.

    The log is this arm's and is truncated at its start, so a name in it is
    this arm's work -- but only if the CUES were written after the arm began
    (codex, 2026-09-12). A runner that reuses or merely mentions an earlier
    episode would otherwise hand back music nothing in this arm rendered.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    seen = []
    for match in re.finditer(r"episodes[\\/]([A-Za-z0-9_\-]+_\d{8}_\d{6})", text):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    for name in reversed(seen):
        candidate = EPISODES / name
        if cues_written_after(candidate, after):
            return candidate
        if (candidate / "audio").is_dir():
            _say("%s is named in this arm's log but its music predates the "
                 "arm -- not measuring it" % name)
    return None


def cues_written_after(episode_dir: Path, after: float) -> bool:
    """Does this episode hold music cues that were written after ``after``?

    NO GRACE WINDOW (codex r2). The first cut allowed a second of slack for
    filesystem granularity, which is both unnecessary -- an arm's music is
    written MINUTES into its render, never in the first second -- and a hole
    wide enough to admit a neighbouring render. A proof with a tolerance is
    not a proof.
    """
    cues = glob.glob(str(episode_dir / "audio" / "music_cue_*.wav"))
    return bool(cues) and all(Path(c).stat().st_mtime >= after for c in cues)


def run_arm(name: str, env_overrides: dict, args, *, server_is_dirty=False) -> dict:
    """Render ONE canonical episode with ``env_overrides`` applied.

    ``server_is_dirty`` means an earlier arm in this run booted a server with
    ITS settings. A shipped-defaults arm that followed one used to reuse that
    server and report the mutated recipe as the baseline -- argv order was
    load-bearing and nothing said so (cursor r3, 2026-09-12). Such an arm now
    boots a clean server of its own.
    """
    log_path = REPO_ROOT / "tmp" / ("_music_ab_%s.log" % "".join(
        c if c.isalnum() else "_" for c in name))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wants_own_server = bool(env_overrides) and needs_its_own_server(env_overrides)
    if wants_own_server or server_is_dirty:
        if args.no_boot:
            raise SystemExit(
                "arm %r needs its own server (%s), but --no-boot was given. "
                "The runner only POSTs to a server that is already running, so "
                "the arm would measure whatever the resident server was booted "
                "with. Drop --no-boot, or boot the server yourself." % (
                    name, ", ".join(sorted(env_overrides)) if wants_own_server
                    else "an earlier arm left this server holding ITS settings"))
        boot_log = REPO_ROOT / "tmp" / ("_music_ab_server_%s.log" % "".join(
            c if c.isalnum() else "_" for c in name))
        if not boot_server(env_overrides, boot_log):
            return {"arm": name, "env": env_overrides, "exit": -2,
                    "minutes": 0.0, "log": boot_log.name, "episode": None,
                    "error": "the server did not come up for this arm"}
    elif not server_is_up():
        raise SystemExit(
            "no server is listening on :8000. Boot one (scripts/"
            "_otr_soak_server_launch.cmd) or let an arm with server-side "
            "settings boot its own.")
    command = [sys.executable, str(RUNNER),
               "--profile", args.profile,
               "--act-count", str(args.acts),
               "--timeout", str(LEG_TIMEOUT_S)]
    if args.bank:
        command += ["--source-bank", args.bank]
    env = dict(os.environ, PYTHONUTF8="1", **env_overrides)
    started = time.time()
    _say("arm %r: %s" % (name, env_overrides or "shipped defaults"))
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("### arm=%s env=%s start=%s\n" % (name, env_overrides, dt.datetime.now()))
        handle.flush()
        try:
            code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                  cwd=str(REPO_ROOT), env=env,
                                  timeout=LEG_TIMEOUT_S + 600).returncode
        except subprocess.TimeoutExpired:
            code = -1
            handle.write("\n### runner timeout\n")
    minutes = round((time.time() - started) / 60.0, 1)
    episode = episode_from_log(log_path, started)
    if episode is not None and not receipt_matches(episode, env_overrides):
        _say("REFUSING %s: its receipt does not carry this arm's settings, "
             "so either it is another render or this arm never reached the "
             "engine. Either way the number would be a lie." % episode.name)
        episode = None
    result = {"arm": name, "env": env_overrides, "exit": code, "minutes": minutes,
              "log": log_path.name, "episode": episode.name if episode else None}
    if episode is not None:
        result["cues"] = measure_episode(episode)
        result["receipt"] = receipt_of(episode)
    _say("arm %r finished: exit=%s %.1f min episode=%s" % (
        name, code, minutes, result["episode"]))
    return result


#: Server-side settings the render receipt RECORDS, and therefore the only
#: ones an arm can be proved to have run (codex r2, 2026-09-12). An arm that
#: sets a server-side variable outside this map is refused rather than
#: measured, because a number nothing can attribute is worse than none.
_RECEIPT_FIELDS = {"OTR_SA3_SAMPLER": "sampler",
                   "OTR_SA3_SCHEDULER": "scheduler",
                   "OTR_SA3_STEPS": "steps",
                   "OTR_SA3_CFG": "cfg",
                   "OTR_SA3_DENOISE": "denoise",
                   "OTR_SA3_NEG_PROMPT": "negative_prompt",
                   "OTR_SA3_CONTEXT_RATIO": "context_ratio"}


def _safe_cue_name(cue_id) -> str:
    """The cue id as the wav writer spells it (`stable_audio_theme.py:595`)."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(cue_id))


def _cue_id_of_wav(path) -> str:
    """`.../music_cue_opening.wav` -> `opening`."""
    return Path(path).stem[len("music_cue_"):]


def receipt_matches(episode_dir: Path, env_overrides: dict) -> bool:
    """Does this episode's music receipt carry the settings THIS arm asked
    for? (codex, 2026-09-12.)

    Naming and timing bind an episode to an arm's WINDOW; this binds it to
    the arm's REQUEST. The engine records what it actually ran, so an arm can
    refuse an episode whose sampler or step count is not the one it asked
    for -- which is also the check that would have caught the inert arms of
    2026-09-12, where the settings never reached the server at all.

    Its reach is exactly the recorded fields, and an arm that reaches past
    them is REFUSED rather than measured (codex r2): a server-side setting
    the receipt does not record cannot be attributed to any episode, and a
    number nothing can attribute is worse than no number. There is no
    exception and no inequality -- every check here is an EQUALITY, because
    the first cut compared the context ratio as a floor and a default 3x
    episode satisfied a 1.0 control arm perfectly (cursor r3). An arm that
    sets nothing server-side has nothing to check and passes.
    """
    unverifiable = [key for key in env_overrides
                    if key.startswith(SERVER_SIDE_PREFIXES)
                    and key not in _RECEIPT_FIELDS]
    if unverifiable:
        _say("REFUSING: %s is read inside the server and the receipt does not "
             "record it, so no episode can be proved to be this arm's. Add it "
             "to the render receipt before A/B-ing it."
             % ", ".join(sorted(unverifiable)))
        return False
    checks = {field: env_overrides[key] for key, field in _RECEIPT_FIELDS.items()
              if key in env_overrides}
    if not checks:
        return True
    receipt = receipt_of(episode_dir)
    if not receipt:
        _say("no music receipt in %s -- cannot prove it is this arm's"
             % episode_dir.name)
        return False
    # EVERY MEASURED CUE NEEDS A RECEIPT ROW (codex r2). The measurement is
    # per wav and the check was per receipt row, so an episode with two cues
    # and one row passed on the strength of half its music.
    measured = {_cue_id_of_wav(path) for path in
                glob.glob(str(episode_dir / "audio" / "music_cue_*.wav"))}
    receipted = {_safe_cue_name(cue) for cue in receipt}
    missing = measured - receipted
    if missing:
        _say("%s measures %s with no receipt row -- cannot prove it is this "
             "arm's" % (episode_dir.name, ", ".join(sorted(missing))))
        return False
    for cue, row in receipt.items():
        params = row.get("params") or {}
        for field, wanted in checks.items():
            if field not in params:
                # FAIL CLOSED ON AN ABSENT FIELD. A receipt that does not
                # record what this arm set cannot prove the arm ran, and
                # "no field" used to read as "no objection".
                _say("%s cue %s has no %s in its receipt, so this arm's "
                     "%s=%s is unproven" % (episode_dir.name, cue, field,
                                            field, wanted))
                return False
            if not _same_setting(params.get(field), wanted):
                _say("%s cue %s ran %s=%s but this arm asked for %s"
                     % (episode_dir.name, cue, field, params.get(field), wanted))
                return False
    return True


def _same_setting(actual, wanted) -> bool:
    """``100`` and ``"100"`` and ``"100.0"`` are the same request."""
    try:
        return abs(float(actual) - float(wanted)) < 1e-6
    except (TypeError, ValueError):
        return str(actual).strip() == str(wanted).strip()


def report(results: list[dict]) -> None:
    print()
    print("=" * 96)
    print("%-14s %-10s %10s %8s %10s %8s %8s" % (
        "arm", "cue", "loopiness", "lag s", "peak dBFS", "rms", "clipped"))
    for row in results:
        for cue in row.get("cues") or []:
            if cue.get("error"):
                print("%-14s %-10s  %s" % (row["arm"], cue["cue"], cue["error"]))
                continue
            print("%-14s %-10s %10s %8s %10.2f %8.2f %8d" % (
                row["arm"], cue["cue"], cue["loopiness"], cue["loop_lag_s"],
                cue["peak_dbfs"], cue["rms_dbfs"], cue["clipped"]))
    print()
    print("loopiness is the strongest repeat in the loudness envelope (0.25-6 s);")
    print("the 2026-09-11 shipped cues measured 0.485 (opening) and 0.713 (closing).")
    print("THIS QUALIFIES NOTHING -- listen to the episodes in otr/obs and decide.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=["shipped"],
                        help="name or name:KEY=VALUE,KEY=VALUE (repeatable)")
    parser.add_argument("--banks", dest="bank", default=None,
                        help="source bank to pin for every arm")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--acts", type=int, default=1)
    parser.add_argument("--measure-only", action="store_true",
                        help="re-measure the newest episodes instead of rendering")
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="with --measure-only: episode directory names")
    parser.add_argument("--no-boot", action="store_true",
                        help="never reset/boot a server; refuses any arm whose "
                             "settings the server would have to be booted with")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    if not CANONICAL.is_file():
        raise SystemExit("the canonical workflow is missing: %s" % CANONICAL)
    _say("canonical workflow: %s" % CANONICAL)

    results = []
    if args.measure_only:
        names = args.episodes or [p.name for p in sorted(
            (Path(p) for p in glob.glob(str(EPISODES / "*")) if Path(p).is_dir()),
            key=lambda p: p.stat().st_mtime)[-len(args.arms):]]
        for name in names:
            episode = EPISODES / name
            results.append({"arm": name[:14], "episode": name,
                            "cues": measure_episode(episode),
                            "receipt": receipt_of(episode)})
    else:
        dirty = False
        for text in args.arms:
            name, env_overrides = parse_arm(text)
            results.append(run_arm(name, env_overrides, args,
                                   server_is_dirty=dirty))
            dirty = dirty or needs_its_own_server(env_overrides)

    report(results)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        _say("wrote %s" % args.json_out)
    return 0 if all(r.get("exit", 0) == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
