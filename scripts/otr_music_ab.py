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
CANONICAL = REPO_ROOT / "workflows" / "otr_canonical.json"
#: Where episodes land. The operator's own output tree, never a test rig
#: (memory: "obs is always the local output folder").
EPISODES = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")
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
    for path in glob.glob(str(episode_dir / "audio" / "*_ledger.json")):
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
    """Reset per CLAUDE.md section 4, then boot the shipped launcher with
    ``env_overrides`` in its environment (section 5: the .cmd is launched
    directly, never through a cmd.exe /c whose quoting eats the log path).

    Kills SELECTIVELY by command line -- a blanket python kill would sever
    the tooling running this harness.
    """
    import signal

    killed = []
    try:
        listing = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get",
             "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001 -- wmic absent on newer Windows
        listing = ""
    for line in listing.splitlines():
        if "main.py" not in line or "ComfyUI" not in line:
            continue
        pid = line.rsplit(",", 1)[-1].strip()
        if pid.isdigit():
            try:
                os.kill(int(pid), signal.SIGTERM)
                killed.append(pid)
            except OSError:
                pass
    if killed:
        _say("reset: stopped server pid(s) %s" % ", ".join(killed))
    time.sleep(6)

    env = dict(os.environ, PYTHONUTF8="1", **env_overrides)
    _say("booting a server for this arm with %s" % (env_overrides or "no overrides"))
    subprocess.Popen([str(LAUNCHER), str(log_path)], cwd=str(REPO_ROOT), env=env,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(5)
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if "To see the GUI go to" in text:
                _say("server up")
                return True
            if "SERVER DID NOT COME UP" in text:
                return False
    return False


def parse_arm(text: str) -> tuple[str, dict]:
    """``name`` or ``name:KEY=VALUE,KEY=VALUE`` -> ``(name, env)``."""
    name, _, rest = text.partition(":")
    env = {}
    for pair in rest.split(",") if rest else []:
        if not pair.strip():
            continue
        key, _, value = pair.partition("=")
        if not key.strip() or not value:
            raise SystemExit("bad arm setting %r in %r (want KEY=VALUE)" % (pair, text))
        env[key.strip()] = value.strip()
    return name.strip() or "arm", env


def episode_from_log(log_path: Path) -> Path | None:
    """The episode THIS arm rendered, taken from the runner's own output.

    Preferred over "the newest directory" because the episode root is shared:
    a second window, an overnight batch or a leg already in flight can finish
    into it while an arm runs, and a timestamp heuristic would then attribute
    somebody else's music to this arm with no error at all.
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
        if (candidate / "audio").is_dir():
            return candidate
    return None


def newest_episode(after: float) -> Path | None:
    """Fallback only, and LOUD about its own ambiguity."""
    candidates = [Path(p) for p in glob.glob(str(EPISODES / "*"))
                  if Path(p).is_dir() and Path(p).stat().st_mtime >= after]
    if not candidates:
        return None
    if len(candidates) > 1:
        _say("WARNING: %d episodes appeared while this arm ran (%s) and the "
             "runner log did not name one -- attributing the newest, which "
             "may be another window's work"
             % (len(candidates), ", ".join(sorted(p.name for p in candidates))[:160]))
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_arm(name: str, env_overrides: dict, args) -> dict:
    """Render ONE canonical episode with ``env_overrides`` applied."""
    log_path = REPO_ROOT / "tmp" / ("_music_ab_%s.log" % "".join(
        c if c.isalnum() else "_" for c in name))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if env_overrides and needs_its_own_server(env_overrides):
        if args.no_boot:
            raise SystemExit(
                "arm %r sets %s, which the SERVER reads, but --no-boot was "
                "given. The runner only POSTs to a server that is already "
                "running, so these settings would be silently ignored and the "
                "arm would measure nothing. Drop --no-boot, or boot the server "
                "yourself with them set." % (
                    name, ", ".join(sorted(env_overrides))))
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
    episode = episode_from_log(log_path) or newest_episode(started)
    result = {"arm": name, "env": env_overrides, "exit": code, "minutes": minutes,
              "log": log_path.name, "episode": episode.name if episode else None}
    if episode is not None:
        result["cues"] = measure_episode(episode)
        result["receipt"] = receipt_of(episode)
    _say("arm %r finished: exit=%s %.1f min episode=%s" % (
        name, code, minutes, result["episode"]))
    return result


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
        for text in args.arms:
            name, env_overrides = parse_arm(text)
            results.append(run_arm(name, env_overrides, args))

    report(results)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1), encoding="utf-8")
        _say("wrote %s" % args.json_out)
    return 0 if all(r.get("exit", 0) == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
