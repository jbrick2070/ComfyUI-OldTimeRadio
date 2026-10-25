"""What moved between two environments. Read-only. (PLAN row 2.2,
planning/creatoros-concepts/PLAN.md.)

Every ledger carries the environment its run rendered on in ``meta.env``
(nodes/_otr_shared/env_fingerprint.py). This compares two of them:

  python scripts/otr_env_drift.py
      this machine's newest stamped episode vs its known-good snapshot
      (``otr/episodes/_shared/state/known_good_env.json``), "worked yesterday,
      broken today"

  python scripts/otr_env_drift.py <episode A> <episode B>
      two episodes, by id, episode folder or ledger path -- also one from
      each box, "works on the 4060, not the 5080"

  python scripts/otr_env_drift.py --freeze <episode>
      make that episode's environment the known-good snapshot by hand

OTR_MasterAudioMux writes the snapshot by itself each time an episode reaches
otr/obs/ (any runner: the canonical script, the app, the Studio). Output: Python, torch, CUDA and the driver first,
then the GPU, the OS and ComfyUI, then packs changed (old -> new), added and
removed. Exit 0 when nothing differs, 1 when something does, 2 when an input
cannot be read. Nothing here writes anywhere except ``--freeze`` (the snapshot file, under the
output tree, never in git).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KNOWN_GOOD_NAME = "known_good_env.json"
#: The order a person reads a drift in: the stack under everything first.
HEADLINE_FIELDS = ("python", "torch", "cuda", "driver", "gpu", "os")


class DriftInputError(RuntimeError):
    """An input that cannot be read, named in plain words."""


def _paths():
    from nodes import _otr_paths
    return _otr_paths


def known_good_path() -> pathlib.Path:
    return pathlib.Path(_paths().otr_state_dir()) / KNOWN_GOOD_NAME


def ledger_path_for(arg: str) -> pathlib.Path:
    """An episode id, an episode folder or a ledger file -> its ledger."""
    candidate = pathlib.Path(arg)
    if candidate.is_file():
        return candidate
    folder = candidate if candidate.is_dir() else (
        pathlib.Path(_paths().otr_episodes_root()) / arg)
    if not folder.is_dir():
        raise DriftInputError("no episode %r (not a ledger, a folder, or an id under %s)"
                              % (arg, _paths().otr_episodes_root()))
    hits = sorted((folder / "audio").glob("*_ledger.json"))
    named = [h for h in hits if h.name.startswith(folder.name)]
    if not (named or hits):
        raise DriftInputError("no ledger under %s" % (folder / "audio"))
    return (named or hits)[0]


def _read_json(path: pathlib.Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DriftInputError("%s cannot be read: %s" % (path, exc)) from None
    if not isinstance(data, dict):
        raise DriftInputError("%s is not a JSON object" % path)
    return data


def env_of_ledger(path: pathlib.Path) -> tuple[str, dict]:
    """``(episode id, meta.env)`` of one ledger."""
    data = _read_json(path)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    env = meta.get("env")
    episode = str(data.get("episode_id") or path.parent.parent.name)
    if not isinstance(env, dict) or not env:
        raise DriftInputError(
            "episode %s has no meta.env: it was rendered before the environment "
            "stamp existed (2026-10-24)" % episode)
    return episode, env


def newest_stamped_ledger() -> pathlib.Path:
    """The most recently written ledger on this machine that carries meta.env."""
    root = pathlib.Path(_paths().otr_episodes_root())
    found = []
    for path in root.glob("*/audio/*_ledger.json"):
        if _paths().is_reserved_episode_entry(path.parent.parent.name):
            continue
        try:
            found.append((path.stat().st_mtime, path))
        except OSError:
            continue
    for _mtime, path in sorted(found, reverse=True):
        try:
            env_of_ledger(path)
        except DriftInputError:
            continue
        return path
    raise DriftInputError(
        "no episode under %s carries meta.env: nothing to compare yet. A full "
        "asset cleanup deletes a published episode's ledger, so this usually "
        "finds the ledger of a run that FAILED -- the one worth comparing with "
        "the known-good snapshot -- and there is none" % root)


def load_known_good() -> tuple[str, dict]:
    path = known_good_path()
    if not path.is_file():
        raise DriftInputError(
            "no known-good snapshot at %s yet: one is written each time an "
            "episode reaches otr/obs, or by --freeze <episode>"
            % path)
    data = _read_json(path)
    env = data.get("env")
    if not isinstance(env, dict) or not env:
        raise DriftInputError("%s holds no env" % path)
    return "known-good %s (%s)" % (data.get("episode"), data.get("recorded_at")), env


def write_known_good(episode: str, env: dict, env_version: Any, how: str) -> pathlib.Path:
    from nodes._otr_shared import env_fingerprint
    return pathlib.Path(env_fingerprint.write_known_good(
        episode, env, env_version, how, path=str(known_good_path())))


def diff_env(old: dict, new: dict) -> list[str]:
    """The differences, one line each, in reading order. Pure."""
    lines = []

    def changed(label, a, b):
        if a != b:
            lines.append("%s: %s -> %s" % (label, "(absent)" if a is None else a,
                                            "(absent)" if b is None else b))

    for key in HEADLINE_FIELDS:
        changed(key, old.get(key), new.get(key))
    old_core = old.get("comfyui") if isinstance(old.get("comfyui"), dict) else {}
    new_core = new.get("comfyui") if isinstance(new.get("comfyui"), dict) else {}
    for key in ("version", "commit"):
        changed("comfyui %s" % key, old_core.get(key), new_core.get(key))
    old_packs = old.get("packs") if isinstance(old.get("packs"), dict) else {}
    new_packs = new.get("packs") if isinstance(new.get("packs"), dict) else {}
    for name in sorted(set(old_packs) & set(new_packs)):
        if old_packs[name] != new_packs[name]:
            lines.append("pack changed %s: %s -> %s" % (name, old_packs[name], new_packs[name]))
    for name in sorted(set(new_packs) - set(old_packs)):
        lines.append("pack added %s: %s" % (name, new_packs[name]))
    for name in sorted(set(old_packs) - set(new_packs)):
        lines.append("pack removed %s: %s" % (name, old_packs[name]))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episodes", nargs="*", metavar="EPISODE",
                        help="none (newest stamped episode vs known-good), or two")
    parser.add_argument("--freeze", metavar="EPISODE", default=None,
                        help="make this episode's environment the known-good snapshot")
    args = parser.parse_args(argv)
    try:
        if args.freeze:
            if args.episodes:
                parser.error("--freeze takes one episode and no others")
            path = ledger_path_for(args.freeze)
            episode, env = env_of_ledger(path)
            meta = _read_json(path).get("meta") or {}
            out = write_known_good(episode, env, meta.get("env_version"), "--freeze")
            print("[env-drift] known-good is now %s (%s)" % (episode, out))
            return 0
        if len(args.episodes) == 2:
            old_name, old_env = env_of_ledger(ledger_path_for(args.episodes[0]))
            new_name, new_env = env_of_ledger(ledger_path_for(args.episodes[1]))
        elif not args.episodes:
            old_name, old_env = load_known_good()
            new_name, new_env = env_of_ledger(newest_stamped_ledger())
        else:
            parser.error("give no episode, or two")
    except DriftInputError as exc:
        print("[env-drift] %s" % exc)
        return 2
    print("[env-drift] %s -> %s" % (old_name, new_name))
    lines = diff_env(old_env, new_env)
    if not lines:
        print("[env-drift] no difference")
        return 0
    for line in lines:
        print("  " + line)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
