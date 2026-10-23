"""Freeze a finished episode into a REPLAY BUNDLE (campaign item 0, 2026-09-02).

    python scripts/otr_freeze_replay_bundle.py <episode dir or episode id> [--out DIR]

The bundle is what the canonical graph replays: the writer's ``replay_from`` widget
(or ``otr_canonical_api_run.py --replay-from``) points at it, the writer short-circuits
authorship, the audio chain passes through, node 7 copies the frozen master, ShotLock
reuses the PLANNED video section, node 91 verifies the imported stills, and only the
video phase and the publish tail run -- the same seeds, the same audio, a new episode.

WHAT IS FROZEN: the durable ledger, the master WAV, ``episode_canon.json``, every file
under ``stills/`` and ``portraits/``. NOT frozen: clips, composites, the published MP4 --
the replay makes its own. The manifest is a safe import format: relative paths only,
sizes and SHA-256 per file, the source episode id, root and commit; built in a temporary
sibling directory and renamed into place only after every file was verified. Freezing is
explicit and operator-visible; nothing freezes itself.

The ledger MUST carry the planned ``video.shots`` (stamped durably by ShotLock since
2026-09-02) -- a replay without the plan would re-author prompts through the LLM, which is
not a replay. Refused otherwise unless ``--allow-no-plan`` is passed for a diagnostic.

The freezing itself lives in ``nodes/_otr_replay_freezer.py`` (2026-10-22), so a node can
freeze a stage inside the server. This script is its command-line front end, re-exports
its functions for the scripts and tests that import them from here, and is the one place
that asks git for the source commit.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_replay_freezer as _freezer  # noqa: E402
from nodes.production_ledger import (  # noqa: E402
    REPLAY_MANIFEST_NAME, REPLAY_MANIFEST_SCHEMA, ReplayBundleError, load_replay_manifest,
)

FROZEN_DIRS = _freezer.FROZEN_DIRS
find_ledger = _freezer.find_ledger
master_digest = _freezer.master_digest
find_master = _freezer.find_master
derive_engine_bundle = _freezer.derive_engine_bundle

__all__ = [
    "FROZEN_DIRS", "REPLAY_MANIFEST_NAME", "REPLAY_MANIFEST_SCHEMA", "ReplayBundleError",
    "load_replay_manifest", "find_ledger", "master_digest", "find_master", "freeze",
    "derive_engine_bundle", "resolve_episode_dir", "main",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(REPO_ROOT), text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def _episodes_root() -> pathlib.Path:
    """The live episodes tree, through the same resolver the ledger uses."""
    from nodes import _otr_paths
    return pathlib.Path(_otr_paths.otr_episodes_root())


def resolve_episode_dir(arg: str) -> pathlib.Path:
    p = pathlib.Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = _episodes_root() / arg
    if cand.is_dir():
        return cand.resolve()
    raise SystemExit("episode dir not found: %r (also tried %s)" % (arg, cand))


def freeze(ep_dir: pathlib.Path, out_root: pathlib.Path, *, allow_no_plan: bool = False,
           **options) -> pathlib.Path:
    """The library freeze, with this checkout's git commit as the source commit
    unless the caller names one. Refusals raise ``ReplayBundleError``."""
    if "source_commit" not in options:            # git only when the caller names none
        options["source_commit"] = _git_commit()
    return _freezer.freeze(ep_dir, out_root, allow_no_plan=allow_no_plan, **options)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode directory, or an episode id under the live episodes root; "
                                    "with --derive-engine, an EXISTING bundle directory")
    ap.add_argument("--out", default="", help="bundle root (default: <episodes root>/_replay)")
    ap.add_argument("--allow-no-plan", action="store_true",
                    help="freeze a ledger without planned video.shots (diagnostic only)")
    ap.add_argument("--derive-engine", default="", dest="derive_engine",
                    help="derive a sibling bundle from an existing bundle that replays on the "
                         "named engine (a registered Ghost sibling of the frozen plan's engine)")
    args = ap.parse_args(argv)
    try:
        if args.derive_engine:
            final = derive_engine_bundle(pathlib.Path(args.episode), args.derive_engine,
                                         pathlib.Path(args.out) if args.out else None)
            man = json.loads((final / REPLAY_MANIFEST_NAME).read_text(encoding="utf-8"))
            print("[derive] %s -> %s (engine_override=%s, %d file(s))"
                  % (args.episode, final, man["engine_override"], len(man["files"])))
            return 0
        ep_dir = resolve_episode_dir(args.episode)
        out_root = pathlib.Path(args.out) if args.out else (_episodes_root() / "_replay")
        out_root.mkdir(parents=True, exist_ok=True)
        final = freeze(ep_dir, out_root, allow_no_plan=args.allow_no_plan)
    except ReplayBundleError as exc:
        raise SystemExit("refused: %s" % exc)
    man = json.loads((final / REPLAY_MANIFEST_NAME).read_text(encoding="utf-8"))
    print("[freeze] %s -> %s (%d file(s), %d planned shot(s), commit %s)"
          % (ep_dir.name, final, len(man["files"]), man["planned_shots"], man["source_commit"] or "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
