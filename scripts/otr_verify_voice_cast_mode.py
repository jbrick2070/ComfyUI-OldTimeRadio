"""Prove, from a published episode, that the DETERMINISTIC SCORER cast its voices.

This is the acceptance gate for the 2026-08-18 hybrid-voice-fit removal. It is
deliberately not a "did the render succeed" check -- the render succeeding proves
the pipeline ran, not that the casting changed.

WHY AN ABSENCE ASSERTION IS NOT ENOUGH. `meta.voice_cast_decision == {}` is
produced by at least two different situations: the pass being disabled, and the
pass being enabled while no char-voice engine resolves. Gating on it would pass
identically in both. So the primary gate is the POSITIVE marker
`meta.voice_cast_mode`, and the decision dict is corroboration only.

WHY A REPLAY. Even a correct marker only records intent. The only artifact-level
proof that the scorer produced a given row is to re-run the scorer from the
episode's own recorded inputs and get the same voice back.

THE FOUR PINS -- each one is a false-FAIL vector if you get it wrong, and all
four were found by review rather than by running it:
  1. `role` is the literal "char_voice" (cast_lock passes that constant), NOT the
     ensemble slot's role. `stable_cast_seed` folds role into the seed identity,
     so the wrong role silently draws a different voice.
  2. Gender goes through `canonical_bank_gender`. A row stamped "woman" replayed
     raw will not match.
  3. The used-set must be rebuilt IN CAST-ROW ORDER. The ladder's first pass
     always excludes already-used voices, and CastLock accumulates that set as it
     stamps each row, so replaying one row in isolation is only valid for the
     first open character row.
  4. Draw-affecting knobs must match the render: OTR_CAST_WEIGHTED and
     OTR_CAST_MIN_TIER_POOL at their defaults, and the same voice bank.

Usage:
    python scripts/otr_verify_voice_cast_mode.py                # newest episode
    python scripts/otr_verify_voice_cast_mode.py --episode <id-or-path>

Exit 0 = the scorer demonstrably cast this episode. Exit 2 = it did not, or the
evidence is missing. CPU only; never touches the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO.parent) not in sys.path:
    sys.path.insert(0, str(REPO.parent))

os.environ.setdefault("OTR_TEST_MODE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

EPISODE_ROOTS = (
    Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"),
    REPO / "otr" / "episodes",
)
#: Knobs that change the scorer's draw. The replay is only valid at these values.
DRAW_AFFECTING_ENV = ("OTR_CAST_WEIGHTED", "OTR_CAST_MIN_TIER_POOL")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  ok    {msg}")


def find_ledger(episode: str | None) -> Path | None:
    candidates: list[Path] = []
    for root in EPISODE_ROOTS:
        if not root.is_dir():
            continue
        if episode:
            direct = Path(episode)
            if direct.is_file():
                return direct
            for hit in root.glob(f"*{episode}*/audio/*_ledger.json"):
                candidates.append(hit)
        else:
            candidates.extend(root.glob("*/audio/*_ledger.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", default=None,
                    help="episode id, substring, or a ledger path (default: newest)")
    ap.add_argument("--expect-mode", default="scorer",
                    choices=("scorer", "hybrid", "hybrid_unavailable"))
    args = ap.parse_args()

    ledger_path = find_ledger(args.episode)
    if ledger_path is None:
        print("FAIL: no episode ledger found")
        return 2
    print(f"episode ledger: {ledger_path}")
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    cast = [r for r in (data.get("cast") or []) if isinstance(r, dict)]
    failures = 0

    # ---- pin 4: the replay is only valid at default draw knobs --------------
    print("\n[0] draw-affecting environment")
    for key in DRAW_AFFECTING_ENV:
        val = os.environ.get(key)
        if val is None:
            _ok(f"{key} unset (default)")
        else:
            _fail(f"{key}={val!r} is set; the replay below would not match the "
                  f"render unless the render used the same value")
            failures += 1

    # ---- gate 1: the POSITIVE marker ---------------------------------------
    print("\n[1] which caster ran (the primary gate)")
    mode = meta.get("voice_cast_mode")
    if mode is None:
        _fail("meta.voice_cast_mode is ABSENT. Either the writer's key-by-key "
              "meta copy is missing its line, or this episode predates the "
              "marker. An absent marker proves nothing either way.")
        failures += 1
    elif mode == "":
        _fail("meta.voice_cast_mode is EMPTY -- the writer copied a missing "
              "upstream stamp (fail-closed working as designed). The marker is "
              "not being produced by lock_cast.")
        failures += 1
    elif mode != args.expect_mode:
        _fail(f"meta.voice_cast_mode == {mode!r}, expected {args.expect_mode!r}")
        failures += 1
    else:
        _ok(f"meta.voice_cast_mode == {mode!r}")

    # ---- gate 2: corroboration, never the gate itself -----------------------
    print("\n[2] corroboration (NOT the gate -- ambiguous on its own)")
    decision = meta.get("voice_cast_decision")
    if args.expect_mode == "scorer":
        if decision == {}:
            _ok("meta.voice_cast_decision == {} (consistent)")
        else:
            n = len(decision) if hasattr(decision, "__len__") else "?"
            _fail(f"voice_cast_decision has {n} entries -- the hybrid pass "
                  f"produced decisions on an episode claiming 'scorer'")
            failures += 1

    # ---- gate 3: the replay ------------------------------------------------
    print("\n[3] scorer replay (the artifact-level proof)")
    from importlib import import_module
    VB = import_module("ComfyUI-OldTimeRadio.nodes._otr_voice_bank")
    RG = import_module("ComfyUI-OldTimeRadio.nodes._otr_roster_gender")

    bank, bank_sha = VB.load_voice_bank()
    stamped_sha = str(meta.get("voice_bank_id") or "")
    if stamped_sha and stamped_sha not in (bank_sha, bank_sha[:16]):
        print(f"  note  episode voice_bank_id={stamped_sha!r} vs on-disk "
              f"{bank_sha[:16]!r}; a bank change since the render can make the "
              f"replay diverge legitimately")

    slots = meta.get("cast_voice_slots") or {}
    episode_seed = meta.get("episode_seed")
    if episode_seed is None:
        _fail("meta.episode_seed missing -- cannot replay")
        return 2

    # PIN 3: walk rows IN ORDER, accumulating the used-set exactly as CastLock does.
    used: set = set()
    checked = matched = 0
    for row in cast:
        vid = str(row.get("voice_ref_id") or "")
        name = str(row.get("name") or "")
        char_id = str(row.get("char_id") or "")
        if not vid:
            continue
        entry = next((e for e in bank if e.voice_ref_id == vid), None)
        if name.upper().startswith("ANNOUNCER"):
            # Announcer has its own pinned path; it is not a scorer draw. Its
            # reference still occupies the used-set in production.
            if entry is not None:
                used.update(VB.voice_ref_usage_keys(entry))
            continue
        slot = slots.get(char_id) or {}
        gender = RG.canonical_bank_gender(row.get("gender"))          # PIN 2
        if not gender:
            print(f"  note  {name}: no canonical gender; skipped (the render "
                  f"took the gender-agnostic fallback, not a scorer draw)")
            if entry is not None:
                used.update(VB.voice_ref_usage_keys(entry))
            continue
        checked += 1
        try:
            replay = VB.assign_voice_for_slot(
                role="char_voice",                                    # PIN 1
                engine=str(row.get("voice_engine") or ""),
                char_id=char_id,
                gender=gender,
                timbre=tuple(slot.get("timbre") or ()),
                age_band=str(slot.get("age_band") or ""),
                episode_seed=episode_seed,
                allow_voice_reuse=True,                               # PIN 4
                used_voice_ref_ids=set(used),
                bank=bank,
            )
        except Exception as exc:  # noqa: BLE001 -- report, never crash the gate
            _fail(f"{name}: replay raised {type(exc).__name__}: {exc}")
            failures += 1
            if entry is not None:
                used.update(VB.voice_ref_usage_keys(entry))
            continue
        if replay.voice_ref_id == vid:
            matched += 1
            _ok(f"{name:22} {vid}")
        else:
            _fail(f"{name:22} stamped {vid}, scorer replay gives "
                  f"{replay.voice_ref_id}")
            failures += 1
        if entry is not None:
            used.update(VB.voice_ref_usage_keys(entry))               # PIN 3

    if checked == 0:
        unstamped = [r for r in cast
                     if not str(r.get("voice_ref_id") or "")
                     and not str(r.get("name") or "").upper().startswith("ANNOUNCER")]
        if unstamped:
            _fail(f"{len(unstamped)} character row(s) carry NO voice_ref_id yet. "
                  f"This is almost always an EARLY ledger: the script writer "
                  f"saves meta (including voice_cast_mode) before CastLock "
                  f"stamps voices, so a mid-render `pending_*` ledger looks like "
                  f"this. Re-run against the FINAL ledger in the renamed episode "
                  f"directory once the leg reports RESULT SUCCESS.")
        else:
            _fail("no replayable character rows -- the replay proved nothing")
        failures += 1
    else:
        print(f"\n  replayed {matched}/{checked} character rows")

    print("\n" + "=" * 62)
    if failures:
        print(f"RESULT: FAIL ({failures} problem(s)) -- {ledger_path.name}")
        print("This episode does NOT demonstrate the scorer cast it.")
        return 2
    print(f"RESULT: PASS -- {ledger_path.name}")
    print("The deterministic scorer demonstrably cast this episode.")
    print("NOTE: one episode proves MECHANISM. The concentration claim "
          "(top-2 42% -> ~9%) stays a corpus measurement until post-flip "
          "episodes accumulate. Do not report one as the other.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
