"""Keep the GPU busy: 1-act episodes across the live banks and visual styles.

WHY THIS SHAPE. The operator asked for continuous test episodes while he is
remote -- "1 act across random local banks, styles and local video/still
models". Banks, styles AND engines are all rotated here.

**THIS PARAGRAPH USED TO SAY ENGINES WERE NOT ROTATED, AND IT WAS STALE
(corrected 2026-08-19).** It read "Engines are NOT [rotated] ... Every leg
therefore runs the canonical engines -- still_flat + z_image_turbo", which
stopped being true the day `PROFILES` was added: `main()` prints "banks x
styles x engine profiles" and the leg loop calls `rng.choice(PROFILES)`. A
docstring that describes a constraint the code no longer has is worse than
none -- it is exactly what a reader consults before deciding whether the
harness can answer their question, and this one said "no" to a question the
code answers "yes".

WHAT IS STILL TRUE, and it is the part worth keeping: engine rotation goes
through CAPABILITY PROFILES because the video/image engine widgets are MANAGED
and `patch_creative` refuses them outright (the BUG-08.06 stranded-COMBO
class). A profile's `role_overrides` is the only sanctioned lever, so widening
the rotation means authoring profiles, not adding a `--set`.

WHAT THE ROTATION COVERS (2026-08-22). `--lanes still` is the historical
CHEAP still/procgen rotation and remains the DEFAULT, so every existing
invocation is byte-identical. `--lanes video` rotates the eight per-engine
video profiles in `VIDEO_PROFILES` -- ghost_signal, wan_ti2v, ltx_video,
ltx_8gb, fastwan, humo, ltx_audio_in and ltx25 -- which is the gap this
paragraph used to name. `--lanes all` runs both.

STILL NOT COVERED, stated so nobody infers it: no profile here carries an
`upscale_stage` section, so the two upscale engines (`off` and
`spandrel_esrgan`) are not exercised by this harness. That is an addition
waiting on its own profile; the harness needs no change to accept one.

`--hours N` gives the run a wall-clock budget for an overnight soak. The leg in
flight always finishes -- a soak that kills its own last render manufactures a
failure it then reports, which is the one result nobody can act on.

SEQUENTIAL BY DESIGN. One GPU, one render at a time (CLAUDE.md scope rule:
no async CUDA streams, no queue refactor). Each leg runs to a terminal state
before the next is submitted, and a failed leg is LOGGED and skipped rather
than stopping the campaign -- an overnight soak that dies on leg 3 is worth
less than one that reports 3 failures out of 20.

A receipt JSON is rewritten after EVERY leg, so killing the run mid-flight
still leaves a complete record of what has finished.

THE HARNESS DOES NOT NAME EPISODES (PBUG-20260817-05). A leg carries a
`leg_label` for the console and the receipt, and that label never reaches the
writer. Titling belongs to the canonical workflow -- which is what "mimic the
entire workflow" means, and it costs a `_generate_title_from_script` LLM call
per leg that this soak had never made. The receipt then records what the
WORKFLOW titled the episode, read back from the ledger the run wrote.

Usage:
    python scripts/otr_gpu_soak_matrix.py --legs 12
    python scripts/otr_gpu_soak_matrix.py --legs 0      # until stopped
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import random
import time
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUNNER = HERE / "otr_canonical_api_run.py"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nodes._otr_paths import (  # noqa: E402  (the one OTR path authority)
    is_reserved_episode_entry,
    otr_episodes_root,
)

#: A HEADLESS run reporting this `title_source` is a contradiction on its
#: face: nobody typed anything. It means a caller planted a run label in the
#: `episode_title` widget, where the writer reads any non-empty value as
#: "user typed a value; respect it verbatim" -- and that string then becomes
#: the on-screen title card, the filenames, the episode folder, the ledger
#: name, the treatment, the canon, the credits and the published obs
#: artifact. One check here would have caught all 17 harness-titled episodes
#: the day they landed (PBUG-20260817-05).
HEADLESS_FORBIDDEN_TITLE_SOURCE = "user"

#: Grace on the read-back window's UPPER bound, in seconds. `finished` is
#: taken the instant the runner returns, but `os.stat` reports a finer
#: resolution than `datetime.now()` -- measured on this box, a ledger written
#: immediately before `finished` reports an mtime GREATER than it by a
#: fraction of a microsecond. A hard upper edge therefore drops the leg's own
#: episode whenever the ledger is the last write before the runner exits.
#: Seconds rather than minutes: the bound exists to keep a LATER concurrent
#: write out of this leg's receipt, and a few seconds does not weaken that.
READBACK_GRACE_S = 5.0

#: The five runnable banks after the 2026-08-16 scifi_news rip. Rotated
#: EXPLICITLY rather than left to the canonical roll, so a finite soak gets
#: even coverage instead of luck -- a roll can hand you four shakespeares.
BANKS = [
    "media_archive",
    "original",
    "scifi_news_pro",
    "public_domain",
    "shakespeare",
]

#: Every non-sentinel visual style the live registry offers.
STYLES = [
    "anime", "archival_documentary", "cartoon", "paper_origami",
    "recur_frac", "sci_fi_radio", "shakespeare_stage_realism",
    "storybook_engraving", "video_art", "visual_storybased",
]

#: Engine rotation goes through CAPABILITY PROFILES, which is the sanctioned
#: surface: the video/image widgets are MANAGED and `patch_creative` refuses
#: them outright so a run cannot strand a COMBO value (BUG-08.06 class). Each
#: profile below differs from `16gb_full` in role_overrides ONLY, so a leg
#: differs from a normal render in exactly the engine under test. They are
#: listed in build_variants.LANE_PRESETS so they never emit a shipping
#: variant -- soak instruments, not platform targets.
PROFILES = [
    "otr_soak_still_flat_z_image_turbo",
    "otr_soak_still_flat_flux_gen1",
    "otr_soak_still_motion_lumina_image",
    "otr_soak_still_motion_flux2_klein",
    "otr_soak_still_pan_flux_gen1",
    "otr_soak_still_pan_ideo",
    "otr_soak_still_word_flux2_klein",
    "otr_soak_still_word_z_image_turbo",
    "otr_soak_word_razzle_ideo",
    "otr_soak_word_razzle_lumina_image",
]

#: THE HEAVY VIDEO LANES -- the gap this harness's own docstring named ("no
#: heavy local video model is in the rotation"). These are the per-engine
#: profiles, one video engine each, so a leg's failure names a lane rather than
#: a mixture.
#:
#: A 24-HOUR SOAK IS A DIFFERENT INSTRUMENT FROM A SMOKE, and this list is why
#: it is worth running. A smoke asks "does this lane render once". A soak asks
#: what only appears on the tenth consecutive episode: a stranded GPU lease, a
#: patcher never detached, VRAM that creeps a little per leg, a cache key that
#: collides on the second episode of the same bank. None of those are visible
#: in a single green leg.
VIDEO_PROFILES = [
    "otr_ghost_signal",
    "otr_g4_wan_ti2v",
    "otr_g4_ltx_video",
    "otr_g4_ltx_8gb",
    "otr_g4_fastwan",
    "otr_g4_humo",
    "otr_g4_ltx_audio_in",
    "otr_ltx25_high_video",
]

#: What ``--lanes`` selects. ``still`` is the historical default and keeps every
#: existing invocation byte-identical.
PROFILE_SETS = {
    "still": PROFILES,
    "video": VIDEO_PROFILES,
    "all": PROFILES + VIDEO_PROFILES,
}


def ledgers_in_window(started: datetime.datetime,
                      finished: datetime.datetime) -> list:
    """EVERY episode ledger written during ``[started, finished]``.

    Returns a LIST on purpose. A single "newest" answer cannot tell a correct
    read from a collision: if anything else wrote an episode into the shared
    tree while the leg ran -- a resident server from a prior run, which is
    exactly the hazard CLAUDE.md section 4 exists to prevent -- then picking
    the newest silently records ANOTHER episode's title. That is the same
    false-title defect this receipt exists to catch, so the caller is given
    the ambiguity and refuses to guess rather than being handed one answer.

    Never raises. A campaign runs for hours against a tree in motion, and a
    file vanishing mid-scan must cost one leg's receipt field, never the run.
    """
    try:
        root = otr_episodes_root()
        if not root.is_dir():
            return []
        entries = list(root.iterdir())
    except Exception:
        return []
    low = started.timestamp()
    high = finished.timestamp() + READBACK_GRACE_S
    found = []
    for entry in entries:
        try:
            if not entry.is_dir() or is_reserved_episode_entry(entry.name):
                continue
            audio = entry / "audio"
            if not audio.is_dir():
                continue
            for led in audio.glob("*_ledger.json"):
                if low <= led.stat().st_mtime <= high:
                    found.append(led)
        except OSError:
            continue
    return found


def ledger_meta(path):
    """The ``meta`` mapping from a ledger, or None if it cannot be read.

    Broad on purpose. A torn ledger is valid JSON of the WRONG SHAPE at least
    as often as it is invalid JSON -- `null` at the top level, or `meta`
    holding a string -- and both raise `AttributeError`, which a narrow
    `(OSError, ValueError)` guard lets straight through into the campaign loop.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    meta = data.get("meta")
    return meta if isinstance(meta, dict) else None


def title_guard_verdict(title_source) -> str:
    """Classify a produced episode's ``title_source`` for a HEADLESS run.

    REPORTS, NEVER FAILS. The episode is already rendered and published by
    the time this runs, and an audit may never fail an episode (operator law
    2026-07-22) -- so a violation is recorded and printed loudly, and the
    leg's ``ok`` keeps meaning exactly one thing: did the render succeed.
    """
    if not isinstance(title_source, str) or not title_source:
        return "unknown"
    if title_source == HEADLESS_FORBIDDEN_TITLE_SOURCE:
        return "VIOLATION_headless_title_source_user"
    return "ok"


def title_receipt(started: datetime.datetime, finished: datetime.datetime,
                  bank: str, style: str) -> dict:
    """What the CANONICAL WORKFLOW titled the episode this leg produced.

    Read back from the ledger rather than asserted by the harness: now that
    the harness has stopped naming episodes the ledger is the only authority
    on the title, and a receipt that records the harness's own label instead
    is a receipt about nothing.

    On more than one candidate it NARROWS by the leg's own parameters and
    then REFUSES. `source_bank` / `visual_style` are a TIE-BREAKER only --
    they are absent on a large share of real completed ledgers, so gating on
    them would report `no_ledger` for perfectly good runs. An honest
    "ambiguous" beats a confident wrong title, which is the whole lesson of
    this defect family.
    """
    candidates = ledgers_in_window(started, finished)
    if not candidates:
        return {"episode_title": None, "title_source": None,
                "title_guard": "no_ledger"}
    if len(candidates) > 1:
        narrowed = [path for path in candidates
                    for meta in [ledger_meta(path)]
                    if meta is not None
                    and meta.get("source_bank") == bank
                    and meta.get("visual_style") == style]
        if len(narrowed) != 1:
            return {"episode_title": None, "title_source": None,
                    "title_guard": f"ambiguous_{len(candidates)}_episodes"}
        candidates = narrowed
    meta = ledger_meta(candidates[0])
    if meta is None:
        return {"episode_title": None, "title_source": None,
                "title_guard": "unreadable_ledger"}
    source = meta.get("title_source")
    return {"episode_title": meta.get("episode_title"),
            "title_source": source,
            "title_guard": title_guard_verdict(source)}


def leg(index, bank, style, profile, timeout) -> dict:
    stamp = datetime.datetime.now().strftime("%H%M%S")
    short = profile.replace("otr_soak_", "")
    # A LABEL FOR THIS LEG, NOT A TITLE FOR THE EPISODE. It names the run in
    # the console and the receipt and goes nowhere near the writer. Passing
    # it as `--title` put it in the `episode_title` widget, which the writer
    # treats as a person naming their episode, and the label became the
    # on-screen title card and the published artifact (PBUG-20260817-05).
    leg_label = f"SOAK{index:02d} {bank} {style} {short}"
    cmd = [
        sys.executable, str(RUNNER),
        "--act-count", "1",
        "--source-bank", bank,
        "--visual-style", style,
        "--profile", profile,
        "--timeout", str(timeout),
    ]
    started = datetime.datetime.now()
    print(f"[soak] leg {index} START {stamp} bank={bank} style={style} "
          f"engines={short}", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                              text=True, timeout=timeout + 600)
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = "RESULT SUCCESS" in out
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        out, ok, rc = "harness timeout", False, -1
    finished = datetime.datetime.now()
    elapsed = (finished - started).total_seconds() / 60.0
    tail = [ln for ln in out.splitlines() if "RESULT" in ln or "Exception" in ln]
    # BELT AND BRACES. The module contract above is that a failed leg is
    # logged and skipped rather than stopping the campaign, and `main`'s loop
    # catches only KeyboardInterrupt -- so anything escaping the read-back
    # would end an overnight run over a receipt FIELD. The helpers below are
    # already written not to raise; this makes that a property of the leg
    # rather than a promise a later edit can quietly break.
    try:
        titled = title_receipt(started, finished, bank, style)
    except Exception as exc:
        titled = {"episode_title": None, "title_source": None,
                  "title_guard": f"readback_error_{type(exc).__name__}"}
    print(f"[soak] leg {index} {'PASS' if ok else 'FAIL'} "
          f"{elapsed:.1f} min rc={rc} {tail[-1][:110] if tail else ''}",
          flush=True)
    print(f"[soak] leg {index} TITLE {titled['episode_title']!r} "
          f"(source={titled['title_source']}, {titled['title_guard']})",
          flush=True)
    return {"leg": index, "bank": bank, "style": style,
            "profile": profile, "ok": ok,
            "rc": rc, "minutes": round(elapsed, 1),
            "leg_label": leg_label, **titled}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--legs", type=int, default=12,
                    help="0 = run until stopped")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lanes", choices=sorted(PROFILE_SETS), default="still",
                    help="which profile rotation (default: still, the "
                         "historical behaviour)")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="wall-clock budget; 0 = no budget. The leg in flight "
                         "is always allowed to FINISH -- a soak that kills its "
                         "own last render manufactures a failure it then "
                         "reports, which is the one result nobody can act on.")
    args = ap.parse_args(argv)

    profiles = PROFILE_SETS[args.lanes]
    rng = random.Random(args.seed)
    deadline = (time.time() + args.hours * 3600.0) if args.hours else None
    print(f"[soak] rotating {len(BANKS)} banks x {len(STYLES)} styles x "
          f"{len(profiles)} {args.lanes} profiles, 1 act per leg", flush=True)
    if deadline:
        print("[soak] wall-clock budget %.1f h; the leg in flight always "
              "finishes" % args.hours, flush=True)

    results, index = [], 0
    out_dir = REPO / "otr_soak_receipts"
    out_dir.mkdir(exist_ok=True)
    receipt = out_dir / (
        "soak_%s.json" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    try:
        while args.legs == 0 or index < args.legs:
            if deadline is not None and time.time() >= deadline:
                print("[soak] wall-clock budget reached after %d leg(s)"
                      % index, flush=True)
                break
            index += 1
            row = leg(index, rng.choice(BANKS), rng.choice(STYLES),
                      rng.choice(profiles), args.timeout)
            results.append(row)
            receipt.write_text(json.dumps(results, indent=1), encoding="utf-8")
    except KeyboardInterrupt:
        print("[soak] interrupted -- receipts kept", flush=True)

    passed = sum(1 for r in results if r["ok"])
    print(f"\n[soak] {passed}/{len(results)} passed. receipt: {receipt}")
    for r in results:
        if not r["ok"]:
            print(f"        FAIL leg {r['leg']}: {r['bank']} + {r['style']} "
                  f"+ {r['profile']}")

    violations = [r for r in results
                  if str(r.get("title_guard", "")).startswith("VIOLATION")]
    if violations:
        print(f"\n[soak] TITLE GUARD: {len(violations)} leg(s) reported "
              f"title_source={HEADLESS_FORBIDDEN_TITLE_SOURCE!r} on a HEADLESS "
              f"run. Nobody typed a title, so a caller planted a run label in "
              f"the episode_title widget and it reached the title card.")
        for r in violations:
            print(f"        leg {r['leg']}: episode_title="
                  f"{r['episode_title']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
