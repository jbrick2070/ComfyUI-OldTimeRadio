"""
watch_full_run.py  --  overnight ledger watcher for v2.0-alpha FULL renders
============================================================================

Run this in a separate cmd window while a FULL workflow is queued so a
status log accumulates as each phase / line completes. The log is
human-readable and scroll-friendly the next morning.

Usage:
    python scripts\\watch_full_run.py
    (or just: watch.cmd  if the wrapper exists)

What it does:
    - Polls output/otr/audio/ for the most recent *_ledger.json every
      30 seconds.
    - Each poll computes a one-line status delta against the previous
      state: which phases have new entries in meta.phase_ms, which
      audio_gates have appeared, how many clips[] are populated, any
      lines[] without text_for_tts (BUG-101 audit), any clip whose
      mp4_dur_s deviates from dur_s by >0.05s (BUG-102 audit).
    - Appends one line per change to:
        scripts/full_run_watch_<YYYYMMDD>.log
    - Stops when the watcher sees `final_video_path` set on the ledger
      (= VideoComposite finished) OR after 6 hours wall-clock as a
      safety stop.

The script never modifies the ledger; read-only.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Bootstrap so we can reuse _otr_paths.find_most_recent_ledger logic.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from nodes._otr_paths import otr_audio_dir  # noqa: E402

POLL_SECONDS = 30
SAFETY_STOP_HOURS = 6


def find_most_recent_ledger() -> Path | None:
    # S28 cleanbreak (completed 2026-05-18 retest #17 follow-up):
    # otr_legacy_audio_dir() was removed; this script's dual-dir
    # walker was missed in the original sweep. Per-episode workspace
    # is now the only contract.
    cands: list[Path] = []
    for d in (otr_audio_dir(),):
        if d.exists():
            cands.extend(
                p for p in d.glob("*_ledger.json")
                if not p.name.startswith("pending_")
            )
    if not cands:
        # also look for pending_* during early phases of a fresh run
        for d in (otr_audio_dir(),):
            if d.exists():
                cands.extend(d.glob("pending_*_ledger.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def snapshot(led: dict) -> dict:
    """Reduce a full ledger to the few fields we audit."""
    meta = led.get("meta") or {}
    phase_ms = meta.get("phase_ms") or {}
    gates = led.get("audio_gates") or []
    lines = led.get("lines") or []
    clips = led.get("clips") or []
    return {
        "schema_version": led.get("schema_version") or meta.get("schema_version"),
        "git_commit": meta.get("git_commit"),
        "phase_ms": dict(phase_ms),
        "audio_gates": [g.get("gate") for g in gates],
        "lines_total": len(lines),
        "lines_with_text_for_tts": sum(
            1 for ln in lines if ln.get("text_for_tts")
        ),
        "lines_with_bark_dur": sum(
            1 for ln in lines if ln.get("bark_wav_dur_s")
        ),
        "clips_count": len(clips),
        "clips_with_warmup_pad": sum(
            1 for cl in clips if cl.get("warmup_pad_ms") is not None
        ),
        "clips_with_mp4_dur": sum(
            1 for cl in clips if cl.get("mp4_dur_s") is not None
        ),
        "final_video_path": led.get("final_video_path"),
        "episode_id": led.get("episode_id"),
    }


def compute_delta(prev: dict | None, curr: dict) -> list[str]:
    """Compose a list of human-readable change lines vs the previous
    snapshot. First call (prev=None) produces the baseline summary."""
    out: list[str] = []
    if prev is None:
        out.append(
            f"  baseline: episode={curr['episode_id']} "
            f"schema={curr['schema_version']} "
            f"git={curr['git_commit']}"
        )
        out.append(
            f"  baseline: lines_total={curr['lines_total']} "
            f"clips_count={curr['clips_count']} "
            f"gates={len(curr['audio_gates'])}"
        )
        return out
    if curr["episode_id"] != prev["episode_id"]:
        out.append(
            f"  EPISODE CHANGED: {prev['episode_id']} -> {curr['episode_id']}"
        )
    if curr["git_commit"] != prev["git_commit"]:
        out.append(
            f"  git_commit: {prev['git_commit']} -> {curr['git_commit']}"
        )
    new_phases = set(curr["phase_ms"]) - set(prev["phase_ms"])
    for ph in sorted(new_phases):
        out.append(f"  phase done: {ph} = {curr['phase_ms'][ph]} ms")
    new_gates = [
        g for g in curr["audio_gates"] if g not in prev["audio_gates"]
    ]
    for g in new_gates:
        out.append(f"  audio gate: {g}")
    if curr["lines_total"] != prev["lines_total"]:
        out.append(
            f"  lines_total: {prev['lines_total']} -> {curr['lines_total']}"
        )
    if curr["lines_with_text_for_tts"] != prev["lines_with_text_for_tts"]:
        out.append(
            f"  lines with text_for_tts: "
            f"{prev['lines_with_text_for_tts']} -> "
            f"{curr['lines_with_text_for_tts']} / {curr['lines_total']}"
        )
    if curr["lines_with_bark_dur"] != prev["lines_with_bark_dur"]:
        out.append(
            f"  lines with bark_wav_dur_s: "
            f"{prev['lines_with_bark_dur']} -> "
            f"{curr['lines_with_bark_dur']} / {curr['lines_total']}"
        )
    if curr["clips_count"] != prev["clips_count"]:
        out.append(
            f"  clips_count: {prev['clips_count']} -> {curr['clips_count']}"
        )
    if curr["clips_with_warmup_pad"] != prev["clips_with_warmup_pad"]:
        out.append(
            f"  clips with warmup_pad_ms: "
            f"{prev['clips_with_warmup_pad']} -> "
            f"{curr['clips_with_warmup_pad']} / {curr['clips_count']}"
        )
    if curr["clips_with_mp4_dur"] != prev["clips_with_mp4_dur"]:
        out.append(
            f"  clips with mp4_dur_s: "
            f"{prev['clips_with_mp4_dur']} -> "
            f"{curr['clips_with_mp4_dur']} / {curr['clips_count']}"
        )
    if curr["final_video_path"] and not prev["final_video_path"]:
        out.append(f"  FINAL: {curr['final_video_path']}")
    return out


def deep_audit(led: dict) -> list[str]:
    """One-shot audit run when final_video_path appears, surfacing any
    BUG-100/101/102 red flags. Lines a sleepy human can read fast."""
    out: list[str] = []
    lines = led.get("lines") or []
    clips = led.get("clips") or []

    # BUG-101: any text_for_tts still containing a paren-form?
    paren_leaks = [
        ln.get("line_id") for ln in lines
        if ln.get("text_for_tts") and "(" in ln["text_for_tts"]
    ]
    if paren_leaks:
        out.append(
            f"  BUG-101 RED FLAG: {len(paren_leaks)} line(s) have "
            f"parens in text_for_tts: {paren_leaks[:5]}"
        )
    else:
        out.append(
            f"  BUG-101 PASS: no parens in any text_for_tts "
            f"({len(lines)} lines audited)"
        )

    # BUG-100: any line whose text starts with the speaker's first
    # name in third person? Heuristic match against cast.
    cast = {c.get("char_id"): (c.get("name") or "") for c in (led.get("cast") or [])}
    narration_leaks = []
    for ln in lines:
        speaker = cast.get(ln.get("char_id"), "")
        text = (ln.get("text_for_tts") or ln.get("text") or "").strip()
        if not speaker or not text:
            continue
        first = text.split()[0].rstrip(".,!?;:'\"").lower()
        sp_first = speaker.split()[0].lower()
        if first == sp_first:
            narration_leaks.append(ln.get("line_id"))
    if narration_leaks:
        out.append(
            f"  BUG-100 RED FLAG: {len(narration_leaks)} line(s) "
            f"appear to start with speaker's first name (narration?): "
            f"{narration_leaks[:5]}"
        )
    else:
        out.append(
            f"  BUG-100 PASS: no third-person narration detected "
            f"({len(lines)} lines audited)"
        )

    # BUG-102: any clip whose mp4_dur_s deviates from dur_s by > 0.05s?
    drift = []
    for cl in clips:
        d = cl.get("dur_s")
        m = cl.get("mp4_dur_s")
        if d is None or m is None:
            continue
        if abs(float(m) - float(d)) > 0.05:
            drift.append((cl.get("line_id"), float(d), float(m)))
    if drift:
        out.append(
            f"  BUG-102 RED FLAG: {len(drift)} clip(s) with "
            f"|mp4_dur_s - dur_s| > 50 ms"
        )
        for lid, d, m in drift[:5]:
            out.append(f"    {lid}: dur_s={d:.3f} mp4_dur_s={m:.3f}")
    else:
        out.append(
            f"  BUG-102 PASS: all {len(clips)} clip(s) within 50 ms "
            f"of expected duration"
        )

    # Pad consistency
    pad_values = {
        cl.get("warmup_pad_ms") for cl in clips
        if cl.get("warmup_pad_ms") is not None
    }
    if pad_values:
        out.append(
            f"  warmup_pad_ms values seen: {sorted(pad_values)} "
            f"(expected: {{200}})"
        )

    # Phase timing summary
    pm = (led.get("meta") or {}).get("phase_ms") or {}
    if pm:
        out.append(f"  phase_ms summary:")
        for k in sorted(pm, key=lambda x: pm[x], reverse=True):
            out.append(f"    {k:>22s} : {pm[k]:>10d} ms")

    # Audio gate hash chain
    gates = led.get("audio_gates") or []
    if gates:
        out.append(f"  audio_gates ({len(gates)}):")
        for g in gates:
            out.append(
                f"    {g.get('gate', '?'):>22s} : "
                f"sha256={g.get('sha256_first_kb', '?')[:16]}... "
                f"dur={g.get('dur_s', 0):.2f}s"
            )

    return out


def main() -> int:
    today = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _REPO_ROOT / "scripts" / f"full_run_watch_{today}.log"
    log_fh = log_path.open("a", encoding="utf-8", buffering=1)

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")

    log(f"watcher started; polling every {POLL_SECONDS}s; "
        f"safety stop at {SAFETY_STOP_HOURS}h")
    log(f"logging to: {log_path}")

    t_start = time.time()
    prev_snap: dict | None = None
    prev_path: Path | None = None
    prev_mtime = 0.0
    final_seen = False
    audit_done = False
    # If the most-recent ledger at watcher start is ALREADY complete
    # (final_video_path set), don't exit on its first read -- wait for
    # the FULL run to produce a fresh ledger. The watcher only treats
    # final_video_path as a stop condition for ledgers it watched go
    # from incomplete -> complete.
    initial_ledger_pinned: Path | None = None
    initial_ledger_was_complete: bool = False

    while True:
        if (time.time() - t_start) > SAFETY_STOP_HOURS * 3600:
            log("SAFETY STOP: 6h elapsed without final_video_path -- exiting")
            break

        try:
            ledger_p = find_most_recent_ledger()
        except Exception as exc:
            log(f"glob error: {exc}")
            ledger_p = None

        if ledger_p is None:
            time.sleep(POLL_SECONDS)
            continue

        if ledger_p != prev_path:
            log(f"new ledger: {ledger_p.name}")
            prev_path = ledger_p
            prev_snap = None
            prev_mtime = 0.0
            # On first sighting of THIS ledger, capture whether it is
            # already complete -- if yes, we need a NEW ledger to
            # appear before any final-audit fires. BUG-104: require
            # BOTH final_video_path AND non-empty clips[]; the path
            # alone can be pre-populated by an upstream node before
            # HuMo actually runs.
            if initial_ledger_pinned is None:
                initial_ledger_pinned = ledger_p
                try:
                    _peek = json.loads(
                        ledger_p.read_text(encoding="utf-8")
                    )
                    if _peek.get("final_video_path") and (_peek.get("clips") or []):
                        initial_ledger_was_complete = True
                        log(
                            f"  initial ledger is already complete "
                            f"(final_video_path set + clips populated) -- "
                            f"waiting for NEW ledger from a fresh run"
                        )
                except Exception:
                    pass

        try:
            mtime = ledger_p.stat().st_mtime
        except OSError:
            time.sleep(POLL_SECONDS)
            continue

        if mtime == prev_mtime:
            time.sleep(POLL_SECONDS)
            continue
        prev_mtime = mtime

        try:
            led = json.loads(ledger_p.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"  ledger parse failed: {exc}")
            time.sleep(POLL_SECONDS)
            continue

        curr = snapshot(led)
        deltas = compute_delta(prev_snap, curr)
        for d in deltas:
            log(d)
        prev_snap = curr

        # BUG-LOCAL-104 (2026-04-29 morning): treat the run as
        # complete only when BOTH final_video_path is set AND clips[]
        # is non-empty. Some upstream node (SignalLostVideo or
        # EpisodeAssembler) pre-populates final_video_path with the
        # expected output location BEFORE BatchHumoRender runs --
        # that left the deep_earth_echoes overnight watcher exiting
        # at 23:03:25 instead of ~03:30 when HuMo actually finished,
        # missing 5+ hours of progress. Requiring clips[] populated
        # ensures HuMo has at least started saving real renders.
        run_complete = (
            curr["final_video_path"]
            and curr["clips_count"] > 0
        )
        if run_complete and not audit_done:
            # Only run audit if THIS ledger started incomplete and is
            # now complete -- skip if it was already complete on first
            # sight (the user started us before queueing the FULL run).
            if initial_ledger_pinned == ledger_p and initial_ledger_was_complete:
                log("  (skipping audit: this ledger was already complete "
                    "when watcher started; waiting for NEW ledger)")
            else:
                log("FULL run complete -- running deep audit:")
                for line in deep_audit(led):
                    log(line)
                audit_done = True
                final_seen = True
                log("watcher exiting cleanly")
                break

        time.sleep(POLL_SECONDS)

    log_fh.close()
    return 0 if final_seen else 1


if __name__ == "__main__":
    sys.exit(main())
