"""Verify a replay against its source WITHOUT ComfyUI (campaign item 0, 2026-09-02).

    python scripts/otr_verify_replay.py <source episode dir|ledger> <replay episode dir|ledger> [...]

Checks, and exits non-zero on the first that fails:
  1. the replay ledger names the source as ``meta.replay_of_episode`` and carries a
     ``replay_workspace_id`` different from the source's (which has none);
  2. ``freeze_timestamp`` is byte-identical (the freeze receipt survives a replay);
  3. every planned shot's ``render_request_hash`` matches, in order (same brief, same cast,
     same beats -> same comparison seeds);
  4. the master audio SHA-256 (``audio.master_audio_sha256``) is identical;
  5. every ``meta.render_trace`` row's ``actual_request_sha`` recomputes from its own causal
     fields (the receipt was not edited after it was hashed), and, when two replays are
     given, their traces agree row for row on ``seed`` and ``actual_request_sha`` (the A/A
     null: identical by construction);
  6. a source trace, when present, is compared to the replay's on ``seed`` per shot.

``--ab`` reads the two replays as an A/B PAIR instead: the same frozen episode composed by two
PROMPT VERSIONS. Then the seeds and the planned request hashes must still match exactly -- the
seed is derived from ``render_request_hash``, which mixes brief, cast, beat and character and
has never included the prompt -- while the prompt text and ``actual_request_sha`` must DIFFER,
because two arms that composed the same text tested nothing. The plate rule stays on the A/A
path only: an A/B legitimately mints a different plate, since the plate follows the prompt.

Prints a one-line verdict per check and a table of per-shot seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes._otr_video_engines.render_driver import _RECEIPT_CAUSAL_KEYS  # noqa: E402


def load_ledger(arg: str) -> dict:
    p = pathlib.Path(arg)
    if p.is_dir():
        hits = sorted((p / "audio").glob("*_ledger.json"))
        named = [h for h in hits if h.name.startswith(p.name)]
        hits = named or hits
        if not hits:
            raise SystemExit("no ledger under %s" % p)
        p = hits[0]
    return json.loads(p.read_text(encoding="utf-8"))


def recompute_sha(row: dict) -> str:
    causal = {k: row.get(k) for k in _RECEIPT_CAUSAL_KEYS}
    return hashlib.sha256(json.dumps(causal, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, default=str).encode("utf-8")).hexdigest()


def shots(led: dict):
    v = led.get("video") if isinstance(led.get("video"), dict) else {}
    return [s for s in (v.get("shots") or []) if isinstance(s, dict)]


def trace(led: dict):
    m = led.get("meta") if isinstance(led.get("meta"), dict) else {}
    return [r for r in (m.get("render_trace") or []) if isinstance(r, dict)]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print("%-52s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("replays", nargs="+")
    ap.add_argument(
        "--ab", action="store_true",
        help=("Read two replays as an A/B PAIR rather than an A/A pair. The "
              "default rule requires both replays to agree on the seed AND the "
              "actual request sha; a prompt-version experiment agrees on the "
              "seed and DIFFERS on the sha by design, so run as-is it would "
              "report FAIL on a correct experiment. Under --ab the seeds and "
              "planned request hashes must still match exactly -- that is what "
              "makes it an honest comparison -- and the prompt text and its "
              "sha must differ, because two arms that composed the same text "
              "did not test anything."))
    args = ap.parse_args(argv)
    src = load_ledger(args.source)
    reps = [load_ledger(r) for r in args.replays]
    ok = True
    src_meta = src.get("meta") or {}
    for i, rep in enumerate(reps, 1):
        meta = rep.get("meta") or {}
        tag = "replay %d" % i
        ok &= check("%s names the source" % tag,
                    str(meta.get("replay_of_episode") or "") == str(src.get("episode_id") or ""),
                    "%r vs %r" % (meta.get("replay_of_episode"), src.get("episode_id")))
        ok &= check("%s has its own workspace id" % tag,
                    bool(meta.get("replay_workspace_id")) and not src_meta.get("replay_workspace_id"))
        ok &= check("%s keeps the freeze receipt" % tag,
                    str(meta.get("freeze_timestamp") or "") == str(src_meta.get("freeze_timestamp") or ""))
        s_hashes = [s.get("render_request_hash") for s in shots(src)]
        r_hashes = [s.get("render_request_hash") for s in shots(rep)]
        ok &= check("%s planned request hashes match (%d shots)" % (tag, len(s_hashes)),
                    bool(s_hashes) and s_hashes == r_hashes)
        ok &= check("%s master audio sha256 identical" % tag,
                    (src.get("audio") or {}).get("master_audio_sha256")
                    == (rep.get("audio") or {}).get("master_audio_sha256"))
        rows = trace(rep)
        ok &= check("%s carries a render trace (%d rows)" % (tag, len(rows)), bool(rows))
        bad = [r.get("shot_id") for r in rows if r.get("actual_request_sha") != recompute_sha(r)]
        ok &= check("%s trace rows recompute their sha" % tag, not bad, ", ".join(map(str, bad[:4])))
        s_rows = trace(src)
        if s_rows:
            by_shot = {(r.get("shot_id"), r.get("segment_index")): r for r in s_rows}
            diff = [r.get("shot_id") for r in rows
                    if by_shot.get((r.get("shot_id"), r.get("segment_index")), {}).get("seed") != r.get("seed")]
            ok &= check("%s seeds equal the source's per shot" % tag, not diff, ", ".join(map(str, diff[:4])))
    if args.ab and len(reps) < 2:
        # A FLAG THAT DID NOTHING MUST SAY SO. Silently skipping the whole A/B
        # section would print a clean PASS for a run that never compared
        # anything, which is the one failure mode a proof tool may not have.
        ok &= check("A/B: --ab needs TWO replays to compare (%d given)"
                    % len(reps), False)
    if len(reps) >= 2 and args.ab:
        # A/B: the SAME episode composed by two prompt versions. The seed is
        # derived from `render_request_hash`, which mixes the brief, the cast,
        # the beat and the character and has never included the prompt -- so a
        # composer change moves the text and leaves the seed alone, and that is
        # exactly the pair this mode asserts.
        a, b = trace(reps[0]), trace(reps[1])
        aligned = len(a) == len(b) and bool(a)
        ok &= check("A/B: both arms rendered the same shots (%d vs %d)"
                    % (len(a), len(b)), aligned)
        if not aligned:
            # THE PER-SHOT LINES ARE SKIPPED, NOT ZIPPED. `zip` stops at the
            # shorter trace, so running them anyway would print three PASSes
            # computed over a prefix while the arms disagree on how many shots
            # they even rendered -- the verdict would be right and every line a
            # reader skims would be wrong.
            print("  (per-shot A/B checks skipped: the traces are different "
                  "lengths, so any per-shot verdict would be computed over a "
                  "prefix)")
        else:
            seed_diff = [x.get("shot_id") for x, y in zip(a, b)
                         if x.get("seed") != y.get("seed")]
            ok &= check("A/B: seeds identical per shot", not seed_diff,
                        ", ".join(map(str, seed_diff[:4])))
            same_text = [x.get("shot_id") for x, y in zip(a, b)
                         if str(x.get("text_prompt") or "") == str(y.get("text_prompt") or "")]
            ok &= check("A/B: every shot's prompt actually differs", not same_text,
                        ", ".join(map(str, same_text[:4])))
            same_sha = [x.get("shot_id") for x, y in zip(a, b)
                        if x.get("actual_request_sha") == y.get("actual_request_sha")]
            ok &= check("A/B: request shas differ (the prompt is causal)",
                        not same_sha, ", ".join(map(str, same_sha[:4])))
    elif len(reps) >= 2:
        a, b = trace(reps[0]), trace(reps[1])
        same = len(a) == len(b) and all(
            x.get("seed") == y.get("seed") and x.get("actual_request_sha") == y.get("actual_request_sha")
            for x, y in zip(a, b))
        ok &= check("A/A: replay 1 and replay 2 traces agree row for row", same)
        # THE PLATE RULE (still-in lab peer, 2026-09-02): a row that minted a
        # plate carries its rendered sha OUTSIDE the causal hash; two A/A rows
        # must therefore agree on it too, or the kernel is not bit-stable and
        # the video difference has a named cause. Applies only where BOTH rows
        # carry a plate (the shipping lane carries none).
        plated = [(x, y) for x, y in zip(a, b)
                  if x.get("plate_sha256") or y.get("plate_sha256")]
        if plated:
            bad_plate = [x.get("shot_id") for x, y in plated
                         if not (x.get("plate_sha256") and y.get("plate_sha256"))
                         or x.get("plate_sha256") != y.get("plate_sha256")]
            ok &= check("A/A: plate hashes present and equal (%d plated rows)" % len(plated),
                        not bad_plate, ", ".join(map(str, bad_plate[:4])))
    print()
    print("%-28s %-12s %-10s %s" % ("shot", "seed", "sha8", "prompt"))
    for r in trace(reps[0]) if reps else []:
        print("%-28s %-12s %-10s %.60s" % (r.get("shot_id"), r.get("seed"),
                                           str(r.get("actual_request_sha") or "")[:8],
                                           r.get("text_prompt") or ""))
    print()
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
