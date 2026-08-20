"""Sweep the ledger archive for cast rows whose description is about SOMEBODY ELSE.

Bug Bible ``11.61`` (promoted from PBUG-20260817-03): two naming authorities are
handed to one prompt and no precedence is stated. An upstream creative pass
invents cast names (``meta.source_meta.selected_concept.cast[].name``); the
casting brief restates them; Python then assigns a DIFFERENT name from the pool.
The description prompt receives both and the model fills its free-text
``<story-linked role>`` slot with the upstream name. Nothing errors: the field is
non-empty, on-format, schema-valid and well written -- it is simply about
somebody else. The contaminated string is then copied verbatim into
``meta.visual_plan.characters[NAME].portrait_prompt``, so the portrait is painted
of that other person too.

THIS IS THE PITCH-SCOPED DETECTOR AND IT IS DELIBERATELY THE CONSERVATIVE ONE.
It only reports a name the UPSTREAM ARTIFACT ITSELF NAMED, found verbatim in a
row the roster gave a different name. It cannot guess, so it cannot invent a
finding. The two cheap alternatives are both wrong and ``11.61`` says so: the
sibling-name check ("no record may name another record") flags correct
relational prose and misses this defect entirely, and a capitalisation-anchored
subject-head check fires on the dominant HEALTHY head style -- a Title-Case
occupation.

WHY ADAPTATION LANES ARE NOT FLAGGED, BY CONSTRUCTION. On ``shakespeare`` and
``public_domain`` the source's own names ARE the roster (``lock_cast`` receives
``source_character_names``), so an upstream name that the roster owns is not
foreign and is never reported. Fidelity to the source is the point of those
lanes; this audit must never argue against it.

NORMALISATION IS THE WHOLE BALLGAME, and getting it wrong is how the first census
scored a contaminated episode clean. ``the_wax_cylinders_whisper`` pitched
``ELIZABETH 'LIZZIE' WALSH`` and the row reads ``Elizabeth 'Lizzie' Walsh``: a
case-sensitive substring test misses BOTH contaminated rows in an episode where
both dramatic rows are wrong. Compare case-folded, quote-unified,
whitespace-collapsed, or do not compare at all.

Read-only. Never writes to a ledger.

Exit codes:
  0  no contaminated row found in a complete scan
  1  at least one contaminated row
  2  the scan is INCOMPLETE and its verdict cannot be trusted -- a bad root,
     an unreadable ledger, or zero ledgers found

The 2 matters: a scan that did not finish is not a pass. A clean report over
nothing looks exactly like a clean report over everything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# The ledger corpus lives under ComfyUI's real output base, not in the repo.
# Both spellings are checked so the script runs from either layout without a
# flag; the resolved root is always PRINTED, because a silently wrong root is
# the one way this audit can lie.
_ROOT_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(_REPO)), "output", "otr", "episodes"),
    os.path.join(_REPO, "otr", "episodes"),
)


class ScanIncomplete(RuntimeError):
    """The sweep could not cover its corpus, so its verdict is meaningless."""


# --------------------------------------------------------------------------- #
# Normalisation and ownership -- IMPORTED, never re-implemented
# --------------------------------------------------------------------------- #

# The detector primitives live with the runtime guard in
# ``nodes/_otr_name_authority.py`` and are imported here on purpose. A second
# copy in this script is exactly how a sweep ends up certifying a different rule
# than the one production enforces -- and this audit's whole value is that its
# green means the same thing runtime's green means.
from nodes._otr_name_authority import (  # noqa: E402
    find_foreign_identities,
    identity_aliases,
    name_tokens,
    normalize_text,
    roster_owns,
    superseded_identities,
)


# --------------------------------------------------------------------------- #
# Ledger reading
# --------------------------------------------------------------------------- #

def iter_ledger_paths(root: str) -> Iterable[str]:
    """Yield every ``*_ledger.json`` under root.

    ``baked_ledger.json`` copies are INCLUDED on purpose: 11.61 records that the
    contamination survives a freeze, so a sweep that skipped the frozen copies
    would under-count exactly the artifacts that ship.
    """
    errors: List[str] = []

    def _on_error(exc: OSError) -> None:
        # os.walk swallows access errors by default, so a directory this process
        # cannot read would silently shrink the corpus and the sweep would report
        # a clean archive it never finished reading.
        errors.append(f"{getattr(exc, 'filename', '?')}: {exc}")

    for dirpath, _dirnames, filenames in os.walk(root, onerror=_on_error):
        for name in sorted(filenames):
            if name.endswith("_ledger.json") or name == "baked_ledger.json":
                yield os.path.join(dirpath, name)
    if errors:
        raise ScanIncomplete("; ".join(errors[:5]))


def read_ledger(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"ledger is not an object: {path}")
    return data


def pitch_names(ledger: Dict[str, Any]) -> List[str]:
    meta = ledger.get("meta") or {}
    source_meta = meta.get("source_meta") or {}
    concept = source_meta.get("selected_concept") or {}
    out: List[str] = []
    for entry in concept.get("cast") or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or "").strip()
            if name:
                out.append(name)
    return out


def roster_rows(ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in (ledger.get("cast") or []) if isinstance(r, dict)]


def portrait_prompts(ledger: Dict[str, Any]) -> Dict[str, str]:
    meta = ledger.get("meta") or {}
    plan = meta.get("visual_plan") or {}
    characters = plan.get("characters") or {}
    out: Dict[str, str] = {}
    if isinstance(characters, dict):
        for key, value in characters.items():
            if isinstance(value, dict):
                out[str(key)] = str(value.get("portrait_prompt") or "")
    return out


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #

def scan_ledger(path: str, ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Return this ledger's cohort membership and every contaminated row."""
    meta = ledger.get("meta") or {}
    pitches = pitch_names(ledger)
    rows = roster_rows(ledger)
    result: Dict[str, Any] = {
        "path": path,
        "episode": str(ledger.get("episode_id") or "").strip(),
        "path_hint": os.path.basename(os.path.dirname(os.path.dirname(path))),
        "bank": str(meta.get("source_bank") or ""),
        "annotated": bool(pitches),
        "pitch_cast": list(pitches),
        "roster": [str(r.get("name") or "") for r in rows],
        "row_count": len([
            r for r in rows if str(r.get("name") or "").upper() != "ANNOUNCER"
        ]),
        "hits": [],
    }
    if not pitches or not rows:
        return result

    roster_names = [str(r.get("name") or "") for r in rows]
    foreign_names = superseded_identities(pitches, roster_names)
    if not foreign_names:
        return result
    result["superseded"] = list(foreign_names)

    prompts = portrait_prompts(ledger)
    for row in rows:
        row_name = str(row.get("name") or "")
        desc = str(row.get("character_description") or "")
        # ALIAS-AWARE, via the same detector runtime uses. Full-string matching
        # missed every short form the archive actually contains -- "Lizzie Gray"
        # for ELIZABETH 'LIZZIE' WALSH, a bare "'Eddie'", "EDWARDM PINCH".
        # speech_signature is checked too: it is a second model-owned prose
        # field and it is demonstrably contaminated in the archive.
        surfaces = {
            "character_description": desc,
            "speech_signature": str(row.get("speech_signature") or ""),
            "portrait_prompt": prompts.get(row_name, ""),
        }
        found = find_foreign_identities(surfaces, foreign_names, roster_names)
        if found:
            in_desc = any(f.field != "portrait_prompt" for f in found)
            in_prompt = any(f.field == "portrait_prompt" for f in found)
            intruders = sorted({f.identity for f in found})
            result["hits"].append({
                "row": row_name,
                "gender": str(row.get("gender") or ""),
                "intruder": intruders[0],
                "intruders": intruders,
                "matched": sorted({f.matched for f in found}),
                "surfaces": sorted({f.field for f in found}),
                "tokens": min(len(name_tokens(normalize_text(i))) for i in intruders),
                "in_description": in_desc,
                "in_portrait_prompt": in_prompt,
                "description": desc[:200],
            })
    return result


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep ledgers for character descriptions about the wrong "
                    "person (Bug Bible 11.61 / PBUG-20260817-03).",
    )
    parser.add_argument(
        "--root", default="",
        help="Directory scanned recursively for *_ledger.json. Defaults to the "
             "ComfyUI output episodes base; the resolved root is always printed.",
    )
    parser.add_argument(
        "--json", dest="json_out", default="",
        help="Write the full per-row findings to this path.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Summary only; omit the per-row listing.",
    )
    args = parser.parse_args(argv)

    root = args.root or next((c for c in _ROOT_CANDIDATES if os.path.isdir(c)), "")
    if not root or not os.path.isdir(root):
        print(
            f"INCOMPLETE: no ledger root found (tried {list(_ROOT_CANDIDATES)})",
            file=sys.stderr,
        )
        return 2

    scanned: List[Dict[str, Any]] = []
    unreadable: List[Tuple[str, str]] = []
    missing_id: List[str] = []
    try:
        # iter_ledger_paths raises ScanIncomplete from INSIDE the generator,
        # after the walk. Uncaught, that escapes main() and Python exits 1 --
        # the code that means "complete scan, contamination found". An
        # unreadable directory would then be indistinguishable from a normal
        # dirty result, which is precisely the confusion the 2 exists to
        # prevent. Catch it around the ITERATION, not around the call.
        for path in iter_ledger_paths(root):
            try:
                # scan_ledger MUST be inside this handler too. A ledger can be
                # valid JSON and still structurally wrong -- `"source_meta":
                # "not-an-object"` makes `.get` raise AttributeError -- and that
                # escaped as process exit 1, which is the code for "complete
                # scan, contamination found". A malformed corpus would have been
                # reported as a normal dirty result.
                ledger = read_ledger(path)
                result = scan_ledger(path, ledger)
            except Exception as exc:  # noqa: BLE001 -- a bad ledger breaks the sweep
                unreadable.append((path, f"{type(exc).__name__}: {exc}"))
                continue
            if not result["episode"]:
                missing_id.append(path)
                continue
            scanned.append(result)
    except ScanIncomplete as exc:
        print(f"INCOMPLETE: directory traversal failed -- {exc}", file=sys.stderr)
        return 2

    if missing_id:
        print(f"INCOMPLETE: {len(missing_id)} ledger(s) carry no durable "
              f"episode_id; identity cannot be inferred from storage layout",
              file=sys.stderr)
        for bad in missing_id[:5]:
            print(f"  NO EPISODE_ID {bad}", file=sys.stderr)
        return 2

    if not scanned:
        print(f"INCOMPLETE: zero ledgers under {root}", file=sys.stderr)
        return 2

    annotated = [s for s in scanned if s["annotated"]]
    dirty = [s for s in scanned if s["hits"]]
    all_hits = [h for s in dirty for h in s["hits"]]

    # DEDUPE THE FROZEN COPIES. A single episode's baked_ledger is written once
    # per bench arm -- one episode appeared in EIGHT copies -- so a raw file
    # count silently multiplies a handful of episodes into an alarming total.
    # This has to live in the instrument: quoting an externally-computed dedupe
    # makes the headline figure unreproducible from the tool that reports it.
    unique_rows = {
        (s["episode"], tuple(s["roster"]), h["row"], tuple(h.get("intruders") or [h["intruder"]]))
        for s in dirty for h in s["hits"]
    }
    unique_episodes = {s["episode"] for s in dirty}

    # BENCH CAMPAIGN COPIES ARE DISTINCT DURABLE EPISODES BUT NOT DISTINCT
    # STORIES. A bench re-renders one authored episode once per arm, and each
    # arm gets its own episode_id, so the durable-id count is correct and yet
    # overstates how many STORIES shipped wrong. Both numbers are printed
    # because quoting either one alone misleads.
    def _is_bench(entry: Dict[str, Any]) -> bool:
        path = entry.get("path", "")
        return ("_shared" in path or "_bench" in path or "measurement" in path)

    bench = [s for s in dirty if _is_bench(s)]
    real = [s for s in dirty if not _is_bench(s)]
    real_rows = sum(len(s["hits"]) for s in real)
    real_episodes = {s["episode"] for s in real}
    multi = [h for h in all_hits if h["tokens"] >= 2]
    single = [h for h in all_hits if h["tokens"] < 2]
    derived = [h for h in all_hits if h["in_portrait_prompt"]]
    annotated_rows = sum(s["row_count"] for s in annotated)

    print(f"root              : {root}")
    print(f"ledgers scanned   : {len(scanned)}")
    print(f"annotated cohort  : {len(annotated)} ledgers carry an upstream pitch cast")
    print(f"production cohort : {len(scanned) - len(annotated)} ledgers carry none "
          f"(this detector is blind to them by design)")
    print(f"rows in cohort    : {annotated_rows} non-announcer rows")
    print(f"CONTAMINATED      : {len(all_hits)} row occurrences in "
          f"{len(dirty)} ledger files")
    print(f"  DEDUPED         : {len(unique_rows)} unique rows in "
          f"{len(unique_episodes)} unique durable episodes")
    print(f"  REAL EPISODES   : {real_rows} rows in {len(real_episodes)} episodes "
          f"<- quote THIS as production impact")
    print(f"  bench arm copies: {len(bench)} ledger files re-rendering stories "
          f"already counted above")
    print(f"  multi-token     : {len(multi)}  (a two-or-more-word upstream name, verbatim)")
    print(f"  single-token    : {len(single)}  (one word -- eyeball these)")
    print(f"  portrait_prompt : {len(derived)} of those also contaminated the derived surface")
    if unreadable:
        print(f"unreadable        : {len(unreadable)}")

    if not args.quiet and dirty:
        print("")
        for entry in sorted(dirty, key=lambda s: s["episode"]):
            print(f"{entry['episode']}  [{entry['bank']}]")
            print(f"    pitch cast : {entry['pitch_cast']}")
            print(f"    roster     : {entry['roster']}")
            for hit in entry["hits"]:
                surfaces = []
                if hit["in_description"]:
                    surfaces.append("description")
                if hit["in_portrait_prompt"]:
                    surfaces.append("portrait_prompt")
                print(f"    ROW {hit['row']} ({hit['gender']}) carries "
                      f"{hit['intruder']!r} in {'+'.join(surfaces)}")

    if args.json_out:
        payload = {
            "root": root,
            "ledgers_scanned": len(scanned),
            "annotated_ledgers": len(annotated),
            "annotated_rows": annotated_rows,
            "contaminated_row_occurrences": len(all_hits),
            "contaminated_ledger_files": len(dirty),
            "unique_contaminated_rows": len(unique_rows),
            "unique_contaminated_episodes": len(unique_episodes),
            "real_episode_rows": real_rows,
            "real_episodes": len(real_episodes),
            "bench_copy_files": len(bench),
            "multi_token_rows": len(multi),
            "single_token_rows": len(single),
            "derived_surface_rows": len(derived),
            "unreadable": unreadable,
            "findings": dirty,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_out}")

    if unreadable:
        for path, err in unreadable[:10]:
            print(f"UNREADABLE {path}: {err}", file=sys.stderr)
        return 2
    return 1 if all_hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
