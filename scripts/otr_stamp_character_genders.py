"""Write the gender a source states about its own characters into its sidecar.

WHY. `cast_source_contract.gender_by_name` shipped `{}` on every public_domain
episode, because the field is fed from a provenance sidecar and 64 of 65 units
had none. Empty field -> blind 40/40/20 roll -> three measured inversions on
three different sources: GERTRUDE male, LORD RONALD female (the script calls the
male-voiced character *"Miss McFiggins"*), and then AHAB, of Moby-Dick, in a
woman's voice.

The fact was never missing. Melville writes "he" of Ahab throughout. What was
missing was the sidecar carrying it, so this tool reads the source the vendor
already downloaded and writes down what the author says.

OWNERSHIP, because a sidecar is shared property (one owner per field):
  * `characters[]` and `gender_ladder` -- THIS tool, sole writer.
  * every base provenance field -- the FETCHER
    (`otr_fetch_public_domain.py` `write_source`). This tool never overwrites
    one. It will FILL a base field that is absent, but only from local truth:
    the manifest row plus the text file on disk. It never invents `fetched_utc`,
    because this tool did not fetch anything and a fabricated timestamp is a
    false receipt.

IDEMPOTENCE, and why `ran_utc` is not part of it. Re-running must be a no-op
when nothing substantive changed, so the staleness identity is the CONTENT --
the character rows and the tier counts -- and explicitly NOT the timestamp.
Treating a clock reading as invalidation state means every run "changes" every
sidecar, which makes the freshness signal worthless and the diffs unreadable.
`ran_utc` is preserved on a no-op and moves only when the content moves.

DECLINING IS AN ANSWER. A candidate the deterministic ladder cannot decide is
OMITTED from `characters` rather than written as `unknown`. Downstream that is a
render-time JOIN MISS, which preserves today's roll -- exactly the behaviour
that name has now, so an omitted name is a no-change rather than a regression.
No consumer has to branch on `unknown`, because it never enters the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes._otr_character_roster import (  # noqa: E402
    infer_gender, parse_character_roster,
)
from nodes._otr_gender_pronoun_scan import scan_gender  # noqa: E402
from nodes._otr_roster_gender import sidecar_path_for_text  # noqa: E402

BANK_DIR = REPO_ROOT / "config" / "source_banks" / "public_domain_story"
MANIFEST = BANK_DIR / "manifest.sample.json"

LADDER_VERSION = "gender_ladder_v1"
SIDECAR_SCHEMA = "otr_source_provenance_v1"

#: Ruling 3: the announcer's gender is random BY DESIGN, so pinning it would
#: remove variety the show wants. `cradle_protocol` really does carry "the
#: Announcer" in its cast_hints, so this exclusion is load-bearing, not defensive.
_EXCLUDED = frozenset({"ANNOUNCER", "THE ANNOUNCER"})


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidates(source: Dict[str, Any]) -> List[str]:
    """The names to decide, from the manifest's own cast hints.

    `cast_hints` is schema-required non-empty and authored in the vendor table,
    and it already names every character the three live failures inverted --
    "Captain Ahab", "Gertrude", "Lord Ronald". A wider net (an LLM pass over the
    full text) is what the SPEC adds for names the briefs invent at render time;
    it is a separate tier and not needed to close what actually broke.
    """
    out: List[str] = []
    for hint in source.get("cast_hints") or []:
        name = str(hint or "").strip()
        if not name or name.upper() in _EXCLUDED:
            continue
        if name not in out:
            out.append(name)
    return out


def _aliases_for(name: str) -> List[str]:
    """Extra join keys, because the render extracts "SCROOGE", not "EBENEZER
    SCROOGE". The existing join tiers cannot bridge that on their own."""
    from nodes._otr_gender_pronoun_scan import mention_forms

    forms = [f for f in mention_forms(name) if f != name.strip().lower()]
    return [f for f in forms if f]


def _decide(name: str, text: str, others: List[str]) -> Tuple[str, str, str]:
    """(gender, gender_source, evidence) -- or ("", "", reason) when declined.

    Tier 1 ROSTER first: free, deterministic, and it makes this ladder uniform
    with the Shakespeare lane that already works. It fires on roughly zero prose
    units (prose has no cast block) and is kept anyway, exactly as specified.
    Tier 2 PRONOUNS second: the author's own words.
    """
    for record in parse_character_roster(text):
        if record.matches(name):
            gender = infer_gender(record.description or record.name)
            if gender in ("male", "female"):
                return gender, "roster", (
                    "source cast list: %s" % (record.description or record.name)
                )
    verdict = scan_gender(name, text, other_names=others)
    if verdict.decided:
        return verdict.gender, "pronouns", verdict.evidence
    return "", "", verdict.evidence


def _base_identity(source: Dict[str, Any], unit: Dict[str, Any],
                   text: str, text_path: pathlib.Path) -> Dict[str, Any]:
    """Base fields derivable from LOCAL truth only -- no network, no clock."""
    rel = text_path.relative_to(REPO_ROOT).as_posix()
    return {
        "schema_version": SIDECAR_SCHEMA,
        "slug": str(source.get("source_id") or ""),
        "unit": str(unit.get("unit_id") or ""),
        "work_title": str(source.get("title") or ""),
        "author": str(source.get("author") or ""),
        "license_label": str(source.get("license_status") or ""),
        "source_url": str(source.get("source_url") or ""),
        "body_sha256": _sha256_text(text),
        "body_bytes": len(text.encode("utf-8")),
        "body_words": len(text.split()),
        "text_path": rel,
    }


def stamp_unit(source: Dict[str, Any], unit: Dict[str, Any],
               *, write: bool) -> Dict[str, Any]:
    """Decide every candidate for one unit and merge the result into its sidecar."""
    text_path = BANK_DIR / str(unit.get("text_path") or "")
    result: Dict[str, Any] = {
        "source_id": str(source.get("source_id") or ""),
        "unit_id": str(unit.get("unit_id") or ""),
        "decided": [], "declined": [], "changed": False,
    }
    if not text_path.is_file():
        result["error"] = "missing text: %s" % (text_path,)
        return result
    text = text_path.read_text(encoding="utf-8", errors="replace")

    names = _candidates(source)
    rows: List[Dict[str, Any]] = []
    tier_counts = {"roster": 0, "pronouns": 0, "llm_web": 0, "name_frequency": 0}
    for name in names:
        gender, tier, evidence = _decide(name, text, names)
        if not gender:
            result["declined"].append({"name": name, "why": evidence})
            continue
        tier_counts[tier] += 1
        row: Dict[str, Any] = {
            "name": name,
            "gender": gender,
            "gender_source": tier,
            # Redundant with gender_source by construction, stamped anyway so a
            # ledger reader never has to carry the mapping table in their head.
            "gender_confidence": "known",
            "evidence": evidence,
        }
        aliases = _aliases_for(name)
        if aliases:
            row["aliases"] = aliases
        rows.append(row)
        result["decided"].append({"name": name, "gender": gender, "tier": tier})

    path = sidecar_path_for_text(text_path)
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            result["error"] = "existing sidecar is not valid JSON: %s" % (path,)
            return result
        if not isinstance(existing, dict):
            result["error"] = "existing sidecar is not an object: %s" % (path,)
            return result

    merged = dict(existing)
    # Fill base identity ONLY where the fetcher has not spoken.
    for key, value in _base_identity(source, unit, text, text_path).items():
        merged.setdefault(key, value)

    previous_ladder = existing.get("gender_ladder")
    previous_rows = existing.get("characters")
    substantive_change = (previous_rows != rows) or (
        not isinstance(previous_ladder, dict)
        or previous_ladder.get("version") != LADDER_VERSION
        or previous_ladder.get("tier_counts") != tier_counts
    )
    merged["characters"] = rows
    ladder = {
        "version": LADDER_VERSION,
        "tier_counts": tier_counts,
        "candidates": len(names),
        "declined": [d["name"] for d in result["declined"]],
    }
    # ran_utc is audit metadata, NOT part of the staleness identity. Preserved
    # verbatim on a no-op so re-running is genuinely idempotent.
    if isinstance(previous_ladder, dict) and previous_ladder.get("ran_utc"):
        ladder["ran_utc"] = previous_ladder["ran_utc"]
    merged["gender_ladder"] = ladder

    result["changed"] = bool(substantive_change)
    result["sidecar"] = str(path)
    if write and substantive_change:
        path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="write sidecars; without it this only reports")
    ap.add_argument("--source", default="",
                    help="limit to one source_id (for iterating on one unit)")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    for source in manifest.get("sources") or []:
        if args.source and source.get("source_id") != args.source:
            continue
        for unit in source.get("units") or []:
            results.append(stamp_unit(source, unit, write=args.write))

    errors = [r for r in results if r.get("error")]
    decided = sum(len(r["decided"]) for r in results)
    declined = sum(len(r["declined"]) for r in results)
    changed = [r for r in results if r.get("changed")]
    total = decided + declined

    print("[gender-stamper] units=%d candidates=%d decided=%d declined=%d "
          "coverage=%.1f%%" % (
              len(results), total, decided, declined,
              100.0 * decided / max(1, total)))
    print("[gender-stamper] sidecars %s: %d" % (
        "written" if args.write else "that WOULD change", len(changed)))
    for r in results:
        for d in r["declined"]:
            print("  DECLINED %s:%s" % (r["source_id"], d["name"]))
    for r in errors:
        print("  ERROR %s: %s" % (r["source_id"], r["error"]))
    # Fail loud in an authoring-time tool: a unit whose text is missing means
    # the corpus is incomplete, and a silent skip there is how 64 empty
    # sidecars went unnoticed for a month in the first place.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
