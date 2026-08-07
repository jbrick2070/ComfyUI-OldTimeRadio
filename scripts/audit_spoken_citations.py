"""Audit published ledgers for source citations that reached the SPOKEN surface.

The defect this exists for: on the fidelity lanes the interpreter was handed the
source URL and asked for an attribution note in the same payload, and that reply
was appended VERBATIM as the announcer's closing line. 84 spoken lines across 30
episodes read a URL or a licence identifier on air, and because the caption
builder copies raw ``lines[].text`` into the ASS cue, it was burned into the video
as well.

The forbidden values are derived PER LEDGER from that episode's own
``source_rights`` / ``provenance`` -- never from a hard-coded list -- so this same
predicate works on any bank and on episodes that have not been produced yet.

Exit codes follow ``scripts/audit_otr_full_run.py``:
    0  clean
    1  violations found
    2  operational error (bad root, unreadable tree)

Usage::

    python scripts/audit_spoken_citations.py --root <output>/otr/episodes
    python scripts/audit_spoken_citations.py --root . --json

Read-only. Never writes to a ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

EXIT_CLEAN = 0
EXIT_VIOLATIONS = 1
EXIT_OPERATIONAL = 2

#: Ledgers written before the spoken-coda receipt existed. They cannot carry
#: ``spoken_coda_source``, so its absence is history rather than a regression.
#: Anything at a LATER schema is required to carry it on a provenance-owned lane.
#:
#: This is the COMPLETE pre-l4 lineage from ``nodes/_otr_ledger.py``. It used to
#: list three versions, one of which -- "l2-2026-05-02" -- never existed: it is
#: l2-2026-04-25 and l3-2026-05-02 mashed together. Four real versions were
#: missing, and ten live ledgers sit at l3-2026-05-08 alone, so the gap was
#: real rather than cosmetic. It was only ever LATENT because :177 requires a
#: provenance block before enforcing and the pre-l3 versions predate it.
#:
#: "l3-2026-05-14" MUST STAY. It was the current version until the l4 bump, and
#: it is the string that holds the boundary: replacing it with the new value
#: would un-legacy 1,587 historical ledgers and legacy every post-bump one --
#: inverting the audit entirely, while the unit tests below stayed green.
LEGACY_SCHEMA_VERSIONS = frozenset({
    "l1-2026-04-24",
    "l2-2026-04-25",
    "l3-2026-04-28",
    "l3-2026-05-02",
    "l3-2026-05-08",
    "l3-2026-05-14",
})

VALID_CODA_SOURCES = frozenset({"provenance", "news_close_brief", "none"})

#: Our own interpreter-prompt scaffold. A model that echoes the block reproduces
#: these labels, so a label-only leak must fail even when no URL survives.
SCAFFOLD_LABELS = ("source:", "url:", "date/rights:")

#: Spoken rows only. A citation in a music cue is not a citation on air.
SPEECH_ROLES = ("character", "announcer")


def _clean_needles(values: Iterable[Any]) -> List[str]:
    """Case-folded, de-duplicated, EMPTY-FREE forbidden values.

    Dropping empties is load-bearing, not hygiene: the public-domain rights
    sidecar carries no ``license_label`` at all, so ``normalize_provenance``
    yields ``""`` for it -- and an empty needle is a substring of every string,
    which would make this audit report every line in the corpus.
    """
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip().casefold()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def forbidden_values(ledger: Dict[str, Any]) -> List[str]:
    """Every token this episode must never SAY, derived from its own metadata."""
    meta = ledger.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    rights = meta.get("source_rights")
    rights = rights if isinstance(rights, dict) else {}
    prov = meta.get("provenance")
    prov = prov if isinstance(prov, dict) else {}

    raw: List[Any] = [
        rights.get("source_url"), rights.get("license_url"),
        rights.get("license_label"),
        prov.get("source_url"), prov.get("license_url"),
        prov.get("license_label"),
    ]

    # Hostnames, parsed only from non-empty URLs. "gutenberg.org" spoken without
    # the scheme is the same leak wearing a hat.
    for url in (rights.get("source_url"), rights.get("license_url"),
                prov.get("source_url"), prov.get("license_url")):
        text = str(url or "").strip()
        if not text:
            continue
        try:
            host = urlparse(text).netloc
        except ValueError:
            continue
        if host:
            raw.append(host)

    return _clean_needles(raw)


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)

#: A FORMAL licence identifier, in any of the shapes a model actually speaks.
#:
#: Deriving needles from the rights sidecar alone is not enough: the stored
#: label is the full "CC BY-NC 3.0 (Folger Shakespeare Library)", while shipped
#: episodes said the short "CC BY-NC 3.0" and even the spelled-out "CC BY-NC
#: three". A longer needle can never match a shorter line, so exact-substring
#: matching missed the very lines this audit exists to find.
#:
#: Deliberately NOT matching the phrase "public domain": the correct
#: deterministic coda is "Tonight's tale was adapted from a work in the public
#: domain." Forbidding that would fail the fix instead of the defect.
_LICENCE_RE = re.compile(
    r"\bCC[ \-]?BY(?:[ \-]?NC)?(?:[ \-]?SA)?"
    r"(?:[ \-]?(?:\d(?:\.\d)?|zero|one|two|three|four))?\b"
    r"|\bCC[ \-]?0\b"
    r"|\bcreativecommons\.org\b",
    re.I,
)


def line_violations(text: str, needles: Iterable[str]) -> List[str]:
    """Which forbidden tokens this spoken line contains. Pure."""
    raw = str(text or "")
    hay = raw.casefold()
    if not hay:
        return []
    hits = [needle for needle in needles if needle in hay]
    hits.extend(label for label in SCAFFOLD_LABELS if label in hay)
    if _URL_RE.search(raw):
        hits.append("<bare-url>")
    licence = _LICENCE_RE.search(raw)
    if licence:
        hits.append("<licence:%s>" % licence.group(0).casefold())
    # Stable, de-duplicated, so two runs over one corpus compare cleanly.
    return sorted(set(hits))


def audit_ledger(ledger: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return ``(violations, receipt_errors)`` for one parsed ledger."""
    meta = ledger.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    needles = forbidden_values(ledger)

    violations: List[Dict[str, Any]] = []
    lines = ledger.get("lines")
    for line in lines if isinstance(lines, list) else []:
        if not isinstance(line, dict) or line.get("skip"):
            continue
        if str(line.get("speaker_role") or "") not in SPEECH_ROLES:
            continue
        hits = line_violations(line.get("text") or "", needles)
        if hits:
            violations.append({
                "line_id": str(line.get("line_id") or ""),
                "speaker_role": str(line.get("speaker_role") or ""),
                "matched": hits,
                "text": str(line.get("text") or "")[:200],
            })

    # The receipt is only REQUIRED on ledgers new enough to have one. On a legacy
    # ledger its absence is history; on a current one it is a regression, because
    # a build that silently stopped stamping it is exactly the failure mode that
    # let the dead provenance coda survive for months.
    receipt_errors: List[str] = []
    schema = str(ledger.get("schema_version") or "")
    is_legacy = (not schema) or schema in LEGACY_SCHEMA_VERSIONS
    if not is_legacy and isinstance(meta.get("provenance"), dict):
        source = meta.get("spoken_coda_source")
        if source is None:
            receipt_errors.append(
                "provenance-owned ledger at schema %r carries no "
                "spoken_coda_source receipt" % schema
            )
        elif source not in VALID_CODA_SOURCES:
            receipt_errors.append(
                "spoken_coda_source %r is outside the closed vocabulary %s"
                % (source, sorted(VALID_CODA_SOURCES))
            )
    return violations, receipt_errors


def iter_ledgers(root: str) -> Iterable[str]:
    """Every ``*_ledger.json`` under root, in deterministic sorted order."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith("_ledger.json"):
                yield os.path.join(dirpath, name)


def main(argv: "List[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit ledgers for source citations on the spoken surface.")
    parser.add_argument("--root", required=True,
                        help="Directory to scan recursively for *_ledger.json.")
    parser.add_argument("--json", action="store_true",
                        help="Emit findings as JSON for machine consumption.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.root):
        print("audit_spoken_citations: not a directory: %s" % args.root,
              file=sys.stderr)
        return EXIT_OPERATIONAL

    scanned = 0
    unreadable: List[str] = []
    findings: List[Dict[str, Any]] = []

    for path in iter_ledgers(args.root):
        try:
            with open(path, encoding="utf-8") as handle:
                ledger = json.load(handle)
        except (OSError, ValueError) as exc:
            unreadable.append("%s: %s" % (path, exc))
            continue
        if not isinstance(ledger, dict):
            unreadable.append("%s: top level is not an object" % path)
            continue
        scanned += 1
        violations, receipt_errors = audit_ledger(ledger)
        if violations or receipt_errors:
            findings.append({
                "episode": os.path.relpath(path, args.root).replace("\\", "/"),
                "schema_version": str(ledger.get("schema_version") or ""),
                "spoken_violations": violations,
                "receipt_errors": receipt_errors,
            })

    findings.sort(key=lambda row: row["episode"])

    if args.json:
        print(json.dumps({
            "scanned": scanned,
            "unreadable": unreadable,
            "findings": findings,
        }, indent=2, sort_keys=True))
    else:
        print("scanned %d ledger(s) under %s" % (scanned, args.root))
        for row in findings:
            print("\n%s  [schema %s]" % (row["episode"], row["schema_version"]))
            for violation in row["spoken_violations"]:
                print("    %s (%s) matched %s"
                      % (violation["line_id"], violation["speaker_role"],
                         ", ".join(violation["matched"])))
                print("        %s" % violation["text"])
            for error in row["receipt_errors"]:
                print("    RECEIPT: %s" % error)
        for problem in unreadable:
            print("UNREADABLE: %s" % problem, file=sys.stderr)
        print("\n%d episode(s) with findings" % len(findings))

    if unreadable and not findings:
        return EXIT_OPERATIONAL
    return EXIT_VIOLATIONS if findings else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
