"""Fetch authentic public-domain source text into the local OTR corpus.

Why this exists
---------------
The `public_domain` and `shakespeare` banks are FIDELITY lanes: the episode must be
true to the real work, inventing nothing. On 2026-08-03 both lanes were found to be
running on placeholder fixtures -- and the Wells fixture contained no Wells at all,
just invented summary prose wrapped in genuine Project Gutenberg START/END markers,
under which the bank stamped an H.G. Wells attribution. That is a false-attribution
defect, so authenticity here is a correctness requirement, not a quality nicety.

Offline-first, by design
------------------------
The render pipeline never reaches the network. This tool crawls ONCE and vendors the
text into the repo, alongside a provenance sidecar (source URL, fetch timestamp,
byte count, SHA-256 of the exact stored body). Attribution downstream can then be
bound to that hash, so a credit line can never outrun the bytes it claims to cite.

Fail-loud
---------
Every failure raises. There is no silent fallback to a shorter body, a cached copy,
or a summary: a bank running on unverified text is exactly the defect this replaces.

Usage
-----
    python scripts/otr_fetch_public_domain.py --gutenberg-id 35 --slug time_machine
    python scripts/otr_fetch_public_domain.py --gutenberg-id 35 --slug time_machine \
        --unit arrival --start-marker "I" --end-marker "II"
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "config" / "source_banks" / "_corpus"

GUTENBERG_TEXT_URL = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"

# Gutenberg wraps the work body in these; the boilerplate outside them is licence
# text, not the author's words, and must never reach a script.
_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE
)

# A real work body is never this small. The 145-word fabricated fixture that
# triggered this tool would fail here, which is the point.
MIN_BODY_WORDS = 2000

USER_AGENT = "OTR-OldTimeRadio/2.0 (local research pipeline; contact via repo)"


class FetchError(RuntimeError):
    """A source could not be fetched or did not survive verification."""


def fetch_url(url: str, *, timeout: int = 60) -> str:
    """GET a URL and decode as UTF-8. Raises FetchError on any failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise FetchError(f"{url}: HTTP {response.status}")
            raw = response.read()
    except FetchError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surface the real cause, never degrade
        raise FetchError(f"{url}: {type(exc).__name__}: {exc}") from exc
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8-sig", errors="strict")


def strip_gutenberg_boilerplate(text: str, *, origin: str) -> str:
    """Return only the work body between the Gutenberg START/END markers."""
    start = _START_RE.search(text)
    if start is None:
        raise FetchError(f"{origin}: no Project Gutenberg START marker found")
    end = _END_RE.search(text, start.end())
    if end is None:
        raise FetchError(f"{origin}: no Project Gutenberg END marker found")
    body = text[start.end():end.start()].strip("\r\n \t")
    if not body:
        raise FetchError(f"{origin}: body between the Gutenberg markers is empty")
    return body


def slice_unit(body: str, *, start_marker: str, end_marker: str, origin: str) -> str:
    """Cut a named unit out of a work body by literal line markers."""
    start_at = body.find(start_marker)
    if start_at < 0:
        raise FetchError(f"{origin}: start marker {start_marker!r} not found in body")
    tail = body[start_at:]
    end_at = tail.find(end_marker, len(start_marker))
    if end_at < 0:
        raise FetchError(f"{origin}: end marker {end_marker!r} not found after start")
    unit = tail[:end_at].strip("\r\n \t")
    if not unit:
        raise FetchError(f"{origin}: sliced unit is empty")
    return unit


def verify_body(body: str, *, origin: str, min_words: int) -> int:
    """Refuse a body too short to be the real work. Returns the word count."""
    words = len(body.split())
    if words < min_words:
        raise FetchError(
            f"{origin}: body is {words} words, below the {min_words}-word floor. "
            f"A stub or summary is not an acceptable source for a fidelity lane."
        )
    return words


def write_source(
    body: str,
    *,
    slug: str,
    unit: str | None,
    source_url: str,
    work_title: str,
    author: str,
    license_label: str,
    word_count: int,
    dest_dir: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the body plus its provenance sidecar. Returns both paths.

    ``dest_dir`` exists because a bank manifest may only reference
    MANIFEST-LOCAL relative paths -- `..` is rejected by the schema validators
    on purpose -- so vendored text has to land inside the bank's own directory
    rather than in a shared corpus one level up.
    """
    root = dest_dir or CORPUS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    stem = slug if unit is None else f"{slug}__{unit}"
    text_path = root / f"{stem}.txt"
    sidecar_path = root / f"{stem}.provenance.json"

    body_bytes = body.encode("utf-8")
    digest = hashlib.sha256(body_bytes).hexdigest()

    text_path.write_bytes(body_bytes)
    provenance = {
        "schema_version": "otr_source_provenance_v1",
        "slug": slug,
        "unit": unit,
        "work_title": work_title,
        "author": author,
        "license_label": license_label,
        "source_url": source_url,
        "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "body_sha256": digest,
        "body_bytes": len(body_bytes),
        "body_words": word_count,
        "text_path": str(text_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    sidecar_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return text_path, sidecar_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gutenberg-id", type=int, required=True,
                        help="Project Gutenberg ebook id (The Time Machine = 35)")
    parser.add_argument("--slug", required=True,
                        help="corpus filename stem, e.g. time_machine")
    parser.add_argument("--work-title", default="", help="title for the provenance sidecar")
    parser.add_argument("--author", default="", help="author for the provenance sidecar")
    parser.add_argument("--license-label", default="Public domain (US)",
                        help="rights label recorded in the sidecar")
    parser.add_argument("--unit", default=None,
                        help="optional named unit; requires --start-marker/--end-marker")
    parser.add_argument("--start-marker", default=None)
    parser.add_argument("--end-marker", default=None)
    parser.add_argument("--min-words", type=int, default=MIN_BODY_WORDS,
                        help="floor below which the body is refused as a stub")
    parser.add_argument("--dest-dir", default=None,
                        help="repo-relative directory to write into; defaults to the "
                             "shared corpus. Bank manifests may only cite "
                             "manifest-local paths, so point this at the bank's own "
                             "directory when vendoring for a bank.")
    args = parser.parse_args(argv)

    if args.unit and not (args.start_marker and args.end_marker):
        parser.error("--unit requires both --start-marker and --end-marker")

    url = GUTENBERG_TEXT_URL.format(gid=args.gutenberg_id)
    origin = f"gutenberg:{args.gutenberg_id}"
    print(f"[fetch] {url}")

    raw = fetch_url(url)
    body = strip_gutenberg_boilerplate(raw, origin=origin)
    full_words = verify_body(body, origin=origin, min_words=args.min_words)
    print(f"[fetch] work body: {full_words} words")

    if args.unit:
        body = slice_unit(
            body,
            start_marker=args.start_marker,
            end_marker=args.end_marker,
            origin=f"{origin}:{args.unit}",
        )
        # A unit is a cut of the work, so the full-work floor does not apply --
        # but an empty or trivial cut is still refused.
        unit_words = verify_body(body, origin=f"{origin}:{args.unit}", min_words=200)
        print(f"[fetch] unit {args.unit!r}: {unit_words} words")
        word_count = unit_words
    else:
        word_count = full_words

    text_path, sidecar_path = write_source(
        body,
        slug=args.slug,
        unit=args.unit,
        source_url=url,
        work_title=args.work_title or args.slug,
        author=args.author,
        license_label=args.license_label,
        word_count=word_count,
        dest_dir=(REPO_ROOT / args.dest_dir) if args.dest_dir else None,
    )
    print(f"[fetch] wrote {text_path}")
    print(f"[fetch] wrote {sidecar_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FetchError as exc:
        print(f"[fetch] FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
