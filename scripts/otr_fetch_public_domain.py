"""Fetch authentic source text into the local OTR corpus (Gutenberg + Folger).

Why this exists
---------------
The `public_domain` and `shakespeare` banks are FIDELITY lanes: the episode must be
true to the real work, inventing nothing. On 2026-08-03 both lanes were found to be
running on placeholder fixtures -- the Wells fixture contained no Wells at all, just
invented summary prose wrapped in genuine Project Gutenberg START/END markers, under
which the bank stamped an H.G. Wells attribution; and all fourteen Shakespeare
fixtures were 93-125 word collages of a scene several thousand words long. Authentic
text here is a correctness requirement, not a quality nicety.

Offline-first, by design
------------------------
The render pipeline never reaches the network. This tool crawls ONCE and vendors the
text into the repo beside a provenance sidecar (source URL, fetch timestamp, byte and
word counts, SHA-256 of the exact stored body). Attribution downstream can then be
bound to that hash, so a credit line can never outrun the bytes it claims to cite.

Fail-loud
---------
Every failure raises. There is no silent fallback to a shorter body, a cached copy, or
a summary: a bank running on unverified text is exactly the defect this replaces.

Usage
-----
    # Gutenberg: a whole work, or a unit sliced by literal markers
    python scripts/otr_fetch_public_domain.py --gutenberg-id 35 --slug time_machine \
        --unit arrival --start-marker "III." --end-marker "IV." \
        --dest-dir config/source_banks/public_domain_story/sources

    # Folger: one act/scene of a play, with its speakers reported
    python scripts/otr_fetch_public_domain.py --folger-play as-you-like-it \
        --act 3 --scene 2 --slug as_you_like_it --unit act3_scene2 \
        --dest-dir config/source_banks/shakespeare/sources
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
# Import the roster module DIRECTLY rather than through the `nodes` package: the
# repo directory name contains a hyphen so it is not importable as a package,
# and `nodes/__init__.py` registers ComfyUI node classes, which this
# authoring-time tool must not need.
sys.path.insert(0, str(REPO_ROOT / "nodes"))

from _otr_character_roster import parse_character_roster  # noqa: E402
CORPUS_ROOT = REPO_ROOT / "config" / "source_banks" / "_corpus"

GUTENBERG_TEXT_URL = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
FOLGER_TEXT_URL = (
    "https://folger-main-site-assets.s3.amazonaws.com/uploads/2022/11/"
    "{play}_TXT_FolgerShakespeare.txt"
)

# Gutenberg wraps the work body in these; the boilerplate outside them is licence
# text, not the author's words, and must never reach a script.
_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE
)
_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE
)

# Folger plain text marks structure on its own lines: "ACT 3" then "Scene 2".
_ACT_LINE_RE = re.compile(r"^ACT\s+(\d+)\s*$", re.MULTILINE)
_SCENE_LINE_RE = re.compile(r"^Scene\s+(\d+)\s*$", re.MULTILINE)

# Folger uses TWO speech-prefix layouts and a parser that knows only one silently
# loses whole casts:
#   verse -- the name alone on its line, speech beneath:   ORLANDO\nHang there, my verse
#   prose -- the name inline, two spaces, then speech:     TOBY  Come thy ways, Signior Fabian.
# Either may carry a stage qualifier: ROSALIND, [as Ganymede] / BENEDICK, [aside]
# Matching only the verse form parsed ZERO speakers out of an all-prose scene
# (Twelfth Night 2.5) and dropped Benedick from the scene named after him.
# The name allows single spaces between capitalised words ("FIRST WITCH",
# "ANTIPHOLUS OF EPHESUS") but never a double space, which is what keeps the
# match from running on into the spoken line that follows it.
_SPEECH_PREFIX_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z'’.\-]*(?: [A-Z][A-Z'’.\-]*)*)"
    r"(?:,\s*\[[^\]]*\])?"
    r"(?:\s*$|\s{2,}(?=\S))",
    re.MULTILINE,
)

# All-caps shapes that are structure or performance directions, never speakers.
_NON_SPEAKER_TOKENS = frozenset({
    "ACT", "SCENE", "FINIS", "THE END", "EPILOGUE", "PROLOGUE", "EXIT", "EXEUNT",
})

# A real work body is never this small. The 145-word fabricated fixture that
# triggered this tool would fail here, which is the point.
MIN_BODY_WORDS = 2000
# A single scene is legitimately shorter than a whole work, but still not a stub.
MIN_UNIT_WORDS = 200

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


def slice_folger_scene(body: str, *, act: int, scene: int, origin: str) -> str:
    """Cut one ACT n / Scene m out of a Folger plain-text play.

    The scene runs to the next "Scene" heading, or to the next "ACT" heading when
    it is the last scene of its act, or to the end of the play.
    """
    act_starts = [(int(m.group(1)), m.start()) for m in _ACT_LINE_RE.finditer(body)]
    if not act_starts:
        raise FetchError(f"{origin}: no ACT headings found -- not a Folger text?")
    act_at = next((pos for num, pos in act_starts if num == act), None)
    if act_at is None:
        raise FetchError(
            f"{origin}: ACT {act} not found (play has acts "
            f"{sorted(n for n, _ in act_starts)})"
        )
    act_end = next((pos for num, pos in act_starts if pos > act_at), len(body))
    act_body = body[act_at:act_end]

    scenes = [(int(m.group(1)), m.start()) for m in _SCENE_LINE_RE.finditer(act_body)]
    scene_at = next((pos for num, pos in scenes if num == scene), None)
    if scene_at is None:
        raise FetchError(
            f"{origin}: ACT {act} has no Scene {scene} (scenes "
            f"{sorted(n for n, _ in scenes)})"
        )
    scene_end = next((pos for num, pos in scenes if pos > scene_at), len(act_body))
    text = act_body[scene_at:scene_end].strip("\r\n \t")
    if not text:
        raise FetchError(f"{origin}: ACT {act} Scene {scene} sliced empty")
    return text


def speakers_in_order(scene_text: str) -> list[str]:
    """Every speaking character in first-appearance order.

    This is why `cast_hints` can be retired on the fidelity lanes: the people in a
    scene are a fact OF the scene, not a hand-curated list that can disagree with
    it. A curated list ordered ['Rosalind','Celia','Orlando'] is what dropped
    Orlando -- the character the scene is about -- and let a writer substitute
    Romeo from another play entirely.
    """
    found: list[str] = []
    for match in _SPEECH_PREFIX_RE.finditer(scene_text):
        name = match.group("name").strip()
        if name in _NON_SPEAKER_TOKENS:
            continue
        if name not in found:
            found.append(name)
    return found


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
    extra: dict | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write the body plus its provenance sidecar. Returns both paths.

    ``dest_dir`` exists because a bank manifest may only reference MANIFEST-LOCAL
    relative paths -- `..` is rejected by the schema validators on purpose -- so
    vendored text has to land inside the bank's own directory rather than in a
    shared corpus one level up.
    """
    root = dest_dir or CORPUS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    stem = slug if unit is None else f"{slug}__{unit}"
    text_path = root / f"{stem}.txt"
    sidecar_path = root / f"{stem}.provenance.json"

    # Newlines are normalized to LF BEFORE hashing, and the bytes are written
    # without platform translation. Gutenberg serves CRLF; git normalizes line
    # endings on commit and may hand a clone something different again. A
    # provenance hash that a checkout can invalidate would be worse than no hash
    # at all -- it would fail exactly when someone tried to verify an attribution.
    body = body.replace("\r\n", "\n").replace("\r", "\n")
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
    if extra:
        provenance.update(extra)
    sidecar_path.write_bytes(
        (json.dumps(provenance, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    return text_path, sidecar_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gutenberg-id", type=int,
                      help="Project Gutenberg ebook id (The Time Machine = 35)")
    mode.add_argument("--folger-play",
                      help="Folger play slug, e.g. as-you-like-it, macbeth")
    parser.add_argument("--slug", required=True,
                        help="corpus filename stem, e.g. time_machine")
    parser.add_argument("--work-title", default="", help="title for the provenance sidecar")
    parser.add_argument("--author", default="", help="author for the provenance sidecar")
    parser.add_argument("--license-label", default="",
                        help="rights label recorded in the sidecar")
    parser.add_argument("--unit", default=None, help="optional named unit")
    parser.add_argument("--start-marker", default=None, help="Gutenberg unit slicing")
    parser.add_argument("--end-marker", default=None, help="Gutenberg unit slicing")
    parser.add_argument("--act", type=int, default=None, help="Folger act number")
    parser.add_argument("--scene", type=int, default=None, help="Folger scene number")
    parser.add_argument("--min-words", type=int, default=MIN_BODY_WORDS,
                        help="floor below which the whole body is refused as a stub")
    parser.add_argument("--dest-dir", default=None,
                        help="repo-relative directory to write into; defaults to the "
                             "shared corpus. Bank manifests may only cite "
                             "manifest-local paths, so point this at the bank's own "
                             "directory when vendoring for a bank.")
    args = parser.parse_args(argv)

    extra: dict = {}

    if args.folger_play:
        if args.act is None or args.scene is None:
            parser.error("--folger-play requires --act and --scene")
        url = FOLGER_TEXT_URL.format(play=args.folger_play)
        origin = f"folger:{args.folger_play}"
        license_label = args.license_label or "CC BY-NC 3.0 (Folger Shakespeare Library)"
        author = args.author or "William Shakespeare"
        print(f"[fetch] {url}")
        body = fetch_url(url)
        verify_body(body, origin=origin, min_words=args.min_words)
        # The cast list lives in the WHOLE play, never in a sliced scene, so it
        # must be read before slicing. Its descriptions carry the gender that
        # casting otherwise assigns by dice roll: every Shakespeare name is
        # absent from the 1930s first-name lexicon, so a female lead draws a
        # male voice on roughly half of seeds without this.
        roster = parse_character_roster(body)
        body = slice_folger_scene(
            body, act=args.act, scene=args.scene,
            origin=f"{origin}:act{args.act}-scene{args.scene}",
        )
        word_count = verify_body(
            body,
            origin=f"{origin}:act{args.act}-scene{args.scene}",
            min_words=MIN_UNIT_WORDS,
        )
        speakers = speakers_in_order(body)
        if not speakers:
            # Never vendor a scene with no parsed cast. Downstream, an empty
            # source-name queue falls through to the pool that INVENTS
            # characters -- the exact defect this corpus exists to end. A parse
            # that finds nobody means the parser is wrong about this text, and
            # that has to surface here, not as a stranger in the episode.
            raise FetchError(
                f"{origin}:act{args.act}-scene{args.scene}: parsed ZERO speakers "
                f"from {word_count} words. Refusing to vendor a scene whose cast "
                f"cannot be derived from its own text."
            )
        def _record_for(prefix: str):
            # Alias-aware: the cast list writes "SIGNIOR BENEDICK" and
            # "COUNT CLAUDIO" while the text speaks as BENEDICK and CLAUDIO.
            return next((r for r in roster if r.matches(prefix)), None)

        characters = []
        for name in speakers:
            record = _record_for(name)
            characters.append({
                "name": name,
                "roster_name": record.name if record else "",
                "description": record.description if record else "",
                "gender": record.gender if record else "unknown",
                "gender_source": (
                    record.gender_source if record else "absent_from_roster"
                ),
            })
        extra = {
            "act": args.act,
            "scene": args.scene,
            "speakers": speakers,
            # Only the characters who actually speak in THIS scene, so the
            # render path reads a roster scoped to what it must cast.
            "characters": characters,
        }
        print(f"[fetch] ACT {args.act} Scene {args.scene}: {word_count} words")
        print(f"[fetch] speakers in order: {', '.join(speakers) or '(none parsed)'}")
        _resolved = sum(
            1 for c in extra["characters"] if c["gender"] in ("male", "female")
        )
        print(
            f"[fetch] roster: {len(roster)} characters in the play; "
            f"gender resolved for {_resolved}/{len(speakers)} of this scene's speakers"
        )
        for entry in extra["characters"]:
            if entry["gender"] == "unknown":
                print(
                    f"[fetch]   UNKNOWN gender: {entry['name']} "
                    f"({entry['gender_source']}) -- recorded, not guessed"
                )
    else:
        url = GUTENBERG_TEXT_URL.format(gid=args.gutenberg_id)
        origin = f"gutenberg:{args.gutenberg_id}"
        license_label = args.license_label or "Public domain (US)"
        author = args.author
        print(f"[fetch] {url}")
        raw = fetch_url(url)
        body = strip_gutenberg_boilerplate(raw, origin=origin)
        full_words = verify_body(body, origin=origin, min_words=args.min_words)
        print(f"[fetch] work body: {full_words} words")
        if args.unit:
            if not (args.start_marker and args.end_marker):
                parser.error("a Gutenberg --unit requires --start-marker and --end-marker")
            body = slice_unit(
                body, start_marker=args.start_marker, end_marker=args.end_marker,
                origin=f"{origin}:{args.unit}",
            )
            word_count = verify_body(
                body, origin=f"{origin}:{args.unit}", min_words=MIN_UNIT_WORDS
            )
            print(f"[fetch] unit {args.unit!r}: {word_count} words")
        else:
            word_count = full_words

    text_path, sidecar_path = write_source(
        body,
        slug=args.slug,
        unit=args.unit,
        source_url=url,
        work_title=args.work_title or args.slug,
        author=author,
        license_label=license_label,
        word_count=word_count,
        dest_dir=(REPO_ROOT / args.dest_dir) if args.dest_dir else None,
        extra=extra,
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
