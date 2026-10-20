"""Character photos: a person's own picture as a character's identity.

OTR_ImageDirector's ``character_photos`` widget takes one line per character,
``Name = file`` -- or ``Name = a.png, b.png`` for up to PHOTOS_PER_CHARACTER
pictures that together make one face (operator, 2026-10-20: "feel free to make
an amalgamation") -- naming pictures in ComfyUI's input folder. The parsed map
rides the image policy as ``character_photos``. MetaBrief matches each name to
the cast ONCE (``match_cast``) and tags that character's portrait object with
``identity_photo``; the dispatcher then hands the photo to the still engine as
the portrait's reference, so the portrait comes out in the episode's style and
every still of the character follows it. The raw photo never appears in the
episode.

Widget values are untrusted: a file is a bare name inside ComfyUI's input
folder, never a path, and is resolved and confined there at use time.

Cold-import clean: stdlib only.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, Mapping

#: At most this many photos per character; each is one more reference latent.
PHOTOS_PER_CHARACTER = 3

#: A bare picture name inside ComfyUI's input folder -- no separator, no "..".
SAFE_PHOTO_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,95}\.(png|jpg|jpeg|webp)$",
                             re.IGNORECASE)


def name_key(name: str) -> str:
    """The match key for a character name: whitespace folded, case folded.
    The same fold the My Story writer uses to find a named person in its cast."""
    return " ".join(str(name or "").split()).casefold()


def parse_photo_lines(text: str) -> "tuple[dict, list]":
    """``({name: [file, ...]}, problems)`` from ``Name = file[, file...]`` lines.

    A malformed line, an unsafe file name or a name given twice is a problem,
    reported and skipped -- the episode still renders. Past
    PHOTOS_PER_CHARACTER, the rest of a line is reported and dropped."""
    photos: dict = {}
    seen: dict = {}
    problems: list = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        name, sep, listed = line.rpartition("=")
        name = " ".join(name.split())
        if not sep or not name:
            problems.append(f"{line!r}: write it as Name = file")
            continue
        files = []
        for filename in (f.strip() for f in listed.split(",")):
            if not filename:
                continue
            if not SAFE_PHOTO_NAME.match(filename) or ".." in filename:
                problems.append(f"{name}: {filename!r} is not a picture name in ComfyUI's "
                                "input folder (png, jpg or webp, no folders)")
            elif filename not in files:
                files.append(filename)
        if len(files) > PHOTOS_PER_CHARACTER:
            problems.append(f"{name}: {len(files)} photos; the first "
                            f"{PHOTOS_PER_CHARACTER} are used")
            files = files[:PHOTOS_PER_CHARACTER]
        if not files:
            continue
        key = name_key(name)
        if key in seen:
            problems.append(f"{name}: given on two lines; neither is used")
            photos.pop(seen[key], None)
            continue
        seen[key] = name
        photos[name] = files
    return photos, problems


def photo_list(value) -> list:
    """A character's photos as a list: a policy written before several photos
    were allowed carries one file name as a string."""
    if isinstance(value, str):
        return [value] if value else []
    return [str(v) for v in (value or []) if v]


def match_cast(cast: Iterable, photos: Mapping) -> "tuple[dict, list]":
    """``({char_id: [file, ...]}, problems)``: each photo's name matched to exactly one
    cast member by ``name_key``. A name no cast member has, or that two share,
    is a problem -- never a guess."""
    by_key: dict = {}
    for row in cast or ():
        if not isinstance(row, Mapping):
            continue
        cid, name = str(row.get("char_id") or ""), str(row.get("name") or "")
        if cid and name:
            by_key.setdefault(name_key(name), []).append(cid)
    matched: dict = {}
    problems: list = []
    for name, filename in (photos or {}).items():
        ids = by_key.get(name_key(name), [])
        if len(ids) == 1:
            matched[ids[0]] = photo_list(filename)
        elif not ids:
            problems.append(f"{name}: no character by that name in this episode's cast; "
                            "the photo is not used")
        else:
            problems.append(f"{name}: {len(ids)} characters share that name; "
                            "the photo is not used")
    return matched, problems


def input_dir() -> str:
    """ComfyUI's input folder (where its LoadImage and /upload/image work)."""
    try:
        import folder_paths  # type: ignore
        return folder_paths.get_input_directory()
    except Exception:  # noqa: BLE001 -- headless or test: ComfyUI/input beside the pack
        here = os.path.abspath(__file__)
        comfy = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(here)))))
        return os.path.join(comfy, "input")


def resolve_photo(filename: str, base: "str | None" = None) -> str:
    """The absolute path of a photo inside the input folder, or "" when the
    name is unsafe, escapes the folder, or names no file."""
    if not SAFE_PHOTO_NAME.match(str(filename or "")) or ".." in str(filename):
        return ""
    root = os.path.realpath(base or input_dir())
    path = os.path.realpath(os.path.join(root, filename))
    if os.path.dirname(path) != root or not os.path.isfile(path):
        return ""
    return path


def photo_hash(path: str) -> str:
    """sha256 of the photo file: its identity in the still cache key."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["SAFE_PHOTO_NAME", "PHOTOS_PER_CHARACTER", "name_key", "parse_photo_lines",
           "photo_list", "match_cast",
           "input_dir", "resolve_photo", "photo_hash"]
