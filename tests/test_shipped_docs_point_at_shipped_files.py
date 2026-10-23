"""Nothing that ships may point a reader at a file that does not.

The registry zip carries README.md, the Studio's README, every apple/*.md not
named in .comfyignore, and the node help pages under js/docs/. On 2026-10-23 a
sweep excluded internal records from the zip and, for one of them, a shipped
guide and a shipped code comment still named it (Codex QA): a reader of the
installed pack would have followed a path to nothing. Every apple/ file a
shipped document names -- as `apple/<file>` or as the bare file name -- must
exist in git and ship. A link into this repository on GitHub is the one way to
name a file that does not ship, and the link's target must exist too.
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GITHUB_BLOB = "https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/"
_REF = re.compile(r"(?<![/\w])apple/([A-Za-z0-9_.\-]+\.(?:md|yaml|json|txt))")
_BARE = re.compile(r"(?<![/\w])([A-Za-z0-9_\-]+(?:--[A-Za-z0-9_.\-]+)?\.(?:md|yaml|json|txt))\b")
_LINK = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")


def _tracked(prefix: str) -> set:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", prefix],
                         capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def _ignore_patterns() -> list:
    return [line.strip() for line in (REPO / ".comfyignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith("!")]


def _excluded(rel: str, patterns: list) -> bool:
    return any((pat.endswith("/") and rel.startswith(pat)) or fnmatch.fnmatch(rel, pat) or rel == pat
               for pat in patterns)


def _tracked_guides() -> list:
    """apple/*.md as git holds them, top level only: an untracked file never
    ships, and apple/evidence/ and the schema folder are excluded whole."""
    return sorted(REPO / rel for rel in _tracked("apple/*.md") if rel.count("/") == 1)


def _shipped_guides() -> list:
    patterns = _ignore_patterns()
    return [p for p in _tracked_guides() if not _excluded(f"apple/{p.name}", patterns)]


def _shipped_documents() -> list:
    docs = [REPO / "README.md", REPO / "contest" / "radio-play-studio" / "README.md"]
    return docs + _shipped_guides() + sorted(REPO / rel for rel in _tracked("js/docs/*.md"))


def _apple_names() -> set:
    """Every file name under apple/ that git holds, at any depth."""
    return {rel.rsplit("/", 1)[-1] for rel in _tracked("apple/")}


def _problems(text: str, patterns: list, names: set, tracked_all: set) -> list:
    problems = []
    for url in _LINK.findall(text):
        if url.startswith(GITHUB_BLOB):
            target = url[len(GITHUB_BLOB):].split("#")[0]
            if target not in tracked_all:
                problems.append(f"GitHub link to {target}, which git does not hold")
        # Other links into the repository -- its root, /issues, a folder on
        # /tree/main -- are the reader's to follow; only a file link is held.
    # A line that links into this repository on GitHub may name the file it links.
    for line in text.splitlines():
        if GITHUB_BLOB in line:
            continue
        for name in set(_REF.findall(line)):
            rel = f"apple/{name}"
            if not (REPO / rel).exists():
                problems.append(f"{rel} does not exist")
            elif _excluded(rel, patterns):
                problems.append(f"{rel} is excluded by .comfyignore")
        # A bare file name is held only when it is an apple/ file's, and every
        # tracked file of that name is excluded: a style's `anime.json` that
        # also ships from config/ is the shipped one.
        for name in {n for n in _BARE.findall(line) if n in names}:
            holders = [rel for rel in tracked_all if rel.rsplit("/", 1)[-1] == name]
            if holders and all(_excluded(rel, patterns) for rel in holders):
                problems.append(f"{name} ({', '.join(sorted(holders))}) is excluded by .comfyignore")
    return sorted(set(problems))


@pytest.mark.parametrize("doc", _shipped_documents(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_apple_file_a_shipped_document_names_exists_and_ships(doc):
    problems = _problems(doc.read_text(encoding="utf-8"), _ignore_patterns(), _apple_names(),
                         _tracked("."))
    assert problems == [], f"{doc.relative_to(REPO).as_posix()} names: " + "; ".join(problems)


def test_a_bare_file_name_and_a_bad_github_link_are_caught():
    names = {"IDEOGRAM4_LICENSE_ATTESTATION.md", "INSTALL.md"}
    patterns = ["apple/IDEOGRAM4_LICENSE_ATTESTATION.md"]
    tracked = {"apple/INSTALL.md", "apple/IDEOGRAM4_LICENSE_ATTESTATION.md"}
    text = ("see `IDEOGRAM4_LICENSE_ATTESTATION.md` and apple/INSTALL.md\n"
            "and [the plan](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/NOPE.md)\n"
            "but [this](" + GITHUB_BLOB + "apple/IDEOGRAM4_LICENSE_ATTESTATION.md) is fine, "
            "as is this IDEOGRAM4_LICENSE_ATTESTATION.md beside it\n")
    assert _problems(text, patterns, names, tracked) == [
        "GitHub link to apple/NOPE.md, which git does not hold",
        "IDEOGRAM4_LICENSE_ATTESTATION.md (apple/IDEOGRAM4_LICENSE_ATTESTATION.md) is "
        "excluded by .comfyignore"]
    # A name that also ships from elsewhere is the shipped one.
    assert _problems("see `anime.json`", ["apple/schema/"], {"anime.json"},
                     {"apple/schema/anime.json", "config/styles/anime.json"}) == []


def test_the_readme_guide_table_is_the_shipped_set():
    """Every guide the README's table links ships, and every shipped guide
    that is not a licence or rights record is on the table."""
    patterns = _ignore_patterns()
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start = readme.index("## The guides that ship with it")
    table = readme[start:readme.index("\n## ", start + 1)]
    linked = set(re.findall(r"\(apple/([A-Za-z0-9_.\-]+\.md)\)", table))
    assert linked, "the README's guide table links nothing"
    for name in sorted(linked):
        assert not _excluded(f"apple/{name}", patterns), f"README links apple/{name}, which does not ship"
    shipped = {p.name for p in _shipped_guides()}
    records = {n for n in shipped if "LICENSE" in n or n.startswith("model-license")}
    off_table = sorted(shipped - linked - records)
    assert off_table == ["ARCHITECTURE.md"], (
        "shipped guides the README's table does not link (ARCHITECTURE.md is the one "
        f"reader's map that is linked from the guides instead): {off_table}")
