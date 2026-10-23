"""No shipped document may point a reader at an apple/ file that does not ship.

The registry zip carries README.md, the Studio's README, every apple/*.md not
named in .comfyignore, and the node help pages under js/docs/. On 2026-10-23 a
sweep excluded internal records from the zip and, for one of them, a shipped
guide and a shipped code comment still named it (Codex QA): a reader of the
installed pack would have followed a path to nothing. Every apple/ path a
shipped document names -- `apple/<file>`, `apple/<folder>/<file>`, or the bare
file name of an apple/ file -- must exist in git and ship. The one way to name
a file that does not ship is to link it on GitHub, on that line, and the link's
target must exist in git. Links to this repository's root, its issues, a folder
on /tree/main or a raw file are the reader's to follow and are not held here;
nor are references to scripts/ or tests/, which the guides for people working
in the git tree name on purpose.
"""
from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GITHUB_BLOB = "https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/"
GITHUB_BLOB_ANY = "https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
_REF = re.compile(r"(?<![/\w])apple/((?:[A-Za-z0-9_.\-]+/)*[A-Za-z0-9_.\-]+\.(?:md|yaml|json|txt))")
_BARE = re.compile(r"(?<![/\w])([A-Za-z0-9_\-]+(?:--[A-Za-z0-9_.\-]+)?\.(?:md|yaml|json|txt))\b")
#: A file link on GitHub: on main, or a permalink to a commit.
_BLOB = re.compile(re.escape(GITHUB_BLOB_ANY) + r"(main|[0-9a-f]{7,40})/([^\s)\]<>\"'#?]+)")


def _held_in_git(ref: str, path: str, tracked_all: set) -> bool:
    """main's tracked files for a main link; the commit's tree for a permalink."""
    if ref == "main":
        return path in tracked_all
    import subprocess
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{ref}:{path}"],
                          capture_output=True).returncode == 0


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
    # A Windows spelling of a path names the same file (Codex QA, 2026-10-23).
    for line in text.replace("\\", "/").splitlines():
        # Every file this line links on GitHub, bare URL or markdown link alike.
        linked = set()
        for ref, target in _BLOB.findall(line):
            target = target.rstrip(".,;:")       # a bare URL at the end of a sentence
            if not _held_in_git(ref, target, tracked_all):
                problems.append(f"GitHub link to {target} at {ref}, which git does not hold")
            linked.add(target)
        # A reference is exempt only when THIS line links that same file.
        for rel in {f"apple/{m}" for m in _REF.findall(line)}:
            if rel in linked:
                continue
            if rel not in tracked_all:
                problems.append(f"{rel} is not in git")
            elif _excluded(rel, patterns):
                problems.append(f"{rel} is excluded by .comfyignore")
        # A bare file name is held only when it is an apple/ file's and every
        # tracked file of that name is excluded: a style's `anime.json` that
        # also ships from config/ is the shipped one.
        for name in {n for n in _BARE.findall(line) if n in names}:
            holders = [rel for rel in tracked_all if rel.rsplit("/", 1)[-1] == name]
            if any(h in linked for h in holders):
                continue
            if holders and all(_excluded(rel, patterns) for rel in holders):
                problems.append(f"{name} ({', '.join(sorted(holders))}) is excluded by .comfyignore")
    return sorted(set(problems))


@pytest.mark.parametrize("doc", _shipped_documents(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_apple_file_a_shipped_document_names_exists_and_ships(doc):
    problems = _problems(doc.read_text(encoding="utf-8"), _ignore_patterns(), _apple_names(),
                         _tracked("."))
    assert problems == [], f"{doc.relative_to(REPO).as_posix()} names: " + "; ".join(problems)


def test_the_rule_catches_what_it_must_and_nothing_else():
    names = {"IDEOGRAM4_LICENSE_ATTESTATION.md", "INSTALL.md", "anime.json", "manifest.json"}
    patterns = ["apple/IDEOGRAM4_LICENSE_ATTESTATION.md", "apple/evidence/", "apple/schema/"]
    tracked = {"apple/INSTALL.md", "apple/IDEOGRAM4_LICENSE_ATTESTATION.md", "apple/evidence/manifest.json",
               "apple/schema/anime.json", "config/styles/anime.json"}
    held = _problems(
        "see `IDEOGRAM4_LICENSE_ATTESTATION.md` and apple/INSTALL.md\n"
        "and [the plan](" + GITHUB_BLOB + "apple/NOPE.md)\n"
        "the receipts in apple/evidence/manifest.json\n"
        # A link on the line exempts only the file it links.
        "[ok](" + GITHUB_BLOB + "apple/INSTALL.md) beside apple/IDEOGRAM4_LICENSE_ATTESTATION.md\n",
        patterns, names, tracked)
    assert held == [
        "GitHub link to apple/NOPE.md at main, which git does not hold",
        "IDEOGRAM4_LICENSE_ATTESTATION.md (apple/IDEOGRAM4_LICENSE_ATTESTATION.md) is excluded by .comfyignore",
        "apple/IDEOGRAM4_LICENSE_ATTESTATION.md is excluded by .comfyignore",
        "apple/evidence/manifest.json is excluded by .comfyignore"]
    # A Windows spelling names the same excluded file (Codex QA, 2026-10-23).
    assert _problems("See apple\\evidence\\manifest.json", patterns, names, tracked) == [
        "apple/evidence/manifest.json is excluded by .comfyignore"]
    # A permalink to a commit is held against that commit's tree: this file at
    # the repository's first commit on main does not exist there.
    import subprocess
    root = subprocess.run(["git", "-C", str(REPO), "rev-list", "--max-parents=0", "HEAD"],
                          capture_output=True, text=True).stdout.split()[0]
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    assert _problems(f"[then]({GITHUB_BLOB_ANY}{root}/tests/test_shipped_docs_point_at_shipped_files.py)",
                     patterns, names, tracked) == [
        f"GitHub link to tests/test_shipped_docs_point_at_shipped_files.py at {root}, which git does not hold"]
    fine = _problems(
        # A bare URL counts as a link; a query string or fragment is not part of the path.
        "it is at " + GITHUB_BLOB + "apple/IDEOGRAM4_LICENSE_ATTESTATION.md?plain=1#top, "
        "the IDEOGRAM4_LICENSE_ATTESTATION.md note\n"
        "see `anime.json`, which ships from config/\n"
        "and https://github.com/jbrick2070/ComfyUI-OldTimeRadio/issues\n"
        f"and a permalink [now]({GITHUB_BLOB_ANY}{head}/tests/test_shipped_docs_point_at_shipped_files.py)\n",
        patterns, names, tracked)
    assert fine == []


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
