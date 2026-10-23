"""Nothing that ships may point a reader at a file that does not.

The registry zip carries README.md, the Studio's README and every apple/*.md
not named in .comfyignore. On 2026-10-23 a sweep excluded internal records from
the zip and, for one of them, a shipped guide and a shipped code comment still
named it (Codex QA): a reader of the installed pack would have followed a path
to nothing. Every `apple/<file>` a shipped document names must exist and ship.
"""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: A relative reference. One that is a GitHub link -- the link text may say
#: apple/X.md, the target is the repository -- reaches a file the reader can
#: open whether or not it ships, so those links are removed before the scan.
_REF = re.compile(r"(?<![/\w])apple/([A-Za-z0-9_.\-]+\.(?:md|yaml|json|txt))")
_GITHUB_LINK = re.compile(r"\[[^\]]*\]\(https://github\.com/[^)]*\)")


def _ignore_patterns() -> list:
    return [line.strip() for line in (REPO / ".comfyignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#") and not line.startswith("!")]


def _excluded(rel: str, patterns: list) -> bool:
    return any((pat.endswith("/") and rel.startswith(pat)) or fnmatch.fnmatch(rel, pat) or rel == pat
               for pat in patterns)


def _tracked_guides() -> list:
    """apple/*.md as git holds them: an untracked file never ships."""
    import subprocess
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", "apple/*.md"],
                         capture_output=True, text=True, check=True).stdout
    # git's pathspec `*` crosses folders; the guides are the top-level files,
    # and apple/evidence/ and the schema folder are excluded whole.
    return sorted(REPO / line.strip() for line in out.splitlines()
                  if line.strip() and line.strip().count("/") == 1)


def _shipped_guides() -> list:
    patterns = _ignore_patterns()
    return [p for p in _tracked_guides() if not _excluded(f"apple/{p.name}", patterns)]


def _shipped_documents() -> list:
    return [REPO / "README.md", REPO / "contest" / "radio-play-studio" / "README.md"] + _shipped_guides()


@pytest.mark.parametrize("doc", _shipped_documents(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_apple_file_a_shipped_document_names_exists_and_ships(doc):
    patterns = _ignore_patterns()
    text = _GITHUB_LINK.sub("", doc.read_text(encoding="utf-8"))
    problems = []
    for name in sorted(set(_REF.findall(text))):
        rel = f"apple/{name}"
        if not (REPO / rel).exists():
            problems.append(f"{rel} does not exist")
        elif _excluded(rel, patterns):
            problems.append(f"{rel} is excluded by .comfyignore")
    assert problems == [], f"{doc.relative_to(REPO).as_posix()} names: " + "; ".join(problems)


def test_the_readme_guide_table_is_the_shipped_set():
    """Every guide the README's table links ships, and every shipped guide
    that is not a licence or rights record is on the table."""
    patterns = _ignore_patterns()
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    linked = set(re.findall(r"\(apple/([A-Za-z0-9_.\-]+\.md)\)", readme))
    for name in sorted(linked):
        assert not _excluded(f"apple/{name}", patterns), f"README links apple/{name}, which does not ship"
    shipped = {p.name for p in _shipped_guides()}
    records = {n for n in shipped if "LICENSE" in n or n.startswith("model-license")}
    off_table = sorted(shipped - linked - records)
    assert off_table == ["ARCHITECTURE.md"], (
        "shipped guides the README's table does not link (ARCHITECTURE.md is the one "
        f"reader's map that is linked from the guides instead): {off_table}")
