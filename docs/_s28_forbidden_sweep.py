# S28 forbidden-pattern sweep (Phase 5 hand-off artifact).
#
# Walks the s27-cleanbreak-tail..HEAD diff of every *.py file and
# pulls every ADDED line ('+') that matches one of the S28 forbidden
# patterns. Then opens the actual source file and checks whether
# that line is inside a docstring (via tokenize), because git diff
# context isn't wide enough to track docstring state across a
# typical hunk.
from __future__ import annotations

import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root

DIFF_PATH = ROOT / "docs" / "s28_diff_tmp.txt"
OUT_PATH = ROOT / "docs" / "2026-05-13-S28-new-forbidden-hits.txt"

forbidden = re.compile(
    r"DeprecationWarning"
    r"|back-compat"
    r"|legacy fallback"
    r"|legacy shape"
    r"|\bshim\b"
    r"|\balias\b"
    r"|\botr_legacy_audio_dir\b"
    r"|budget is None"
)
diff_file_re = re.compile(r"^\+\+\+ b/(.+)$")
hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def classify_lines(path: Path) -> dict[int, str]:
    """Return a map of 1-indexed line number to a classification:
    "code", "comment", "string" (docstring or triple-quoted)."""
    src = path.read_text(encoding="utf-8", errors="replace")
    classifications: dict[int, str] = {}
    try:
        toks = list(tokenize.generate_tokens(iter(src.splitlines(keepends=True)).__next__))
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        # Fall back to "code" for every line if tokenize chokes.
        for i in range(1, src.count("\n") + 2):
            classifications[i] = "code"
        return classifications
    string_lines: set[int] = set()
    comment_lines: set[int] = set()
    for tok in toks:
        if tok.type == tokenize.STRING and "\n" in tok.string:
            start_l, _ = tok.start
            end_l, _ = tok.end
            for L in range(start_l, end_l + 1):
                string_lines.add(L)
        elif tok.type == tokenize.STRING:
            # Single-line string — these don't span lines so just
            # mark the one.
            start_l, _ = tok.start
            string_lines.add(start_l)
        elif tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])
    total = src.count("\n") + 1
    for i in range(1, total + 1):
        if i in string_lines:
            classifications[i] = "string"
        elif i in comment_lines:
            classifications[i] = "comment"
        else:
            classifications[i] = "code"
    return classifications


# Walk the diff, collecting (file, new_line_number, line_text) for
# every `+` line that hits a forbidden pattern.
hits_raw: list[tuple[str, int, str]] = []
current_file: str | None = None
current_line: int | None = None
for raw in DIFF_PATH.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True):
    m = diff_file_re.match(raw.rstrip("\n"))
    if m:
        current_file = m.group(1)
        current_line = None
        continue
    if raw.startswith("---"):
        continue
    mh = hunk_re.match(raw.rstrip("\n"))
    if mh:
        current_line = int(mh.group(1))
        continue
    if current_line is None or current_file is None:
        continue
    side = raw[:1]
    if side == "+":
        if forbidden.search(raw):
            hits_raw.append((current_file, current_line, raw))
        current_line += 1
    elif side == " ":
        current_line += 1
    # `-` lines do not advance the new-side counter

# For each hit, look at the actual file and classify.
classified_cache: dict[str, dict[int, str]] = {}
forensic: list[str] = []
runtime: list[str] = []
for fpath, lineno, raw in hits_raw:
    abs_path = ROOT / fpath
    if not abs_path.exists():
        runtime.append(f"{fpath}:{lineno} [missing-file] {raw}")
        continue
    cls = classified_cache.get(fpath)
    if cls is None:
        cls = classify_lines(abs_path)
        classified_cache[fpath] = cls
    kind = cls.get(lineno, "code")
    line_str = f"{fpath}:{lineno} [{kind}] {raw}"
    if kind in ("string", "comment"):
        forensic.append(line_str)
    else:
        runtime.append(line_str)

OUT_PATH.write_text("".join(runtime), encoding="utf-8")
print(f"HITS: {len(hits_raw)}  forensic: {len(forensic)}  runtime: {len(runtime)}")
print(f"OUT:  {OUT_PATH}")
if forensic:
    print("--- forensic (suppressed from output) ---")
    for l in forensic:
        print("  " + l.rstrip("\n"))
if runtime:
    print("--- runtime (in output file) ---")
    for l in runtime:
        print("  " + l.rstrip("\n"))
