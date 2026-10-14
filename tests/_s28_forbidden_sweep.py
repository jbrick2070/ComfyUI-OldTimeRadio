# S28 forbidden-pattern sweep (Phase 5 hand-off artifact).
#
# Walks the s27-cleanbreak-tail..HEAD diff of every *.py file and
# pulls every ADDED line ('+') that matches one of the S28 forbidden
# patterns. Then opens the actual source file and checks whether
# that line is inside a docstring (via tokenize), because git diff
# context isn't wide enough to track docstring state across a
# typical hunk.
from __future__ import annotations

import os
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root

# The diff/out paths default to docs/ but are overridable via env. The
# `docs/` tree lives under a OneDrive-synced Documents folder; OneDrive
# holds the multi-MB diff file open mid-sync, so a later truncating
# re-open can fail with OSError(22) (caught 2026-05-31 -- forbidden
# sweep red baseline). Callers (the regression test) point these at the
# OS temp dir, which is not synced, to make the sweep deterministic.
DIFF_PATH = Path(os.environ.get("OTR_S28_DIFF_PATH") or (ROOT / "apple" / "s28_diff_tmp.txt"))
OUT_PATH = Path(os.environ.get("OTR_S28_OUT_PATH") or (ROOT / "tmp" / "s28_new_forbidden_hits.txt"))

forbidden = re.compile(
    # No back-compat shims, aliases or deprecation paths (clean-break law).
    r"DeprecationWarning"
    r"|back-compat"
    r"|legacy fallback"
    r"|legacy shape"
    r"|\bshim\b"
    # "alias" as prose (comments, strings) or a bare name -- but not the
    # API spellings: pydantic's alias= keyword, an .alias attribute, or the
    # ast alias node (`alias.name`, `for alias in`). Those are vocabulary,
    # not a back-compat alias, and they matter now that a hit is judged by
    # where it sits on the line rather than by whether the line has a string.
    r"|(?<!\.)\balias\b(?!=[^=])(?!\.name)(?!\s+in\b)"
    # No hardcoded developer path in .py files.
    r"|C:/Users/jeffr"
    # One generate surface (Hard rule #5): no wrapper-by-another-name on top
    # of `make_generate_fn`.
    r"|\bgenerate_text\b"
    r"|\bgenerate_with_llm\b"
    # No period-locked language in prompts (era literal cleanbreak).
    # Tokenize suppression handles forensic mentions in docstrings and
    # comments.
    r"|\b1940s\b"
    r"|1980s broadcast"
    r"|1950s Americana"
    r"|golden.age radio"
    r"|\bOmni.Retro\b"
    r"|\bOrson Welles\b"
    r"|\bNorman Corwin\b"
    r"|\bLucille Fletcher\b"
    # Router substring-dispatch guard: exact-match dispatch only.
    r"|if.+talkie.+in repo_id"
    # The validator path widget must not default to a hardcoded operator path.
    r"|workflow_json_path.+default.+C:"
)
diff_file_re = re.compile(r"^\+\+\+ b/(.+)$")
hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


_STRING_TOKEN_TYPES = (
    tokenize.STRING,
    # Python 3.12 splits an f-string into START / MIDDLE / END tokens; the
    # expressions inside its braces arrive as ordinary code tokens.
    getattr(tokenize, "FSTRING_START", -1),
    getattr(tokenize, "FSTRING_MIDDLE", -1),
    getattr(tokenize, "FSTRING_END", -1),
)


def string_and_comment_spans(path: Path) -> dict[int, list[tuple[int, int]]] | None:
    """Map each 1-indexed line to the column spans that are string or comment.

    A multi-line string covers every line it touches (to end of line on all
    but its last). ``None`` when tokenize cannot read the file, which the
    caller treats as all code -- a gate that cannot see must not wave a hit
    through.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        toks = list(tokenize.generate_tokens(
            iter(src.splitlines(keepends=True)).__next__))
    except (tokenize.TokenError, SyntaxError):  # IndentationError is a SyntaxError
        return None
    spans: dict[int, list[tuple[int, int]]] = {}
    for tok in toks:
        if tok.type not in _STRING_TOKEN_TYPES and tok.type != tokenize.COMMENT:
            continue
        (start_l, start_c), (end_l, end_c) = tok.start, tok.end
        for line in range(start_l, end_l + 1):
            spans.setdefault(line, []).append((
                start_c if line == start_l else 0,
                end_c if line == end_l else 1 << 30,
            ))
    return spans


def classify_hit(spans: dict[int, list[tuple[int, int]]] | None,
                 lineno: int, text: str) -> str:
    """``"code"`` when ANY forbidden match on the line falls outside every
    string and comment span, else ``"forensic"``.

    Per MATCH, not per line: a line holding both a string and executable code
    used to be called a string line as a whole, so ``x = generate_text("hi")``
    passed as a forensic mention.
    """
    if spans is None:
        return "code"
    covered = spans.get(lineno, ())
    for m in forbidden.finditer(text):
        if not any(a <= m.start() and m.end() <= b for a, b in covered):
            return "code"
    return "forensic"


def main() -> None:
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

    # For each hit, look at the actual file and classify each match on it.
    spans_cache: dict[str, dict[int, list[tuple[int, int]]] | None] = {}
    forensic: list[str] = []
    runtime: list[str] = []
    for fpath, lineno, raw in hits_raw:
        abs_path = ROOT / fpath
        if not abs_path.exists():
            runtime.append(f"{fpath}:{lineno} [missing-file] {raw}")
            continue
        if fpath not in spans_cache:
            spans_cache[fpath] = string_and_comment_spans(abs_path)
        kind = classify_hit(spans_cache[fpath], lineno, raw[1:].rstrip("\n"))
        line_str = f"{fpath}:{lineno} [{kind}] {raw}"
        if kind == "forensic":
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


if __name__ == "__main__":
    main()
