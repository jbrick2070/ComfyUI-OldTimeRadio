"""Local replica of the Comfy Registry's lexical YARA scan -- our own oracle.

WHY THIS EXISTS. The registry promotes a version to Active iff its private
scanner returns an empty body, and there is NO way to run that scanner locally
(``comfy node validate`` runs Ruff S102/S307/E702 only). But the scan is a
lexical string match over the shipped ``node.zip``, and the registry leaks the
exact matched pattern per finding, so we can replicate it precisely from the
outside and iterate to zero findings WITHOUT burning version strings.

Calibrated against measured verdicts (docs/2026-09-05-scanner-research/):
* it scans the PACKED archive -- git-tracked minus ``.comfyignore`` -- not the
  repo tree;
* it is spelling-sensitive: it keys on the literal API text;
* comments are stripped in CODE files (.py/.js) but NOT in data files, where
  prose that quotes a literal still flags;
* one finding of any severity flags the whole version, so this gate is
  ALL-OR-NOTHING and fails closed.

The patterns below are the twelve identifiers the registry reported against
OTR alpha.21, plus the conservative supersets artokun's public replica proved
necessary (e.g. reads count on the socket rule). This is a MODEL of a private
rule, so it errs toward over-reporting: a hit here is a line to fix, and a clean
run here is necessary but not a guarantee.

Usage:
    python scripts/otr_registry_scan_oracle.py            # scan the packed set
    python scripts/otr_registry_scan_oracle.py --json     # machine-readable
    python scripts/otr_registry_scan_oracle.py --all-tracked  # ignore .comfyignore
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess  # noqa: S404 -- this tool is DEV-ONLY and .comfyignore'd; it never ships
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: rule id -> (compiled pattern, human label). Ordered by the registry's own
#: rule families. Each pattern is the measured literal or its conservative
#: superset; see the synthesis doc for the evidence behind each one.
RULES = [
    # --- python_environment_manipulation ---
    ("$env_read/mod", re.compile(r"os\.environ\b"), "os.environ[ / .get( / [x]= / .pop("),
    ("$env_read3", re.compile(r"os\.getenv\s*\("), "os.getenv("),
    # --- python_network_operations ---
    ("$http1", re.compile(r"requests\.(?:get|post|put|patch|delete|head)\s*\("), "requests.<method>("),
    ("$http2", re.compile(r"urllib\.request\.urlopen\s*\("), "urllib.request.urlopen("),
    ("$http5", re.compile(r"aiohttp\s*\.\s*ClientSession"), "aiohttp.ClientSession"),
    ("$socket1", re.compile(r"socket\.socket\s*\("), "socket.socket("),
    ("$socket2", re.compile(r"socket\.create_connection\s*\("), "socket.create_connection("),
    ("$socket3", re.compile(r"\.connect\s*\("), ".connect("),
    # --- python_command_injection_risk ---
    ("$subprocess_direct",
     re.compile(r"subprocess\.(?:Popen|run|call|check_output|check_call)\s*\(|os\.system\s*\("),
     "subprocess.<spawn>( / os.system("),
]

_CODE_EXT = {".py", ".pyw", ".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"}
_DATA_EXT = {".json", ".md", ".txt", ".yaml", ".yml", ".cfg", ".ini", ".toml"}


def _shipped_files(all_tracked: bool):
    """The files a publish would pack: git-tracked minus ``.comfyignore``."""
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    tracked = [p for p in tracked if p]
    if all_tracked:
        return tracked
    try:
        import pathspec  # type: ignore
    except ImportError:
        print("pathspec not installed; scanning ALL tracked files", file=sys.stderr)
        return tracked
    ci = REPO / ".comfyignore"
    if not ci.exists():
        return tracked
    spec = pathspec.PathSpec.from_lines("gitwildmatch", ci.read_text(encoding="utf-8").splitlines())
    return [p for p in tracked if not spec.match_file(p)]


def _strip_python(text: str) -> str:
    """Blank COMMENT and STRING tokens in .py, preserving line count.

    CALIBRATED: the registry did NOT flag files where ``os.environ`` appears only
    in a docstring or a comment (hf_env.py, _otr_determinism.py, eng_ltx_*.py),
    but DID flag files where it appears in executable code (env.py, prestartup).
    So the scanner excludes comments AND string literals in code. Python's own
    tokenizer draws that line exactly, so we blank COMMENT and STRING tokens and
    keep the code, mapping each blanked token back onto its own lines.
    """
    import io
    import tokenize
    # f-strings tokenize as FSTRING_* on 3.12+, not STRING; blank those too, or
    # `os.environ` inside an f-string message reads as a false finding (the real
    # scanner treats f-string text as a string literal and does not flag it).
    _BLANK = {tokenize.COMMENT, tokenize.STRING}
    for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        _t = getattr(tokenize, _name, None)
        if _t is not None:
            _BLANK.add(_t)
    lines = text.splitlines()
    blanked = list(lines)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text  # fail toward over-reporting on unparseable source
    for tok in toks:
        if tok.type not in _BLANK:
            continue
        (sr, sc), (er, ec) = tok.start, tok.end
        for row in range(sr, er + 1):
            i = row - 1
            if i < 0 or i >= len(blanked):
                continue
            line = blanked[i]
            a = sc if row == sr else 0
            b = ec if row == er else len(line)
            blanked[i] = line[:a] + (" " * (b - a)) + line[b:]
    return "\n".join(blanked)


def _blank_line_comments(text: str, ext: str) -> str:
    """Blank comments/strings in CODE files, preserving offsets; DATA files raw.

    A literal that lives only in a code comment or string does not flag; one in
    a data file's prose does (measured: a URL in a requirements.txt comment
    flagged comfyui-vosr2). Python uses the real tokenizer; other code files get
    a simpler ``//`` line-comment blank (string-aware for the comment start).
    """
    if ext not in _CODE_EXT:
        return text
    if ext in {".py", ".pyw"}:
        return _strip_python(text)
    out = []
    for line in text.splitlines():
        q = None
        i = 0
        cut = None
        while i < len(line):
            c = line[i]
            if q:
                if c == "\\":
                    i += 2
                    continue
                if c == q:
                    q = None
            elif c in "\"'":
                q = c
            elif c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def scan(all_tracked: bool = False):
    findings = []
    for rel in _shipped_files(all_tracked):
        p = REPO / rel
        ext = p.suffix.lower()
        if ext not in _CODE_EXT and ext not in _DATA_EXT:
            continue
        try:
            raw = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned = _blank_line_comments(raw, ext)
        lines = scanned.splitlines()
        raw_lines = raw.splitlines()
        for n, line in enumerate(lines, 1):
            for rule_id, pat, label in RULES:
                if pat.search(line):
                    findings.append({
                        "file": rel, "line": n, "rule": rule_id, "label": label,
                        "text": (raw_lines[n - 1].strip() if n <= len(raw_lines) else "")[:100],
                    })
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all-tracked", action="store_true",
                    help="ignore .comfyignore (scan every tracked file)")
    args = ap.parse_args()
    findings = scan(all_tracked=args.all_tracked)
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        by_rule = {}
        for f in findings:
            by_rule.setdefault(f["rule"], 0)
            by_rule[f["rule"]] += 1
        for f in findings:
            print("%-46s :%-4d  %-20s %s" % (f["file"], f["line"], f["rule"], f["text"]))
        print("\n%d finding(s) in the packed set; by rule: %s"
              % (len(findings), dict(sorted(by_rule.items()))))
        print("REGISTRY VERDICT MODEL:",
              "FLAGGED" if findings else "clean (necessary, not a guarantee)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
