"""Local replica of the Comfy Registry's lexical YARA scan -- our own oracle.

WHY THIS EXISTS. The registry promotes a version to Active iff its private
scanner returns an empty body, and there is NO way to run that scanner locally
(``comfy node validate`` runs Ruff S102/S307/E702 only). But the scan is a
lexical string match over the shipped ``node.zip``, and the registry leaks the
exact matched pattern per finding, so we can replicate it precisely from the
outside and iterate to zero findings WITHOUT burning version strings.

Calibrated against measured verdicts from the 2026-09-05 scanner research:
* it scans the PACKED archive -- git-tracked minus ``.comfyignore`` -- not the
  repo tree;
* it is spelling-sensitive: it keys on the literal API text;
* comments are stripped in CODE files (.py/.js) but NOT in data files, where
  prose that quotes a literal still flags;
* one finding of any severity flags the whole version, so this gate is
  ALL-OR-NOTHING and fails closed.

The patterns below are the twelve identifiers the registry reported against
OTR alpha.21, plus the conservative supersets artokun's public replica proved
necessary (e.g. reads count on the socket rule). Two more joined on 2026-10-30,
read off the structured verdicts the registry now writes for other packs:
``__import__(`` (YARA ``$import_func_direct``, python_bytecode_manipulation)
and a run-time ``pip install`` spawned from Python -- the registry's second,
pylint-based pass reports that as ``subprocess-pip-install`` (severity high),
and because it is an AST check it is modelled below as one, on raw source.
That second rule is the one that earned OTR 2.3.34 and 2.3.35 a
``policy-v0.7: code-install`` verdict (PBUG-20261030-01). This is a MODEL of
private rules, so it errs toward over-reporting: a hit here is a line to fix,
and a clean run here is necessary but not a guarantee.

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
    # --- python_bytecode_manipulation (import evasion) ---
    # Read 2026-10-30 off the registry's structured verdicts (rgthree-comfy,
    # ComfyUI-DistorchMemoryManager): `__import__(` matches $import_func_direct.
    ("$import_func_direct", re.compile(r"__import__\s*\("), "__import__("),
]

#: ``pip install`` as shell text: a pip token, any options, then ``install``.
_PIP_SHELL_RE = re.compile(r"\bpip\d*(?:\.\d+)*(?:\.exe)?\s+(?:-[^\s]+\s+)*install\b")
#: ``pip``, ``pip3``, ``pip3.12``, ``pip.exe`` as one argv token.
_PIP_TOKEN_RE = re.compile(r"^pip\d*(?:\.\d+)*(?:\.exe)?$", re.IGNORECASE)
#: Callees that run what they are handed, by name, whatever module owns them.
_SPAWN_CALLEES = {"run", "Popen", "call", "check_call", "check_output", "system",
                  "popen", "spawn", "exec"}
#: Callees handed a list literal without ever running it. A list that grows
#: through ``extend``/``append``/``insert`` is followed into the NAME instead.
_INERT_CALLEES = {"print", "extend", "append", "insert", "join", "format", "debug",
                  "info", "warning", "error", "exception", "log", "len", "sorted",
                  "set", "list", "tuple", "repr", "str"}


def _argv_spells_pip_install(words) -> bool:
    """True when the FIRST pip token is followed, options allowed between, by
    ``install``. ``["pip", "uninstall", "x"]`` is not an install;
    ``["pip", "--no-cache-dir", "install", "x"]`` is."""
    for i, word in enumerate(words):
        if not _PIP_TOKEN_RE.match(word):
            continue
        for later in words[i + 1:]:
            if later.startswith("-"):
                continue
            return later == "install"
        return False
    return False


def _literal_words(node, names) -> list:
    """The string constants an argv expression carries, in order: a list or
    tuple literal (a starred element is followed), a ``+`` of such, or a NAME
    assigned or grown from them anywhere in the same file."""
    import ast
    if isinstance(node, (ast.List, ast.Tuple)):
        out = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            elif isinstance(elt, ast.Starred):
                out.extend(_literal_words(elt.value, names))
            elif isinstance(elt, ast.Name):
                # `PIP = "pip"; run([..., PIP, "install", x])` (Sonnet, 2026-10-30)
                out.extend(names.get(elt.id, ()))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_words(node.left, names) + _literal_words(node.right, names)
    if isinstance(node, ast.Name):
        return list(names.get(node.id, ()))
    return []


def _collect_argv_names(tree) -> dict:
    """NAME -> the string constants its list carries. Scope-blind on purpose
    (one map per file): ``cmd = [...]``, ``cmd += [...]``, ``cmd.extend([...])``
    and ``cmd.append("x")`` all feed the name, so a spawn handed ``cmd`` is
    judged on what ``cmd`` was built from. A reader that must guess guesses
    toward reporting."""
    import ast
    names: dict = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                names[node.targets[0].id] = [node.value.value]
                continue
            words = _literal_words(node.value, names)
            if words:
                names[node.targets[0].id] = words
        elif (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                and isinstance(node.op, ast.Add)):
            names[node.target.id] = (list(names.get(node.target.id, ()))
                                     + _literal_words(node.value, names))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.attr in ("extend", "append", "insert")):
            added = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    added.append(arg.value)
                else:
                    added.extend(_literal_words(arg, names))
            if added:
                target = node.func.value.id
                names[target] = list(names.get(target, ())) + added
    return names


def _pip_install_spawns(rel: str, raw: str) -> list:
    """The registry's pylint-scanner rule ``subprocess-pip-install``, modelled.

    An AST check, not a lexical one. Two shapes fire:

    * ARGV: a call -- any callee except the inert ones above, because this pack
      spawns through its own boundary (``otr_proc.run``) and the reviewer read
      through it -- handed, positionally or by keyword, a list whose first pip
      token is followed by ``install`` (options allowed between). The list may
      be a literal, a ``+`` of literals, a starred splice, or a NAME the file
      assigned or grew from literals (``cmd = [...]``, ``cmd.extend([...])``);
      a NAME inside the list that the file assigned a string resolves to it
      (``PIP = "pip"``).
    * SHELL TEXT: a spawn-named callee (run, Popen, call, check_call,
      check_output, system, popen, spawn, exec) with ``pip ... install`` in any
      string constant anywhere in its arguments -- a plain string, an f-string,
      ``"pip install " + pkg``, or ``shlex.split("pip install x")``.

    An error message that merely quotes a pip line is neither shape. What is
    still over-reported, on purpose: a helper that builds a hint from a pip
    argv (``fmt(["pip", "install", x])``) fires, because the reader cannot tell
    a hint builder from a spawn wrapper -- a hit here is a line to look at.
    Measured 2026-10-30 against the shapes two reviewers (Sonnet, Codex)
    listed as misses; `tests/test_registry_scan_oracle_rules.py` pins them.
    """
    import ast
    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return []
    names = _collect_argv_names(tree)
    raw_lines = raw.splitlines()
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        callee = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if callee in _INERT_CALLEES:
            continue
        values = list(node.args) + [kw.value for kw in node.keywords]
        hit = any(_argv_spells_pip_install(_literal_words(v, names)) for v in values)
        if not hit and callee in _SPAWN_CALLEES:
            hit = any(
                isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                and _PIP_SHELL_RE.search(sub.value)
                for v in values for sub in ast.walk(v))
        if hit:
            out.append({
                "file": rel, "line": node.lineno, "rule": "$subprocess_pip_install",
                "label": "a pip install spawned at run time (pylint subprocess-pip-install)",
                "text": (raw_lines[node.lineno - 1].strip()
                         if node.lineno <= len(raw_lines) else "")[:100],
            })
    return out

#: The registry's YARA rule `contains_custom_url_dependency` ("Detects custom
#: wheel or URL dependencies", strings `$plaintext_url` and `$whl_string`),
#: read 2026-10-31 off three packs' structured verdicts. It matches TEXT, so a
#: web address in a requirements comment counts (comfyui-prompt-tag-editor
#: 1.5.0), and that one finding alone flagged comfyui-visage 1.0.0. In
#: pyproject.toml it fired on a dependency line (comfyui-soda-storyboard);
#: the [project.urls] and [tool.comfy] Icon addresses every pack carries did
#: not flag this pack's 2.3.15-2.3.20, so only the dependency arrays count there.
_URL_DEP_RE = re.compile(r"://|\.whl\b|\bgit\+", re.IGNORECASE)
_REQUIREMENTS_NAME_RE = re.compile(r"(^|/)requirements[^/]*\.txt$", re.IGNORECASE)


def _custom_url_dependencies(rel: str, raw: str) -> list:
    """`contains_custom_url_dependency`, modelled: any address or wheel file
    name in a requirements file, comments included; in pyproject.toml, one
    inside a `dependencies` or `optional-dependencies` array."""
    name = rel.replace("\\", "/")
    out = []
    if _REQUIREMENTS_NAME_RE.search(name):
        for n, line in enumerate(raw.splitlines(), 1):
            if _URL_DEP_RE.search(line):
                out.append({"file": rel, "line": n, "rule": "$custom_url_dependency",
                            "label": "a web address or wheel in a requirements file "
                                     "(YARA contains_custom_url_dependency)",
                            "text": line.strip()[:100]})
    elif name.lower().endswith("pyproject.toml"):
        import tomllib
        try:
            project = tomllib.loads(raw).get("project", {})
        except tomllib.TOMLDecodeError:
            return out
        specs = list(project.get("dependencies") or [])
        for group in (project.get("optional-dependencies") or {}).values():
            specs.extend(group or [])
        raw_lines = raw.splitlines()
        for spec in specs:
            if not isinstance(spec, str) or not _URL_DEP_RE.search(spec):
                continue
            n = next((i for i, line in enumerate(raw_lines, 1) if spec in line), 0)
            out.append({"file": rel, "line": n, "rule": "$custom_url_dependency",
                        "label": "a web address or wheel in a pyproject dependency "
                                 "(YARA contains_custom_url_dependency)",
                        "text": spec[:100]})
    return out


_CODE_EXT = {".py", ".pyw", ".js", ".mjs", ".cjs", ".ts", ".mts", ".cts"}
_DATA_EXT = {".json", ".md", ".txt", ".yaml", ".yml", ".cfg", ".ini", ".toml"}


def _shipped_files(all_tracked: bool):
    """The files a publish would pack: git-tracked minus ``.comfyignore``."""
    # -z: without it git quotes a non-ASCII name ("caf\303\251.mp4"), which then
    # matches no .comfyignore line and no path prefix a caller checks.
    tracked = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.split("\0")
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
        if ext in {".py", ".pyw"}:
            findings.extend(_pip_install_spawns(rel, raw))
        findings.extend(_custom_url_dependencies(rel, raw))
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
