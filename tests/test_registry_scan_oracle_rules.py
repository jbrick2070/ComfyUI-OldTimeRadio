"""The scan replica's two 2026-10-30 rules fire on what the registry flags.

`tests/test_registry_scan_oracle_clean.py` asserts the tree is clean, which a
neutered rule also passes. These tests pin the rules themselves: the AST model
of the registry's pylint `subprocess-pip-install` check (the rule that cost
2.3.34 and 2.3.35 a `policy-v0.7: code-install` verdict, PBUG-20261030-01) and
the lexical `__import__(` rule (YARA `$import_func_direct`). The shapes come
from the two contrarian reviews of 68d0a1417: every miss and every false
positive they named is a case here, so the rule cannot drift back.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _oracle():
    spec = importlib.util.spec_from_file_location(
        "otr_registry_scan_oracle_rules", REPO / "scripts" / "otr_registry_scan_oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The engine exactly as 2.3.35 shipped it, trimmed to the spawn.
_OLD_ENGINE = '''
def _pip_install_nvidia_vfx():
    import sys
    from .._otr_shared import proc as otr_proc

    def attempt(argv):
        r = otr_proc.run(argv, capture_output=True, text=True)
        return r.returncode == 0, r.stdout

    ok, detail = attempt([sys.executable, "-m", "pip", "install", "--no-input",
                          "--disable-pip-version-check", "--only-binary", ":all:",
                          AUTO_INSTALL_REQUIREMENT])
    return ok, detail
'''

FIRES = {
    "the 2.3.35 engine argv": _OLD_ENGINE,
    "shell string, shell=True": 'subprocess.run("pip install foo", shell=True)\n',
    "os.system with concatenation": 'os.system("pip install " + pkg)\n',
    "os.system with an f-string": 'os.system(f"pip install {pkg}")\n',
    "shlex.split under run": 'subprocess.run(shlex.split("pip install x"))\n',
    "keyword argv": 'subprocess.run(args=[sys.executable, "-m", "pip", "install", "x"])\n',
    "argv assembled in a name": 'cmd = [sys.executable, "-m", "pip", "install", pkg]\notr_proc.run(cmd)\n',
    "argv grown with extend": 'cmd = [sys.executable, "-m"]\ncmd.extend(["pip", "install", pkg])\nrun(cmd)\n',
    "argv grown with +=": 'cmd = ["uv"]\ncmd += ["pip", "install", "x"]\nrun(cmd)\n',
    "list concatenation": 'run([sys.executable, "-m"] + ["pip", "install", "x"])\n',
    "starred prefix": 'run([*prefix, "pip", "install", "x"])\n',
    "uv pip": 'run([uv_exe, "pip", "install", "--python", sys.executable, req])\n',
    "pip3": 'subprocess.check_call(["pip3", "install", "x"])\n',
    "pip.exe": 'subprocess.check_call(["pip.exe", "install", "x"])\n',
    "options between pip and install": 'run(["pip", "--no-cache-dir", "--quiet", "install", "x"])\n',
    "shell text with an option": 'subprocess.call("pip --no-input install x", shell=True)\n',
    "pip token through a constant name": 'PIP = "pip"\nrun([sys.executable, "-m", PIP, "install", x])\n',
}

STAYS_QUIET = {
    "an error message that quotes the line": (
        'raise EpisodeLanguageError(\n'
        '    "install it with this ComfyUI Python: python -m pip install %r" % extra)\n'),
    "a hint in a dict": 'HINT = {"z": "pip install jieba pypinyin"}\n',
    "a print of the words": 'print(["pip", "install", "x"])\n',
    "a log line": 'log.warning("run pip install %s first", pkg)\n',
    "pip uninstall": 'run(["pip", "uninstall", "-y", "x"])\n',
    "uninstall before another pip word": 'run(["pip", "uninstall", "pip", "install"])\n',
    "a list that only mentions pip": 'run([sys.executable, "-m", "pip", "--version"])\n',
    "extend alone, never run": 'cmd.extend(["pip", "install", "x"])\n',
    "a non-spawn callee with shell text": 'fmt("please run pip install foo first")\n',
}


@pytest.mark.parametrize("shape", sorted(FIRES))
def test_a_runtime_pip_install_fires(shape):
    hits = _oracle()._pip_install_spawns("x.py", FIRES[shape])
    assert hits, shape
    assert all(h["rule"] == "$subprocess_pip_install" for h in hits)


@pytest.mark.parametrize("shape", sorted(STAYS_QUIET))
def test_a_mention_of_pip_stays_quiet(shape):
    assert _oracle()._pip_install_spawns("x.py", STAYS_QUIET[shape]) == [], shape


# `contains_custom_url_dependency`, from three packs' structured verdicts read
# 2026-10-31: comfyui-visage 1.0.0 was flagged on this finding ALONE.
URL_DEP_FIRES = {
    "git direct reference (comfyui-visage 1.0.0)": (
        "requirements.txt", "visage-image @ git+https://github.com/x/Visage-Image@main\n"),
    "a wheel URL (this pack's 2.3.45/2.3.46 spaCy line)": (
        "requirements.txt", 'en_core_web_sm @ https://github.com/explosion/spacy-models/'
        'releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
        ' ; python_version < "3.13"\n'),
    "an address in a COMMENT (comfyui-prompt-tag-editor 1.5.0)": (
        "requirements.txt", "numpy\n#   https://github.com/JamePeng/llama-cpp-python/releases\n"),
    "an index flag": ("requirements-gpu.txt", "--extra-index-url https://download.example/whl\n"),
    "a pyproject dependency (comfyui-soda-storyboard 0.1.1)": (
        "pyproject.toml", '[project]\nname = "x"\ndependencies = [\n'
        '    "soda @ https://github.com/x/soda/releases/download/v1/soda-1-py3-none-any.whl",\n]\n'),
}

URL_DEP_QUIET = {
    "plain names and markers": (
        "requirements.txt", 'nvidia-vfx>=0.2,<0.3; (sys_platform == "win32" and platform_machine == "AMD64")\n'
        "# spaCy publishes the model as a GitHub release wheel, not on PyPI\n"),
    "pyproject metadata addresses every pack carries (2.3.15-2.3.20 passed with them)": (
        "pyproject.toml", '[project]\nname = "x"\ndependencies = ["numpy>=1.24"]\n'
        '[project.urls]\nRepository = "https://github.com/x/y"\n'
        '[tool.comfy]\nIcon = "https://raw.githubusercontent.com/x/y/main/icon.png"\n'),
    "a URL in a node's code is another rule's business": ("nodes/x.py", 'URL = "https://example.com/a.whl"\n'),
}


@pytest.mark.parametrize("shape", sorted(URL_DEP_FIRES))
def test_a_custom_url_dependency_fires(shape):
    rel, raw = URL_DEP_FIRES[shape]
    hits = _oracle()._custom_url_dependencies(rel, raw)
    assert hits and all(h["rule"] == "$custom_url_dependency" for h in hits), shape


@pytest.mark.parametrize("shape", sorted(URL_DEP_QUIET))
def test_a_plain_dependency_stays_quiet(shape):
    rel, raw = URL_DEP_QUIET[shape]
    assert _oracle()._custom_url_dependencies(rel, raw) == [], shape


def test_the_shipped_dependency_files_carry_no_address():
    """The real files, through the same rule the clean test runs over the
    packed set -- stated here so a failure names the rule."""
    oracle = _oracle()
    for rel in ("requirements.txt", "pyproject.toml"):
        raw = (REPO / rel).read_text(encoding="utf-8")
        assert oracle._custom_url_dependencies(rel, raw) == [], rel


def test_the_old_engine_is_reported_at_its_spawn_line():
    (hit,) = _oracle()._pip_install_spawns("eng.py", _OLD_ENGINE)
    assert "attempt([sys.executable" in hit["text"]


def _lexical(oracle, source: str):
    scanned = oracle._blank_line_comments(source, ".py")
    return sorted({rule for line in scanned.splitlines()
                   for rule, pat, _label in oracle.RULES if pat.search(line)})


def test_the_october_rules_fire_on_the_flagged_shapes_only():
    """$importlib_direct and $pathlib2, from the registry's own verdicts:
    the shapes that flagged versions alone fire; the shapes that 60 packs
    carried through a passing scan since 2026-10-08 stay quiet."""
    oracle = _oracle()
    assert "$importlib_direct" in _lexical(oracle, 'm = importlib.import_module("nodes")\n')
    assert "$importlib_direct" in _lexical(oracle, "m = importlib.import_module('a.b', pkg)\n")
    for quiet in ('m = importlib.import_module(name)\n',
                  'm = importlib.import_module(f"misaki.{sub}")\n',
                  'm = importlib.import_module(".nodes." + modname, pkg)\n',
                  '# importlib.import_module("nodes") in a comment\n'):
        assert "$importlib_direct" not in _lexical(oracle, quiet), quiet
    assert "$pathlib2" in _lexical(oracle, 'text = Path(path).read_text()\n')
    assert "$pathlib2" in _lexical(oracle, 'text = Path(p).read_text(encoding="utf-8")\n')
    assert "$pathlib2" in _lexical(oracle, 'raw = Path(p).read_bytes()\n')
    for quiet in ('with open(p, encoding="utf-8") as fh:\n    text = fh.read()\n',
                  'text = p.read_text(encoding="utf-8")\n'):
        assert "$pathlib2" not in _lexical(oracle, quiet), quiet


def test_import_func_direct_fires_in_code_not_in_prose():
    oracle = _oracle()
    assert "$import_func_direct" in _lexical(oracle, 'exe = __import__("sys").executable\n')
    assert _lexical(oracle, '# __import__("sys") in a comment\nx = "__import__(\'sys\')"\n') == []
    assert _lexical(oracle, "import importlib\nmod = importlib.import_module(name)\n") == []


def _rules_in(oracle, source: str, ext: str):
    """The rule ids that fire on `source` read as a file with extension `ext`."""
    scanned = oracle._blank_line_comments(source, ext)
    return sorted({rule for line in scanned.splitlines()
                   for rule, pat, _label in oracle.RULES if pat.search(line)})


def test_the_socket_rules_read_browser_calls_as_sockets():
    """From comfyui-logblaster 1.1.1's structured verdict (2026-10-31): a dotted
    bind call in a .js file matched `$socket4`, and Web Audio wiring matched
    `$socket3`. The respelling LogBlaster 1.1.2 shipped stays quiet."""
    oracle = _oracle()
    assert "$socket4" in _rules_in(oracle, "const orig = api.queuePrompt.bind(api);\n", ".js")
    assert "$socket4" in _lexical(oracle, "server.bind((host, port))\n")
    assert "$socket3" in _rules_in(oracle, "synthMasterGain.connect(analyser);\n", ".js")
    for quiet, ext in (("// api.queuePrompt.bind(api) in a comment\n", ".js"),
                       ("# sock.bind(addr) in a comment\n", ".py"),
                       ("address = bind_address(host)\n", ".py"),
                       ("const orig = api.queuePrompt;\nawait orig.apply(api, args);\n", ".js"),
                       ("const wireAudio = (a, b) => AudioNode.prototype.connect.call(a, b);\n", ".js")):
        found = _rules_in(oracle, quiet, ext)
        assert "$socket3" not in found and "$socket4" not in found, quiet


def test_html_pages_are_scanned_and_read_raw(tmp_path):
    """LogBlaster 1.1.1 was flagged on lines inside web_app/index.html, so the
    replica scans .html and .htm. Whether the scanner strips HTML comments is
    unknown, so a page is read raw: a literal in a comment still counts."""
    import subprocess

    oracle = _oracle()
    # Line 2 is code, line 3 a script comment, line 5 a page comment: read raw,
    # all three count.
    page = ("<script>\n  gain.connect(ctx.destination);\n  // osc.connect(gain) was the old wiring\n"
            "</script>\n<!-- node.connect(other) -->\n")
    # A small packed set of its own, run through the replica's real scan():
    # both pages are read, raw, and a file type the scanner skips is not.
    repo = tmp_path
    (repo / "web_app").mkdir()
    (repo / "web_app" / "index.html").write_text(page, encoding="utf-8")
    (repo / "web_app" / "old.htm").write_text(page, encoding="utf-8")
    (repo / "theme.css").write_text("a.connect(b);\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "web_app/index.html", "web_app/old.htm", "theme.css"], cwd=repo, check=True)
    found = sorted((f["file"], f["line"]) for f in oracle.scan(repo=repo) if f["rule"] == "$socket3")
    assert found == [("web_app/index.html", 2), ("web_app/index.html", 3), ("web_app/index.html", 5),
                     ("web_app/old.htm", 2), ("web_app/old.htm", 3), ("web_app/old.htm", 5)]
