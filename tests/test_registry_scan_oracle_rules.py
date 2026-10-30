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


def test_the_old_engine_is_reported_at_its_spawn_line():
    (hit,) = _oracle()._pip_install_spawns("eng.py", _OLD_ENGINE)
    assert "attempt([sys.executable" in hit["text"]


def _lexical(oracle, source: str):
    scanned = oracle._blank_line_comments(source, ".py")
    return sorted({rule for line in scanned.splitlines()
                   for rule, pat, _label in oracle.RULES if pat.search(line)})


def test_import_func_direct_fires_in_code_not_in_prose():
    oracle = _oracle()
    assert "$import_func_direct" in _lexical(oracle, 'exe = __import__("sys").executable\n')
    assert _lexical(oracle, '# __import__("sys") in a comment\nx = "__import__(\'sys\')"\n') == []
    assert _lexical(oracle, "import importlib\nmod = importlib.import_module(name)\n") == []
