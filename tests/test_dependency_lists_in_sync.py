"""``pyproject.toml`` and ``requirements.txt`` must declare the SAME dependencies.

THE TWO LISTS ARE MAINTAINED BY HAND AND CANNOT BE DERIVED FROM EACH OTHER,
because they are read by two different consumers for two different purposes:

* **The Comfy Registry reads ``pyproject.toml``'s literal static list.** It does
  NOT evaluate setuptools' ``dynamic = ["dependencies"]``. Proven on
  2.0.0-alpha.3, which published with ``dependencies: []`` recorded -- so a
  registry install got the pack's CODE and none of its LIBRARIES.
* **Comfy-Org's ``node-pack-extract`` installs from ``requirements.txt``**, in a
  headless Linux container under ``set -e``, and reads ``/object_info`` to
  publish the node list the registry card renders as "N Nodes".

So a line present in only one of them fails in a way the other cannot reveal,
and both failure modes are SILENT: alpha.14 shipped kokoro and pyloudnorm fixed
in code and unreachable from a registry install, because only requirements.txt
carried them. A fresh install reached its first spoken line and died.

This test is the guard that was missing. It is cheap, it is exact, and it turns
a hand-sync into a checked one. ``tests/`` is stripped by ``.comfyignore``, so
nothing here ships.

NOT ASSERTED HERE, deliberately: that either list is CORRECT. Only that they
AGREE. What the pack actually imports is a different question with a different
owner (``scripts/otr_venv_audit.py``).
"""
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _normalize(spec: str) -> str:
    """One comparable form: collapsed whitespace, one quote style.

    Environment markers are part of the requirement and are COMPARED, not
    stripped -- ``bitsandbytes`` without its ``sys_platform != 'darwin'``
    marker blocked the entire macOS registry install (2026-09-01 ship audit),
    and a marker that drifts between the two files is exactly that defect
    reappearing on one consumer and not the other.
    """
    return re.sub(r"\s+", " ", spec.replace('"', "'")).strip()


def _pyproject_deps() -> set:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {_normalize(d) for d in data["project"]["dependencies"]}


def _requirements_deps() -> set:
    out = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(_normalize(line))
    return out


def test_every_requirements_line_is_in_pyproject():
    missing = sorted(_requirements_deps() - _pyproject_deps())
    assert not missing, (
        "declared in requirements.txt but NOT in pyproject.toml, so a Comfy "
        "Registry install receives the pack without them (the alpha.14 cold-"
        "install failure): %s" % missing)


def test_every_pyproject_line_is_in_requirements():
    missing = sorted(_pyproject_deps() - _requirements_deps())
    assert not missing, (
        "declared in pyproject.toml but NOT in requirements.txt, so Comfy-Org's "
        "node-pack-extract container installs without them and the published "
        "node list can come back empty: %s" % missing)


def test_neither_list_is_empty():
    """The alpha.3 shape: a list that PARSES and is empty publishes happily."""
    assert _pyproject_deps(), "pyproject declares no dependencies (the alpha.3 defect)"
    assert _requirements_deps(), "requirements.txt declares no dependencies"


def test_torch_is_declared_nowhere():
    """ComfyUI manages its own torch; declaring it can fight the host install."""
    for name, deps in (("pyproject.toml", _pyproject_deps()),
                       ("requirements.txt", _requirements_deps())):
        offenders = [d for d in deps if re.match(r"^torch(\W|$)", d)]
        assert not offenders, "%s declares torch: %s" % (name, offenders)


@pytest.mark.parametrize("package", ["kokoro", "kokoro-onnx", "pycairo", "bitsandbytes"])
def test_marker_bearing_lines_carry_the_same_marker_in_both_files(package):
    """The four lines whose ENVIRONMENT MARKER is the whole point.

    Each of these has already cost a real install when its marker was absent or
    differed: kokoro/kokoro-onnx are complementary by Python version (a bare
    kokoro line made `pip install -r requirements.txt` install NONE of the 18
    packages on Python 3.13), pycairo is win32-only because there are zero Linux
    wheels and the sdist build killed the extractor's container boot, and
    bitsandbytes without its marker blocked macOS entirely.
    """
    def pick(deps):
        return {d for d in deps if re.match(r"^%s(\W|$)" % re.escape(package), d)}
    pj, rq = pick(_pyproject_deps()), pick(_requirements_deps())
    assert pj, "%s missing from pyproject.toml" % package
    assert rq, "%s missing from requirements.txt" % package
    assert pj == rq, ("%s declares different specs/markers in the two files, so "
                      "the registry and the extractor would resolve it "
                      "differently: pyproject=%s requirements=%s"
                      % (package, sorted(pj), sorted(rq)))
