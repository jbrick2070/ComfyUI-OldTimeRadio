"""The GGUF lane's version pin must survive, in the two places users meet it.

PBUG-20260829-12. llama-cpp-python 0.3.35 dies with STATUS_ILLEGAL_INSTRUCTION
inside llama_init_from_model, reproduced at n_gpu_layers=0 -- so the fault is
in the CPU backend and no GPU avoids it. 0.3.33 loads and generates, and the
two builds were confirmed byte-identical across two machines by SHA-256.

An unpinned `pip install llama-cpp-python` resolves to the broken one, so a
fresh install of the GGUF lane is broken by default. The dependency is
deliberately NOT in requirements.txt -- it is an opt-in lane and the CUDA wheel
is ~945 MB -- which makes the error message and the README the only two places
the pin can live.
"""
from __future__ import annotations

import inspect
import pathlib

from nodes import _otr_gguf_backend as ggf

GOOD, BAD = "0.3.33", "0.3.35"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_import_error_names_the_working_version():
    src = inspect.getsource(ggf._import_llama_cpp)
    assert GOOD in src, (
        "the failure a user actually hits does not name the version that "
        "works -- they will pip install the latest and get the broken one")
    assert BAD in src, (
        "the error does not warn against %s by name; 'install llama-cpp-python' "
        "resolves to it" % BAD)


def test_the_readme_documents_the_pin():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "llama-cpp-python==%s" % GOOD in readme, (
        "README lost the pinned install command")
    assert BAD in readme, "README does not warn against the broken version"


def test_the_pin_is_not_silently_added_to_requirements():
    """It is opt-in on purpose: a ~945 MB CUDA wheel for a lane most users
    never select. If this ever changes it should be a deliberate decision,
    not a drive-by -- and pyproject edits auto-fire a registry publish."""
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "llama" not in reqs.lower(), (
        "llama-cpp-python appeared in requirements.txt -- that forces a very "
        "large optional wheel on every installer; if intended, delete this "
        "test in the same commit and say why")


def test_the_bare_import_trap_is_documented():
    """A bare `import llama_cpp` fails on a WORKING install, because OTR
    preloads CUDA DLLs first. Both boxes lost time to this."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "_import_llama_cpp" in readme, (
        "README does not tell users to test through OTR's own import path, so "
        "they will diagnose a working install as broken")
