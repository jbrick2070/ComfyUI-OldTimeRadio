"""The GGUF lane's version pin, kept where it can still be reached.

PBUG-20260829-12. llama-cpp-python 0.3.35 dies with STATUS_ILLEGAL_INSTRUCTION
inside llama_init_from_model, reproduced at n_gpu_layers=0 -- so the fault is
in the CPU backend and no GPU avoids it. 0.3.33 loads and generates, and the
two builds were confirmed byte-identical across two machines by SHA-256.

THE LANE NO LONGER SHIPS A WRITER ROW (operator directive 2026-09-06,
hardened 2026-09-17). `GGUF_ROWS = ()`, the catalog does not inject a
peer, and `validate_model_id` rejects `*-GGUF` / `*.gguf` writer ids.
No *-GGUF model reaches the writer dropdown.

So the two README assertions that used to live here are GONE. bd106b06
deleted that README section on purpose, and re-adding it would document an
install path to a lane the pack does not offer -- worse than silent, because it
reads as a supported route. Do not restore them, and do not revive the lane as
an installer-ergonomics fix; what would have to change first is that the user
CHOOSES the weights.

What remains is the pair that still guards something real: the error message a
reader of the code meets, and the promise that a ~945 MB wheel never quietly
becomes a hard dependency.
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


def test_the_pin_is_not_silently_added_to_requirements():
    """It is opt-in on purpose: a ~945 MB CUDA wheel for a lane most users
    never select. If this ever changes it should be a deliberate decision,
    not a drive-by -- and pyproject edits auto-fire a registry publish."""
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "llama" not in reqs.lower(), (
        "llama-cpp-python appeared in requirements.txt -- that forces a very "
        "large optional wheel on every installer; if intended, delete this "
        "test in the same commit and say why")
