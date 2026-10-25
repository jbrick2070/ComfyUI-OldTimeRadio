"""Known platform failures -> a one-line hint that points at an OTR tool.

WHY THIS EXISTS. `describe_execution_error` already names the node, the
exception and the traceback tail, and OTR's own refusals already say what to do.
The failures that do NOT are the platform ones: the GPU ran out of memory, a
model file ComfyUI does not list, a half-downloaded weights file, a module the
venv does not have, CUDA unusable, a VAE or LoRA from another model family, the
server gone. The raw message is accurate and tells a person nothing about which
script fixes it. (Idea taken from CreatorOS-Studio's error-hint table; see
planning/creatoros-concepts/PLAN.md row 1.2.)

HOW IT IS USED. `hint_for(text)` is pure: text in, one line or None out. Rows
are tried in order and the FIRST match wins, so the narrow rows come first.
`scripts/otr_api.py` appends the result to `describe_execution_error` and to the
`node_errors` refusal in `submit_prompt`.

WHAT IT WILL NOT DO. It never changes a result, never gates or retries a render
and never invents a reference: `refs` are Bug Bible ids (BUG_BIBLE.yaml in the
survival-guide repo) or PBUG ids (apple/PROD_BUG_LOG.md) that
tests/test_otr_known_failures.py proves exist. A failure class with no entry in
either has an empty `refs`, not a plausible-looking id. Text that matches no row
gets no hint, and the original message is reported exactly as before.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class KnownFailure(NamedTuple):
    """One recognisable failure class."""

    key: str
    pattern: "re.Pattern[str]"
    hint: str
    refs: tuple[str, ...]


def _rx(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern, re.IGNORECASE)


#: Order matters: first match wins. A dead server is tried first because its
#: message can also carry words the later rows look for.
KNOWN_FAILURES: tuple[KnownFailure, ...] = (
    KnownFailure(
        key="server_gone",
        pattern=_rx(
            r"SERVER-GONE"
            r"|ConnectionRefusedError"
            r"|Max retries exceeded with url"
            r"|Failed to establish a new connection"
        ),
        hint=(
            "the ComfyUI server is not answering: read the watchdog verdict "
            "(scripts/otr_render_watchdog.ps1) and the server log before the leg "
            "log, then reset the box per CLAUDE.md section 4 before relaunching"
        ),
        refs=(),
    ),
    KnownFailure(
        key="out_of_memory",
        pattern=_rx(
            r"OutOfMemoryError"
            r"|CUDA out of memory"
            r"|CUDA error: out of memory"
            r"|Allocation on device"
            r"|not enough memory"
        ),
        hint=(
            "the GPU ran out of memory: close other GPU programs, or run the "
            "smaller-tier workflow JSON for this card (workflows/otr_8gb_*.json "
            "or workflows/otr_16gb_*.json)"
        ),
        refs=("07.01", "07.03"),
    ),
    KnownFailure(
        key="value_not_in_list",
        pattern=_rx(r"value_not_in_list|Value not in list"),
        hint=(
            "ComfyUI does not list that option or model file: run "
            "scripts/otr_asset_index.py to see which lane needs it, install it "
            "with scripts/otr_fetch_lane_weights.py or scripts/otr_provision.py, "
            "then restart ComfyUI so it re-scans its folders"
        ),
        refs=("03.04", "04.08"),
    ),
    KnownFailure(
        key="truncated_weights",
        pattern=_rx(
            r"Error while deserializing header"
            r"|MetadataIncompleteBuffer"
            r"|HeaderTooLarge"
            r"|InvalidHeaderDeserialization"
            r"|PytorchStreamReader failed reading zip archive"
            r"|failed finding central directory"
            r"|invalid load key"
        ),
        hint=(
            "a model file is incomplete or corrupt, usually an interrupted "
            "download: delete that file and fetch it again with "
            "scripts/otr_fetch_lane_weights.py"
        ),
        refs=(),
    ),
    KnownFailure(
        key="cuda_device_assert",
        pattern=_rx(r"device-side assert"),
        hint=(
            "a CUDA device-side assert is usually one model, card and "
            "quantization combination failing at inference, not a broken GPU: "
            "try another quantization of the same model before changing drivers"
        ),
        refs=("06.06",),
    ),
    KnownFailure(
        key="cuda_unavailable",
        pattern=_rx(
            r"Torch not compiled with CUDA enabled"
            r"|no CUDA GPUs are available"
            r"|CUDA driver version is insufficient"
            r"|CUDA initialization"
            r"|No CUDA-capable device"
        ),
        hint=(
            "CUDA is not usable from the Python that runs ComfyUI: check "
            "nvidia-smi, then run scripts/otr_venv_audit.py with that same "
            "interpreter to see which torch build it has"
        ),
        refs=(),
    ),
    KnownFailure(
        key="missing_module",
        pattern=_rx(r"ModuleNotFoundError|No module named"),
        hint=(
            "a Python module is missing from the interpreter that runs ComfyUI: "
            "run scripts/otr_venv_audit.py with that interpreter, and install "
            "with that interpreter's own `-m pip`, not the system pip"
        ),
        refs=("02.01",),
    ),
    KnownFailure(
        key="model_family_mismatch",
        pattern=_rx(
            r"Error\(s\) in loading state_dict"
            r"|size mismatch for"
            r"|mat1 and mat2 shapes cannot be multiplied"
            r"|Given groups=\d+, weight of size"
        ),
        hint=(
            "a VAE, LoRA or checkpoint from a different model family than the "
            "graph expects: compare the loader widgets with the files the lane "
            "needs (scripts/otr_asset_index.py)"
        ),
        refs=(),
    ),
)


def match_for(text: object) -> KnownFailure | None:
    """The first known failure whose pattern is found in ``text``, or None."""
    if text is None:
        return None
    haystack = text if isinstance(text, str) else str(text)
    for row in KNOWN_FAILURES:
        if row.pattern.search(haystack):
            return row
    return None


def hint_for(text: object) -> str | None:
    """One line telling a person what to do about ``text``, or None.

    The line carries the Bug Bible / PBUG ids as ``(see 07.01, 07.03)`` when the
    row has any, so a reader can open the full entry.
    """
    row = match_for(text)
    if row is None:
        return None
    if row.refs:
        return "%s (see %s)" % (row.hint, ", ".join(row.refs))
    return row.hint
