"""The writer dropdowns offer BARE model ids (operator, 2026-10-14).

Until then each label carried a size/fit badge ("Qwen/Qwen3.5-4B (8.7 GB
download, mac16-tight nv8-nf4 nv16 nv24)"), so correcting a size changed the
SAVED value and orphaned every saved graph. Now the label is the id, the size
and fit sit in the tooltip, and an older graph's badged value keeps working
three ways: the writer's VALIDATE_INPUTS admits it, js/workflow_schema_core.js
rewrites it to the bare option on load, and the run path strips it as before.

The published filename's writer code (the `__cg4e2__` in an otr/obs name) is
computed from the bare id the ledger stamps, so it does not move -- pinned
below for every label the dropdown offered before the change.
"""
from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import pytest

from nodes import _otr_model_catalog as C
from nodes._otr_shared import shortcodes as SC

_ROOT = Path(__file__).resolve().parents[1]

#: Exactly what dropdown_choices() offered before the change (HEAD b7f45918).
OLD_LABELS = (
    "Qwen/Qwen3.5-4B (8.7 GB download, mac16-tight nv8-nf4 nv16 nv24)",
    "Qwen/Qwen3.8-27B (51.8 GB download, nv24-nf4)",
    "unsloth/Llama-3.2-3B-Instruct (6.4 GB download, mac16 nv8 nv16 nv24)",
    "mistralai/Mistral-Nemo-Instruct-2407 (24.0 GB download, nv16-nf4 nv24-nf4)",
    "google/gemma-4-E2B-it (6.0 GB download, mac16-tight nv8 nv16 nv24)",
    "google/gemma-4-E4B-it (9.0 GB download, mac16-tight nv8-nf4 nv16 nv24)",
    "google/gemma-4-12b-it (23.9 GB download, nv16-nf4 nv24-nf4)",
    "google/gemma-2-2b-it (5.2 GB download, gated mac16 nv8 nv16 nv24)",
    "comfy_native:gemma4-e2b-it-int8-convrot (4.8 GB download)",
    "comfy_native:gemma4-e4b-it-int8-convrot (7.5 GB download)",
    "comfy_native:gemma4-12b-int8-convrot (11.2 GB download)",
    "openrouter:slot-a", "openrouter:slot-b", "comfy:slot-a", "comfy:slot-b",
    "google_api:slot-a", "google_api:slot-b",
)
#: The published-filename code each writer had before the change.
OLD_CODES = ("q354b", "q3827", "lla3", "nemo", "g4e2", "g4e4", "g412", "g22b",
             "cg4e2", "cg4e4", "cg412", "orsa", "orsb", "cmsa", "cmsb", "gasa", "gasb")


def _writer():
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
    return OTR_LedgerScriptWriter


def test_every_dropdown_label_is_the_bare_id():
    for entry in C.build_dropdown_choices():
        assert entry.label == entry.repo_id, entry
    assert C.default_llm_option() == C.DEFAULT_LLM
    assert C.fresh_llm_option() in C.dropdown_choices()


def test_the_old_labels_name_exactly_todays_options():
    """The catalog's own rows only: a causal LM the machine happens to have
    cached also appears in the dropdown, and must not make this test depend
    on the box it runs on."""
    curated = [e.label for e in C.build_dropdown_choices() if e.curated]
    assert [C._strip_label_suffix(l) for l in OLD_LABELS] == curated


def test_the_published_filename_code_and_run_id_do_not_move():
    for old, code in zip(OLD_LABELS, OLD_CODES):
        bare = C._strip_label_suffix(old)
        assert SC.code_for("llm", bare, "nollm") == code, bare
        assert SC.code_for("llm", old, "nollm") == code, old
        assert C.validate_model_id(old, auto_download_enabled=False) == \
            C.validate_model_id(bare, auto_download_enabled=False)


def test_validate_inputs_names_only_the_two_writer_inputs():
    """docs.comfy.org: inputs named in the signature skip ComfyUI's own list
    check; **kwargs would skip it for EVERY widget on the node."""
    spec = inspect.getfullargspec(_writer().VALIDATE_INPUTS)
    assert spec.args == ["cls", "creative_writing_model", "technical_model"]
    assert spec.varkw is None


def test_validate_inputs_admits_old_spellings_and_refuses_strangers():
    v = _writer().VALIDATE_INPUTS
    assert v() is True                                    # linked or absent: None
    for old in OLD_LABELS:
        assert v(creative_writing_model=old, technical_model=old) is True, old
    assert v(technical_model="Qwen/Qwen3.5-4B:nf4") is True       # retired spelling
    assert v(technical_model="Qwen/Qwen3.5-4B [LOCAL HF]") is True
    bad = v(creative_writing_model="nobody/no-such-writer (1.0 GB download)")
    assert isinstance(bad, str) and bad.startswith("creative_writing_model:")
    assert isinstance(v(technical_model=7), str)


def test_the_widgets_offer_bare_ids_and_carry_the_size_table():
    optional = _writer().INPUT_TYPES()["optional"]
    for name in ("creative_writing_model", "technical_model"):
        choices, opts = optional[name]
        assert not any("GB download" in c for c in choices), name
        assert opts["default"] in choices, name
        assert C.writer_tooltip_table() in opts["tooltip"], name


def test_the_cli_validator_admits_old_labels_exactly_as_the_server_does():
    """scripts/otr_api.py validates --set values before submitting; it mirrors
    the catalog's strip so an old badged label is not refused client-side
    while the server would run it."""
    import sys
    sys.path.insert(0, str(_ROOT / "scripts"))
    import otr_api
    choices = C.dropdown_choices()
    for label in OLD_LABELS + ("Qwen/Qwen3.5-4B:nf4", "Qwen/Qwen3.5-4B [LOCAL HF]",
                               "mistralai/Mistral-Nemo-Instruct-2407 [NOT DOWNLOADED]"):
        assert otr_api._writer_bare_id(label) == C._canonical_qwen_id(label), label
        otr_api._validate_widget_value("OTR_LedgerScriptWriter", "creative_writing_model",
                                       [choices, {}], label)
    with pytest.raises(ValueError):
        otr_api._validate_widget_value("OTR_LedgerScriptWriter", "creative_writing_model",
                                       [choices, {}], "nobody/no-such-writer (1.0 GB download)")
    # the admission is for the two writer widgets only
    with pytest.raises(ValueError):
        otr_api._validate_widget_value("X", "source_bank", [["original"], {}], "original (badge)")


def _node_exe():
    for candidate in ("node", r"C:\Program Files\nodejs\node.exe"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=30, check=True)
            return candidate
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def test_the_load_time_rewrite_reads_a_label_as_the_backend_does():
    """js/workflow_schema_core.js writerName == _canonical_qwen_id, on every old
    label plus the edge spellings, run under node against the shipped file."""
    node = _node_exe()
    if node is None:
        pytest.skip("node is not installed; the JS reader cannot run here")
    labels = list(OLD_LABELS) + [
        "Qwen/Qwen3.5-4B:nf4", "Qwen/Qwen3.5-4B:nf4 (8.7 GB, mac16-tight)",
        "mistralai/Mistral-Nemo-Instruct-2407 [NOT DOWNLOADED]",
        "Qwen/Qwen3.5-4B [LOCAL HF]", "google/gemma-4-12b-it (11.9 GB)", "a (b) (c)",
        "  google/gemma-2-2b-it  ",
    ]
    core = (_ROOT / "js" / "workflow_schema_core.js").as_uri()
    script = ("import { writerName } from %s;" % json.dumps(core)
              + "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>"
                "process.stdout.write(JSON.stringify(JSON.parse(s).map(writerName))));")
    proc = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(labels),
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout) == [C._canonical_qwen_id(l) for l in labels]
    # the one retired spelling the JS hard-codes is the catalog's own
    js = (_ROOT / "js" / "workflow_schema_core.js").read_text(encoding="utf-8")
    assert '"%s" ? "%s"' % (C.DEFAULT_LLM_NF4, C.QWEN_LLM) in js
