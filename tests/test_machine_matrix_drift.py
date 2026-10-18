"""The generated machine matrix must not drift from config/workflow_matrix.json.

WHY THIS TEST EXISTS, and it is not hypothetical. README told users an 8 GB card
"has rendered **nothing**" and marked the haunted lane "?" at 8 GB, for days
after that exact card published six documented episodes across five source banks. Both
statements were true when written. Nobody re-read them.

A compatibility claim is the worst kind of documentation to hand-maintain,
because it goes stale in the direction that costs a reader the most: not "this
might work and doesn't", but "your card cannot do this" about the thing it has
already done. So the table is generated, and this test is what makes the
generation binding rather than advisory.

THERE ARE NO MACHINE CLASSES (operator, 2026-10-18: "no machine classes, no
profiles, just JSONs"). The workflow JSONs are the whole machine answer; the
receipts sit beside the workflows in config/workflow_matrix.json.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "otr_machine_matrix.py")
_MATRIX = os.path.join(_REPO, "config", "workflow_matrix.json")


def _generator():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_matrix as M          # noqa: E402
    return M


def test_matrix_and_readme_are_in_sync_with_the_workflows():
    """`--check` writes nothing and fails if either surface is stale."""
    r = subprocess.run([sys.executable, _SCRIPT, "--check"],
                       capture_output=True, text=True, cwd=_REPO)
    if r.returncode == 3:
        # The generator reads the live audio registry, so an interpreter
        # without torch renders an incomplete doc. That is not drift.
        pytest.skip(r.stdout.strip() or "registry unavailable in this interpreter")
    assert r.returncode == 0, (
        "apple/MACHINE_MATRIX.md or README's generated block no longer matches "
        "config/workflow_matrix.json. Regenerate with:\n"
        "    python scripts/otr_machine_matrix.py\n\n" + r.stdout + r.stderr)


def test_no_machine_class_or_profile_layer_exists():
    """The class file, its helper and the --machine selector are gone for good."""
    assert not os.path.exists(os.path.join(_REPO, "config", "machine_classes.json"))
    assert not os.path.exists(os.path.join(_REPO, "scripts", "otr_machine_profile.py"))
    for script in ("otr_provision.py", "otr_canonical_api_run.py",
                   "otr_machine_matrix.py", "otr_pod_provision.sh"):
        text = io.open(os.path.join(_REPO, "scripts", script), encoding="utf-8").read()
        assert "machine_classes" not in text, script
        assert '"--machine"' not in text and "--machine " not in text, script


def test_the_front_door_names_workflows_and_sends_mid_cards_to_8gb():
    M = _generator()
    text = M.render()
    assert "## Which workflow to open" in text
    assert ("| 10-15 GB NVIDIA (RTX 4070, 3080) | an `otr_8gb_*` workflow -- "
            "or try an `otr_16gb_*` one at your own risk |") in text
    assert "scripts/otr_provision.py --profile otr_8gb_low --list" in text
    assert "--machine" not in text
    assert "machine key" not in text
    assert "`widget_mapping`" not in text


def test_workflows_are_grouped_by_the_card_their_name_names():
    M = _generator()
    assert M._group("otr_8gb_low") == "8 GB NVIDIA"
    assert M._group("otr_16gb_video") == "16 GB NVIDIA"
    assert M._group("otr_mac16_low") == "Apple Silicon (16 GB)"
    assert M._group("otr_canonical") == "Any machine"
    text = M.render()
    eight = text.index("## 8 GB NVIDIA")
    sixteen = text.index("## 16 GB NVIDIA")
    mac = text.index("## Apple Silicon (16 GB)")
    assert eight < text.index("| `otr_8gb_low` |") < sixteen
    assert mac < text.index("| `otr_mac16_low` |")
    assert "## 10-15 GB" not in text


def test_shipping_status_is_separate_from_install_ownership():
    M = _generator()
    profiles = {row["id"]: row for row in M.load_profiles()}
    # EVERY SHIPPED WORKFLOW HAS AN EXACT INSTALL OWNER (2026-09-25).
    shipping = [pid for pid, row in profiles.items() if row["status"] == "shipping"]
    assert shipping, "the matrix ships nothing; this test would prove nothing"
    unowned = sorted(pid for pid in shipping
                     if "missing exact owner" in profiles[pid]["install_recipe"])
    assert unowned == [], unowned
    assert profiles["otr_8gb_low"]["install_recipe"].startswith("complete")

    text = M.render()
    assert "| extra install | confidence | install recipe |" in text
    assert "runtime-ready on a preloaded machine" in text
    for pid in shipping:
        row_start = text.index("| `%s` |" % pid)
        row_end = text.index("\n", row_start)
        assert "missing exact owner" not in text[row_start:row_end], pid
    # AnimateDiff needs a third-party node pack; its rows must say so.
    row_start = text.index("| `otr_8gb_animatediff` |")
    assert "ComfyUI-AnimateDiff-Evolved" in text[row_start:text.index("\n", row_start)]


def test_hardware_receipts_carry_their_evidence():
    """PROVEN is the strongest claim the table makes; it must be checkable."""
    with io.open(_MATRIX, encoding="utf-8") as fh:
        matrix = json.load(fh)
    receipts = matrix.get("hardware_receipts") or []
    assert receipts, "the matrix's strongest column is empty"
    for receipt in receipts:
        for field in ("workflows", "hardware", "episodes", "scope", "evidence"):
            assert receipt.get(field), "receipt missing %r: %r" % (field, receipt)
        assert int(receipt["episodes"]) > 0, receipt

    M = _generator()
    text = M.render()
    assert "## Hardware episode receipts, with their exact scope" in text
    # 2026-09-13: four more 4060 episodes, one per 8 GB workflow in the shipping set.
    assert "on RTX 4060 8 GB -- 11 episode(s)" in text


def test_engine_evidence_separates_published_and_lab_proof():
    """A physical lab clip must be visible without becoming episode proof."""
    M = _generator()
    rows = M.load_engine_evidence()
    assert rows
    assert {row["level"] for row in rows} <= {"PROVEN", "LAB-PROVEN"}

    h3_4060 = [row for row in rows
               if row["engine"] == "minimax_h3_fl2va_raw_recipe"
               and "4060" in row["hardware"]]
    assert len(h3_4060) == 1
    assert h3_4060[0]["level"] == "LAB-PROVEN"
    assert int(h3_4060[0]["artifacts"]) == 3
    assert "not the OTR engine adapter or a canonical episode" in h3_4060[0]["scope"]

    text = M.render()
    assert "## Engine proof by hardware" in text
    assert "| `minimax_h3_fl2va_raw_recipe` (MiniMax H3 FL2VA raw ComfyUI recipe with native audio) | **LAB-PROVEN** | RTX 4060 Laptop, 8 GB, Ada |" in text
    assert "setup, model load, queued prompt, reserve clamp" in text


def test_measurements_are_structured_and_rendered_with_conditions():
    M = _generator()
    rows = M.load_measurements()
    assert rows
    assert all(row["engine"] and row["conditions"] and row["measured"]
               for row in rows)
    text = M.render()
    for row in rows:
        assert "| `%s` | %s | %s |" % (
            row["engine"], row["conditions"], row["measured"]) in text


def test_known_limits_render_from_the_workflow_matrix():
    M = _generator()
    limits = M.load_known_limits()
    assert limits
    text = M.render()
    assert "## Known limits, written down when they were found" in text
    for limit in limits:
        assert "* %s" % limit in text
