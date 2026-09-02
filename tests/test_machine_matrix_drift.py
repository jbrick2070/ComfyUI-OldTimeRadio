"""The generated machine matrix must not drift from its structured sources.

WHY THIS TEST EXISTS, and it is not hypothetical. README told users an 8 GB card
"has rendered **nothing**" and marked the haunted lane "?" at 8 GB, for days
after that exact card published six documented episodes across five source banks. Both
statements were true when written. Nobody re-read them.

A compatibility claim is the worst kind of documentation to hand-maintain,
because it goes stale in the direction that costs a reader the most: not "this
might work and doesn't", but "your card cannot do this" about the thing it has
already done. So the table is generated, and this test is what makes the
generation binding rather than advisory.
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


def test_matrix_and_readme_are_in_sync_with_the_profiles():
    """`--check` writes nothing and fails if either surface is stale."""
    r = subprocess.run([sys.executable, _SCRIPT, "--check"],
                       capture_output=True, text=True, cwd=_REPO)
    assert r.returncode == 0, (
        "docs/MACHINE_MATRIX.md or README's generated block no longer matches "
        "config/profiles/. Regenerate with:\n"
        "    python scripts/otr_machine_matrix.py\n\n" + r.stdout + r.stderr)


def test_declared_machine_classes_are_complete_and_unique():
    """Every stranger-facing row is a complete `--machine` selection."""
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_matrix as M          # noqa: E402
    import otr_machine_profile as P         # noqa: E402
    import otr_provision as provision       # noqa: E402

    rows = M.load_classes()
    assert len(rows) == len({row["key"] for row in rows})
    assert all(row["writer"] and row["writer_model"] for row in rows)
    assert next(row for row in rows if row["key"] == "16gb")["vram_max_gb"] is None

    text = M.render()
    for row in rows:
        command = ("<ComfyUI Python> scripts/otr_provision.py --machine %s "
                   "--list" % row["key"])
        assert command in text
        # The printed stranger-facing command must lead to a real install plan,
        # not merely to a well-formed row with a misspelled engine id.
        profile = P.build_profile(row, P.load_matrix())
        plan = provision.profile_lanes(profile)
        assert set(plan) == {"automatic", "manual"}
    assert "Use the profile named for your machine" not in text
    assert "`widget_mapping`" not in text


def test_machine_rows_drive_the_exact_writer_model_into_both_slots():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_profile as P          # noqa: E402

    matrix = P.load_matrix()
    for row in P.rows(matrix):
        profile = P.build_profile(row, matrix)
        assert profile["llm"]["creative_model"] == row["writer_model"]
        assert profile["llm"]["technical_model"] == row["writer_model"]


def test_machine_selector_accepts_only_exact_public_keys():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_profile as P          # noqa: E402

    assert P.resolve("8gb")["key"] == "8gb"
    for invalid in ("4060", "8GB", " 8gb", "8gb "):
        with pytest.raises(SystemExit, match="no machine"):
            P.resolve(invalid)


def test_experimental_profile_tiers_include_the_middle_vram_band():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_matrix as M          # noqa: E402

    assert M._tier(8) == "8 GB"
    assert M._tier(10.5) == "10-15 GB"
    assert M._tier(14.5) == "16 GB+"
    text = M.render()
    middle = text.index("## 10-15 GB")
    high = text.index("## 16 GB+")
    assert middle < text.index("`otr_nv40_12gb`") < high


def test_experimental_shipping_status_is_separate_from_install_ownership():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_matrix as M          # noqa: E402

    profiles = {row["id"]: row for row in M.load_profiles()}
    missing = {
        "otr_g4_fastwan",
        "otr_g4_ltx_audio_in",
        "otr_g4_ltx_video",
        "otr_w45_fastwan",
        "otr_w45_ltx_audio_in",
        "otr_w45_ltx_video",
        "otr_w45_mesh_stage",
    }
    assert all(profiles[pid]["status"] == "shipping" for pid in missing)
    assert all(profiles[pid]["install_recipe"] == "missing exact owner"
               for pid in missing)
    assert profiles["otr_4060_floor"]["install_recipe"] == "complete"

    text = M.render()
    assert "| confidence | install recipe |" in text
    assert "runtime-ready on a preloaded machine" in text
    for pid in missing:
        row_start = text.index("| `%s` |" % pid)
        row_end = text.index("\n", row_start)
        assert "missing exact owner" in text[row_start:row_end]


def test_every_machine_profile_is_schema_valid_and_applies_to_canonical(tmp_path):
    """The displayed machine command must reach the real graph end to end."""
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_profile as P          # noqa: E402
    from nodes._otr_shared.capability_profiles import validate_profile_shape

    runner = os.path.join(_REPO, "scripts", "otr_canonical_api_run.py")
    matrix = P.load_matrix()
    for row in P.rows(matrix):
        profile = P.build_profile(row, matrix)
        validate_profile_shape(profile, source="machine:%s" % row["key"])
        assert "proven" not in profile

        dump = tmp_path / ("machine-%s.json" % row["key"])
        result = subprocess.run(
            [sys.executable, runner, "--machine", row["key"], "--dry-run",
             "--offline-schemas", "--dump-prompt", str(dump)],
            capture_output=True, text=True, cwd=_REPO,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        prompt_text = json.dumps(json.loads(dump.read_text(encoding="utf-8")))
        assert row["writer_model"] in prompt_text


def test_amd_machine_declares_an_unproven_rocm_candidate_policy():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_profile as P          # noqa: E402

    matrix = P.load_matrix()
    row = P.resolve("amd", matrix)
    profile = P.build_profile(row, matrix)
    assert profile["gpu_vendor"] == "amd"
    assert profile["status"] == "draft"
    assert profile["platform"] == "linux"
    assert profile["device_backend"] == "cuda"  # PyTorch ROCm API spelling
    assert profile["llm"]["device"] == "cuda"
    assert profile["llm"]["quant_policy"] == "none"
    assert profile["llm"]["creative_model"] == "google/gemma-4-E2B-it"
    assert profile["video"]["device_policy"] == "cuda"
    assert profile["video"]["dtype_policy"] == "no_fp8"
    assert profile["image"]["dtype_policy"] == "no_fp8"
    assert profile["audio"]["voice_device"] == "cuda"


def test_proven_receipts_carry_their_evidence():
    """PROVEN is the strongest claim the table makes; it must be checkable.

    Receipts live on the MATRIX ROW, not on a profile file -- a profile has a
    declared shape and an extra key there broke `build_variants --check`. A
    receipt without hardware, a count and a pointer to evidence is an opinion
    wearing a verdict's badge, which is what the hand-written table had become.
    """
    import json
    with io.open(os.path.join(_REPO, "config", "machine_classes.json"),
                 encoding="utf-8") as fh:
        matrix = json.load(fh)

    seen = 0
    for row in matrix.get("classes", []):
        for receipt in (row.get("proven") or []):
            seen += 1
            for field in ("hardware", "episodes", "scope", "evidence"):
                assert receipt.get(field), (
                    "%s has a proven receipt missing %r: %r"
                    % (row.get("key"), field, receipt))
            assert int(receipt["episodes"]) > 0, (
                "%s claims proof with zero episodes" % row.get("key"))
    assert seen, ("no machine class carries a proven receipt -- the matrix's "
                  "strongest column is empty")


def test_engine_evidence_separates_published_and_lab_proof():
    """A physical lab clip must be visible without becoming episode proof."""
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_matrix as M          # noqa: E402

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
    assert "## Hardware episode receipts, with their exact scope" in text
    assert "RTX 4060 8 GB -- 6 episode(s)" in text
    assert "image lane unexercised" in text
    assert "exact row tuple and unlisted cards unproven" in text


def test_readme_4060_episode_count_matches_the_structured_receipt():
    """The video detail repeats the count, so the matrix must still own it."""
    with io.open(os.path.join(_REPO, "config", "machine_classes.json"),
                 encoding="utf-8") as fh:
        matrix = json.load(fh)
    floor = next(row for row in matrix["classes"] if row["key"] == "8gb")
    receipts = [receipt for receipt in floor["proven"]
                if "4060" in receipt["hardware"]]
    assert len(receipts) == 1
    count = int(receipts[0]["episodes"])

    with io.open(os.path.join(_REPO, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    assert "has published %d documented full OTR episodes" % count in readme


def test_measurements_are_structured_and_rendered_with_conditions():
    sys.path.insert(0, os.path.join(_REPO, "scripts"))
    import otr_machine_matrix as M          # noqa: E402

    rows = M.load_measurements()
    assert rows
    assert all(row["engine"] and row["conditions"] and row["measured"]
               for row in rows)
    text = M.render()
    for row in rows:
        assert "| `%s` | %s | %s |" % (
            row["engine"], row["conditions"], row["measured"]) in text
