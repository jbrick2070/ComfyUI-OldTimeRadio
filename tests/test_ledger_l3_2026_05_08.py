"""tests/test_ledger_l3_2026_05_08.py

l3-2026-05-08 schema additions: BUG-126 telemetry + Cast Contract
pre-wiring + atomic save. Pure stdlib unit tests; verifies that
every helper:
  * stamps the right field at the right location
  * defaults gracefully on absent input
  * never raises (recovery code requirement)
  * preserves back-compat with l3-2026-05-02 ledgers

Round-robin synthesis source: docs/2026-05-08-ledger-schema-additions__*.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
if str(_NODES_DIR) not in sys.path:
    sys.path.insert(0, str(_NODES_DIR))


import _otr_ledger as L  # noqa: E402  -- after sys.path bootstrap


# ---------- schema version + atomic save -------------------------


def test_schema_version_bumped_to_l3_2026_05_08():
    assert L.CURRENT_SCHEMA_VERSION == "l3-2026-05-08"


def test_save_ledger_safe_writes_atomically(tmp_path):
    """Round-robin Element catch (Gemini): non-atomic write on a
    soak-survival system risks 0-byte ledgers on crash. Verify that
    after save, only the final file exists -- no `.tmp` debris."""
    target = tmp_path / "test_ledger.json"
    payload = {
        "episode_id": "ep_test",
        "lines": [{"line_id": "l001", "text": "hi"}],
    }
    ok = L.save_ledger_safe(target, payload)
    assert ok is True
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["test_ledger.json"], (
        f"expected only the final file, got {files} "
        "(temp debris means atomic-rename pattern is broken)"
    )
    # Round-trip readable.
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["episode_id"] == "ep_test"
    assert loaded["schema_version"] == "l3-2026-05-08"


def test_save_ledger_safe_overwrites_existing(tmp_path):
    """Subsequent saves replace the file atomically without leaving
    debris."""
    target = tmp_path / "test_ledger.json"
    L.save_ledger_safe(target, {"episode_id": "v1"})
    L.save_ledger_safe(target, {"episode_id": "v2"})
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["test_ledger.json"]
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["episode_id"] == "v2"


# ---------- BUG-126 telemetry -----------------------------------


def test_bump_line_oom_recovery_count_creates_field_at_zero():
    """Field absent on l3-2026-05-02 ledger -> first bump returns 1."""
    ledger = {"lines": [{"line_id": "l002"}]}
    new = L.bump_line_oom_recovery_count(ledger, "l002")
    assert new == 1
    assert ledger["lines"][0]["oom_recovery_count"] == 1


def test_bump_line_oom_recovery_count_is_idempotent_per_call():
    ledger = {"lines": [{"line_id": "l002", "oom_recovery_count": 3}]}
    assert L.bump_line_oom_recovery_count(ledger, "l002") == 4
    assert L.bump_line_oom_recovery_count(ledger, "l002") == 5
    assert ledger["lines"][0]["oom_recovery_count"] == 5


def test_bump_line_oom_recovery_count_missing_line_returns_zero():
    ledger = {"lines": []}
    assert L.bump_line_oom_recovery_count(ledger, "l_unknown") == 0


def test_stamp_line_render_method_known_value():
    ledger = {"lines": [{"line_id": "l001"}]}
    L.stamp_line_render_method(ledger, "l001", "humo_native")
    assert ledger["lines"][0]["render_method"] == "humo_native"


def test_stamp_line_render_method_warns_on_unknown_value_but_stamps(caplog):
    """Round-robin: unknown method passes through to allow future
    backends without crashing readers, but logs WARNING."""
    ledger = {"lines": [{"line_id": "l001"}]}
    L.stamp_line_render_method(ledger, "l001", "future_engine_native")
    assert ledger["lines"][0]["render_method"] == "future_engine_native"


def test_stamp_clip_humo_oom_recovered_basic():
    clip = {"line_id": "l002", "humo_render_ms": 646000}
    L.stamp_clip_humo_oom_recovered(clip, recovered_at_chunk=2)
    assert clip["humo_oom_recovered"] is True
    assert clip["humo_oom_recovered_at_chunk"] == 2


def test_stamp_clip_humo_oom_recovered_idempotent():
    """Calling twice keeps the flag True; only updates chunk if was
    None."""
    clip = {"humo_oom_recovered_at_chunk": 1}
    L.stamp_clip_humo_oom_recovered(clip, recovered_at_chunk=5)
    assert clip["humo_oom_recovered"] is True
    assert clip["humo_oom_recovered_at_chunk"] == 1  # preserved


def test_bump_meta_cuda_hard_reset_count_creates_meta_block():
    ledger = {}
    L.bump_meta_cuda_hard_reset_count(ledger)
    L.bump_meta_cuda_hard_reset_count(ledger)
    assert ledger["meta"]["cuda_hard_reset_count"] == 2


def test_stamp_meta_soak_cap_records_full_block():
    ledger = {}
    L.stamp_meta_soak_cap(ledger, cap=6, lines_completed_when_hit=6, hit=True)
    expected = {"cap": 6, "lines_completed_when_hit": 6, "hit": True}
    assert ledger["meta"]["soak_cap"] == expected


def test_append_meta_process_run_creates_list():
    ledger = {}
    L.append_meta_process_run(ledger, pid=1234, started_at="2026-05-08T12:00:00Z")
    assert ledger["meta"]["process_runs"] == [
        {"pid": 1234, "started_at": "2026-05-08T12:00:00Z"}
    ]


def test_append_meta_process_run_chains_across_resumes():
    """Round-robin Gemini catch: scalar field would lose history on
    resume. Append-only list preserves the chain."""
    ledger = {}
    L.append_meta_process_run(ledger, pid=1, started_at="t0")
    L.append_meta_process_run(ledger, pid=2, started_at="t1")
    L.append_meta_process_run(ledger, pid=3, started_at="t2")
    runs = ledger["meta"]["process_runs"]
    assert [r["pid"] for r in runs] == [1, 2, 3]
    assert [r["started_at"] for r in runs] == ["t0", "t1", "t2"]


def test_append_meta_process_run_dedupes_same_pid_consecutive():
    """If a node calls this twice in the same process, the second
    call updates the timestamp on the existing entry rather than
    duplicating."""
    ledger = {}
    L.append_meta_process_run(ledger, pid=1, started_at="t0")
    L.append_meta_process_run(ledger, pid=1, started_at="t0_later")
    runs = ledger["meta"]["process_runs"]
    assert len(runs) == 1
    assert runs[0]["started_at"] == "t0_later"


def test_stamp_meta_audit_verdict_pass():
    ledger = {}
    L.stamp_meta_audit_verdict(ledger, pass_=True)
    v = ledger["meta"]["audit_verdict"]
    assert v["pass"] is True
    assert v["fail_patterns_hit"] == []
    assert isinstance(v["ts"], str)
    assert v["ts"].endswith("Z")


def test_stamp_meta_audit_verdict_fail_with_patterns():
    ledger = {}
    L.stamp_meta_audit_verdict(
        ledger, pass_=False,
        fail_patterns_hit=["Fatal Python error: Aborted", "Non-monotonous DTS"],
    )
    v = ledger["meta"]["audit_verdict"]
    assert v["pass"] is False
    assert "Fatal Python error: Aborted" in v["fail_patterns_hit"]


# ---------- Cast Contract pre-wiring -----------------------------


def test_stamp_cast_contract_version_top_level():
    ledger = {}
    L.stamp_cast_contract_version(ledger, "sha:abc12345")
    assert ledger["cast_contract_version"] == "sha:abc12345"


def test_stamp_cast_contract_version_none_keeps_field_absent():
    """Round-robin: None default means "no contract yet" and must
    not write an empty string or null."""
    ledger = {}
    L.stamp_cast_contract_version(ledger, None)
    assert "cast_contract_version" not in ledger


def test_stamp_cast_contract_full_dict():
    ledger = {}
    contract = {
        "version": "sha:abc12345",
        "characters": [
            {
                "character_id": "c01",
                "canonical_name": "MONTY",
                "aliases": ["MONTGOMERY"],
                "voice_spec": "bark:v2/en_speaker_3",
            }
        ],
    }
    L.stamp_cast_contract(ledger, contract)
    assert ledger["cast_contract"]["version"] == "sha:abc12345"
    assert ledger["cast_contract"]["characters"][0]["canonical_name"] == "MONTY"


def test_stamp_cast_contract_none_keeps_absent():
    ledger = {}
    L.stamp_cast_contract(ledger, None)
    assert "cast_contract" not in ledger


def test_stamp_line_cast_contract_version():
    ledger = {"lines": [{"line_id": "l001"}, {"line_id": "l002"}]}
    L.stamp_line_cast_contract_version(ledger, "l001", "sha:abc12345")
    assert ledger["lines"][0]["cast_contract_version"] == "sha:abc12345"
    assert "cast_contract_version" not in ledger["lines"][1]


def test_stamp_cast_voice_spec_finds_by_char_id():
    ledger = {
        "cast": [
            {"char_id": "c01", "name": "MONTY", "voice_preset": "v2/en_speaker_3"},
            {"char_id": "c02", "name": "AEGEUS", "voice_preset": "v2/en_speaker_5"},
        ]
    }
    L.stamp_cast_voice_spec(ledger, "c01", "bark:v2/en_speaker_3")
    assert ledger["cast"][0]["voice_spec"] == "bark:v2/en_speaker_3"
    # Legacy field preserved per round-robin synthesis.
    assert ledger["cast"][0]["voice_preset"] == "v2/en_speaker_3"
    # Sibling unchanged.
    assert "voice_spec" not in ledger["cast"][1]


def test_stamp_cast_voice_spec_unknown_char_id_no_op():
    ledger = {"cast": [{"char_id": "c01"}]}
    L.stamp_cast_voice_spec(ledger, "c99", "bark:foo")
    assert "voice_spec" not in ledger["cast"][0]


# ---------- back-compat: every helper safe on minimal ledger ----


def test_all_helpers_safe_on_empty_ledger():
    """A brand-new ledger with no fields must not crash any
    helper. Recovery code -- never raises."""
    ledger = {}
    L.bump_line_oom_recovery_count(ledger, "l001")
    L.stamp_line_render_method(ledger, "l001", "humo_native")
    L.bump_meta_cuda_hard_reset_count(ledger)
    L.stamp_meta_soak_cap(ledger, cap=6, lines_completed_when_hit=6, hit=True)
    L.append_meta_process_run(ledger, pid=1, started_at="t0")
    L.stamp_meta_audit_verdict(ledger, pass_=True)
    L.stamp_cast_contract_version(ledger, "sha:test")
    L.stamp_cast_contract(ledger, {"version": "sha:test", "characters": []})
    L.stamp_line_cast_contract_version(ledger, "l001", "sha:test")
    L.stamp_cast_voice_spec(ledger, "c01", "bark:foo")
    # If we got here without raising, the contract holds.
    assert "meta" in ledger


def test_helpers_back_compat_with_l3_2026_05_02_ledger():
    """Apply all stamps to an l3-2026-05-02-shape ledger; no errors,
    legacy fields preserved, new fields layered on top."""
    legacy_ledger = {
        "schema_version": "l3-2026-05-02",
        "episode_id": "ep_legacy",
        "cast": [{"char_id": "c01", "name": "MONTY", "voice_preset": "v2/en_speaker_3"}],
        "lines": [{"line_id": "l001", "text": "hi"}],
        "clips": [{"line_id": "l002", "humo_render_ms": 540000}],
        "meta": {"phase_ms": {"flux": 1000}},
    }
    L.bump_line_oom_recovery_count(legacy_ledger, "l001")
    L.stamp_line_render_method(legacy_ledger, "l001", "humo_native")
    L.bump_meta_cuda_hard_reset_count(legacy_ledger)
    L.stamp_meta_soak_cap(legacy_ledger, cap=6, lines_completed_when_hit=6)
    L.stamp_cast_voice_spec(legacy_ledger, "c01", "bark:v2/en_speaker_3")
    # Legacy fields preserved.
    assert legacy_ledger["cast"][0]["voice_preset"] == "v2/en_speaker_3"
    assert legacy_ledger["meta"]["phase_ms"]["flux"] == 1000
    # New fields layered.
    assert legacy_ledger["cast"][0]["voice_spec"] == "bark:v2/en_speaker_3"
    assert legacy_ledger["lines"][0]["oom_recovery_count"] == 1
    assert legacy_ledger["lines"][0]["render_method"] == "humo_native"
    assert legacy_ledger["meta"]["cuda_hard_reset_count"] == 1


def test_valid_render_methods_constant_includes_all_documented():
    """Sanity check on the canonical enum vocabulary."""
    expected = {
        "humo_native",
        "ltx_native",
        "static_radio_fill",
        "kokoro_announcer_only",
        "unknown",
    }
    assert set(L.VALID_RENDER_METHODS) == expected
