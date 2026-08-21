"""Positive-evidence audit for the selected LTX 2.5 HQ graph."""

from __future__ import annotations

import json

import pytest

from scripts import otr_ltx25_two_stage_audit as audit


def _record(node_id, class_name, ordinal, shape):
    return "[OTR graph-exec] " + json.dumps({
        "class_name": class_name,
        "node_id": node_id,
        "ordinal": ordinal,
        "output_shapes": [shape],
    }, separators=(",", ":"))


def _good_shot(frames=97):
    return [
        "[OTR video] ltx25_video PLAN source=832x480 output=1664x960 frames=%d" % frames,
        _record("latent_upscale", "LTXVLatentUpsampler", 1,
                [1, 128, 13, 30, 52]),
        _record("refine_sampler", "SamplerCustomAdvanced", 2,
                [1, 128, 13, 30, 52]),
        _record("decode", "VAEDecodeTiled", 3, [frames, 960, 1664, 3]),
    ]


def _write(tmp_path, lines):
    path = tmp_path / "server.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_every_shot_must_carry_executor_owned_stage_two_proof(tmp_path):
    path = _write(tmp_path, _good_shot() + _good_shot())
    assert audit.audit_log(path, expect_shots=2) == 2


@pytest.mark.parametrize("drop", [1, 2, 3])
def test_a_missing_stage_or_decode_is_a_refusal(tmp_path, drop):
    lines = _good_shot()
    del lines[drop]
    with pytest.raises(audit.AuditFailure, match="expected 3"):
        audit.audit_log(_write(tmp_path, lines))


def test_stage_order_cannot_be_faked_by_three_unordered_records(tmp_path):
    lines = _good_shot()
    lines[1], lines[2] = lines[2], lines[1]
    with pytest.raises(audit.AuditFailure, match="record 1"):
        audit.audit_log(_write(tmp_path, lines))


def test_decode_must_be_the_upscaled_canvas(tmp_path):
    lines = _good_shot()
    lines[-1] = _record("decode", "VAEDecodeTiled", 3, [97, 480, 832, 3])
    with pytest.raises(audit.AuditFailure, match="does not match plan"):
        audit.audit_log(_write(tmp_path, lines))


def test_plan_and_decode_must_name_the_same_selected_canvas(tmp_path):
    lines = _good_shot()
    lines[0] = lines[0].replace("output=1664x960", "output=1920x1080")
    with pytest.raises(audit.AuditFailure, match="plan canvas"):
        audit.audit_log(_write(tmp_path, lines))


def test_an_adapter_pass_line_without_executor_records_is_not_evidence(tmp_path):
    lines = [_good_shot()[0],
             "[OTR video] ltx25_video TWO-STAGE PASS nodes=3 decode=1664x960"]
    with pytest.raises(audit.AuditFailure, match="expected 3"):
        audit.audit_log(_write(tmp_path, lines))


def test_expected_shot_count_is_exact(tmp_path):
    with pytest.raises(audit.AuditFailure, match="expected 2"):
        audit.audit_log(_write(tmp_path, _good_shot()), expect_shots=2)
