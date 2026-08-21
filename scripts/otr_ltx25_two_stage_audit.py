#!/usr/bin/env python3
"""Fail closed unless every logged LTX 2.5 shot executed the HQ second stage.

The evidence comes from ``wrapper_bridge.run_graph`` after each real Comfy node
returns. Adapter plan/pass messages are boundaries and summaries, never proof.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PLAN_RE = re.compile(
    r"\[OTR video\]\s+ltx25_video\s+PLAN\b.*"
    r"\bsource=(\d+)x(\d+)\s+output=(\d+)x(\d+).*\bframes=(\d+)\b")
MARKER = "[OTR graph-exec] "
EXPECTED_SOURCE = (832, 480)
EXPECTED_OUTPUT = (1664, 960)
EXPECTED = (
    ("latent_upscale", "LTXVLatentUpsampler"),
    ("refine_sampler", "SamplerCustomAdvanced"),
    ("decode", "VAEDecodeTiled"),
)


class AuditFailure(RuntimeError):
    """The server log does not positively prove the selected HQ graph."""


def _validate_shot(number, shot, records):
    frames = shot["frames"]
    source = shot["source"]
    output = shot["output"]
    if source != EXPECTED_SOURCE or output != EXPECTED_OUTPUT:
        raise AuditFailure(
            "shot %d: plan canvas source=%dx%d output=%dx%d is not the "
            "selected 832x480 -> 1664x960 recipe"
            % (number, source[0], source[1], output[0], output[1]))
    if len(records) != len(EXPECTED):
        raise AuditFailure(
            "shot %d: expected 3 execution records, found %d"
            % (number, len(records)))
    for ordinal, (record, wanted) in enumerate(zip(records, EXPECTED), 1):
        observed = (record.get("node_id"), record.get("class_name"),
                    record.get("ordinal"))
        expected = (wanted[0], wanted[1], ordinal)
        if observed != expected:
            raise AuditFailure(
                "shot %d record %d: expected %r, found %r"
                % (number, ordinal, expected, observed))

    shapes = records[-1].get("output_shapes")
    shape = shapes[0] if isinstance(shapes, list) and shapes else None
    if not (isinstance(shape, list) and len(shape) == 4
            and shape[0] == frames
            and shape[1:3] == [output[1], output[0]]
            and shape[3] in (3, 4)):
        raise AuditFailure(
            "shot %d: decode shape %r does not match plan [%d,%d,%d,3|4]"
            % (number, shape, frames, output[1], output[0]))


def audit_log(path, expect_shots=None):
    lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    shots = []
    current = None
    for line_number, line in enumerate(lines, 1):
        plan = PLAN_RE.search(line)
        if plan:
            if current is not None:
                _validate_shot(len(shots) + 1, current, current["records"])
                shots.append(current)
            current = {
                "source": (int(plan.group(1)), int(plan.group(2))),
                "output": (int(plan.group(3)), int(plan.group(4))),
                "frames": int(plan.group(5)),
                "records": [],
            }
            continue
        if MARKER not in line or current is None:
            continue
        payload = line.split(MARKER, 1)[1].strip()
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AuditFailure(
                "line %d: malformed graph execution JSON: %s"
                % (line_number, exc)) from exc
        if record.get("node_id") in {item[0] for item in EXPECTED}:
            current["records"].append(record)

    if current is not None:
        _validate_shot(len(shots) + 1, current, current["records"])
        shots.append(current)
    if not shots:
        raise AuditFailure("no ltx25_video PLAN line found")
    if expect_shots is not None and len(shots) != int(expect_shots):
        raise AuditFailure(
            "expected %d LTX 2.5 shot(s), found %d"
            % (int(expect_shots), len(shots)))
    return len(shots)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--expect-shots", type=int)
    args = parser.parse_args(argv)
    try:
        count = audit_log(args.server_log, args.expect_shots)
    except (AuditFailure, OSError, UnicodeError) as exc:
        parser.exit(1, "LTX25 TWO-STAGE AUDIT FAIL: %s\n" % exc)
    print("LTX25 TWO-STAGE AUDIT PASS: %d shot(s)" % count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
