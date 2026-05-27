"""Quick debug to see why the BUG-286 test is failing."""
import json as _j
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes import _otr_ledger_reviewer as _lr
from nodes._otr_ledger_reviewer import (
    LineDiagnosis,
    ScriptDoctorDiagnosis,
    run_script_doctor_edits,
)

empty_err = _j.JSONDecodeError("Expecting value", "", 0)
print(f"empty_err msg: {empty_err}")
print(f"is line 1 col 1 (char 0)?: {'line 1 column 1 (char 0)' in str(empty_err)}")

exc = _lr.StructuredCallFailedError(attempts=3, last_error=empty_err)
print(f"exc.last_error: {exc.last_error}")

with patch.object(_lr, "structured_call", side_effect=exc):
    try:
        report = run_script_doctor_edits(
            lambda *a, **k: "",
            {"lines": [{"line_id": "b001", "speaker_role": "announcer", "text": "x"}], "cast": []},
            [],
            ScriptDoctorDiagnosis(diagnoses=[
                LineDiagnosis(line_id="b001", failure="flat_exposition"),
            ]),
            10,
        )
        print(f"verdict: {report.overall_verdict}")
        print(f"edits: {report.edits}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"failed with {type(e).__name__}: {e}")
