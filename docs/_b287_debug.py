"""Run the failing test inline so we see the real assertion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch
from tests.test_freeze_cascade_meta_persistence import (
    _RebindingLedger,
    _clean_ledger_data,
    _stub_reviewer_disposition,
    _critic_with_targets,
)
from nodes import _otr_freeze_cascade as _LFC
from nodes import _otr_freeze_cascade as _LFC_ORCH

# Mirror the test exactly
led = _RebindingLedger(_clean_ledger_data())

def fake_review(_fn, led_):
    led_.save()
    return _stub_reviewer_disposition("clean_no_edits")

def fake_critic(_fn, _candidate_ledger, _cast_rows):
    return _critic_with_targets()

seen = {"reroll_targets": None, "arc_verdict": None}
original_reroll = _LFC_ORCH._OTRRR.run_targeted_reroll

def spy_reroll(generate_fn, led_):
    from nodes._otr_reroll import _coerce_report
    meta = led_.data.setdefault("meta", {})
    report = _coerce_report(meta.get("story_critic_report"))
    seen["reroll_targets"] = list(report.reroll_targets)
    seen["arc_verdict"] = report.arc_verdict
    return original_reroll(generate_fn, led_)

with patch.object(_LFC_ORCH._OTRLR, "review_ledger", side_effect=fake_review), \
     patch.object(_LFC_ORCH._OTRSC, "run_story_critic", side_effect=fake_critic), \
     patch.object(_LFC_ORCH._OTRRR, "run_targeted_reroll", side_effect=spy_reroll):
    _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

print(f"seen.arc_verdict = {seen['arc_verdict']!r}")
print(f"seen.reroll_targets = {seen['reroll_targets']!r}")
print(f"led.data.meta keys = {sorted(led.data.get('meta', {}).keys())}")
print(f"reroll_escalation = {led.data.get('meta', {}).get('reroll_escalation')}")
print(f"freeze_verdict = {led.data.get('meta', {}).get('freeze_verdict')}")
