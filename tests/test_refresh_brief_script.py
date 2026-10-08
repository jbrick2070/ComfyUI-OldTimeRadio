"""The /refresh skill's brief.py runs from any directory and reports the
current shipping set.

It used to ast.literal_eval() build_variants.py's SHIPPING_SET -- which became
a call, `shipping_ids()`, so the script raised ValueError -- and then read
config/profiles/<id>.json, a directory that no longer exists. It now asks the
matrix API itself.
"""
import subprocess
import sys
from pathlib import Path

from nodes._otr_shared.capability_profiles import shipping_ids

ROOT = Path(__file__).resolve().parents[1]
BRIEF = ROOT / ".cursor" / "skills" / "refresh" / "scripts" / "brief.py"


def test_brief_runs_outside_the_repo_and_counts_the_shipping_set(tmp_path):
    result = subprocess.run(
        [sys.executable, str(BRIEF)], cwd=str(tmp_path),
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    shipping = shipping_ids()
    local = [p for p in shipping if not p.startswith("otr_cloud_")]
    cloud = [p for p in shipping if p.startswith("otr_cloud_")]
    assert ("COUNTS canonical=1 local=%d cloud=%d shipping=%d"
            % (len(local), len(cloud), len(shipping))) in result.stdout
    for pid in shipping:
        assert ("  %s  " % pid) in result.stdout
