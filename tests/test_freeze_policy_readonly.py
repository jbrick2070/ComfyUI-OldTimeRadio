"""Freeze-policy resolution for the live story banks, and the no-acquisition proof.

The policy decides which receipt the retired same-story cleanup phase stamps
(inline banks: retired; producer-owned banks: not applicable). It never
converts word count, visual vocabulary, style, or craft observations into an
episode veto. The fresh-process test at the end proves the real canonical
freeze node acquires no model.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as LFC
from nodes import _otr_story_routing as ROUTING


@pytest.fixture(autouse=True)
def _fresh_registry():
    ROUTING._REGISTRY = None
    yield
    ROUTING._REGISTRY = None


@pytest.mark.parametrize(
    "bank_id",
    ["media_archive", "original", "public_domain", "shakespeare"],
)
def test_inline_banks_receive_same_story_cleanup_policy(bank_id):
    policy = LFC.resolve_freeze_policy({"source_bank": bank_id})
    assert policy.name == "inline_safety_cleanup"
    assert policy.run_inline_safety_cleanup is True
    assert policy.terminal_error == ""


# A bank is content-owned when its pack declares NO line_composer_system
# seam -- the lane wrote its own lines, so the freeze verifies them instead
# of running the inline mutators over them. Derived from the pack, never
# from a bank-id list in the code.
@pytest.mark.parametrize("bank_id", ["scifi_news_pro", "my_story"])
def test_fixed_topology_banks_are_content_owned(bank_id):
    policy = LFC.resolve_freeze_policy({"source_bank": bank_id})
    assert policy.name == "content_owned_readonly"
    assert policy.run_inline_safety_cleanup is False
    assert policy.terminal_error == ""


def test_untagged_ledger_uses_inline_compatibility_policy():
    policy = LFC.resolve_freeze_policy({})
    assert policy.name == "inline_safety_cleanup"
    assert policy.run_inline_safety_cleanup is True


def test_unknown_declared_bank_fails_as_structural_configuration():
    policy = LFC.resolve_freeze_policy({"source_bank": "no_such_bank"})
    assert policy.name == "policy_resolution_failed"
    assert policy.run_inline_safety_cleanup is False
    assert policy.terminal_error


# ---------------------------------------------------------------------------
# Fresh-process proof: the canonical freeze node acquires no model
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "workflows" / "otr_canonical.json"
FREEZE_SOURCES = (
    "nodes/OTR_LedgerFreezeCascade.py",
    "nodes/_otr_freeze_cascade.py",
    "nodes/_otr_ledger_freeze.py",
    "nodes/_otr_model_loader.py",
)
EXPECTED_CLEANUP_STATUS = {
    "original": "retired_no_content_policy",
    "scifi_news_pro": "not_applicable_content_owned",
}

# The child program runs the real registered freeze node against a fresh,
# valid ledger with the loader's acquisition entry points poisoned. It imports
# the pack exactly the way the fresh canonical audio check does (package
# loader in scripts/otr_canonical_audio_check.py), so the poisoned module is
# THAT package's loader, never a stock ``nodes`` module on sys.path.
_FRESH_FREEZE_CHILD = r'''
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
bank = sys.argv[3]
os.environ.update(PYTHONUTF8="1", CUDA_VISIBLE_DEVICES="",
                  PYTHONDONTWRITEBYTECODE="1", OTR_TEST_MODE="1",
                  OTR_OUTPUT_DIR=str(output), OTR_EXTRA_OUTPUT_ROOTS=str(output))
os.environ.pop("OTR_OBS_DIR", None)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


spec = importlib.util.spec_from_file_location(
    "otr_canonical_audio_check", root / "scripts" / "otr_canonical_audio_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
package = check.load_package()
namespace = package.__name__ + ".nodes."
pl = importlib.import_module(namespace + "production_ledger")
loader = importlib.import_module(namespace + "_otr_model_loader")
cascade = importlib.import_module(namespace + "_otr_freeze_cascade")
authorship = importlib.import_module(namespace + "_otr_content_authorship")

canonical = root / "workflows" / "otr_canonical.json"
raw = canonical.read_bytes()
graph = json.loads(raw)
nodes = {n["id"]: n for n in graph["nodes"]}
links = {link[0]: link for link in graph["links"]}
active = [n for n in graph["nodes"]
          if n["type"] == "OTR_LedgerFreezeCascade" and n.get("mode", 0) == 0]
require(len(active) == 1, "Expected exactly one active canonical OTR_LedgerFreezeCascade")
node = active[0]
cls = package.NODE_CLASS_MAPPINGS["OTR_LedgerFreezeCascade"]
require(cls.__module__ == namespace + "OTR_LedgerFreezeCascade",
        f"Registered freeze class comes from {cls.__module__}, not this package")
schema = cls.INPUT_TYPES()
declared = {**schema.get("required", {}), **schema.get("optional", {})}
require([i["name"] for i in node["inputs"]] == list(declared),
        "Canonical freeze inputs differ from live INPUT_TYPES")
require(len(cls.RETURN_TYPES) == 7, "Freeze node no longer declares seven outputs")
widgets = [i for i in node["inputs"] if i.get("widget")]
require(len(widgets) == len(node["widgets_values"]), "Positional widget count differs")
bound = dict(zip([w["name"] for w in widgets], node["widgets_values"]))
require(all(isinstance(v, bool) for v in bound.values()), f"Widgets are not readiness toggles: {bound}")
linked = {}
for index, item in enumerate(node["inputs"]):
    if item.get("link") is None:
        continue
    link = links[item["link"]]
    require(link[3:5] == [node["id"], index], f"{item['name']} link target differs")
    source = nodes[link[1]]["outputs"][link[2]]
    require(link[0] in (source.get("links") or []), f"{item['name']} missing source fan-out")
    require(source["type"] == link[5] == item["type"], f"{item['name']} link type differs")
    linked[item["name"]] = {"link": link, "source_type": nodes[link[1]]["type"],
                            "source_output": source["name"]}
require(set(linked) == {"script_text", "script_json", "news_used",
                        "estimated_minutes", "technical_model"},
        f"Freeze socket set changed: {sorted(linked)}")
require(linked["technical_model"]["source_type"] == "OTR_LedgerScriptWriter"
        and linked["technical_model"]["source_output"] == "technical_model",
        "technical_model no longer comes from the writer broadcast; requalify the boundary")

episode_id = f"freeze_fixture_{bank}"
text = "The signal is clear."


def fixture_ledger(episode):
    led = pl.new_ledger(episode, str(output / "otr" / "episodes" / episode / "audio"))
    meta = led.data.setdefault("meta", {})
    meta["source_bank"] = bank
    meta["episode_title"] = "Freeze fixture"
    meta["style"] = "radio_drama"
    led.data["cast"] = [{"char_id": "c01", "name": "Mira", "traits": "calm"}]
    led.data["lines"] = [{"line_id": "l001", "char_id": "c01", "speaker_role": "character",
                          "text": text, "beat_id": "b001"}]
    led.data["beats"] = [{"beat_id": "b001"}]
    if bank == "scifi_news_pro":
        # Declared synthetic authored asset; exercises real voice coverage and
        # authorship validation, not accepted production provenance.
        authorship.stamp_receipt(led.data, owner_bank=bank,
                                 accepted_artifacts={"probe": {"text": text}})
    led.save()
    require(Path(led.path).is_file(), "Fixture ledger did not save")
    return led


counters = {"request_slot": 0, "make_generate_fn": 0}


def poison(name):
    def sentinel(*args, **kwargs):
        counters[name] += 1
        raise AssertionError(f"freeze must not call {name}")
    return sentinel


loader.request_slot = poison("request_slot")
loader.make_generate_fn = poison("make_generate_fn")

led = fixture_ledger(episode_id)
boundaries = {"script_text": text, "script_json": json.dumps(led.data),
              "news_used": "fixture", "estimated_minutes": 1,
              "technical_model": "unused-model-boundary"}
result = getattr(cls(), cls.FUNCTION)(**bound, **boundaries)
require(counters == {"request_slot": 0, "make_generate_fn": 0}, f"Acquisition attempted: {counters}")
require(len(result) == 7, f"Expected seven outputs, got {len(result)}")
require(result[1] == result[6], "script_json and v2_ledger_json differ")
require(result[4] in ("frozen_clean", "frozen_with_warns"), f"Unexpected verdict {result[4]!r}")
require(result[2] == "fixture", "news_used was not preserved")
frozen = json.loads(result[1])
require(frozen["meta"]["freeze_unload_ok"] is True, "freeze_unload_ok stamp missing or false")
require(frozen["lines"][0]["text"] == text, "Authored text changed")
cleanup_status = frozen["meta"]["same_story_safety_cleanup"]["status"]
require(Path(led.path).is_file(), "Persisted ledger missing after freeze")

# Secondary public-API proof: the real orchestrator with a poisoned callback.
callback_calls = {"n": 0}


def poisoned_callback(*args, **kwargs):
    callback_calls["n"] += 1
    raise AssertionError("run_freeze_cascade must not invoke its callback")


api_led = fixture_ledger(episode_id + "_api")
disposition = cascade.run_freeze_cascade(
    poisoned_callback, api_led,
    enable_phase_7_audio_readiness=bound["enable_phase_7_audio_readiness"],
    enable_phase_8_video_readiness=bound["enable_phase_8_video_readiness"])
require(callback_calls["n"] == 0, "The cascade invoked its generation callback")
require(disposition.verdict in ("frozen_clean", "frozen_with_warns"),
        f"Orchestrator verdict {disposition.verdict!r}")
require(disposition.gap_audit_post is not None and not disposition.gap_audit_post.errors,
        f"Post-audit errors: {getattr(disposition.gap_audit_post, 'errors', None)}")
require(api_led.data["lines"][0]["text"] == text, "Orchestrator changed authored text")

receipt = {
    "bank": bank,
    "canonical_sha256": hashlib.sha256(raw).hexdigest(),
    "sources": {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest()
                for rel in sys.argv[4:]},
    "package": package.__name__,
    "loader_module": loader.__name__,
    "invoked": {"class": cls.__name__, "module": cls.__module__, "function": cls.FUNCTION},
    "node": {"id": node["id"], "widgets": bound,
             "links": {name: info["link"] for name, info in linked.items()}},
    "declared_boundaries": boundaries,
    "freeze_verdict": result[4],
    "freeze_unload_ok": frozen["meta"]["freeze_unload_ok"],
    "cleanup_status": cleanup_status,
    "counters": counters,
    "callback_calls": callback_calls["n"],
    "orchestrator_verdict": disposition.verdict,
    "ledger": led.path,
}
receipt_path = output / "freeze_no_acquisition_receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(str(receipt_path))
'''


def _head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _hashes():
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            for rel in (("workflows/otr_canonical.json",) + FREEZE_SOURCES)}


@pytest.mark.parametrize("bank", ["original", "scifi_news_pro"])
def test_canonical_freeze_without_acquisition_in_fresh_process(tmp_path, bank):
    """The real canonical freeze node runs with poisoned acquisition.

    Launched as a fresh interpreter so ComfyUI-style package loading, the real
    cascade, policy resolution, readiness passes and ledger persistence all run
    unmocked; only ``request_slot`` / ``make_generate_fn`` are replaced with
    counters that raise. With the acquisition call still present in the node
    this child fails; with it removed the freeze completes and reports zero
    calls. Not a scheduler-cache or replay qualification.
    """
    head_before = _head()
    hashes_before = _hashes()
    output = tmp_path / "fresh_freeze"
    env = {**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1", "OTR_TEST_MODE": "1",
           "OTR_OUTPUT_DIR": str(output), "OTR_EXTRA_OUTPUT_ROOTS": str(output)}
    env.pop("OTR_OBS_DIR", None)
    command = [sys.executable, "-", str(ROOT), str(output), bank, *FREEZE_SOURCES]
    result = subprocess.run(command, input=_FRESH_FREEZE_CHILD, cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8",
                            timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt_path = Path(result.stdout.strip().splitlines()[-1])
    assert receipt_path.is_file(), result.stdout
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["bank"] == bank
    assert receipt["counters"] == {"request_slot": 0, "make_generate_fn": 0}
    assert receipt["callback_calls"] == 0
    assert receipt["invoked"] == {
        "class": "OTR_LedgerFreezeCascade",
        "module": receipt["package"] + ".nodes.OTR_LedgerFreezeCascade",
        "function": "run",
    }
    assert receipt["freeze_verdict"] in {"frozen_clean", "frozen_with_warns"}
    assert receipt["orchestrator_verdict"] in {"frozen_clean", "frozen_with_warns"}
    assert receipt["freeze_unload_ok"] is True
    assert receipt["cleanup_status"] == EXPECTED_CLEANUP_STATUS[bank]
    assert receipt["node"]["links"]["technical_model"] == [115, 1, 4, 62, 4, "STRING"]
    assert receipt["node"]["widgets"] == {
        "enable_phase_7_audio_readiness": True,
        "enable_phase_8_video_readiness": True,
    }
    assert receipt["declared_boundaries"]["technical_model"] == "unused-model-boundary"
    assert receipt["loader_module"] == receipt["package"] + ".nodes._otr_model_loader"
    assert Path(receipt["ledger"]).is_file()
    assert receipt["canonical_sha256"] == hashes_before["workflows/otr_canonical.json"]
    assert receipt["sources"] == {rel: hashes_before[rel] for rel in FREEZE_SOURCES}
    assert _head() == head_before
    assert _hashes() == hashes_before
