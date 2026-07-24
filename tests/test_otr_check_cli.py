"""
Independent source banks v1, wave 4: the `otr_check bank` activation CLI.

The law under test: activation is the CONSENT ACT. It refuses with a NAMED,
client-actionable code rather than a traceback; it never writes a receipt for a
bundle it could not import; it publishes the snapshot BEFORE the receipt; and
what it publishes is exactly what boot's own `discover()` will admit -- the CLI
owns no second copy of the format.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_story_routing as routing  # noqa: E402
from nodes import _otr_user_banks as ub  # noqa: E402


def _load_cli():
    """Load scripts/otr_check.py under its own module name.

    By file location on purpose: the checker is a script, not an installed
    package, and the test must exercise the very file the .bat launcher runs."""
    path = REPO_ROOT / "scripts" / "otr_check.py"
    spec = importlib.util.spec_from_file_location("otr_check_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


ROW = {
    "source_bank_id": "client_probe",
    "label": "Client Probe",
    "source_kind": "article",
    "interpreter": "self",
    "fetcher": "self",
    "default_story_model": "client_model",
    "default_story_pipeline": "legacy_many_pass_adapt",
    "defaults": {},
    "required_seams": [],
    "runnable": True,
    "guide_ref": "",
}

GOOD_MODULE = (
    "def fetch_source(**kwargs):\n    raise NotImplementedError\n\n\n"
    "def interpret_source(**kwargs):\n    raise NotImplementedError\n\n\n"
    "def check_compatibility(**kwargs):\n    return True\n"
)

MODEL_ID = "client_model"
# The fixture pack is DERIVED from a shipped one rather than hand-rolled: the
# checker now runs the authority's real pack sweep, so a stub pack would test
# the checker against a pack production would reject. Deriving also means the
# fixture follows the pack schema instead of drifting from it.
_SHIPPED_PACK = (REPO_ROOT / "nodes" / "story_packs" / "public_domain" /
                 "faithful_radio_adaptation.json")


def _pack_for(bank_id: str) -> dict:
    pack = json.loads(_SHIPPED_PACK.read_text(encoding="utf-8"))
    pack["source_bank_id"] = bank_id
    pack["story_model_id"] = MODEL_ID
    return pack


def _bundle(base: Path, bank_id: str = "client_probe", *,
            module: str = GOOD_MODULE, row: "dict | None" = None,
            story_pack: bool = True) -> Path:
    """Author a complete client bundle under a throwaway base. NOT activated --
    earning activation is what these tests are about."""
    root = ub.user_banks_root(base) / bank_id
    root.mkdir(parents=True)
    if story_pack:
        (root / ub.STORY_PACKS_DIRNAME).mkdir()
        (root / ub.STORY_PACKS_DIRNAME / f"{MODEL_ID}.json").write_text(
            json.dumps(_pack_for(bank_id), indent=2), encoding="utf-8")
    payload = dict(ROW if row is None else row)
    payload.setdefault("source_bank_id", bank_id)
    if row is None:
        payload["source_bank_id"] = bank_id
    (root / ub.BANK_JSON_FILENAME).write_text(
        json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                    "bank": payload}, indent=2), encoding="utf-8")
    (root / f"{bank_id}.py").write_text(module, encoding="utf-8")
    return root


def _run(args: "list[str]", base: Path, capsys) -> "tuple[int, list[dict]]":
    code = cli.main(["bank", *args, "--base", str(base), "--json"])
    out = capsys.readouterr().out.strip()
    return code, (json.loads(out) if out else [])


def _activate(root: Path, base: Path, capsys, *extra: str):
    return _run([str(root), "--activate", *extra], base, capsys)


def _receipt_exists(root: Path) -> bool:
    return (root / ub.RECEIPT_FILENAME).exists()


def _discover(base: Path):
    """Boot's own view, through the one authority's parser."""
    parse_row, shipped = routing.shipped_bank_seed()
    return ub.discover(parse_row=parse_row, protected_ids=shipped, root=base)


# -- status without activation: reads bytes, executes nothing ----------------

def test_unactivated_bundle_reports_unchecked(tmp_path, capsys):
    root = _bundle(tmp_path)
    code, reports = _run([str(root)], tmp_path, capsys)
    assert code == cli.EXIT_REFUSED
    assert reports[0]["status"] == ub.UNCHECKED
    assert reports[0]["activated"] is False
    assert "--activate" in reports[0]["detail"]


def test_status_never_writes_anything(tmp_path, capsys):
    root = _bundle(tmp_path)
    _run([str(root)], tmp_path, capsys)
    assert not _receipt_exists(root)
    assert not ub.snapshots_root(tmp_path).exists()


def test_missing_path_and_missing_all_is_a_usage_error(tmp_path, capsys):
    assert cli.main(["bank", "--base", str(tmp_path)]) == cli.EXIT_USAGE


# -- every refusal is NAMED, and none of them writes a receipt --------------

@pytest.mark.parametrize("bank_id,kwargs,code", [
    ("original", {}, "protected_id"),
    ("Client_Probe", {}, "bad_bank_id"),
    ("client_probe", {"story_pack": False}, "missing_story_pack"),
])
def test_authoring_refusals_are_named(tmp_path, capsys, bank_id, kwargs, code):
    root = _bundle(tmp_path, bank_id, **kwargs)
    exit_code, reports = _run([str(root)], tmp_path, capsys)
    assert exit_code == cli.EXIT_REFUSED
    assert reports[0]["code"] == code
    assert reports[0]["status"] == ub.BROKEN
    assert not _receipt_exists(root)


def test_missing_module_is_named(tmp_path, capsys):
    root = _bundle(tmp_path)
    (root / "client_probe.py").unlink()
    _, reports = _run([str(root)], tmp_path, capsys)
    assert reports[0]["code"] == "missing_module"


def test_bundle_outside_the_bank_root_is_refused(tmp_path, capsys):
    """A folder boot will never look at cannot be activated. Handing back a
    receipt for it would be a checker that says yes to nothing."""
    stray = tmp_path / "somewhere_else" / "client_probe"
    stray.mkdir(parents=True)
    _, reports = _run([str(stray)], tmp_path, capsys)
    assert reports[0]["code"] == "outside_bank_root"


def test_a_file_is_not_a_bundle(tmp_path, capsys):
    ub.user_banks_root(tmp_path).mkdir(parents=True)
    stray = ub.user_banks_root(tmp_path) / "client_probe"
    stray.write_text("not a directory", encoding="utf-8")
    _, reports = _run([str(stray)], tmp_path, capsys)
    assert reports[0]["code"] == "not_a_bundle_dir"


def test_bad_row_is_refused_by_the_one_authority(tmp_path, capsys):
    """The CLI does not judge rows -- routing does. A row the authority
    rejects must be refused HERE, or activation would bless a bank that
    quarantines at boot."""
    row = dict(ROW)
    row.pop("label")
    root = _bundle(tmp_path, row=row)
    _, reports = _run([str(root)], tmp_path, capsys)
    assert reports[0]["code"] == "bad_row"


def test_all_never_activates(tmp_path, capsys):
    _bundle(tmp_path)
    assert cli.main(["bank", "--all", "--activate", "--base", str(tmp_path)]) \
        == cli.EXIT_USAGE


# -- activation: the consent act -------------------------------------------

def test_activation_publishes_snapshot_and_receipt(tmp_path, capsys):
    root = _bundle(tmp_path)
    code, reports = _activate(root, tmp_path, capsys)
    report = reports[0]
    assert code == cli.EXIT_OK
    assert report["status"] == ub.ACTIVATED and report["activated"] is True
    receipt = json.loads((root / ub.RECEIPT_FILENAME).read_text(encoding="utf-8"))
    assert set(receipt) == set(ub.RECEIPT_KEYS)
    assert receipt["schema_version"] == ub.RECEIPT_SCHEMA_VERSION
    assert receipt["source_bank_id"] == "client_probe"
    assert receipt["digest"] == ub.bundle_digest(root)
    assert receipt["snapshot"] == ub.snapshot_dirname("client_probe",
                                                      receipt["digest"])
    assert (ub.snapshots_root(tmp_path) / receipt["snapshot"]).is_dir()


def test_the_snapshot_holds_the_authored_bytes(tmp_path, capsys):
    """Content-addressed means what it says: the snapshot is a copy, not an
    empty marker directory."""
    root = _bundle(tmp_path)
    _, reports = _activate(root, tmp_path, capsys)
    snapshot = ub.snapshots_root(tmp_path) / reports[0]["snapshot"]
    assert (snapshot / ub.BANK_JSON_FILENAME).read_bytes() == \
        (root / ub.BANK_JSON_FILENAME).read_bytes()
    assert (snapshot / "client_probe.py").read_bytes() == \
        (root / "client_probe.py").read_bytes()
    assert not (snapshot / ub.RECEIPT_FILENAME).exists()


def test_activation_is_what_boot_admits(tmp_path, capsys):
    """The acceptance property for the whole wave: after --activate, the ONE
    authority's own discovery admits the bank with no issues."""
    root = _bundle(tmp_path)
    assert _discover(tmp_path)[0] == ()
    _activate(root, tmp_path, capsys)
    admitted, issues = _discover(tmp_path)
    assert issues == ()
    assert [bundle.bank_id for bundle in admitted] == ["client_probe"]


def test_reactivating_unchanged_bytes_is_a_no_op(tmp_path, capsys):
    root = _bundle(tmp_path)
    _, first = _activate(root, tmp_path, capsys)
    receipt_bytes = (root / ub.RECEIPT_FILENAME).read_bytes()
    snapshots = sorted(p.name for p in ub.snapshots_root(tmp_path).iterdir())
    code, second = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_OK
    assert second[0]["snapshot"] == first[0]["snapshot"]
    assert (root / ub.RECEIPT_FILENAME).read_bytes() == receipt_bytes
    assert sorted(p.name for p in ub.snapshots_root(tmp_path).iterdir()) \
        == snapshots


def test_editing_the_bundle_goes_stale_until_reactivated(tmp_path, capsys):
    root = _bundle(tmp_path)
    _activate(root, tmp_path, capsys)
    pack = _pack_for("client_probe")
    pack["label"] = "Edited After Activation"
    (root / ub.STORY_PACKS_DIRNAME / f"{MODEL_ID}.json").write_text(
        json.dumps(pack, indent=2), encoding="utf-8")
    code, reports = _run([str(root)], tmp_path, capsys)
    assert code == cli.EXIT_REFUSED
    assert reports[0]["status"] == ub.STALE
    assert _discover(tmp_path)[1][0].code == "stale_receipt"
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_OK and reports[0]["status"] == ub.ACTIVATED


def test_a_deleted_snapshot_goes_stale(tmp_path, capsys):
    import shutil

    root = _bundle(tmp_path)
    _, reports = _activate(root, tmp_path, capsys)
    shutil.rmtree(ub.snapshots_root(tmp_path) / reports[0]["snapshot"])
    _, after = _run([str(root)], tmp_path, capsys)
    assert after[0]["status"] == ub.STALE


# -- the bounded probe: the client's Python runs, and only under --activate --

def test_a_module_that_raises_on_import_is_refused_without_a_receipt(
        tmp_path, capsys):
    """The property that matters most: a bundle that cannot be imported must
    never end up with a receipt saying it was checked."""
    root = _bundle(tmp_path, module="raise RuntimeError('boom at import')\n")
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_REFUSED
    # Wave 3's loader already names this precisely; the CLI reports its
    # sentence rather than inventing a second, vaguer one.
    assert reports[0]["code"] == "bundle_execution"
    assert "boom at import" in reports[0]["detail"]
    assert not _receipt_exists(root)
    assert not ub.snapshots_root(tmp_path).exists()


def test_a_declared_lane_with_no_function_is_refused(tmp_path, capsys):
    """The row routes fetch_source to the bundle with `self`; the module does
    not define it. Wave 3 raises UserBankExecutionError at run time -- wave 4's
    job is to catch it before the client ever presses go."""
    root = _bundle(tmp_path,
                   module="def interpret_source(**kwargs):\n    return None\n")
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_REFUSED
    assert reports[0]["code"] == "bundle_execution"
    assert ub.FETCH_ENTRY_ATTR in reports[0]["detail"]
    assert not _receipt_exists(root)


def test_a_lane_that_cannot_take_the_writers_keywords_is_refused(
        tmp_path, capsys):
    """Importable is not enough. A fetch_source that binds the wrong keywords
    dies minutes into a render, after the operator committed a GPU -- so
    activation binds the writer's real keyword set without calling anything."""
    root = _bundle(tmp_path, module=(
        "def fetch_source(bank, request=None):\n    return None\n\n\n"
        "def interpret_source(**kwargs):\n    return None\n"))
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_REFUSED
    assert reports[0]["code"] == "bad_entry_signature"
    assert "technical_model" in reports[0]["detail"]
    assert not _receipt_exists(root)


def test_named_keywords_and_kwargs_both_satisfy_the_binding(tmp_path, capsys):
    explicit = _bundle(tmp_path, "client_named", module=(
        "def fetch_source(*, bank, technical_model, source_ref, load_config,"
        " policy):\n    return None\n\n\n"
        "def interpret_source(*, bank, payload, technical_fn, model_id):\n"
        "    return None\n"))
    assert _activate(explicit, tmp_path, capsys)[0] == cli.EXIT_OK
    catch_all = _bundle(tmp_path, "client_kwargs", module=GOOD_MODULE)
    assert _activate(catch_all, tmp_path, capsys)[0] == cli.EXIT_OK


def test_check_compatibility_is_reserved_and_never_examined(tmp_path, capsys):
    """Option A: it has no runtime consumer, so there is nothing true for
    activation to assert about it. A bundle whose check_compatibility is a
    plain integer still activates -- rejecting one would be enforcing an
    interface production never touches."""
    root = _bundle(tmp_path, module=GOOD_MODULE + "\ncheck_compatibility = 3\n")
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_OK
    assert "compat" not in reports[0]

    bare = _bundle(tmp_path, "client_bare", module=(
        "def fetch_source(**kwargs):\n    return None\n\n\n"
        "def interpret_source(**kwargs):\n    return None\n"))
    assert _activate(bare, tmp_path, capsys)[0] == cli.EXIT_OK


def test_a_bundle_boot_would_quarantine_never_gets_a_receipt(tmp_path, capsys):
    """The parity property. `discover` does NOT run the pack sweep or the row
    cross-references -- `_admit_user_banks` does, afterwards. Without those the
    checker would hand a receipt to a bank that quarantines at boot as
    bad_bundle_contract, i.e. say yes to a bank nobody can select."""
    row = dict(ROW)
    row["default_story_pipeline"] = "no_such_pipeline"
    root = _bundle(tmp_path, row=row)
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_REFUSED
    assert reports[0]["code"] == "bad_bundle_contract"
    assert not _receipt_exists(root)
    # and the same refusal without --activate, before any code runs
    assert _run([str(root)], tmp_path, capsys)[1][0]["code"] \
        == "bad_bundle_contract"


def test_a_slow_import_is_bounded(tmp_path, capsys):
    """A hung bundle costs itself a named refusal, never the checker."""
    root = _bundle(tmp_path, module="import time\ntime.sleep(30)\n" + GOOD_MODULE)
    code, reports = _activate(root, tmp_path, capsys, "--timeout", "3")
    assert code == cli.EXIT_REFUSED
    assert reports[0]["code"] == "probe_timeout"
    assert not _receipt_exists(root)


# -- fixtures are judged by the RUNTIME's validator, not a checker copy ------

def test_a_recorded_fixture_is_validated_with_normalize_fetch_result(
        tmp_path, capsys):
    root = _bundle(tmp_path)
    fixtures = root / ub.FIXTURES_DIRNAME
    fixtures.mkdir()
    (fixtures / "sample.json").write_text(json.dumps({
        "headline": "H", "summary": "S", "full_text": "F", "source": "src",
        "date": "2026-07-24", "link": "https://example.invalid/a",
        "seed_text": "seed"}), encoding="utf-8")
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_OK
    assert reports[0]["fixtures"] == 1


def test_a_broken_fixture_is_refused_without_a_receipt(tmp_path, capsys):
    root = _bundle(tmp_path)
    fixtures = root / ub.FIXTURES_DIRNAME
    fixtures.mkdir()
    (fixtures / "sample.json").write_text(
        json.dumps({"headline": "only this"}), encoding="utf-8")
    code, reports = _activate(root, tmp_path, capsys)
    assert code == cli.EXIT_REFUSED
    assert reports[0]["code"] == "bad_fixture"
    assert "sample.json" in reports[0]["detail"]
    assert not _receipt_exists(root)


# -- publication order: snapshot first, receipt second ----------------------

def test_a_failed_snapshot_leaves_no_receipt(tmp_path, monkeypatch):
    """The ordering law. If the snapshot cannot be written, the bundle stays
    UNCHECKED -- the reverse order would leave a receipt naming a snapshot
    that never existed, which reads at boot as a bundle the client broke."""
    root = _bundle(tmp_path)
    digest = ub.bundle_digest(root)

    def _explode(_root):
        raise OSError("disk went away")

    monkeypatch.setattr(ub, "_authoring_files", _explode)
    with pytest.raises(ub.UserBankActivationError):
        ub.write_activation(root, digest=digest, base=tmp_path)
    assert not _receipt_exists(root)
    assert not (ub.snapshots_root(tmp_path) /
                ub.snapshot_dirname("client_probe", digest)).exists()


def test_the_receipt_is_staged_outside_the_bundle(tmp_path):
    """A temp file inside the bundle would join the authoring bytes and change
    the very digest the receipt records."""
    root = _bundle(tmp_path)
    digest = ub.bundle_digest(root)
    ub.write_activation(root, digest=digest, base=tmp_path)
    assert ub.bundle_digest(root) == digest
    assert sorted(p.name for p in root.iterdir() if p.name.endswith(".tmp")) == []


def test_status_after_a_manual_receipt_edit_is_broken(tmp_path, capsys):
    root = _bundle(tmp_path)
    _activate(root, tmp_path, capsys)
    (root / ub.RECEIPT_FILENAME).write_text(
        json.dumps({"schema_version": "v1.0"}), encoding="utf-8")
    _, reports = _run([str(root)], tmp_path, capsys)
    assert reports[0]["status"] == ub.BROKEN


def test_all_reports_every_bundle_and_fails_if_any_is_not_ready(
        tmp_path, capsys):
    ready = _bundle(tmp_path, "client_ready")
    _activate(ready, tmp_path, capsys)
    _bundle(tmp_path, "client_pending")
    code = cli.main(["bank", "--all", "--base", str(tmp_path), "--json"])
    reports = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_REFUSED
    assert {r["bank_id"] for r in reports} == {"client_ready", "client_pending"}
    assert {r["status"] for r in reports} == {ub.ACTIVATED, ub.UNCHECKED}
