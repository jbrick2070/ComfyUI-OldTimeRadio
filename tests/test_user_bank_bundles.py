"""Client-authored source-bank bundles: integrity, admission, quarantine.

Independent source banks v1, wave 1. The law under test: a broken CLIENT
bundle is QUARANTINED (named issue, boot survives, healthy siblings still
admit) and NEVER raises -- while the shipped seed's fail-loud posture is
untouched.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_user_banks as ub


SHIPPED_IDS = frozenset({
    "media_archive", "original", "public_domain", "shakespeare",
    "scifi_news_pro", "custom_source_bank",
})


def _row(bank_id: str) -> dict:
    return {
        "source_bank_id": bank_id,
        "label": "Client Bank",
        "source_kind": "article",
        "interpreter": "",
        "fetcher": "",
        "default_story_model": "client_model",
        "default_story_pipeline": "legacy_many_pass",
        "defaults": {},
        "required_seams": [],
        "runnable": True,
        "guide_ref": "",
    }


def _passthrough_parse(row: dict, origin: str):
    """Stand-in for the routing authority's one parser."""
    del origin
    return dict(row)


def _make_bundle(base: Path, bank_id: str, *, activate: bool = True,
                 row: "dict | None" = None) -> Path:
    """Author a complete client bundle under `base` and (optionally) activate
    it -- i.e. write the content-addressed snapshot + receipt the way
    `otr_check bank --activate` will."""
    root = ub.user_banks_root(base) / bank_id
    (root / ub.STORY_PACKS_DIRNAME).mkdir(parents=True)
    (root / ub.FIXTURES_DIRNAME).mkdir(parents=True)
    (root / ub.BANK_JSON_FILENAME).write_text(
        json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                    "bank": row if row is not None else _row(bank_id)},
                   indent=2),
        encoding="utf-8")
    (root / f"{bank_id}.py").write_text(
        "def fetch_source(**kwargs):\n    raise NotImplementedError\n",
        encoding="utf-8")
    (root / ub.STORY_PACKS_DIRNAME / "default.json").write_text(
        json.dumps({"story_model_id": "client_model"}), encoding="utf-8")
    if activate:
        _activate(base, root, bank_id)
    return root


def _activate(base: Path, root: Path, bank_id: str) -> str:
    digest = ub.bundle_digest(root)
    snapshot = ub.snapshot_dirname(bank_id, digest)
    (ub.snapshots_root(base) / snapshot).mkdir(parents=True, exist_ok=True)
    (root / ub.RECEIPT_FILENAME).write_text(
        json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                    "source_bank_id": bank_id, "digest": digest,
                    "snapshot": snapshot}),
        encoding="utf-8")
    return digest


def _discover(base: Path):
    return ub.discover(parse_row=_passthrough_parse,
                       protected_ids=SHIPPED_IDS, root=base)


# -- (a) the absent / empty case: no client banks is not an error -----------

def test_absent_root_admits_nothing_and_reports_nothing(tmp_path):
    assert _discover(tmp_path) == ((), ())


def test_empty_root_admits_nothing(tmp_path):
    ub.user_banks_root(tmp_path).mkdir(parents=True)
    assert _discover(tmp_path) == ((), ())


# -- (b) the happy path ------------------------------------------------------

def test_activated_bundle_admits(tmp_path):
    _make_bundle(tmp_path, "client_noir")
    admitted, issues = _discover(tmp_path)
    assert issues == ()
    assert [b.bank_id for b in admitted] == ["client_noir"]
    bundle = admitted[0]
    assert bundle.module_path.name == "client_noir.py"
    assert bundle.row["source_bank_id"] == "client_noir"
    assert bundle.digest == ub.bundle_digest(bundle.root)


def test_two_bundles_admit_in_sorted_order(tmp_path):
    _make_bundle(tmp_path, "bank_b")
    _make_bundle(tmp_path, "bank_a")
    admitted, issues = _discover(tmp_path)
    assert issues == ()
    assert [b.bank_id for b in admitted] == ["bank_a", "bank_b"]


# -- (c) the digest: content-addressed, timestamp-free ----------------------

def test_digest_is_stable_across_repeated_reads(tmp_path):
    root = _make_bundle(tmp_path, "client_noir")
    assert ub.bundle_digest(root) == ub.bundle_digest(root)


def test_digest_changes_when_any_authored_byte_changes(tmp_path):
    root = _make_bundle(tmp_path, "client_noir")
    before = ub.bundle_digest(root)
    (root / "client_noir.py").write_text("# edited\n", encoding="utf-8")
    assert ub.bundle_digest(root) != before


def test_digest_ignores_the_receipt_itself(tmp_path):
    root = _make_bundle(tmp_path, "client_noir", activate=False)
    bare = ub.bundle_digest(root)
    _activate(tmp_path, root, "client_noir")
    assert ub.bundle_digest(root) == bare


# -- (d) quarantine: every failure mode is named, none of them raise --------

def _only_issue(tmp_path):
    admitted, issues = _discover(tmp_path)
    assert admitted == ()
    assert len(issues) == 1
    return issues[0]


def test_missing_bank_json_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir")
    (root / ub.BANK_JSON_FILENAME).unlink()
    _activate(tmp_path, root, "client_noir")
    assert _only_issue(tmp_path).code == "missing_bank_json"


def test_malformed_bank_json_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir", activate=False)
    (root / ub.BANK_JSON_FILENAME).write_text("{not json", encoding="utf-8")
    _activate(tmp_path, root, "client_noir")
    assert _only_issue(tmp_path).code == "malformed_bank_json"


def test_unknown_envelope_key_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir", activate=False)
    (root / ub.BANK_JSON_FILENAME).write_text(
        json.dumps({"schema_version": ub.RECEIPT_SCHEMA_VERSION,
                    "bank": _row("client_noir"), "extra": 1}),
        encoding="utf-8")
    _activate(tmp_path, root, "client_noir")
    assert _only_issue(tmp_path).code == "malformed_bank_json"


def test_id_mismatch_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir", activate=False,
                        row=_row("something_else"))
    _activate(tmp_path, root, "client_noir")
    assert _only_issue(tmp_path).code == "id_mismatch"


@pytest.mark.parametrize("shipped", sorted(SHIPPED_IDS))
def test_shipped_id_cannot_be_shadowed(tmp_path, shipped):
    _make_bundle(tmp_path, shipped)
    assert _only_issue(tmp_path).code == "protected_id"


@pytest.mark.parametrize("bad", ["Client", "9bank", "client-noir", "_client"])
def test_unsafe_folder_name_quarantines(tmp_path, bad):
    _make_bundle(tmp_path, bad)
    assert _only_issue(tmp_path).code == "bad_bank_id"


def test_missing_module_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir", activate=False)
    (root / "client_noir.py").unlink()
    _activate(tmp_path, root, "client_noir")
    assert _only_issue(tmp_path).code == "missing_module"


def test_missing_module_message_names_only_real_requirements(tmp_path):
    """The message a client reads must be TRUE.

    It used to say the module "must ship one module with fetch_source +
    interpret_source + check_compatibility". A bundle that defines no
    `check_compatibility` at all activates cleanly -- nothing inspects that
    name, not even for callability -- so the sentence demanded something the
    code has never required. Naming a reserved-but-unwired entry point in a
    contract-bearing string is how a placeholder becomes a phantom rule that
    clients write code to satisfy."""
    root = _make_bundle(tmp_path, "client_noir", activate=False)
    (root / "client_noir.py").unlink()
    _activate(tmp_path, root, "client_noir")
    detail = _only_issue(tmp_path).detail
    assert ub.COMPAT_ENTRY_ATTR not in detail
    assert ub.FETCH_ENTRY_ATTR in detail
    assert ub.INTERPRET_ENTRY_ATTR in detail


def test_missing_story_pack_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir", activate=False)
    (root / ub.STORY_PACKS_DIRNAME / "default.json").unlink()
    _activate(tmp_path, root, "client_noir")
    assert _only_issue(tmp_path).code == "missing_story_pack"


def test_unactivated_bundle_quarantines(tmp_path):
    _make_bundle(tmp_path, "client_noir", activate=False)
    assert _only_issue(tmp_path).code == "missing_receipt"


def test_edit_after_activation_goes_stale(tmp_path):
    root = _make_bundle(tmp_path, "client_noir")
    (root / "client_noir.py").write_text("# edited after activation\n",
                                         encoding="utf-8")
    issue = _only_issue(tmp_path)
    assert issue.code == "stale_receipt"
    assert "--activate" in issue.detail


def test_missing_snapshot_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir")
    receipt = json.loads((root / ub.RECEIPT_FILENAME).read_text(encoding="utf-8"))
    (ub.snapshots_root(tmp_path) / receipt["snapshot"]).rmdir()
    assert _only_issue(tmp_path).code == "missing_snapshot"


def test_receipt_naming_another_bank_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir")
    receipt = json.loads((root / ub.RECEIPT_FILENAME).read_text(encoding="utf-8"))
    receipt["source_bank_id"] = "someone_else"
    (root / ub.RECEIPT_FILENAME).write_text(json.dumps(receipt), encoding="utf-8")
    assert _only_issue(tmp_path).code == "receipt_id_mismatch"


def test_malformed_receipt_quarantines(tmp_path):
    root = _make_bundle(tmp_path, "client_noir")
    (root / ub.RECEIPT_FILENAME).write_text(
        json.dumps({"schema_version": "v0.0"}), encoding="utf-8")
    assert _only_issue(tmp_path).code == "malformed_receipt"


def test_bad_row_quarantines_with_the_authority_message(tmp_path):
    _make_bundle(tmp_path, "client_noir")

    def _reject(row, origin):
        raise ValueError(f"{origin}: runnable must be a bool")

    admitted, issues = ub.discover(parse_row=_reject,
                                   protected_ids=SHIPPED_IDS, root=tmp_path)
    assert admitted == ()
    assert issues[0].code == "bad_row"
    assert "runnable must be a bool" in issues[0].detail


def test_stray_file_at_the_bank_root_quarantines(tmp_path):
    ub.user_banks_root(tmp_path).mkdir(parents=True)
    (ub.user_banks_root(tmp_path) / "README.txt").write_text("hi", encoding="utf-8")
    assert _only_issue(tmp_path).code == "not_a_bundle_dir"


# -- (e) isolation: one bad bundle never takes a healthy one down ------------

def test_broken_sibling_does_not_block_a_healthy_bank(tmp_path):
    _make_bundle(tmp_path, "healthy_bank")
    _make_bundle(tmp_path, "broken_bank", activate=False)
    admitted, issues = _discover(tmp_path)
    assert [b.bank_id for b in admitted] == ["healthy_bank"]
    assert [(i.bank_id, i.code) for i in issues] == [
        ("broken_bank", "missing_receipt")]


def test_issue_renders_the_bank_the_code_and_the_path(tmp_path):
    _make_bundle(tmp_path, "broken_bank", activate=False)
    rendered = _only_issue(tmp_path).render()
    assert "broken_bank" in rendered
    assert "missing_receipt" in rendered
    assert "QUARANTINED" in rendered


# -- (f) posture pins --------------------------------------------------------

def test_discover_requires_a_real_parser():
    with pytest.raises(ub.UserBankLayoutError):
        ub.discover(parse_row=None, protected_ids=SHIPPED_IDS)


def test_module_import_does_zero_file_io():
    """Lazy law: the module body may only import, define, and assign
    literals -- ComfyUI custom-node import isolation."""
    source = (Path(ub.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        assert isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef,
                                 ast.FunctionDef, ast.Assign, ast.AnnAssign,
                                 ast.Expr)), f"module-level {type(node).__name__}"
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant), "only the docstring"


def test_repo_root_points_at_the_custom_node_repo():
    assert (ub.repo_root() / "nodes" / "_otr_story_routing.py").is_file()
    assert ub.user_banks_root().name == ub.SOURCE_BANKS_DIRNAME
