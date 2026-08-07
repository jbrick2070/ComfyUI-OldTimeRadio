"""Public-domain source-bank skeleton tests."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nodes import _otr_public_domain_sources as pd
from nodes import _otr_source_payload as osp

REPO = Path(__file__).resolve().parents[1]
# A DEDICATED FIXTURE, NOT THE LIVE MANIFEST (2026-08-04). These tests pin a
# manifest's exact contents -- ids, labels, word budgets -- which is the right
# thing to pin for FORMAT tests and the wrong thing to couple to production
# data. While the bank shipped one book the two were indistinguishable; the
# moment the library grew to 28 works, eight of these broke for no reason but
# the shelf being fuller. The live library is asserted separately, on its own
# terms, in tests/test_public_domain_library_roll.py.
SAMPLE_MANIFEST = (
    REPO / "tests" / "fixtures" / "public_domain_bank" / "manifest.json"
)
MODULE_PATH = REPO / "nodes" / "_otr_public_domain_sources.py"


@pytest.fixture(autouse=True)
def _production_story_routing_root(monkeypatch):
    """Keep these tests independent of synthetic registry tests run earlier."""
    from nodes import _otr_story_routing as routing

    monkeypatch.setattr(
        routing,
        "_STORY_PACKS_ROOT",
        REPO / "nodes" / "story_packs",
    )
    routing._clear_caches()
    yield
    routing._clear_caches()


def _manifest():
    return pd.load_public_domain_manifest(SAMPLE_MANIFEST)


def test_sample_manifest_validates():
    manifest = _manifest()
    assert manifest["schema_version"] == "v1"
    source = manifest["sources"][0]
    assert source["license_status"] == "public_domain_us"
    assert source["recommended_word_budget"] == 300
    assert source["units"][0]["unit_id"] == "arrival"


def test_manifest_unknown_keys_fail_loud():
    manifest = _manifest()
    manifest["sources"][0]["unexpected"] = "nope"
    with pytest.raises(pd.PublicDomainManifestError, match="unknown"):
        pd.validate_public_domain_manifest(manifest)


def test_manifest_requires_rights_metadata():
    manifest = _manifest()
    del manifest["sources"][0]["license_url"]
    with pytest.raises(pd.PublicDomainManifestError, match="license_url"):
        pd.validate_public_domain_manifest(manifest)


def test_manifest_rejects_absolute_or_parent_text_paths():
    manifest = _manifest()
    manifest["sources"][0]["units"][0]["text_path"] = "../escape.txt"
    with pytest.raises(pd.PublicDomainManifestError, match="relative"):
        pd.validate_public_domain_manifest(manifest)


def test_manifest_rejects_duplicate_source_or_unit_ids():
    manifest = _manifest()
    manifest["sources"].append(dict(manifest["sources"][0]))
    with pytest.raises(pd.PublicDomainManifestError, match="duplicate source_id"):
        pd.validate_public_domain_manifest(manifest)

    manifest = _manifest()
    manifest["sources"][0]["units"].append(dict(manifest["sources"][0]["units"][0]))
    with pytest.raises(pd.PublicDomainManifestError, match="duplicate unit_id"):
        pd.validate_public_domain_manifest(manifest)


def test_resolve_source_ref():
    resolved = pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:arrival")
    assert resolved.source_ref == "gutenberg-time-machine-sample:arrival"
    assert resolved.source["title"] == "The Time Machine"
    assert resolved.unit["label"] == "The impossible arrival"


def test_resolve_source_ref_fails_loud():
    with pytest.raises(pd.PublicDomainSourceRefError, match="source_id:unit_id"):
        pd.resolve_manifest_unit(_manifest(), "arrival")
    with pytest.raises(pd.PublicDomainSourceRefError, match="unknown"):
        pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:nope")


def test_text_canonicalizer_strips_gutenberg_boilerplate_and_limits():
    raw = """
    *** START OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***
    First line.\n\nSecond&nbsp;line.
    *** END OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***
    License tail.
    """
    assert pd.canonicalize_public_domain_text(raw, max_chars=80) == "First line. Second line."


def test_payload_from_manifest_unit_has_exact_legacy_keys():
    resolved = pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:arrival")
    payload = pd.payload_from_manifest_unit(
        resolved,
        text="The machine stood in the room. The witnesses doubted him.",
    )
    assert set(payload) == osp.SOURCE_PAYLOAD_KEYS
    assert payload["headline"] == "The Time Machine - The impossible arrival"
    assert payload["source"] == "Project Gutenberg"
    assert "H. G. Wells" in payload["seed_text"]


def test_sidecars_are_separate_from_payload():
    resolved = pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:arrival")
    rights = pd.source_rights_from_unit(resolved)
    meta = pd.source_meta_from_unit(resolved)
    assert rights["license_status"] == "public_domain_us"
    assert meta["source_ref"] == resolved.source_ref
    payload = pd.payload_from_manifest_unit(resolved, text="A compact excerpt.")
    assert "source_rights" not in payload
    assert "source_meta" not in payload


def test_fetch_public_domain_source_uses_default_ref_and_sidecars():
    """Fixed-mode selection against the fixture bank.

    This used to run the LIVE bank with a blank ref and assert The Time
    Machine, which held only because that bank had one book and no selector.
    The live bank now deals at random by design, so the deterministic contract
    is exercised where it is still deterministic: an explicit fixed ref.
    """
    from types import SimpleNamespace

    bank = SimpleNamespace(
        source_bank_id="public_domain",
        defaults={"manifest_path": str(SAMPLE_MANIFEST),
                  "selection_mode": "fixed",
                  "source_ref": "gutenberg-time-machine-sample:arrival"},
    )
    result = pd.fetch_public_domain_source(bank=bank)

    payload, meta, rights = osp.normalize_fetch_result(
        result, origin="public_domain_source")
    assert set(payload) == osp.SOURCE_PAYLOAD_KEYS
    assert payload["headline"] == "The Time Machine - The impossible arrival"
    assert payload["summary"].startswith("A shaken traveler")
    assert "START OF THE PROJECT GUTENBERG" not in payload["full_text"]
    assert payload["source"] == "Project Gutenberg"
    assert payload["date"] == "1895"
    assert payload["link"] == "https://www.gutenberg.org/ebooks/35"
    assert "The Time Machine by H. G. Wells" in payload["seed_text"]
    assert "Unit: The impossible arrival" in payload["seed_text"]
    assert meta["source_ref"] == "gutenberg-time-machine-sample:arrival"
    assert meta["recommended_word_budget"] == 300
    assert rights["license_status"] == "public_domain_us"


def test_fetch_public_domain_source_honors_explicit_ref():
    """An explicit ref pins, and it OUTRANKS the bank's selection mode --
    which is the property that matters now the default mode is random."""
    from types import SimpleNamespace

    bank = SimpleNamespace(
        source_bank_id="public_domain",
        defaults={"manifest_path": str(SAMPLE_MANIFEST),
                  "selection_mode": "random"},
    )
    result = pd.fetch_public_domain_source(
        bank=bank,
        source_ref="gutenberg-time-machine-sample:arrival",
    )
    assert result.source_meta["source_ref"] == "gutenberg-time-machine-sample:arrival"


def test_fetch_public_domain_source_missing_defaults_fail_loud():
    from types import SimpleNamespace

    bank = SimpleNamespace(source_bank_id="public_domain_story", defaults={})
    with pytest.raises(pd.PublicDomainManifestError, match="manifest_path"):
        pd.fetch_public_domain_source(bank=bank)

    # A blank ref under FIXED selection still refuses -- there is no fallback.
    # (Under the default "random" mode a blank ref is legitimate and deals from
    # the library; that path is covered in test_public_domain_library_roll.py.)
    bank = SimpleNamespace(
        source_bank_id="public_domain_story",
        defaults={"manifest_path": str(SAMPLE_MANIFEST),
                  "selection_mode": "fixed"},
    )
    with pytest.raises(pd.PublicDomainSourceRefError, match="source_ref"):
        pd.fetch_public_domain_source(bank=bank)


def test_cache_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_SOURCE_BANK_CACHE_DIR", str(tmp_path / "cache"))
    assert pd.source_bank_cache_root() == tmp_path / "cache"


def test_atomic_write_json_replaces_existing_file(tmp_path):
    target = tmp_path / "cache" / "manifest.json"
    pd.atomic_write_json(target, {"version": 1})
    pd.atomic_write_json(target, {"version": 2, "items": ["a"]})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "items": ["a"],
        "version": 2,
    }
    assert not list(target.parent.glob("*.tmp"))


#: The child program that proves the import is lazy. It runs in a SEPARATE
#: interpreter, so whatever it does to class objects cannot reach this one.
#: The repo root arrives as ``argv[1]`` rather than being interpolated into the
#: source: a Windows path is full of backslashes and every one of them would
#: need escaping inside a ``-c`` string.
_LAZY_IMPORT_PROBE = """
import os
import sys

REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, REPO)

from pathlib import Path

_ORIGINAL_READ_TEXT = Path.read_text


def _guarded_read_text(self, *args, **kwargs):
    # SCOPED TO THE REPO, AND THAT SCOPE IS THE POINT. A blanket ban on
    # Path.read_text fails for a reason that has nothing to do with this
    # module: importing pydantic runs its plugin loader, which reads
    # `entry_points.txt` out of site-packages metadata. In the old in-process
    # test that never showed up, because pydantic was already initialised in
    # the parent pytest process by the time the reload ran -- a fresh child
    # initialises it for the first time. Banning third-party metadata reads
    # would test the interpreter, not us.
    #
    # The invariant this test actually owns: the module must not read ITS OWN
    # DATA FILES at import time.
    if os.path.abspath(str(self)).startswith(REPO):
        raise AssertionError("import-time repo file read attempted: %s" % (self,))
    return _ORIGINAL_READ_TEXT(self, *args, **kwargs)


# INSTALLED IN THE CHILD, ON PURPOSE. pytest's monkeypatch fixture is
# in-process and cannot reach across a process boundary, so the guard has to
# be armed here, before the import under test.
Path.read_text = _guarded_read_text

import nodes._otr_public_domain_sources  # noqa: F401,E402

print("LAZY_IMPORT_OK")
"""


def test_module_import_is_lazy():
    """The module reads no files at IMPORT time -- proven in a CHILD process.

    WHY A SUBPROCESS RATHER THAN ``importlib.reload`` (2026-08-07). This test
    used to reload the module twice with ``Path.read_text`` patched. Reload
    REPLACES the module's class objects, and
    ``tests/test_public_domain_interpreter.py`` imports its exception classes at
    COLLECTION time -- so once this test had run, that file's ``except`` clauses
    no longer matched instances raised by the reloaded module and
    ``test_empty_cast_is_rejected_and_retried_to_failure`` failed whenever the
    two files ran adjacently. It passed alone, and full-suite ordering hid it,
    so every targeted run over these two files reported a red line that had to
    be re-diagnosed as benign.

    Class identity cannot be restored by a fixture -- no teardown can un-rebind
    a class object -- so the reload leaves this process entirely. The assertion
    below is the actual invariant: this interpreter's exception class must be
    the SAME OBJECT afterwards.
    """
    before = pd.PublicDomainInterpreterError

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LAZY_IMPORT_PROBE, str(REPO)],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        # A hung import is a real failure mode and `check=True` would say
        # nothing useful about it, so surface whatever the child managed to emit.
        raise AssertionError(
            "lazy-import probe timed out after 120s\n"
            "stdout: %s\nstderr: %s" % (exc.stdout, exc.stderr)
        ) from exc

    assert proc.returncode == 0, (
        "lazy-import probe failed (exit %d) -- the module read a file at import "
        "time, or would not import at all\nstdout: %s\nstderr: %s"
        % (proc.returncode, proc.stdout, proc.stderr)
    )
    assert "LAZY_IMPORT_OK" in proc.stdout, (
        "lazy-import probe exited 0 without reaching its marker\n"
        "stdout: %s\nstderr: %s" % (proc.stdout, proc.stderr)
    )

    # THE REGRESSION. Nothing this test did may rebind the class objects the
    # rest of the suite already imported.
    assert pd.PublicDomainInterpreterError is before


def test_module_has_no_network_or_heavy_imports():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    banned = {"requests", "urllib", "torch", "transformers", "feedparser"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (names & banned)
