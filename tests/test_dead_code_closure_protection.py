"""scripts/dead_code_closure.py applies protection DURING the sweep.

Protection used to filter only the printed result, after the fixpoint, so a
protected symbol was still removed from the model and its own dependencies
were printed as removable (CanonicalImage under the ruled-on
ImageLedgerSection). These tests run the sweep on a synthetic tree.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def closure(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "dead_code_closure_under_test", ROOT / "scripts" / "dead_code_closure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "nodes").mkdir()
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "CODE_DIRS", ["nodes"])
    monkeypatch.setattr(mod, "ROOT_FILES", [])
    monkeypatch.setattr(mod, "PROTECTIVE_DOCS", ["RULINGS.md"])
    monkeypatch.setattr(mod, "PROTECTED_MODULES", {})
    return mod, tmp_path


def _run(mod):
    definitions, rounds, flagged = mod.sweep(False, mod.protection_check())
    names = lambda keys: {definitions[k][1] for k in keys}  # noqa: E731
    return [names(batch) for batch in rounds], names(flagged)


def test_a_protected_symbol_keeps_its_dependencies_alive(closure):
    mod, root = closure
    (root / "nodes" / "schemas.py").write_text(
        "class CanonicalItem:\n"
        "    pass\n"
        "\n"
        "class LedgerSection:\n"
        "    items: 'list[CanonicalItem]' = []\n",
        encoding="utf-8")
    (root / "RULINGS.md").write_text(
        "LedgerSection stays: a declared contract shape.\n", encoding="utf-8")
    rounds, flagged = _run(mod)
    assert flagged == {"LedgerSection"}
    assert all("CanonicalItem" not in batch for batch in rounds)


def test_an_unprotected_chain_is_still_followed(closure):
    mod, root = closure
    (root / "nodes" / "chain.py").write_text(
        "def top():\n"
        "    return middle()\n"
        "\n"
        "def middle():\n"
        "    return 1\n",
        encoding="utf-8")
    rounds, flagged = _run(mod)
    assert rounds == [{"top"}, {"middle"}]
    assert flagged == set()


def test_known_limit_mutual_orphans_are_missed(closure):
    # Leaf peeling, not root reachability: two dead functions that call each
    # other stay "mentioned" by each other. Missed, never wrongly proposed.
    mod, root = closure
    (root / "nodes" / "cycle.py").write_text(
        "def ping():\n"
        "    return pong()\n"
        "\n"
        "def pong():\n"
        "    return ping()\n",
        encoding="utf-8")
    rounds, flagged = _run(mod)
    assert rounds == [] and flagged == set()


def test_known_limit_a_same_named_live_symbol_hides_a_dead_one(closure):
    # Edges are bare names: the live helper in b.py keeps a.py's dead helper
    # mentioned. Missed, never wrongly proposed.
    mod, root = closure
    (root / "nodes" / "a.py").write_text(
        "def helper():\n    return 1\n", encoding="utf-8")
    (root / "nodes" / "b.py").write_text(
        "def helper():\n    return 2\n\nhelper()\n", encoding="utf-8")
    rounds, flagged = _run(mod)
    assert rounds == [] and flagged == set()
