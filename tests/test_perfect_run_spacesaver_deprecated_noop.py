"""Queue item 8 (2026-08-08): perfect_run_spacesaver disposition guard.

The widget stays as a NO-OP SENTINEL on OTR_LedgerScriptWriter -- removing it
would shift widget indices 10..34 on every saved workflow (BUG-LOCAL-097).
Its CONSUMER (`_spacesaver_cleanup_if_flagged` in the retired rtx_upscale.py)
is gone; the tooltip is updated to say "DEPRECATED".

Test scope (Codex r4 MF-17): grep is limited to production modules under
``nodes/*.py`` and ``nodes/_otr_*/*.py``, EXCLUDES ``OTR_LedgerScriptWriter.py``
(the widget's own definition), EXCLUDES ``tests/`` (this file + others may
mention the term), EXCLUDES ``docs/``.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NODES_DIR = REPO / "nodes"


def test_widget_still_defined_on_writer():
    """The widget MUST exist in OTR_LedgerScriptWriter's INPUT_TYPES -- removing
    it shifts every later widget index and breaks saved workflows."""
    # DO NOT pop + re-import the writer module here. The original code did
    # (`sys.modules.pop(...)` then `importlib.import_module(...)`) to "avoid a
    # stale singleton", but this test only reads INPUT_TYPES off the class --
    # a fresh module instance buys nothing, and installing a SECOND writer
    # module for the rest of the session is cross-test pollution: a later test
    # patches one module object while the writer it invokes lives in the other,
    # so the patch silently does nothing. That bit on 2026-08-16, when a rename
    # moved test_scifi_news_pro_tail_context.py past this file alphabetically
    # and its title-regen spy stopped firing. Import normally.
    import nodes.OTR_LedgerScriptWriter as m

    _assert_widget_present(m)


def _assert_widget_present(m) -> None:
    cls = None
    for name in dir(m):
        obj = getattr(m, name)
        if hasattr(obj, "INPUT_TYPES") and callable(getattr(obj, "INPUT_TYPES", None)):
            if "LedgerScriptWriter" in name or "OTR_LedgerScriptWriter" in name:
                cls = obj
                break
    assert cls is not None, "could not resolve the writer class"
    spec = cls.INPUT_TYPES()
    optional = spec.get("optional") or {}
    assert "perfect_run_spacesaver" in optional, (
        "perfect_run_spacesaver widget MUST remain on OTR_LedgerScriptWriter; "
        "removing it shifts widget indices 10..34 and breaks saved workflows.")
    # Default must remain False so behavior is byte-identical (widget is a no-op).
    tp, opts = optional["perfect_run_spacesaver"]
    assert tp == "BOOLEAN", f"widget type changed to {tp!r}; must stay BOOLEAN"
    assert opts.get("default") is False, "default must stay False"


def test_tooltip_marked_deprecated():
    """The tooltip must SAY it's deprecated so a user does not wonder why
    ticking the box has no effect."""
    src = (NODES_DIR / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
    assert "DEPRECATED" in src and "perfect_run_spacesaver" in src, (
        "tooltip does not name the widget as DEPRECATED -- users would "
        "see a live-looking cleanup toggle that does nothing")


def test_no_production_consumer_remains():
    """Grep production modules (excluding the writer itself + tests + docs)
    for any code that READS or STAMPS `perfect_run_spacesaver`. The retired
    consumer was `_spacesaver_cleanup_if_flagged` in rtx_upscale.py; if any
    other module starts reading the flag, the sentinel silently regains a
    side effect."""
    STAMP_PATTERNS = [
        # Attribute-style reads.
        re.compile(r"\bperfect_run_spacesaver\b"),
    ]
    hits = []
    for p in NODES_DIR.rglob("*.py"):
        rel = p.relative_to(REPO)
        # Skip the widget's own definition (writer file).
        if p.name == "OTR_LedgerScriptWriter.py":
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        for pat in STAMP_PATTERNS:
            if pat.search(src):
                hits.append(str(rel))
                break
    assert not hits, (
        f"perfect_run_spacesaver referenced in production modules besides "
        f"OTR_LedgerScriptWriter.py: {hits!r}. If a new consumer wants this "
        f"flag, revive it with a real design pass -- do not let the sentinel "
        f"silently regain side effects.")
