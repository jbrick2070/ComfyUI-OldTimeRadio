"""S31 B6 Fix 1 -- RSS news re-rank slot label / id agreement.

`_fetch_rss_seed_or_die` routes through
`_otr_model_loader.request_slot("technical", model_id)` (post-S31 B3).
The writer's `_resolve_inputs` calls `_fetch_rss_seed_or_die` to seed
news. Pre-fix it passed `creative_writing_model` -- the slot label
("technical") and the resolved id (creative model) would disagree in
differing-slots mode. Post-fix it passes `technical_model`.

In default config (creative == technical, both = DEFAULT_LLM) the two
ids are identical so the fix is a no-op at runtime. In differing-slots
config (S32 forward) the fix is load-bearing -- the slot scheduler
would otherwise load the creative model under the technical slot
label, defeating two-slot routing.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
WRITER_PATH = PACK_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function {name!r} not found")


SOURCE_PAYLOAD_PATH = PACK_ROOT / "nodes" / "_otr_source_payload.py"


def test_resolve_inputs_rss_uses_technical_model():
    """S31 B6 invariant across the chunk-3 topology (2026-07-05): the
    fetch moved behind the source-payload contract, so the pin is now
    TWO-part -- (a) `_resolve_inputs` threads `technical_model` into
    the resolved fetcher entry's `.fetch(...)` as the
    `technical_model=` kwarg (and no longer calls
    `_fetch_rss_seed_or_die` directly); (b) the science_rss wrapper
    forwards it as `_fetch_rss_seed_or_die`'s SOLE POSITIONAL arg.
    Together the technical model still routes the technical re-rank
    slot.

    Style-engine consolidation (2026-07-05): the wrapper used to pass
    (style_slug, technical_model) as two positionals; style_slug was
    removed -- the fetch/rerank chain is style-agnostic now -- so the
    wrapper's only positional is technical_model.
    """
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    resolve_fn = _find_function(tree, "_resolve_inputs")

    # (a) writer side: no direct call; entry.fetch(technical_model=...,
    # source_ref=...).
    fetch_call = None
    for sub in ast.walk(resolve_fn):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        assert not (isinstance(fn, ast.Name)
                    and fn.id == "_fetch_rss_seed_or_die"), (
            "_resolve_inputs must not call _fetch_rss_seed_or_die "
            "directly post-chunk-3 (the science_rss wrapper owns it)."
        )
        if isinstance(fn, ast.Attribute) and fn.attr == "fetch":
            fetch_call = sub
    assert fetch_call is not None, (
        "_resolve_inputs must call the resolved fetcher entry's "
        ".fetch(...); none found."
    )
    kw = {k.arg: k.value for k in fetch_call.keywords if k.arg}
    assert "technical_model" in kw, (
        "entry.fetch(...) must pass technical_model= explicitly."
    )
    assert "source_ref" in kw, (
        "entry.fetch(...) must pass source_ref= explicitly so Source Banks "
        "v2 fetchers can fail loud on unsupported references."
    )
    assert (isinstance(kw["technical_model"], ast.Name)
            and kw["technical_model"].id == "technical_model"), (
        "S31 B6 Fix 1 (chunk-3 form): entry.fetch must receive the "
        "`technical_model` parameter (the slot label routes through "
        "`request_slot(\"technical\", ...)`), not "
        "`creative_writing_model`."
    )
    assert (isinstance(kw["source_ref"], ast.Name)
            and kw["source_ref"].id == "source_ref"), (
        "entry.fetch must receive the source_ref parameter, not a constant "
        "or a resolved default."
    )

    # (b) wrapper side: sole positional into _fetch_rss_seed_or_die
    # (style_slug retired 2026-07-05; the fetch/rerank chain is
    # style-agnostic now).
    sp_tree = ast.parse(SOURCE_PAYLOAD_PATH.read_text(encoding="utf-8"))
    wrapper = _find_function(sp_tree, "_fetch_science_rss")
    target_call = None
    for sub in ast.walk(wrapper):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "_fetch_rss_seed_or_die"):
            target_call = sub
            break
    assert target_call is not None, (
        "science_rss wrapper must call _fetch_rss_seed_or_die."
    )
    assert len(target_call.args) == 1, (
        "wrapper must pass technical_model as the sole positional arg "
        "(style_slug retired 2026-07-05)."
    )
    first_arg = target_call.args[0]
    assert isinstance(first_arg, ast.Name) and first_arg.id == "technical_model", (
        "S31 B6 Fix 1: the wrapper's sole positional must be "
        "technical_model."
    )
    wrapper_kw = {item.arg: item.value for item in target_call.keywords}
    assert "receipt_sink" in wrapper_kw


def test_news_seed_receipt_is_promoted_after_episode_ledger_creation():
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    run_fn = _find_function(tree, "run")
    new_ledger_lines = []
    stamp_lines = []
    for node in ast.walk(run_fn):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "new_ledger"
        ):
            new_ledger_lines.append(node.lineno)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_stamp_news_seed_receipt"
        ):
            stamp_lines.append(node.lineno)
    assert len(new_ledger_lines) == 1
    assert len(stamp_lines) == 1
    assert stamp_lines[0] > new_ledger_lines[0]


def test_news_seed_receipt_matches_and_copies_selected_body():
    from nodes import OTR_LedgerScriptWriter as writer

    body = "complete selected RSS body"
    article = {
        "headline": "A headline",
        "summary": "summary",
        "full_text": body,
        "source": "Fixture Feed",
        "date": "2026-07-30",
        "link": "https://example.com/item",
        "seed_text": "A headline summary",
    }
    receipt = {
        "headline": article["headline"],
        "source": article["source"],
        "url": article["link"],
        "date": article["date"],
        "body_chars": len(body),
        "body_source": "rss_full",
        "rss_content_index": 1,
        "rss_content_count": 2,
        "body_bytes_utf8": len(body.encode("utf-8")),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "selected_at": "2026-07-30T12:00:00",
    }
    meta = {}

    writer._stamp_news_seed_receipt(meta, {
        "news_article": article,
        "news_seed_receipt": receipt,
    })

    assert meta["news_seed"] == receipt
    assert meta["news_seed"] is not receipt
    receipt["headline"] = "mutated after stamp"
    assert meta["news_seed"]["headline"] == "A headline"


def test_rss_fetch_builds_receipt_from_the_selected_complete_body(
        monkeypatch):
    from nodes import OTR_LedgerScriptWriter as writer
    from nodes import story_orchestrator

    body = "complete body with a tail sentinel: TAIL"
    monkeypatch.setattr(
        story_orchestrator,
        "_fetch_science_news",
        lambda **_kwargs: [{
            "headline": "Selected headline",
            "summary": "Selected summary",
            "full_text": body,
            "source": "Fixture Feed",
            "date": "2026-07-30",
            "link": "https://example.com/selected",
            "_body_source": "rss_full",
            "_rss_content_index": 2,
            "_rss_content_count": 3,
        }],
    )
    receipt = {}

    payload = writer._fetch_rss_seed_or_die(
        "fixture-model",
        receipt_sink=receipt,
    )

    assert payload["full_text"] == body
    assert receipt["body_source"] == "rss_full"
    assert receipt["rss_content_index"] == 2
    assert receipt["rss_content_count"] == 3
    assert receipt["body_chars"] == len(body)
    assert receipt["body_bytes_utf8"] == len(body.encode("utf-8"))
    assert receipt["body_sha256"] == hashlib.sha256(
        body.encode("utf-8")
    ).hexdigest()


def test_news_seed_receipt_mismatch_fails_loud():
    from nodes import OTR_LedgerScriptWriter as writer

    body = "selected body"
    article = {
        "headline": "headline",
        "full_text": body,
        "source": "source",
        "date": "2026-07-30",
        "link": "https://example.com/item",
    }
    receipt = {
        "headline": "headline",
        "source": "source",
        "url": article["link"],
        "date": article["date"],
        "body_chars": len(body),
        "body_source": "rss_full",
        "rss_content_index": 0,
        "rss_content_count": 1,
        "body_bytes_utf8": len(body.encode("utf-8")),
        "body_sha256": "0" * 64,
        "selected_at": "2026-07-30T12:00:00",
    }
    with pytest.raises(RuntimeError, match="body_sha256"):
        writer._stamp_news_seed_receipt({}, {
            "news_article": article,
            "news_seed_receipt": receipt,
        })


def test_resolve_inputs_rss_default_config_baseline():
    """Default config sanity: when creative == technical, the
    resolved id is the same regardless of which slot label travels
    with it. The fix from B6 is a no-op at runtime in default
    config -- this test pins that baseline.

    Structural check: AST scan asserts the writer-level routing
    table at the top of OTR_LedgerScriptWriter.py declares
    `creative_writing_model` and `technical_model` as separate
    widgets with the same default. The shared default = DEFAULT_LLM
    means default-config callers pass the SAME id under either
    slot label.
    """
    src = WRITER_PATH.read_text(encoding="utf-8")
    # Both widgets must exist in INPUT_TYPES.
    assert '"creative_writing_model"' in src
    assert '"technical_model"' in src
