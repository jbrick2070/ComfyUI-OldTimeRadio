"""Bake-off source-snapshot replay layer (r3 rulings B7 + B8).

Proves the frozen-source replay path added to ``_resolve_inputs``:
  * the manifest is keyed by BASE bank, so one snapshot serves the base/_v2/_v3
    triplet;
  * an envelope whose declared base != ``base_source_bank_id(selected)`` is
    REJECTED loudly (B7) -- never a silent fall-through to live sourcing;
  * a valid snapshot BYPASSES the three live source branches (entropy /
    custom-premise / external fetch) inside ``_resolve_inputs``;
  * malformed envelopes and an altered-payload receipt fail loud;
  * ``cast_hints`` supplied at the top level lands in the ``source_meta`` sidecar
    where the adaptation consumer reads it.

Pure Python; no GPU, no LLM, no network (the RSS fetcher is asserted UNREACHED).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_source_snapshot as SS  # noqa: E402
from nodes import _otr_source_payload as SP  # noqa: E402
from nodes import OTR_LedgerScriptWriter as LSW  # noqa: E402


def _payload(seed="the frozen source seed text"):
    return {
        "headline": "Frozen Headline",
        "summary": "A frozen summary.",
        "full_text": "The full frozen source body.",
        "source": "Snapshot Source",
        "date": "2026-07-15",
        "link": "",
        "seed_text": seed,
    }


def _envelope(base="science_news", **overrides):
    env = {
        "base_source_bank_id": base,
        "captured_from_bank_id": base,
        "payload": _payload(),
        "seed_source": "science_rss_frozen",
        "source_meta": {"kind": "frozen_rss"},
        "source_rights": {"license_label": "frozen"},
        "seed_receipt": {
            "OTR_CAST_SEED": "42",
            "OTR_STYLE_SEED": "42",
            "OTR_FABLE2_SEED": "42",
        },
    }
    env.update(overrides)
    return env


def _write_manifest(tmp_path, snapshots, *, name="manifest.json", allow_partial=False):
    path = tmp_path / name
    body = {"schema_version": 1, "snapshots": snapshots}
    if allow_partial:
        body["allow_partial"] = True
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


@pytest.fixture(autouse=True)
def _clean_snapshot_env(monkeypatch):
    monkeypatch.delenv(SS.SNAPSHOT_MANIFEST_ENV, raising=False)
    SS._clear_cache()
    yield
    SS._clear_cache()


# ---------------------------------------------------------------------------
# 1. sha256 helper
# ---------------------------------------------------------------------------

def test_payload_sha256_is_canonical_and_key_order_independent():
    a = SS.snapshot_payload_sha256({"a": "1", "b": "2"})
    b = SS.snapshot_payload_sha256({"b": "2", "a": "1"})
    assert a == b
    assert len(a) == 64


# ---------------------------------------------------------------------------
# 2. load_snapshot_for_bank -- None paths (no replay requested)
# ---------------------------------------------------------------------------

def test_no_manifest_env_returns_none():
    assert SS.load_snapshot_for_bank("science_news") is None


def test_manifest_configured_but_base_absent_raises_by_default(tmp_path, monkeypatch):
    # Codex r4 MUST-FIX 1: a configured manifest that lacks the selected base
    # must NOT silently source live -- that invalidates the bake-off control.
    manifest = _write_manifest(tmp_path, {"media_archive": _envelope("media_archive")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    with pytest.raises(SS.SourceSnapshotError, match="no .*entry for base"):
        SS.load_snapshot_for_bank("science_news")


def test_allow_partial_sources_absent_base_live(tmp_path, monkeypatch):
    manifest = _write_manifest(
        tmp_path, {"media_archive": _envelope("media_archive")}, allow_partial=True)
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    assert SS.load_snapshot_for_bank("science_news") is None


# ---------------------------------------------------------------------------
# 3. load_snapshot_for_bank -- valid + base-keyed triplet sharing
# ---------------------------------------------------------------------------

def test_valid_snapshot_is_returned_with_validated_fields(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {"science_news": _envelope("science_news")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    snap = SS.load_snapshot_for_bank("science_news")
    assert snap is not None
    assert snap.base_source_bank_id == "science_news"
    assert snap.seed_source == "science_rss_frozen"
    assert snap.payload["seed_text"] == "the frozen source seed text"
    assert snap.source_rights == {"license_label": "frozen"}
    assert snap.payload_sha256 == SS.snapshot_payload_sha256(_payload())


@pytest.mark.parametrize("selected", ["science_news_v2", "science_news_v3"])
def test_variant_id_resolves_to_base_keyed_snapshot(tmp_path, monkeypatch, selected):
    manifest = _write_manifest(tmp_path, {"science_news": _envelope("science_news")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    snap = SS.load_snapshot_for_bank(selected)
    assert snap is not None
    assert snap.base_source_bank_id == "science_news"


# ---------------------------------------------------------------------------
# 4. load_snapshot_for_bank -- loud rejections (B7 + shape)
# ---------------------------------------------------------------------------

def test_base_mismatch_is_rejected_loud(tmp_path, monkeypatch):
    # Envelope keyed under science_news but DECLARES a different base.
    manifest = _write_manifest(
        tmp_path, {"science_news": _envelope(base="media_archive")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    with pytest.raises(SS.SourceSnapshotError, match="mismatched source"):
        SS.load_snapshot_for_bank("science_news")


def test_variant_selecting_wrong_base_snapshot_is_rejected(tmp_path, monkeypatch):
    # science_news_v2 -> base science_news, but the entry declares media_archive.
    manifest = _write_manifest(
        tmp_path, {"science_news": _envelope(base="media_archive")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    with pytest.raises(SS.SourceSnapshotError):
        SS.load_snapshot_for_bank("science_news_v2")


def test_missing_seed_source_is_rejected(tmp_path, monkeypatch):
    env = _envelope("science_news")
    del env["seed_source"]
    manifest = _write_manifest(tmp_path, {"science_news": env})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    with pytest.raises(SS.SourceSnapshotError, match="seed_source"):
        SS.load_snapshot_for_bank("science_news")


def test_missing_payload_key_is_rejected(tmp_path, monkeypatch):
    env = _envelope("science_news")
    del env["payload"]["seed_text"]
    manifest = _write_manifest(tmp_path, {"science_news": env})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    with pytest.raises(SS.SourceSnapshotError, match="missing key"):
        SS.load_snapshot_for_bank("science_news")


def test_altered_payload_sha_receipt_is_rejected(tmp_path, monkeypatch):
    env = _envelope("science_news")
    env["seed_receipt"]["payload_sha256"] = "deadbeef" * 8
    manifest = _write_manifest(tmp_path, {"science_news": env})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    with pytest.raises(SS.SourceSnapshotError, match="altered after capture"):
        SS.load_snapshot_for_bank("science_news")


def test_matching_payload_sha_receipt_passes(tmp_path, monkeypatch):
    env = _envelope("science_news")
    env["seed_receipt"]["payload_sha256"] = SS.snapshot_payload_sha256(_payload())
    manifest = _write_manifest(tmp_path, {"science_news": env})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    assert SS.load_snapshot_for_bank("science_news") is not None


# ---------------------------------------------------------------------------
# 5. cast_hints sidecar merge + path-pointer entry + cache busting
# ---------------------------------------------------------------------------

def test_top_level_cast_hints_merge_into_source_meta(tmp_path, monkeypatch):
    env = _envelope("shakespeare")
    env["source_meta"] = {}
    env["cast_hints"] = ["MACBETH", "BANQUO"]
    manifest = _write_manifest(tmp_path, {"shakespeare": env})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    snap = SS.load_snapshot_for_bank("shakespeare")
    assert snap.source_meta["cast_hints"] == ["MACBETH", "BANQUO"]


def test_existing_source_meta_cast_hints_not_clobbered(tmp_path, monkeypatch):
    env = _envelope("shakespeare")
    env["source_meta"] = {"cast_hints": ["DUNCAN"]}
    env["cast_hints"] = ["MACBETH"]
    manifest = _write_manifest(tmp_path, {"shakespeare": env})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    snap = SS.load_snapshot_for_bank("shakespeare")
    assert snap.source_meta["cast_hints"] == ["DUNCAN"]


def test_path_pointer_entry_resolves_relative_to_manifest_dir(tmp_path, monkeypatch):
    (tmp_path / "science_news_env.json").write_text(
        json.dumps(_envelope("science_news")), encoding="utf-8")
    manifest = _write_manifest(
        tmp_path, {"science_news": {"path": "science_news_env.json"}})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    snap = SS.load_snapshot_for_bank("science_news")
    assert snap is not None
    assert snap.seed_source == "science_rss_frozen"


def test_manifest_rewrite_is_seen_immediately(tmp_path, monkeypatch):
    manifest = _write_manifest(
        tmp_path, {"science_news": _envelope("science_news")}, allow_partial=True)
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    assert SS.load_snapshot_for_bank("media_archive") is None
    # A live re-pin of the manifest is honored on the next call (no stale cache).
    Path(manifest).write_text(
        json.dumps({"schema_version": 1, "allow_partial": True,
                    "snapshots": {"media_archive": _envelope("media_archive")}}),
        encoding="utf-8",
    )
    snap = SS.load_snapshot_for_bank("media_archive")
    assert snap is not None
    assert snap.base_source_bank_id == "media_archive"


# ---------------------------------------------------------------------------
# 6. _resolve_inputs integration -- the replay BYPASSES live sourcing
# ---------------------------------------------------------------------------

def test_resolve_inputs_replays_snapshot_and_bypasses_rss(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {"science_news": _envelope("science_news")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)

    def _boom(*_a, **_k):
        raise AssertionError("live RSS fetcher must not run under a snapshot")

    monkeypatch.setattr(SP, "resolve_fetcher", _boom)

    out = LSW._resolve_inputs(source_bank="science_news", target_words=120)
    assert out["seed_source"] == "science_rss_frozen"
    assert out["news_article"] == _payload()
    assert out["news_seed"] == "the frozen source seed text"
    assert out["source_meta"] == {"kind": "frozen_rss"}
    assert out["source_rights"] == {"license_label": "frozen"}


def test_resolve_inputs_variant_shares_base_snapshot(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {"science_news": _envelope("science_news")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    monkeypatch.setattr(
        SP, "resolve_fetcher",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("must not fetch under snapshot")),
    )
    out = LSW._resolve_inputs(source_bank="science_news_v2", target_words=120)
    assert out["source_bank"] == "science_news_v2"
    assert out["seed_source"] == "science_rss_frozen"


def test_resolve_inputs_propagates_base_mismatch_loud(tmp_path, monkeypatch):
    manifest = _write_manifest(
        tmp_path, {"science_news": _envelope(base="media_archive")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    with pytest.raises(SS.SourceSnapshotError):
        LSW._resolve_inputs(source_bank="science_news", target_words=120)


def test_resolve_inputs_without_snapshot_is_unaffected(tmp_path, monkeypatch):
    # No manifest -> the snapshot loader returns None -> the custom-premise
    # branch runs exactly as before (hermetic, no network).
    out = LSW._resolve_inputs(
        source_bank="science_news",
        custom_premise="a custom seed premise",
        target_words=120,
    )
    assert out["seed_source"] == "custom_premise"
    assert out["news_seed"] == "a custom seed premise"


def _no_fetch(*_a, **_k):
    raise AssertionError("live fetcher must not run under a snapshot")


def test_resolve_inputs_replay_warns_without_c7_seeds(tmp_path, monkeypatch, caplog):
    import logging
    manifest = _write_manifest(tmp_path, {"science_news": _envelope("science_news")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    monkeypatch.delenv("OTR_CAST_SEED", raising=False)
    monkeypatch.delenv("OTR_STYLE_SEED", raising=False)
    monkeypatch.setattr(SP, "resolve_fetcher", _no_fetch)
    with caplog.at_level(logging.WARNING):
        LSW._resolve_inputs(source_bank="science_news", target_words=120)
    assert any("without C7 seed pinning" in r.getMessage() for r in caplog.records)


def test_resolve_inputs_replay_quiet_with_c7_seeds(tmp_path, monkeypatch, caplog):
    import logging
    manifest = _write_manifest(tmp_path, {"science_news": _envelope("science_news")})
    monkeypatch.setenv(SS.SNAPSHOT_MANIFEST_ENV, manifest)
    monkeypatch.setenv("OTR_CAST_SEED", "42")
    monkeypatch.setenv("OTR_STYLE_SEED", "42")
    monkeypatch.setattr(SP, "resolve_fetcher", _no_fetch)
    with caplog.at_level(logging.WARNING):
        LSW._resolve_inputs(source_bank="science_news", target_words=120)
    assert not any("without C7 seed pinning" in r.getMessage() for r in caplog.records)
