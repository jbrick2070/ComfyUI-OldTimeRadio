"""S2 -- OpenRouter virtual catalog rows are enabled-gated.

Pins the contract: the two virtual rows (openrouter:slot-a|b) appear in
both writer dropdowns and pass validate_model_id Path 1 ONLY when remote
is enabled (OPENROUTER_API_KEY + OTR_ENABLE_OPENROUTER=1); when disabled
they are absent everywhere and validate raises cleanly -- with NO change
to the validator's admit-paths (FC4). The dropdown shows the named
handle, never the real model slug.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from nodes import _otr_model_catalog as cat
from nodes import _otr_openrouter_backend as orb
from nodes._otr_model_inputs import UnknownModelError


SLOT_A = "openrouter:slot-a"
SLOT_B = "openrouter:slot-b"


@pytest.fixture
def disabled(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "OTR_ENABLE_OPENROUTER"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "openai/gpt-4o")


# ---------------------------------------------------------------------------
# Disabled (default) -- rows absent everywhere
# ---------------------------------------------------------------------------


def test_rows_absent_from_by_repo_id_when_disabled(disabled):
    ids = cat._by_repo_id()
    assert SLOT_A not in ids
    assert SLOT_B not in ids


def test_rows_absent_from_dropdown_when_disabled(disabled, tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert all("openrouter:" not in c for c in choices)


def test_validate_raises_when_disabled(disabled):
    with pytest.raises(UnknownModelError):
        cat.validate_model_id(SLOT_A)
    with pytest.raises(UnknownModelError):
        cat.validate_model_id(SLOT_B)


def test_flag_without_key_stays_disabled(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    assert SLOT_A not in cat._by_repo_id()
    with pytest.raises(UnknownModelError):
        cat.validate_model_id(SLOT_A)


# ---------------------------------------------------------------------------
# Enabled -- rows present, validated, FC4 schema
# ---------------------------------------------------------------------------


def test_rows_present_in_by_repo_id_when_enabled(enabled):
    ids = cat._by_repo_id()
    assert SLOT_A in ids
    assert SLOT_B in ids


def test_rows_present_in_both_dropdowns_when_enabled(enabled, tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    assert SLOT_A in choices
    assert SLOT_B in choices
    # The same builder feeds BOTH writer dropdowns, so presence here ==
    # presence in creative + technical.


def test_validate_admits_unchanged_when_enabled(enabled):
    assert cat.validate_model_id(SLOT_A) == SLOT_A
    assert cat.validate_model_id(SLOT_B) == SLOT_B


def test_virtual_rows_match_fc4_schema(enabled):
    row = cat._by_repo_id()[SLOT_A]
    assert row.loader_backend == "openrouter_http"
    assert row.provider == "openrouter"
    assert row.vram_fit_tier == "PASS"
    assert row.approx_safetensors_gb == 0.0
    assert row.context_window == 8192
    assert row.requires_auth is False


def test_dropdown_label_has_no_not_downloaded_suffix(enabled, tmp_path):
    choices = cat.dropdown_choices(hub_root=tmp_path)
    # The remote handle is shown clean -- it is not a local download.
    assert SLOT_A in choices
    assert (SLOT_A + cat.NOT_DOWNLOADED_SUFFIX) not in choices


def test_dropdown_never_shows_real_slug(enabled, tmp_path):
    """The named handle is shown; the operator's real model slug (bound
    in env) must never leak into the dropdown."""
    choices = cat.dropdown_choices(hub_root=tmp_path)
    joined = " ".join(choices)
    assert "anthropic/claude-3.5-sonnet" not in joined
    assert "openai/gpt-4o" not in joined


# ---------------------------------------------------------------------------
# FC4: no validator surgery needed -- the id is structurally safe
# ---------------------------------------------------------------------------


def test_structural_reject_passes_openrouter_ids():
    assert cat._structural_reject(SLOT_A) is None
    assert cat._structural_reject(SLOT_B) is None


def test_gated_set_unaffected_by_virtual_rows(enabled):
    """GATED_CURATED_MODELS is keyed off the real gated set; the
    requires_auth=False virtual rows must never appear there."""
    assert SLOT_A not in cat.GATED_CURATED_MODELS
    assert SLOT_B not in cat.GATED_CURATED_MODELS


# ===========================================================================
# S1 -- openrouter_catalog_dropdown_choices (the slot-A/B slug pickers)
# ===========================================================================
#
# Contract (2026-06-01 plan SS4 + SS7): the catalog feeds ONLY the two slot
# pickers, never creative/technical; disabled -> sentinel; enabled -> tiered,
# filtered slug list led by the recommended default; per-slot filters narrow
# only their slot. INPUT_TYPES-safe (cache reads only, no network).

# Recommended defaults must match the backend constants; include both so each
# slot's lead is a real cache hit. One non-JSON model proves REQUIRE_JSON.
_CATALOG_MODELS = [
    {"id": "anthropic/claude-opus-4.8", "name": "Claude Opus 4.8",
     "provider": "anthropic", "created": 400, "context_length": 200000,
     "pricing": {"prompt": "0.000005", "completion": "0.000025"},
     "supported_parameters": ["response_format", "structured_outputs"],
     "supports_json": True},
    {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro",
     "provider": "deepseek", "created": 300, "context_length": 128000,
     "pricing": {"prompt": "0.00000043", "completion": "0.00000087"},
     "supported_parameters": ["response_format"], "supports_json": True},
    {"id": "openai/gpt-4o", "name": "GPT-4o",
     "provider": "openai", "created": 200, "context_length": 128000,
     "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
     "supported_parameters": ["response_format", "structured_outputs"],
     "supports_json": True},
    {"id": "x-ai/grok-2-mini", "name": "Grok 2 Mini",
     "provider": "x-ai", "created": 100, "context_length": 32768,
     "pricing": {"prompt": "0.000001", "completion": "0.000002"},
     "supported_parameters": ["tools"], "supports_json": False},
]


def test_slim_model_preserves_reasoning_capability_contract():
    slim = orb._slim_model({
        "id": "aion-labs/aion-3.0-mini",
        "reasoning": {
            "supported_efforts": ["high", "medium", "low"],
            "default_effort": "medium",
            "default_enabled": True,
            "mandatory": True,
            "supports_max_tokens": False,
        },
    })

    assert slim["reasoning"] == {
        "supported_efforts": ["high", "medium", "low"],
        "default_effort": "medium",
        "default_enabled": True,
        "mandatory": True,
        "supports_max_tokens": False,
    }

_FILTER_ENV = (
    "OTR_OPENROUTER_MODEL_ALLOWLIST", "OTR_OPENROUTER_MODEL_DENYLIST",
    "OTR_OPENROUTER_PROVIDER_FILTER", "OTR_OPENROUTER_SLOT_A_REQUIRE_JSON",
    "OTR_OPENROUTER_SLOT_B_REQUIRE_JSON", "OTR_OPENROUTER_FAVORITES",
    "OTR_OPENROUTER_SLOT_A_DEFAULT", "OTR_OPENROUTER_SLOT_B_DEFAULT",
    # 2026-08-07: both of these change list LENGTH and ORDER, so without them
    # the fixture never actually establishes the "no filter env" state it
    # advertises -- and an exact-ordered-list assertion would be asserting
    # against whatever leaked in from the host.
    "OTR_OPENROUTER_FULL_CATALOG", "OTR_OPENROUTER_ALLOW_NONTEXT",
)


def _write_cache(cache_dir, models, *, source="live"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": orb.CATALOG_SCHEMA_VERSION,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": source, "count": len(models), "models": models,
    }
    (cache_dir / "openrouter_models.json").write_text(
        json.dumps(payload), encoding="utf-8")


@pytest.fixture
def enabled_cached(monkeypatch, tmp_path):
    """Remote enabled + a populated catalog cache in tmp_path. Clears any
    filter env that might leak from the live host (remote is enabled there)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-opus-4.8")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "deepseek/deepseek-v4-pro")
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))
    for k in _FILTER_ENV:
        monkeypatch.delenv(k, raising=False)
    _write_cache(tmp_path, _CATALOG_MODELS)
    return tmp_path


# --- disabled -> sentinel --------------------------------------------------


def test_slot_picker_sentinel_when_disabled(disabled):
    assert cat.openrouter_catalog_dropdown_choices("a") == [cat.OPENROUTER_ENABLE_SENTINEL]
    assert cat.openrouter_catalog_dropdown_choices("b") == [cat.OPENROUTER_ENABLE_SENTINEL]


def test_slot_picker_invalid_slot_raises():
    with pytest.raises(ValueError):
        cat.openrouter_catalog_dropdown_choices("c")


# --- enabled -> catalog, led by the recommended default --------------------


def test_slot_picker_shows_curated_set_when_enabled(enabled_cached):
    """The default view is the lead + the curated aliases -- and NOTHING from
    the cache. The 2026-08-07 curation deleted the 'recent' tier, so cached
    rows no longer leak into the default dropdown; asserting their ABSENCE is
    the point, because a test that only checked presence would have stayed
    green while the tier silently returned."""
    a = cat.openrouter_catalog_dropdown_choices("a")
    assert a[0] == cat.OPENROUTER_ENABLE_SENTINEL  # BUG-LOCAL-400
    assert a[1] == orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT
    for slug in cat.OPENROUTER_CURATED_ALIASES:
        assert slug in a
    # Cache-only rows must NOT appear without FULL_CATALOG / narrowing.
    for cache_only in ("openai/gpt-4o", "x-ai/grok-2-mini"):
        assert cache_only not in a


def test_slot_a_lead_is_recommended_creative(enabled_cached):
    # BUG-LOCAL-400: choices[0] is the enable-sentinel (off default); the first
    # REAL slug is the recommended creative default.
    a = cat.openrouter_catalog_dropdown_choices("a")
    assert a[0] == cat.OPENROUTER_ENABLE_SENTINEL
    assert a[1] == orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT


def test_no_pinned_concrete_contenders_remain(enabled_cached):
    """2026-08-07 curation: the pinned-creative-contender MECHANISM is gone.

    It carried `tencent/hy3:free` (a promo slug that stopped resolving when the
    promo ended) and `aion-labs/aion-3.0-mini` (dropped by operator ruling: it
    never won a model contest -- none was ever run -- and its live record is
    PBUG-20260713-20 plus an episode-aborting finish_reason=length). With both
    gone the mechanism had nothing to carry, so the constant, its filter path
    and its cold-cache branch were deleted rather than left empty."""
    for slot in ("a", "b"):
        choices = cat.openrouter_catalog_dropdown_choices(slot)
        assert "tencent/hy3:free" not in choices
        assert "aion-labs/aion-3.0-mini" not in choices
    assert not hasattr(cat, "_PINNED_CREATIVE_CONTENDER_ROWS")


def test_slot_b_lead_is_recommended_technical(enabled_cached):
    # BUG-LOCAL-400: enable-sentinel leads; recommended technical is first real.
    b = cat.openrouter_catalog_dropdown_choices("b")
    assert b[0] == cat.OPENROUTER_ENABLE_SENTINEL
    assert b[1] == orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT


def test_per_slot_default_override_leads_when_present(enabled_cached, monkeypatch):
    monkeypatch.setenv("OTR_OPENROUTER_SLOT_A_DEFAULT", "openai/gpt-4o")
    # BUG-LOCAL-400: sentinel is choices[0]; the override leads the REAL slugs.
    a = cat.openrouter_catalog_dropdown_choices("a")
    assert a[0] == cat.OPENROUTER_ENABLE_SENTINEL
    assert a[1] == "openai/gpt-4o"


def test_per_slot_default_override_ignored_when_absent(enabled_cached, monkeypatch):
    """Override set but not in the cache -> fall back to the recommended
    constant for the dropdown lead (discovery can't show what isn't cached;
    S3 still honours an explicit saved slug at resolution)."""
    monkeypatch.setenv("OTR_OPENROUTER_SLOT_A_DEFAULT", "ghost/not-in-cache")
    # BUG-LOCAL-400: sentinel leads; recommended is the first real slug.
    a = cat.openrouter_catalog_dropdown_choices("a")
    assert a[0] == cat.OPENROUTER_ENABLE_SENTINEL
    assert a[1] == orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT


# --- the catalog NEVER appears in creative/technical -----------------------


def test_catalog_absent_from_creative_technical(enabled_cached):
    choices = cat.dropdown_choices(hub_root=enabled_cached)
    joined = " ".join(choices)
    for mid in ("anthropic/claude-opus-4.8", "deepseek/deepseek-v4-pro",
                "openai/gpt-4o", "x-ai/grok-2-mini"):
        assert mid not in joined
    # the routing handles are still present (they are not catalog slugs)
    assert SLOT_A in choices and SLOT_B in choices


# --- per-slot REQUIRE_JSON narrows ONLY its slot ---------------------------


def test_require_json_narrows_only_that_slot(enabled_cached, monkeypatch):
    """REQUIRE_JSON narrows the CACHED CATALOG rows for its slot only.

    FULL_CATALOG=1 is required to make cached rows visible at all: the
    2026-08-07 curation removed the 'recent' tier, so without the opt-in
    neither id below appears in either slot and the assertions would pass
    vacuously."""
    monkeypatch.setenv("OTR_OPENROUTER_FULL_CATALOG", "1")
    monkeypatch.setenv("OTR_OPENROUTER_SLOT_B_REQUIRE_JSON", "1")
    a = cat.openrouter_catalog_dropdown_choices("a")
    b = cat.openrouter_catalog_dropdown_choices("b")
    # grok-2-mini has supports_json False: present in A (no filter), gone in B.
    assert "x-ai/grok-2-mini" in a
    assert "x-ai/grok-2-mini" not in b
    # JSON-capable models remain in both.
    assert "openai/gpt-4o" in a and "openai/gpt-4o" in b


def test_require_json_does_not_narrow_curated_aliases(enabled_cached, monkeypatch):
    """PINNED BEHAVIOUR, not an oversight: the curated aliases bypass
    _filter_catalog_models, so REQUIRE_JSON does not remove them. Filtering
    them against a cache that may be cold would empty the curated block in
    exactly the state it exists to survive. The slot-B tooltip and
    docs/openrouter-setup.md say so explicitly."""
    monkeypatch.setenv("OTR_OPENROUTER_SLOT_B_REQUIRE_JSON", "1")
    b = cat.openrouter_catalog_dropdown_choices("b")
    for slug in cat.OPENROUTER_CURATED_ALIASES:
        assert slug in b


# --- allow / deny / provider filters ---------------------------------------


def test_allowlist_narrows(enabled_cached, monkeypatch):
    monkeypatch.setenv("OTR_OPENROUTER_MODEL_ALLOWLIST",
                       "anthropic/claude-opus-4.8,openai/gpt-4o")
    a = cat.openrouter_catalog_dropdown_choices("a")
    # BUG-LOCAL-400: the enable-sentinel is always present alongside the filtered set.
    # The recommended default is ALSO always present -- the tier-1 lead is added
    # before any narrowing precisely so the slot's own default value stays
    # selectable (see the docstring). That used to be invisible here because the
    # default was `anthropic/claude-opus-4.8`, which the allowlist already
    # contained; chunk B made the default an alias, so it now shows up plainly.
    assert set(a) == {cat.OPENROUTER_ENABLE_SENTINEL,
                      orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT,
                      "anthropic/claude-opus-4.8", "openai/gpt-4o"}


def test_denylist_removes(enabled_cached, monkeypatch):
    """Visible-before / removed-after. Asserting only the absence would pass
    vacuously after the recent-tier cut, because the target is not in the
    default view at all -- the test would keep passing with the denylist
    entirely broken."""
    monkeypatch.setenv("OTR_OPENROUTER_FULL_CATALOG", "1")
    assert "openai/gpt-4o" in cat.openrouter_catalog_dropdown_choices("a")
    monkeypatch.setenv("OTR_OPENROUTER_MODEL_DENYLIST", "openai/gpt-4o")
    assert "openai/gpt-4o" not in cat.openrouter_catalog_dropdown_choices("a")


def test_provider_filter_narrows(enabled_cached, monkeypatch):
    monkeypatch.setenv("OTR_OPENROUTER_PROVIDER_FILTER", "anthropic")
    a = cat.openrouter_catalog_dropdown_choices("a")
    # BUG-LOCAL-400: sentinel leads even a single-provider filtered list, and the
    # tier-1 lead follows it -- always offered so the slot default stays
    # selectable through a filter. Then the filtered catalog itself.
    assert a == [cat.OPENROUTER_ENABLE_SENTINEL,
                 orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT,
                 "anthropic/claude-opus-4.8"]


def test_favorites_float_after_lead(enabled_cached, monkeypatch):
    monkeypatch.setenv("OTR_OPENROUTER_FAVORITES", "openai/gpt-4o")
    a = cat.openrouter_catalog_dropdown_choices("a")
    # BUG-LOCAL-400: sentinel leads; then the recommended default; then favorite.
    assert a[0] == cat.OPENROUTER_ENABLE_SENTINEL
    assert a[1] == orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT
    assert a[2] == "openai/gpt-4o"


def test_default_view_is_exactly_sentinel_lead_and_curated_aliases(enabled_cached):
    """THE contract this curation exists to create, asserted as an EXACT
    ORDERED list rather than a count: none of it from the cache. Before
    2026-08-07 this was 22 entries, of which 8 were whatever the disk cache
    happened to hold.

    The expectation must DE-DUPLICATE, because since chunk B (2026-08-09) slot
    A's lead is itself a curated pointer (`~anthropic/claude-opus-latest`) and
    the builder adds each id once. Slot B's lead is still a concrete pin, so it
    is a genuinely extra entry there. That asymmetry is the point of building
    `expected` this way rather than hardcoding two lengths.
    """
    for slot, lead in (
        ("a", orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT),
        ("b", orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT),
    ):
        expected = [cat.OPENROUTER_ENABLE_SENTINEL, lead]
        expected += [mid for mid in cat.OPENROUTER_CURATED_ALIASES
                     if mid not in expected]
        assert cat.openrouter_catalog_dropdown_choices(slot) == expected


def test_full_catalog_optin_restores_cache_rows(enabled_cached, monkeypatch):
    """Absent-before / present-after on a CACHE-ONLY concrete id. A superset
    assertion alone would pass even if FULL_CATALOG did nothing."""
    pruned = cat.openrouter_catalog_dropdown_choices("a")
    assert "openai/gpt-4o" not in pruned
    monkeypatch.setenv("OTR_OPENROUTER_FULL_CATALOG", "1")
    full = cat.openrouter_catalog_dropdown_choices("a")
    assert "openai/gpt-4o" in full
    assert set(pruned).issubset(set(full))
    assert "~anthropic/claude-opus-latest" in full


# --- enabled but cold/empty cache ------------------------------------------


def test_empty_cache_when_enabled_leads_and_points_at_refresh(monkeypatch, tmp_path):
    """Cold cache: the slot's default stays selectable and the operator is
    pointed at a refresh. The curated aliases are deliberately NOT listed here
    -- the cold branch is a 'go refresh' signal, not a browse view."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))  # no cache file
    for k in _FILTER_ENV:
        monkeypatch.delenv(k, raising=False)
    a = cat.openrouter_catalog_dropdown_choices("a")
    assert a == [cat.OPENROUTER_ENABLE_SENTINEL,
                 orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT,
                 cat.OPENROUTER_EMPTY_CACHE_SENTINEL]
    b = cat.openrouter_catalog_dropdown_choices("b")
    assert b == [cat.OPENROUTER_ENABLE_SENTINEL,
                 orb.OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT,
                 cat.OPENROUTER_EMPTY_CACHE_SENTINEL]


# --- INPUT_TYPES safety: the builder never touches the network -------------


def test_slot_picker_is_network_free(enabled_cached, monkeypatch):
    def _boom(**kw):
        raise AssertionError("network fetch fired from a dropdown build")
    monkeypatch.setattr(orb, "_fetch_models_json", _boom)
    # Must build purely from the disk cache -- no fetch, no raise. The real
    # assertion is the _boom monkeypatch not firing; the membership check just
    # proves a REAL list came back rather than an empty/sentinel-only degrade.
    # It deliberately checks the recommended default rather than a specific
    # cached id: in the default warm view cache rows are pruned, so naming one
    # would couple this test to the curation instead of to the network.
    out = cat.openrouter_catalog_dropdown_choices("a")
    assert orb.OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT in out
    assert len(out) > 2


# --- BUG-LOCAL-400: enable-sentinel stays a valid choice when ENABLED -------


def test_enable_sentinel_leads_when_enabled_bug400(enabled_cached):
    """BUG-LOCAL-400: with the lane ENABLED and the catalog populated, the
    enable-sentinel MUST remain choices[0] (the slot's saved 'off' value + its
    INPUT_TYPES default). If it falls out of the list, a saved workflow storing
    the sentinel fails ComfyUI COMBO validation and every output is dropped."""
    for slot in ("a", "b"):
        choices = cat.openrouter_catalog_dropdown_choices(slot)
        assert choices[0] == cat.OPENROUTER_ENABLE_SENTINEL
        assert choices.count(cat.OPENROUTER_ENABLE_SENTINEL) == 1
