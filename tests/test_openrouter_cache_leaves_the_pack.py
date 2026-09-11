"""The OpenRouter discovery cache must survive a registry update.

It lived at `<repo>/models/openrouter_models.json` -- INSIDE the installed pack --
so every registry update replaced the directory and threw the warmed catalog away.
The user then silently ran at `DEFAULT_CONTEXT_WINDOW` (8192) instead of each
model's real window, while the empty-cache sentinel told them to "run
refresh_catalog_cache" using a script that was not in the bundle either until
2026-09-11.

It now resolves under `otr_shared_cache_dir()`, inside the user's OWN ComfyUI
output tree. That tier's contract is "a cache entry is NEVER the only copy",
which this satisfies exactly: the catalog is rebuilt wholesale by one idempotent
`refresh_catalog_cache()` call and a missing one degrades to an empty catalog.
It is also not janitor-swept -- `_otr_janitor` sweeps `episodes/_shared/tmp` ONLY
and refuses any other root.

WHY THE SECOND HALF OF THIS FILE EXISTS. `_catalog_cache_path()` is read while
node dropdowns are built, where nothing may raise. The old version was pure Path
arithmetic and could not throw; the new one consults the output-tree contract and
can. `load_catalog_cache()` called it OUTSIDE its own try, so a resolver failure
would have propagated straight into `INPUT_TYPES`. Found by the r1 that designed
the move, before it shipped.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from nodes import _otr_openrouter_backend as backend


# ---------------------------------------------------------------------------
# 1. It leaves the pack
# ---------------------------------------------------------------------------
def test_the_DEFAULT_path_is_outside_the_installed_pack(monkeypatch):
    """The defect, inverted. No env override -- the shipping user's case, which
    is the branch the suite never exercised."""
    monkeypatch.delenv("OTR_OPENROUTER_CACHE_DIR", raising=False)

    resolved = Path(backend._catalog_cache_path()).resolve()
    pack_dir = Path(backend.__file__).resolve().parent.parent

    assert pack_dir not in resolved.parents, (
        "the cache is inside the installed pack at %s; a registry update "
        "replaces that directory and wipes it" % resolved)


def test_the_DEFAULT_path_lands_in_the_shared_cache_tier(monkeypatch):
    """And specifically in the tier whose contract it satisfies."""
    monkeypatch.delenv("OTR_OPENROUTER_CACHE_DIR", raising=False)

    resolved = Path(backend._catalog_cache_path()).resolve()
    parts = [p.lower() for p in resolved.parts]

    for expected in ("otr", "episodes", "_shared", "cache", "openrouter"):
        assert expected in parts, (
            "%r missing from %s -- the cache is not in the shared cache tier"
            % (expected, resolved))
    assert resolved.name == "openrouter_models.json"


def test_the_ENV_OVERRIDE_still_wins(monkeypatch, tmp_path):
    """Tests and relocation depend on it, and the move must not break it."""
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))

    resolved = Path(backend._catalog_cache_path()).resolve()
    assert resolved.parent == tmp_path.resolve()
    assert resolved.name == "openrouter_models.json"


# ---------------------------------------------------------------------------
# 2. It cannot break a dropdown
# ---------------------------------------------------------------------------
def test_a_RAISING_resolver_never_reaches_INPUT_TYPES(monkeypatch):
    """THE REGRESSION GUARD. The path is resolved while node dropdowns build,
    and the new resolver consults the output-tree contract, so it CAN throw
    where the old pure-Path version could not."""
    import nodes._otr_paths as paths

    def boom():
        raise RuntimeError("output tree contract refused")

    monkeypatch.delenv("OTR_OPENROUTER_CACHE_DIR", raising=False)
    monkeypatch.setattr(paths, "otr_shared_cache_dir", boom, raising=False)

    # The resolver itself absorbs it and degrades to the old in-pack location.
    resolved = backend._catalog_cache_path()
    assert resolved.name == "openrouter_models.json"


def test_load_catalog_cache_survives_a_resolver_that_explodes(monkeypatch):
    """Belt and braces: even if the resolver did propagate, the reader must
    return a well-formed empty catalog rather than raise. `load_catalog_cache`
    resolved the path OUTSIDE its own try until 2026-09-11."""
    monkeypatch.setattr(
        backend, "_catalog_cache_path",
        lambda: (_ for _ in ()).throw(RuntimeError("resolver exploded")))

    catalog = backend.load_catalog_cache()

    assert isinstance(catalog, dict)
    assert catalog.get("models") == []
    assert catalog.get("count") == 0
    assert catalog.get("source") in ("missing", "corrupt"), catalog


def test_a_real_cache_still_reads_back(monkeypatch, tmp_path):
    """The move must not break the ordinary path."""
    monkeypatch.setenv("OTR_OPENROUTER_CACHE_DIR", str(tmp_path))
    payload = {
        "schema_version": backend.CATALOG_SCHEMA_VERSION,
        "fetched_at": "2026-09-11T00:00:00Z",
        "source": "live",
        "count": 1,
        "models": [{"id": "vendor/model-a"}],
    }
    io.open(tmp_path / "openrouter_models.json", "w", encoding="utf-8").write(
        json.dumps(payload))

    catalog = backend.load_catalog_cache()
    assert catalog["count"] == 1
    assert catalog["models"][0]["id"] == "vendor/model-a"
