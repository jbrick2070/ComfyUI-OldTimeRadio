"""Cloud media billing cache -- S0 chunk 2.

Global, idempotent, cross-episode (pass04 sec 4): identical requests
never re-bill. Layout under resolve_cache_root():

    <root>/<key[:2]>/<key>/manifest.json     (written LAST = commit mark)
    <root>/<key[:2]>/<key>/asset.<ext>

Key discipline: RequestCacheKey is built from REQUEST material only --
computable BEFORE submit. content_sha256 (the canonicalized OUTPUT
hash) lives in the manifest/ledger, never in the lookup key.

Fail-closed: an entry with a missing/mismatched asset is a MISS (logged,
quarantined by rename), never served. Stores are atomic (temp+rename;
manifest written last). Assets are COPIED into episode dirs (no
hardlinks -- Windows edge cases; GPT R4 CUT-2).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .cloud_media_backend import (
    CloudErrorCode,
    CloudMediaError,
    resolve_cache_root,
)

__all__ = [
    "RequestCacheKey",
    "CachedAsset",
    "streaming_sha256",
    "cache_lookup",
    "cache_store",
    "needs_recanonicalize",
    "copy_into_episode",
]

_HASH_CHUNK = 1024 * 1024


def streaming_sha256(path: Path) -> str:
    """Chunked SHA-256 -- never whole-media-in-memory (GPT R3 S-9)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RequestCacheKey (pass04 sec 4 -- pre-submit, request material only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestCacheKey:
    row_id: str
    resolved_slug: str
    params: tuple  # sorted (name, json-str) pairs -- see build()
    input_hashes: tuple  # sorted (input_name, sha256) pairs
    seed: Optional[int]  # ONLY when the pinned schema says seed_supported
    output_contract_version: int
    adapter_version: int
    canonicalizer_version: int
    schema_version: str

    @staticmethod
    def build(
        row_id: str,
        resolved_slug: str,
        params: dict,
        input_hashes: dict,
        *,
        seed: Optional[int],
        output_contract_version: int,
        adapter_version: int,
        canonicalizer_version: int,
        schema_version: str,
    ) -> "RequestCacheKey":
        norm_params = tuple(sorted(
            (str(k), json.dumps(v, sort_keys=True, ensure_ascii=True))
            for k, v in params.items()
        ))
        norm_inputs = tuple(sorted(
            (str(k), str(v)) for k, v in input_hashes.items()
        ))
        return RequestCacheKey(
            row_id=row_id, resolved_slug=resolved_slug, params=norm_params,
            input_hashes=norm_inputs, seed=seed,
            output_contract_version=output_contract_version,
            adapter_version=adapter_version,
            canonicalizer_version=canonicalizer_version,
            schema_version=schema_version,
        )

    def hexdigest(self) -> str:
        payload = json.dumps({
            "row_id": self.row_id,
            "resolved_slug": self.resolved_slug,
            "params": self.params,
            "input_hashes": self.input_hashes,
            "seed": self.seed,
            "output_contract_version": self.output_contract_version,
            "adapter_version": self.adapter_version,
            "canonicalizer_version": self.canonicalizer_version,
            "schema_version": self.schema_version,
        }, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedAsset:
    path: Path
    manifest: dict


# ---------------------------------------------------------------------------
# Per-key locks (GPT R4/R3: double-checked validation under the key lock)
# ---------------------------------------------------------------------------

_KEY_LOCKS_GUARD = threading.Lock()
_KEY_LOCKS: dict = {}


def _key_lock(key_hex: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key_hex)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key_hex] = lock
        return lock


def _entry_dir(key_hex: str, cache_root: Optional[Path]) -> Path:
    root = cache_root or resolve_cache_root()
    return root / key_hex[:2] / key_hex


_REQUIRED_MANIFEST_FIELDS = (
    "content_sha256", "asset_filename", "media_type",
    "canonicalizer_version", "row_id", "created_at",
)


def _load_valid_manifest(entry: Path) -> Optional[dict]:
    mpath = entry / "manifest.json"
    if not mpath.is_file():
        return None
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if any(f not in manifest for f in _REQUIRED_MANIFEST_FIELDS):
        return None
    return manifest


def _quarantine(entry: Path, reason: str) -> None:
    target = entry.with_name(entry.name + f".corrupt.{int(time.time())}")
    try:
        entry.rename(target)
        print(f"[cloud_media_cache] quarantined {entry.name}: {reason}")
    except OSError as exc:  # never let quarantine kill a render
        print(f"[cloud_media_cache] quarantine FAILED for {entry.name}: "
              f"{reason} ({exc})")


def cache_lookup(key: RequestCacheKey,
                 cache_root: Optional[Path] = None) -> Optional[CachedAsset]:
    """Validated hit or None. Integrity: manifest fields + asset present +
    streaming sha matches manifest.content_sha256. Invalid = quarantine +
    miss (fail-closed; a wrong paid asset must never be served)."""
    key_hex = key.hexdigest()
    with _key_lock(key_hex):
        entry = _entry_dir(key_hex, cache_root)
        manifest = _load_valid_manifest(entry)
        if manifest is None:
            if entry.exists():
                _quarantine(entry, "missing/invalid manifest")
            return None
        asset = entry / manifest["asset_filename"]
        if not asset.is_file():
            _quarantine(entry, "asset file missing")
            return None
        if streaming_sha256(asset) != manifest["content_sha256"]:
            _quarantine(entry, "content sha mismatch")
            return None
        return CachedAsset(path=asset, manifest=manifest)


def cache_store(key: RequestCacheKey, canonical_path: Path, manifest: dict,
                cache_root: Optional[Path] = None) -> CachedAsset:
    """Atomic store of a CANONICALIZED asset: copy to temp, rename asset
    into place, write manifest LAST (temp+rename = commit marker).
    manifest must NOT pre-fill integrity fields -- they are computed here
    (sha of the canonicalized bytes; Gemini R3 S-3)."""
    canonical_path = Path(canonical_path)
    if not canonical_path.is_file():
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"canonical asset missing: {canonical_path}")
    key_hex = key.hexdigest()
    with _key_lock(key_hex):
        entry = _entry_dir(key_hex, cache_root)
        entry.mkdir(parents=True, exist_ok=True)
        asset_name = "asset" + canonical_path.suffix.lower()
        final_asset = entry / asset_name
        tmp_asset = entry / (asset_name + ".tmp")
        shutil.copyfile(canonical_path, tmp_asset)
        os.replace(tmp_asset, final_asset)

        full = dict(manifest)
        full["asset_filename"] = asset_name
        full["content_sha256"] = streaming_sha256(final_asset)
        full["created_at"] = time.time()
        full["cache_key"] = key_hex
        missing = [f for f in _REQUIRED_MANIFEST_FIELDS if f not in full]
        if missing:
            raise CloudMediaError(CloudErrorCode.MALFORMED_CONFIG,
                                  f"manifest missing fields: {missing}")
        tmp_manifest = entry / "manifest.json.tmp"
        tmp_manifest.write_text(
            json.dumps(full, sort_keys=True, ensure_ascii=True, indent=1),
            encoding="utf-8")
        os.replace(tmp_manifest, entry / "manifest.json")
        return CachedAsset(path=final_asset, manifest=full)


def needs_recanonicalize(manifest: dict, current_canonicalizer_version: int,
                         *, must_strip_audio: bool = False) -> bool:
    """Stale-entry policy (GPT R3 #10): older canonicalizer version, or a
    strip-required row whose manifest lacks the strip proof."""
    if int(manifest.get("canonicalizer_version", -1)) != int(current_canonicalizer_version):
        return True
    if must_strip_audio and not manifest.get("audio_stripped_proof"):
        return True
    return False


def copy_into_episode(cached: CachedAsset, episode_dir: Path,
                      filename: str) -> Path:
    """Deliverable invariant: episode assets live IN the episode dir.
    Copy (never hardlink), atomic via temp+rename, sha re-verified."""
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    dest = episode_dir / filename
    tmp = episode_dir / (filename + ".tmp")
    shutil.copyfile(cached.path, tmp)
    if streaming_sha256(tmp) != cached.manifest["content_sha256"]:
        tmp.unlink(missing_ok=True)
        raise CloudMediaError(CloudErrorCode.CORRUPT_OUTPUT,
                              f"copy verification failed for {filename}")
    os.replace(tmp, dest)
    return dest
