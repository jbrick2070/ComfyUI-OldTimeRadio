"""Where the writer LLM lives as REAL FILES -- the ComfyUI ``LLM`` model folder.

WHY THIS EXISTS (operator, 2026-09-25: "real folder", "what's best practice").
The writer used to download through ``snapshot_download(cache_dir=...)`` into
the Hugging Face hub cache, whose ``blobs`` + ``snapshots`` layout is built from
symlinks. On Windows without Developer Mode that prints a warning on every first
download and can duplicate multi-GB files, and the cache sits somewhere a
ComfyUI user never looks. Well-behaved ComfyUI packs keep model weights in a
model category under ComfyUI's models tree, as ordinary files, relocatable with
``extra_model_paths.yaml``; Kokoro and the visual assets already do.

THE RULE, in one place:
* New writer downloads go to ``<first LLM path>/<org>--<name>/`` through
  ``snapshot_download(local_dir=...)``: real files, no symlinks.
* The ``LLM`` category is registered with default ``_models_root()/LLM``. A
  user's ``extra_model_paths.yaml`` loads BEFORE custom nodes, so an ``LLM:``
  entry there registers first and stays first -- the visual assets' "native
  order wins" rule.
* A model already complete in the hub cache is still found and still loads.
  Nothing is migrated, moved or deleted, ever. This module only adds a place to
  look; ``_otr_hf_env`` and the catalog keep their hub readers.
* A plain folder counts as COMPLETE only when (a) the pack's own receipt,
  written after ``snapshot_download`` returned, lists files that are all
  present at their recorded sizes, or (b) with no receipt (a folder placed by
  hand), a weight index is present and every shard it names is on disk.
  Neither means incomplete, and re-running the download resumes it. The receipt
  exists because huggingface_hub downloads files concurrently: a first shard
  can land before its index, and "any nonzero weight file" would then call a
  half-finished model complete and short-circuit the download that would
  finish it (Grok QA, 2026-09-25).

UTF-8, no BOM, ASCII-only source. Stdlib only; ``folder_paths`` and
``_models_root`` are imported lazily and guarded, so ``_otr_hf_env`` -- which
must import before the catalog -- can depend on this module.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.llm_folder")

#: The ComfyUI model category. Shared with other LLM packs on purpose: that is
#: what makes the folder a ComfyUI convention rather than ours alone. Their bare
#: folder names (no ``--``) are ignored by ``repo_id_from_folder``.
LLM_CATEGORY = "LLM"

#: The pack's completion receipt, written inside the model folder.
RECEIPT_NAME = ".otr_download_complete.json"
RECEIPT_SCHEMA = "otr_llm_download_v1"

_WEIGHT_SUFFIXES = (".safetensors", ".bin")
_WEIGHT_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #


def folder_name(repo_id: str) -> str:
    """``google/gemma-4-12b-it`` -> ``google--gemma-4-12b-it``. The org stays in
    the name so two orgs' same-named models never share a folder."""
    return str(repo_id).strip().replace("/", "--")


def repo_id_from_folder(name: str) -> Optional[str]:
    """The inverse of :func:`folder_name`, or None for a folder that is not ours
    to read (a bare name another LLM pack placed, a hidden folder)."""
    if not name or name.startswith(".") or "--" not in name:
        return None
    org, _, rest = name.partition("--")
    if not org or not rest:
        return None
    return "%s/%s" % (org, rest)


# --------------------------------------------------------------------------- #
# roots
# --------------------------------------------------------------------------- #


def default_llm_root() -> Optional[Path]:
    """``_models_root()/LLM``, or None when this box honestly has no models root."""
    try:
        try:
            from ._otr_models_root import _models_root
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_models_root import _models_root  # type: ignore
        return Path(_models_root()) / LLM_CATEGORY
    except Exception as exc:  # noqa: BLE001 -- ModelsRootUnresolved and friends
        log.info("[OTR.llm_folder] no models root, so no LLM folder: %s", exc)
        return None


def register_llm_category() -> Optional[Path]:
    """Register the ``LLM`` category with ComfyUI at pack load. Never raises.

    Appends (``is_default=False``): a user's ``extra_model_paths.yaml`` entry is
    already registered by then and keeps its place in front."""
    root = default_llm_root()
    if root is None:
        return None
    try:
        import folder_paths  # ComfyUI's own; absent under bare pytest
        folder_paths.add_model_folder_path(LLM_CATEGORY, str(root))
    except Exception as exc:  # noqa: BLE001 -- no ComfyUI, nothing to register
        log.debug("[OTR.llm_folder] LLM category not registered: %s", exc)
    return root


def llm_roots() -> List[Path]:
    """Every folder a writer model may live in as real files, in priority order.

    ``OTR_LLM_DIR`` pins one explicitly (tests, headless runs). Under
    ``OTR_TEST_MODE=1`` without that pin the answer is EMPTY, so a test that
    builds a hub fixture is never contaminated by real models on the box --
    the same hermetic rule the ledger's mtime walker follows. Otherwise:
    ComfyUI's registered ``LLM`` paths, else the default root."""
    pinned = str(otr_env.get("OTR_LLM_DIR", "") or "").strip()
    if pinned:
        return [Path(pinned)]
    if otr_env.get("OTR_TEST_MODE") == "1":
        return []
    try:
        import folder_paths
        paths = [Path(p) for p in folder_paths.get_folder_paths(LLM_CATEGORY)]
        if paths:
            return paths
    except Exception:  # noqa: BLE001 -- no ComfyUI, or the category is unregistered
        pass
    root = default_llm_root()
    return [root] if root is not None else []


def download_destination(repo_id: str) -> Optional[Path]:
    """Where a NEW download of ``repo_id`` goes: the first ``LLM`` path. None
    when there is no root at all, and the caller keeps the hub cache."""
    roots = llm_roots()
    return (roots[0] / folder_name(repo_id)) if roots else None


# --------------------------------------------------------------------------- #
# completeness and the receipt
# --------------------------------------------------------------------------- #


def _shards_named_by_index(folder: Path) -> Optional[set]:
    for index_name in _WEIGHT_INDEX_NAMES:
        index_path = folder / index_name
        try:
            if not index_path.is_file():
                continue
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        except (OSError, ValueError, AttributeError):
            return set()
        if not isinstance(weight_map, dict) or not weight_map:
            return set()
        return {str(n) for n in weight_map.values() if isinstance(n, str)}
    return None


def _receipt(folder: Path) -> Optional[dict]:
    try:
        data = json.loads((folder / RECEIPT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != RECEIPT_SCHEMA:
        return None
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        return None
    return data


def plain_folder_complete(folder: Path) -> bool:
    """True only for a folder holding a COMPLETE weight set. Fails closed."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    receipt = _receipt(folder)
    if receipt is not None:
        weights = 0
        for rel, size in receipt["files"].items():
            try:
                if (folder / rel).stat().st_size != int(size):
                    return False
            except (OSError, TypeError, ValueError):
                return False
            if Path(rel).suffix.lower() in _WEIGHT_SUFFIXES:
                weights += 1
        return weights > 0
    declared = _shards_named_by_index(folder)
    if not declared:
        # No receipt and no readable index: a single-file folder placed by hand
        # is indistinguishable from a half-finished download, so it is not
        # trusted. Re-running the download resumes it in place.
        return False
    for shard in declared:
        try:
            if (folder / shard).stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _receipt_files(folder: Path) -> dict:
    """Every file under ``folder`` except huggingface_hub's transfer metadata
    and our own receipt, with its size."""
    out = {}
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder)
        if not path.is_file() or rel.parts[0] == ".cache" or rel.name == RECEIPT_NAME:
            continue
        out[rel.as_posix()] = path.stat().st_size
    return out


def write_receipt(folder: Path, repo_id: str) -> bool:
    """Record that ``snapshot_download(local_dir=folder)`` RETURNED. Call it only
    after a successful download. Atomic; never raises; False on failure."""
    folder = Path(folder)
    try:
        payload = {
            "schema": RECEIPT_SCHEMA,
            "repo_id": str(repo_id),
            "completed": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "files": _receipt_files(folder),
        }
        tmp = folder / (RECEIPT_NAME + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, folder / RECEIPT_NAME)
        return True
    except OSError as exc:
        log.warning("[OTR.llm_folder] completion receipt not written in %s: %s", folder, exc)
        return False


# --------------------------------------------------------------------------- #
# lookups
# --------------------------------------------------------------------------- #


def plain_folders_for(repo_id: str, roots: Optional[Iterable[Path]] = None) -> List[Path]:
    """Existing folders for ``repo_id`` across every root, complete or not."""
    name = folder_name(repo_id)
    out = []
    for root in (llm_roots() if roots is None else roots):
        candidate = Path(root) / name
        if candidate.is_dir():
            out.append(candidate)
    return out


def find_plain_model(repo_id: str, roots: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """The first COMPLETE plain folder for ``repo_id``, or None."""
    for candidate in plain_folders_for(repo_id, roots):
        if plain_folder_complete(candidate):
            return candidate
    return None


def find_plain_file(repo_id: str, filename: str,
                    roots: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """A non-empty ``filename`` in any plain folder for ``repo_id``, complete or
    not -- optional metadata (``chat_template.jinja``) is resolved on its own,
    never by treating a metadata-only folder as the model (Bug Bible 02.16)."""
    for candidate in plain_folders_for(repo_id, roots):
        path = candidate / filename
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path
        except OSError:
            continue
    return None


def scan_plain_models(roots: Optional[Iterable[Path]] = None) -> List[Tuple[str, Path, bool]]:
    """``(repo_id, folder, complete)`` for every folder of ours in every root.
    First root wins for a repo present in more than one."""
    seen = set()
    out = []
    for root in (llm_roots() if roots is None else roots):
        try:
            children = sorted(p for p in Path(root).iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            repo_id = repo_id_from_folder(child.name)
            if repo_id is None or repo_id in seen:
                continue
            seen.add(repo_id)
            out.append((repo_id, child, plain_folder_complete(child)))
    return out


__all__ = [
    "LLM_CATEGORY", "RECEIPT_NAME", "folder_name", "repo_id_from_folder",
    "default_llm_root", "register_llm_category", "llm_roots",
    "download_destination", "plain_folder_complete", "write_receipt",
    "plain_folders_for", "find_plain_model", "find_plain_file", "scan_plain_models",
]
