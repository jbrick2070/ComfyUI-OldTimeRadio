"""Story-path routing -- Stage 2 of the multi-modal story schema.

Registries live in JSON (nodes/story_packs/banks.json + pipelines.json); this
module is pure BEHAVIOR: strict fail-loud registry loading + validation + the
source_bank/story_model/story_pipeline resolution. Zero content literals.
Stdlib only, same posture as _otr_story_pack.py.

Law: JSON owns content/config; Python owns validation/routing/execution; no
fallbacks; unknown id = hard error.

LAZY: importing this module performs ZERO file I/O (ComfyUI custom-node import
isolation). The registries load + the pack-directory sweep runs on the first
get_bank / get_pipeline / resolve_story_pack call, then cache.

Run gating: `bank.runnable` is the ONLY runtime gate (require_runnable_bank).
`pipeline.executable` is metadata -- it documents whether a JSON-directed pass
runner exists and is NEVER consulted at run time (kibitz r2 M2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ._otr_story_pack import (
    KNOWN_SCHEMA_VERSIONS,
    PRODUCTION_SEAM_ALLOWLIST,
    StoryPack,
    StoryPackError,
    _reject_dup_keys,
    load_pack_with_seams,
)
from . import _otr_story_pack as _sp

_STORY_PACKS_ROOT = Path(__file__).resolve().parent / "story_packs"
_BANKS_FILENAME = "banks.json"
_PIPELINES_FILENAME = "pipelines.json"
_REGISTRY_FILENAMES = frozenset({_BANKS_FILENAME, _PIPELINES_FILENAME})

_VALID_PASS_SLOTS = frozenset({"creative", "technical"})


class StoryRoutingError(Exception):
    """Base: any fail-loud story-routing problem."""


class UnknownBankError(StoryRoutingError):
    """source_bank_id is not registered in banks.json."""


class UnknownStoryModelError(StoryRoutingError):
    """story_model_id has no pack file under the bank's directory."""


class UnknownPipelineError(StoryRoutingError):
    """story_pipeline_id is not registered in pipelines.json."""


class RegistryValidationError(StoryRoutingError):
    """banks.json / pipelines.json / the pack sweep violates the contract."""


class StoryBankNotRunnableError(StoryRoutingError):
    """Run-intent on a bank whose execution lane does not exist yet."""


@dataclass(frozen=True)
class PipelinePass:
    pass_id: str
    slot: str
    seam_refs: "tuple[str, ...]"
    description: str


@dataclass(frozen=True)
class StoryPipeline:
    story_pipeline_id: str
    label: str
    executable: bool  # metadata-only; NEVER a runtime gate
    declared_seams: frozenset
    passes: "tuple[PipelinePass, ...]"
    notes: "tuple[str, ...]" = ()


@dataclass(frozen=True)
class SourceBank:
    source_bank_id: str
    label: str
    source_kind: str
    interpreter: str
    fetcher: str
    default_story_model: str
    default_story_pipeline: str
    defaults: "dict[str, object]" = field(default_factory=dict)
    required_seams: "tuple[str, ...]" = ()
    runnable: bool = False
    guide_ref: str = ""


@dataclass(frozen=True)
class _Registry:
    banks: "dict[str, SourceBank]"
    pipelines: "dict[str, StoryPipeline]"


# Lazy singleton -- built on first access, never at import time.
_REGISTRY: "_Registry | None" = None


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------

def _read_registry_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RegistryValidationError(f"registry file not found: {path}") from exc
    except OSError as exc:
        raise RegistryValidationError(f"cannot read registry {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise RegistryValidationError(f"registry {path} is not valid UTF-8: {exc}") from exc
    try:
        data = json.loads(text, object_pairs_hook=_reject_dup_keys)
    except StoryPackError as exc:  # duplicate key via _reject_dup_keys
        raise RegistryValidationError(f"registry {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryValidationError(f"malformed JSON in registry {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryValidationError(f"registry {path}: top level must be an object")
    return data


def _require_str(obj: dict, key: str, origin: str, *, allow_empty: bool = False) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or (not allow_empty and not val.strip()):
        raise RegistryValidationError(
            f"{origin}: {key} must be a non-empty string" if not allow_empty
            else f"{origin}: {key} must be a string"
        )
    return val


def _require_str_list(obj: dict, key: str, origin: str) -> "tuple[str, ...]":
    val = obj.get(key)
    if not isinstance(val, list) or any(not isinstance(s, str) or not s.strip() for s in val):
        raise RegistryValidationError(
            f"{origin}: {key} must be a list of non-empty strings"
        )
    return tuple(val)


def _check_schema_version(data: dict, origin: str) -> None:
    ver = data.get("schema_version")
    if ver not in KNOWN_SCHEMA_VERSIONS:
        raise RegistryValidationError(
            f"{origin}: unknown schema_version {ver!r}; known: "
            f"{sorted(KNOWN_SCHEMA_VERSIONS)}"
        )


def _check_unknown_keys(obj: dict, known: frozenset, origin: str) -> None:
    unknown = sorted(set(obj) - set(known))
    if unknown:
        raise RegistryValidationError(
            f"{origin}: unknown key(s) {unknown}; known: {sorted(known)}"
        )


# --------------------------------------------------------------------------
# registry parsing
# --------------------------------------------------------------------------

_BANK_KEYS = frozenset({
    "source_bank_id", "label", "source_kind", "interpreter", "fetcher",
    "default_story_model", "default_story_pipeline", "defaults",
    "required_seams", "runnable", "guide_ref",
})

_PIPELINE_KEYS = frozenset({
    "story_pipeline_id", "label", "executable", "declared_seams",
    "passes", "notes",
})

_PASS_KEYS = frozenset({"pass_id", "slot", "seam_refs", "description"})


def _parse_bank(obj: dict, origin: str) -> SourceBank:
    if not isinstance(obj, dict):
        raise RegistryValidationError(f"{origin}: each bank must be an object")
    _check_unknown_keys(obj, _BANK_KEYS, origin)
    bank_id = _require_str(obj, "source_bank_id", origin)
    origin = f"{origin} bank {bank_id!r}"
    defaults = obj.get("defaults", {})
    if not isinstance(defaults, dict) or any(
        not isinstance(v, (str, int, float, bool)) for v in defaults.values()
    ):
        raise RegistryValidationError(
            f"{origin}: defaults must be an object of scalar values"
        )
    runnable = obj.get("runnable")
    if not isinstance(runnable, bool):
        raise RegistryValidationError(f"{origin}: runnable must be a bool")
    required_seams = _require_str_list(obj, "required_seams", origin)
    bad = sorted(set(required_seams) - PRODUCTION_SEAM_ALLOWLIST)
    if bad:
        raise RegistryValidationError(
            f"{origin}: required_seams contain non-production seam(s) {bad}"
        )
    return SourceBank(
        source_bank_id=bank_id,
        label=_require_str(obj, "label", origin),
        source_kind=_require_str(obj, "source_kind", origin),
        interpreter=_require_str(obj, "interpreter", origin, allow_empty=True),
        fetcher=_require_str(obj, "fetcher", origin, allow_empty=True),
        default_story_model=_require_str(obj, "default_story_model", origin),
        default_story_pipeline=_require_str(obj, "default_story_pipeline", origin),
        defaults=dict(defaults),
        required_seams=required_seams,
        runnable=runnable,
        guide_ref=_require_str(obj, "guide_ref", origin, allow_empty=True),
    )


def _parse_pipeline(obj: dict, origin: str) -> StoryPipeline:
    if not isinstance(obj, dict):
        raise RegistryValidationError(f"{origin}: each pipeline must be an object")
    _check_unknown_keys(obj, _PIPELINE_KEYS, origin)
    pipe_id = _require_str(obj, "story_pipeline_id", origin)
    origin = f"{origin} pipeline {pipe_id!r}"
    executable = obj.get("executable")
    if not isinstance(executable, bool):
        raise RegistryValidationError(f"{origin}: executable must be a bool")
    declared_raw = obj.get("declared_seams")
    if not isinstance(declared_raw, list):
        raise RegistryValidationError(f"{origin}: declared_seams must be a list")
    declared = (frozenset(_require_str_list(obj, "declared_seams", origin))
                if declared_raw else frozenset())
    overlap = sorted(declared & PRODUCTION_SEAM_ALLOWLIST)
    if overlap:
        raise RegistryValidationError(
            f"{origin}: declared_seams shadow production seam(s) {overlap}"
        )
    passes_raw = obj.get("passes")
    if not isinstance(passes_raw, list) or not passes_raw:
        raise RegistryValidationError(f"{origin}: passes must be a non-empty list")
    seam_universe = PRODUCTION_SEAM_ALLOWLIST | declared
    passes = []
    for i, p in enumerate(passes_raw):
        porigin = f"{origin} passes[{i}]"
        if not isinstance(p, dict):
            raise RegistryValidationError(f"{porigin}: must be an object")
        _check_unknown_keys(p, _PASS_KEYS, porigin)
        slot = _require_str(p, "slot", porigin)
        if slot not in _VALID_PASS_SLOTS:
            raise RegistryValidationError(
                f"{porigin}: slot {slot!r} not in {sorted(_VALID_PASS_SLOTS)}"
            )
        seam_refs = tuple(p.get("seam_refs", []))
        bad = sorted(set(seam_refs) - seam_universe)
        if bad:
            raise RegistryValidationError(
                f"{porigin}: seam_refs {bad} not in production allowlist "
                f"union declared_seams"
            )
        passes.append(PipelinePass(
            pass_id=_require_str(p, "pass_id", porigin),
            slot=slot,
            seam_refs=seam_refs,
            description=_require_str(p, "description", porigin),
        ))
    notes = tuple(obj.get("notes", []))
    return StoryPipeline(
        story_pipeline_id=pipe_id,
        label=_require_str(obj, "label", origin),
        executable=executable,
        declared_seams=declared,
        passes=tuple(passes),
        notes=notes,
    )


def _parse_id_list(items, parse_fn, id_attr: str, origin: str) -> dict:
    """Parse a registry LIST enforcing element-level id uniqueness
    (_reject_dup_keys cannot see duplicate ids across list elements)."""
    if not isinstance(items, list) or not items:
        raise RegistryValidationError(f"{origin}: must be a non-empty list")
    out: dict = {}
    for i, raw in enumerate(items):
        row = parse_fn(raw, f"{origin}[{i}]")
        rid = getattr(row, id_attr)
        if rid in out:
            raise RegistryValidationError(f"{origin}: duplicate id {rid!r}")
        out[rid] = row
    return out


# --------------------------------------------------------------------------
# sweep + cross-ref validation
# --------------------------------------------------------------------------

def _pack_path(bank_id: str, model_id: str) -> Path:
    return _STORY_PACKS_ROOT / bank_id / f"{model_id}.json"


def _load_routed_pack(path: Path, pipelines: "dict[str, StoryPipeline]") -> StoryPack:
    """Load a pack with its OWN pipeline's declared seams (two-phase: peek the
    header for story_pipeline_id, then validate with that pipeline's seams)."""
    header = _sp._read_pack_data(path)
    if not isinstance(header, dict):
        raise RegistryValidationError(f"{path}: pack top level must be an object")
    pipe_id = header.get("story_pipeline_id")
    if not isinstance(pipe_id, str) or pipe_id not in pipelines:
        raise UnknownPipelineError(
            f"{path}: story_pipeline_id {pipe_id!r} is not a registered pipeline"
        )
    return load_pack_with_seams(path, pipelines[pipe_id].declared_seams)


def _sweep_and_crossref(banks: "dict[str, SourceBank]",
                        pipelines: "dict[str, StoryPipeline]") -> None:
    # Every immediate SUBDIRECTORY of story_packs/ must be a registered bank.
    for entry in sorted(_STORY_PACKS_ROOT.iterdir()):
        if entry.is_file():
            if entry.name not in _REGISTRY_FILENAMES:
                raise RegistryValidationError(
                    f"unexpected file in {_STORY_PACKS_ROOT}: {entry.name} "
                    f"(only {sorted(_REGISTRY_FILENAMES)} live at the top level)"
                )
            continue
        bank_id = entry.name
        if bank_id not in banks:
            raise RegistryValidationError(
                f"pack directory {entry} is not a registered source_bank_id "
                f"(known: {sorted(banks)})"
            )
        # Every pack inside must validate + match its path coordinates.
        for pack_file in sorted(entry.glob("*.json")):
            pack = _load_routed_pack(pack_file, pipelines)
            model_id = pack_file.stem
            if pack.source_bank_id != bank_id or pack.story_model_id != model_id:
                raise RegistryValidationError(
                    f"{pack_file}: header triple ({pack.source_bank_id!r}, "
                    f"{pack.story_model_id!r}) does not match path coordinates "
                    f"({bank_id!r}, {model_id!r})"
                )
    # Cross-refs per bank.
    for bank in banks.values():
        origin = f"banks.json bank {bank.source_bank_id!r}"
        if bank.default_story_pipeline not in pipelines:
            raise UnknownPipelineError(
                f"{origin}: default_story_pipeline "
                f"{bank.default_story_pipeline!r} not in pipelines.json"
            )
        default_path = _pack_path(bank.source_bank_id, bank.default_story_model)
        if not default_path.is_file():
            raise RegistryValidationError(
                f"{origin}: default_story_model {bank.default_story_model!r} "
                f"has no pack at {default_path}"
            )
        pack = _load_routed_pack(default_path, pipelines)
        # Precedence equality (kibitz r2 M4): no "which wins" ambiguity.
        if pack.story_pipeline_id != bank.default_story_pipeline:
            raise RegistryValidationError(
                f"{origin}: default pack pipeline {pack.story_pipeline_id!r} "
                f"!= bank default_story_pipeline {bank.default_story_pipeline!r}"
            )
        missing = sorted(set(bank.required_seams) - set(pack.prompt_stages))
        if missing:
            raise RegistryValidationError(
                f"{origin}: required_seams {missing} absent from default pack "
                f"{default_path.name}"
            )


def _ensure_loaded() -> _Registry:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    banks_data = _read_registry_json(_STORY_PACKS_ROOT / _BANKS_FILENAME)
    _check_unknown_keys(banks_data, frozenset({"schema_version", "banks"}), _BANKS_FILENAME)
    _check_schema_version(banks_data, _BANKS_FILENAME)
    banks = _parse_id_list(
        banks_data.get("banks"), _parse_bank, "source_bank_id", "banks.json banks")
    pipes_data = _read_registry_json(_STORY_PACKS_ROOT / _PIPELINES_FILENAME)
    _check_unknown_keys(pipes_data, frozenset({"schema_version", "pipelines"}), _PIPELINES_FILENAME)
    _check_schema_version(pipes_data, _PIPELINES_FILENAME)
    pipelines = _parse_id_list(
        pipes_data.get("pipelines"), _parse_pipeline, "story_pipeline_id",
        "pipelines.json pipelines")
    _sweep_and_crossref(banks, pipelines)
    _REGISTRY = _Registry(banks=banks, pipelines=pipelines)
    return _REGISTRY


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def get_bank(source_bank_id: str) -> SourceBank:
    """Registered bank row; unknown id = hard error."""
    reg = _ensure_loaded()
    bank = reg.banks.get(source_bank_id)
    if bank is None:
        raise UnknownBankError(
            f"unknown source_bank {source_bank_id!r}; known: {sorted(reg.banks)}"
        )
    return bank


def get_pipeline(story_pipeline_id: str) -> StoryPipeline:
    """Registered pipeline row; unknown id = hard error."""
    reg = _ensure_loaded()
    pipe = reg.pipelines.get(story_pipeline_id)
    if pipe is None:
        raise UnknownPipelineError(
            f"unknown story_pipeline {story_pipeline_id!r}; "
            f"known: {sorted(reg.pipelines)}"
        )
    return pipe


def list_bank_ids() -> "tuple[str, ...]":
    """Registered bank ids in registry order (for dropdown surfaces)."""
    return tuple(_ensure_loaded().banks)


def resolve_story_pack(source_bank_id: str, story_model_id: str | None = None) -> StoryPack:
    """Resolve (bank, model) -> the validated StoryPack. Model defaults from
    the bank. Unknown bank/model = hard error; no fallback."""
    reg = _ensure_loaded()
    bank = get_bank(source_bank_id)
    model_id = story_model_id if story_model_id is not None else bank.default_story_model
    path = _pack_path(bank.source_bank_id, model_id)
    if not path.is_file():
        raise UnknownStoryModelError(
            f"unknown story_model {model_id!r} for bank "
            f"{bank.source_bank_id!r}: no pack at {path}"
        )
    return _load_routed_pack(path, reg.pipelines)


def require_runnable_bank(source_bank_id: str) -> SourceBank:
    """Run-intent gate: raise LOUD if the bank's execution lane is not built.
    bank.runnable is the ONLY runtime gate (pipeline.executable is metadata)."""
    bank = get_bank(source_bank_id)
    if not bank.runnable:
        raise StoryBankNotRunnableError(
            f"source_bank {bank.source_bank_id!r} is not runnable: its "
            f"interpreter/fetcher/pass-runner lane is not built yet "
            f"(runnable=false in banks.json). Pick a runnable bank; there is "
            f"no fallback."
        )
    return bank


def _clear_caches() -> None:
    """Test hook: reset the routing registry AND both pack caches."""
    global _REGISTRY
    _REGISTRY = None
    _sp._PACK_CACHE.clear()
    _sp._PACK_CACHE_WITH_SEAMS.clear()


__all__ = [
    "PipelinePass",
    "RegistryValidationError",
    "SourceBank",
    "StoryBankNotRunnableError",
    "StoryPipeline",
    "StoryRoutingError",
    "UnknownBankError",
    "UnknownPipelineError",
    "UnknownStoryModelError",
    "get_bank",
    "get_pipeline",
    "list_bank_ids",
    "require_runnable_bank",
    "resolve_story_pack",
]
