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

Client banks (independent source banks v1): bundles under
user_packs/source_banks/ are admitted ALONGSIDE the shipped rows in this one
authority, held to the same row/pack/pipeline contracts. The asymmetry is
deliberate -- a broken SHIPPED seed fails node registration LOUD; a broken
CLIENT bundle is QUARANTINED (absent from every dropdown, boot survives) and
reported via list_validation_issues().

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
# Chunk 3 (source-payload contracts): stdlib-only at import, zero I/O -- safe
# top-level (the lazy posture is test-pinned on both edges).
from . import _otr_source_payload as _osp
# Independent source banks v1: client bundle discovery/integrity/quarantine.
# Stdlib-only leaf, zero I/O at import -- same lazy posture. It owns bundle
# INTEGRITY; this module stays the ONE bank-row parser (injected below), so a
# client row can never drift from a shipped one.
from . import _otr_user_banks as _oub

_STORY_PACKS_ROOT = Path(__file__).resolve().parent / "story_packs"
_BANKS_FILENAME = "banks.json"
_PIPELINES_FILENAME = "pipelines.json"
_REGISTRY_FILENAMES = frozenset({_BANKS_FILENAME, _PIPELINES_FILENAME})
_PACK_SIDECAR_FILENAMES_BY_BANK = {
    "media_archive": frozenset({"drama_seeds.json"}),
    # original (renamed from original_radio 2026-07-19; kibitz r2-r4): the spark
    # deck is entropy data for the concept pass, not a story pack --
    # _otr_original_radio validates it with its own schema at load.
    "original": frozenset({"spark_deck.json"}),
    # scifi_news_pro: the frame-card + stance deck is entropy data for the P1
    # pitch pass, not a story pack -- its dedicated internal runner validates
    # it at load.
    "scifi_news_pro": frozenset({"frame_deck.json"}),
}

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
    # Chunk 3: does this pipeline's execution path consume the bank's
    # fetcher/interpreter source-payload contract? VALIDATION-time metadata
    # only (sweep rule), never consulted at run time -- same law as executable.
    requires_source_contract: bool = False
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
    # bank_id -> the directory that owns its story packs. Shipped banks own a
    # subdirectory of nodes/story_packs; a client bank owns its own bundle's
    # story_packs/ folder. Resolution routes by OWNER, never by guessing.
    pack_dirs: "dict[str, Path]"
    # Admitted client bundles, keyed by bank id (empty on a stock install).
    user_bundles: "dict[str, object]"
    # Quarantined client bundles: reported, never raised.
    issues: "tuple[object, ...]"


# Lazy singleton -- built on first access, never at import time.
_REGISTRY: "_Registry | None" = None
# Atomic-publish guard: the registry is assigned in ONE statement once fully
# built, so no consumer can ever observe a half-loaded authority. This flag
# only catches RE-ENTRY (a consumer importing back into the load), which would
# otherwise recurse until the stack died with an unreadable traceback.
_LOADING = False


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
    "passes", "requires_source_contract", "notes",
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
    # Validated scalar bank defaults. They live inside `defaults` (already
    # scalar-checked above) and carry stricter contracts for structural routing.
    _spc = defaults.get("style_pool_class")
    if _spc is not None and _spc not in ("media", "adaptation", "generic"):
        raise RegistryValidationError(
            f"{origin}: defaults.style_pool_class must be one of "
            f"'media'|'adaptation'|'generic', got {_spc!r}"
        )
    for _bkey in (
        "propagate_adaptation_cast",
        "provenance_normalize",  # v4 P1(viii): opt-in source-provenance normalizer
        "scene_coherence_check",  # v4 P1(vi): opt-in header<->scene structural gate
    ):
        _bval = defaults.get(_bkey)
        if _bval is not None and not isinstance(_bval, bool):
            raise RegistryValidationError(
                f"{origin}: defaults.{_bkey} must be a bool, got {_bval!r}"
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
    requires_source_contract = obj.get("requires_source_contract")
    if not isinstance(requires_source_contract, bool):
        raise RegistryValidationError(
            f"{origin}: requires_source_contract must be a bool"
        )
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
    # notes is a defaulted field: validate list-of-str ONLY when present.
    notes = _require_str_list(obj, "notes", origin) if "notes" in obj else ()
    return StoryPipeline(
        story_pipeline_id=pipe_id,
        label=_require_str(obj, "label", origin),
        executable=executable,
        declared_seams=declared,
        passes=tuple(passes),
        requires_source_contract=requires_source_contract,
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

def _shipped_pack_dirs(bank_ids) -> "dict[str, Path]":
    """The shipped owner map: every bank owns nodes/story_packs/<bank_id>."""
    return {bank_id: _STORY_PACKS_ROOT / bank_id for bank_id in bank_ids}


def _pack_path(pack_dirs: "dict[str, Path]", bank_id: str, model_id: str) -> Path:
    """Resolve a pack by OWNER -- shipped root or the client's own bundle."""
    return pack_dirs[bank_id] / f"{model_id}.json"


def _is_pack_sidecar(bank_id: str, filename: str) -> bool:
    return filename in _PACK_SIDECAR_FILENAMES_BY_BANK.get(bank_id, frozenset())


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


def _sweep_pack_dir(bank_id: str, pack_dir: Path,
                    pipelines: "dict[str, StoryPipeline]") -> None:
    """Every pack in a bank's own directory validates + matches its path
    coordinates. Same rule for a shipped bank and a client bundle."""
    for pack_file in sorted(pack_dir.glob("*.json")):
        if _is_pack_sidecar(bank_id, pack_file.name):
            continue
        pack = _load_routed_pack(pack_file, pipelines)
        model_id = pack_file.stem
        if pack.source_bank_id != bank_id or pack.story_model_id != model_id:
            raise RegistryValidationError(
                f"{pack_file}: header triple ({pack.source_bank_id!r}, "
                f"{pack.story_model_id!r}) does not match path coordinates "
                f"({bank_id!r}, {model_id!r})"
            )


def _sweep_and_crossref(banks: "dict[str, SourceBank]",
                        pipelines: "dict[str, StoryPipeline]",
                        pack_dirs: "dict[str, Path] | None" = None) -> None:
    if pack_dirs is None:
        pack_dirs = _shipped_pack_dirs(banks)
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
        _sweep_pack_dir(bank_id, entry, pipelines)
    # Cross-refs per bank.
    for bank in banks.values():
        _crossref_bank(bank, pipelines, pack_dirs)


def _crossref_bank(bank: SourceBank, pipelines: "dict[str, StoryPipeline]",
                   pack_dirs: "dict[str, Path]",
                   origin: "str | None" = None,
                   *, is_client: bool = False) -> None:
    """Every contract a bank row owes its pipeline + default pack.

    Identical for a shipped bank and a client bundle -- only the ORIGIN label,
    the pack OWNER, and the one entry-point exemption below differ, so a client
    bank is held to exactly the same standard as the shipped six.

    `is_client` is the ONLY thing that unlocks the reserved self-owned
    entry-point id (wave 3). It is passed explicitly rather than inferred from
    `origin`, so no future caller can widen the exemption by relabelling."""
    if origin is None:
        origin = f"banks.json bank {bank.source_bank_id!r}"
    if bank.default_story_pipeline not in pipelines:
        raise UnknownPipelineError(
            f"{origin}: default_story_pipeline "
            f"{bank.default_story_pipeline!r} not in pipelines.json"
        )
    default_path = _pack_path(pack_dirs, bank.source_bank_id,
                              bank.default_story_model)
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
    pipe = pipelines[bank.default_story_pipeline]
    pass_refs = {
        seam for pass_row in pipe.passes for seam in pass_row.seam_refs
    }
    absent_refs = sorted(pass_refs - set(pack.prompt_stages))
    if absent_refs:
        raise RegistryValidationError(
            f"{origin}: pipeline pass seam_refs {absent_refs} absent from "
            f"default pack {default_path.name}"
        )
    custom_pack_keys = set(pack.prompt_stages) - PRODUCTION_SEAM_ALLOWLIST
    custom_pass_refs = pass_refs - PRODUCTION_SEAM_ALLOWLIST
    if (custom_pack_keys != set(pipe.declared_seams)
            or custom_pass_refs != set(pipe.declared_seams)):
        raise RegistryValidationError(
            f"{origin}: custom seam three-way parity failed: "
            f"pack={sorted(custom_pack_keys)} "
            f"declared={sorted(pipe.declared_seams)} "
            f"pass_refs={sorted(custom_pass_refs)}"
        )
    # Chunk 3 source-payload contract rules (after the precedence check so
    # bank.default_story_pipeline is already proven consistent). IDS ONLY
    # -- never executes wrapper bodies (lazy law).
    # (a) Dangling non-empty ids are typos -- loud, on EVERY bank.
    # A CLIENT row may instead route an entry point to its OWN bundle with the
    # reserved self id (wave 3): the bundle module owns fetch_source /
    # interpret_source. The shipped registries never learn that id, so a
    # SHIPPED row declaring it still dies here as an unregistered typo, and a
    # client bank can neither shadow nor extend a shipped entry point.
    _self_id = _oub.SELF_ENTRY_POINT
    _self_hint = (
        f"; a client bundle may also declare {_self_id!r} to own it"
        if is_client else ""
    )
    _allowed_fetchers = set(_osp.registered_fetcher_ids())
    _allowed_interpreters = set(_osp.registered_interpreter_ids())
    if is_client:
        _allowed_fetchers.add(_self_id)
        _allowed_interpreters.add(_self_id)
    if bank.fetcher and bank.fetcher not in _allowed_fetchers:
        raise RegistryValidationError(
            f"{origin}: fetcher {bank.fetcher!r} is not registered in "
            f"_otr_source_payload (registered: "
            f"{sorted(_osp.registered_fetcher_ids())}){_self_hint}"
        )
    if bank.interpreter and bank.interpreter not in _allowed_interpreters:
        raise RegistryValidationError(
            f"{origin}: interpreter {bank.interpreter!r} is not registered "
            f"in _otr_source_payload (registered: "
            f"{sorted(_osp.registered_interpreter_ids())}){_self_hint}"
        )
    # (b) A runnable bank must have a REAL execution lane: either its
    # pipeline consumes the source-payload contract and both ids are
    # declared (registered per (a)), or the pipeline brings its own runner
    # (executable=true -- a validation-time read; the "executable is never
    # a RUNTIME gate" law stands).
    if bank.runnable:
        if pipe.requires_source_contract:
            if not bank.fetcher or not bank.interpreter:
                raise RegistryValidationError(
                    f"{origin}: runnable bank on source-contract pipeline "
                    f"{pipe.story_pipeline_id!r} must declare BOTH fetcher "
                    f"and interpreter (lane-enablement checklist 4b item 3)"
                )
        elif not pipe.executable:
            raise RegistryValidationError(
                f"{origin}: runnable bank on non-source-contract pipeline "
                f"{pipe.story_pipeline_id!r} requires that pipeline's "
                f"runner to exist (executable=true); flipping runnable "
                f"without a lane is forbidden"
            )


def _admit_user_banks(banks: "dict[str, SourceBank]",
                      pipelines: "dict[str, StoryPipeline]",
                      pack_dirs: "dict[str, Path]"
                      ) -> "tuple[dict[str, object], tuple]":
    """Fold client bundles into the authority ALONGSIDE the shipped rows.

    Asymmetry by law: a shipped-seed problem already raised above (node
    registration dies loud); a CLIENT problem quarantines that bundle only and
    every other bank -- shipped or client -- still loads."""
    bundles, issues = _oub.discover(parse_row=_parse_bank,
                                    protected_ids=frozenset(banks))
    admitted: "dict[str, object]" = {}
    quarantined = list(issues)
    for bundle in bundles:
        origin = f"{bundle.root}/{_oub.BANK_JSON_FILENAME}"
        try:
            _sweep_pack_dir(bundle.bank_id, bundle.story_packs_dir, pipelines)
            _crossref_bank(bundle.row, pipelines,
                           {**pack_dirs, bundle.bank_id: bundle.story_packs_dir},
                           origin=origin, is_client=True)
        except Exception as exc:  # any contract failure -> quarantine, not boot death
            quarantined.append(_oub.ValidationIssue(
                bank_id=bundle.bank_id, path=str(bundle.root),
                code="bad_bundle_contract",
                detail=f"{type(exc).__name__}: {exc}"))
            continue
        banks[bundle.bank_id] = bundle.row
        pack_dirs[bundle.bank_id] = bundle.story_packs_dir
        admitted[bundle.bank_id] = bundle
    for issue in quarantined:
        print(issue.render())
    return admitted, tuple(quarantined)


def _ensure_loaded() -> _Registry:
    global _REGISTRY, _LOADING
    if _REGISTRY is not None:
        return _REGISTRY
    if _LOADING:
        raise RegistryValidationError(
            "story-routing registry re-entered while loading: a consumer "
            "imported back into the authority mid-build. Move that import "
            "inside the function that needs it."
        )
    _LOADING = True
    try:
        return _build_registry()
    finally:
        _LOADING = False


def _build_registry() -> _Registry:
    global _REGISTRY
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
    pack_dirs = _shipped_pack_dirs(banks)
    # Shipped seed: fail LOUD (node registration dies rather than boot wrong).
    _sweep_and_crossref(banks, pipelines, pack_dirs)
    # Client bundles: admitted alongside, quarantined individually.
    user_bundles, issues = _admit_user_banks(banks, pipelines, pack_dirs)
    # Atomic publish: one assignment, fully built.
    _REGISTRY = _Registry(banks=banks, pipelines=pipelines,
                          pack_dirs=pack_dirs, user_bundles=user_bundles,
                          issues=issues)
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


def _parse_shipped_seed() -> "tuple[dict, dict]":
    """The shipped banks + pipelines, parsed fresh, with NO client admission.

    Deliberately not `_ensure_loaded()`: the checker calls this while judging a
    bundle that has not earned admission yet, so going through the registry
    would be circular."""
    banks_data = _read_registry_json(_STORY_PACKS_ROOT / _BANKS_FILENAME)
    _check_unknown_keys(banks_data, frozenset({"schema_version", "banks"}),
                        _BANKS_FILENAME)
    _check_schema_version(banks_data, _BANKS_FILENAME)
    banks = _parse_id_list(
        banks_data.get("banks"), _parse_bank, "source_bank_id", "banks.json banks")
    pipes_data = _read_registry_json(_STORY_PACKS_ROOT / _PIPELINES_FILENAME)
    _check_unknown_keys(pipes_data, frozenset({"schema_version", "pipelines"}),
                        _PIPELINES_FILENAME)
    _check_schema_version(pipes_data, _PIPELINES_FILENAME)
    pipelines = _parse_id_list(
        pipes_data.get("pipelines"), _parse_pipeline, "story_pipeline_id",
        "pipelines.json pipelines")
    return banks, pipelines


def shipped_bank_seed() -> "tuple[object, frozenset]":
    """The ONE bank-row parser plus the SHIPPED bank ids.

    `otr_check bank --activate` needs exactly this pair: the parser this
    authority uses -- so the checker can never approve a row boot would refuse,
    nor refuse one it would admit -- and the id set a client bundle may not
    collide with."""
    banks, _ = _parse_shipped_seed()
    return _parse_bank, frozenset(banks)


def validate_client_bundle_contract(bundle) -> None:
    """The REST of what admission demands of a client bundle: its story packs
    validate against their path coordinates, and its row's cross-references
    (pipeline, default pack, required seams, execution lane) resolve.

    `_admit_user_banks` runs exactly these two functions after `discover`, so
    without this the checker would publish a receipt for a bundle that then
    quarantines at boot as `bad_bundle_contract` -- an activation that says yes
    to a bank the operator can never select. Raises the authority's own error;
    the caller reports it under that same stable code."""
    banks, pipelines = _parse_shipped_seed()
    pack_dirs = _shipped_pack_dirs(banks)
    _sweep_pack_dir(bundle.bank_id, bundle.story_packs_dir, pipelines)
    _crossref_bank(bundle.row, pipelines,
                   {**pack_dirs, bundle.bank_id: bundle.story_packs_dir},
                   origin=f"{bundle.root}/{_oub.BANK_JSON_FILENAME}",
                   is_client=True)


def resolve_story_pack(source_bank_id: str, story_model_id: str | None = None) -> StoryPack:
    """Resolve (bank, model) -> the validated StoryPack. Model defaults from
    the bank. Unknown bank/model = hard error; no fallback."""
    reg = _ensure_loaded()
    bank = get_bank(source_bank_id)
    model_id = story_model_id if story_model_id is not None else bank.default_story_model
    path = _pack_path(reg.pack_dirs, bank.source_bank_id, model_id)
    if (not path.is_file()
            or _is_pack_sidecar(bank.source_bank_id, path.name)):
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


def list_validation_issues() -> "tuple[object, ...]":
    """Quarantined CLIENT bundles: what was refused and why.

    Empty on a stock install. These are reported, never raised -- a broken
    client bundle costs its own dropdown row and nothing else."""
    return _ensure_loaded().issues


def user_bank_bundle(source_bank_id: str):
    """The admitted client bundle behind a bank id, or None if it is shipped."""
    return _ensure_loaded().user_bundles.get(source_bank_id)


def _clear_caches() -> None:
    """Test hook: reset the routing registry AND both pack caches."""
    global _REGISTRY, _LOADING
    _REGISTRY = None
    _LOADING = False
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
    "list_validation_issues",
    "require_runnable_bank",
    "resolve_story_pack",
    "shipped_bank_seed",
    "user_bank_bundle",
    "validate_client_bundle_contract",
]
