"""Story-content RULE packs -- Stage 4 of the multi-modal story schema.

Rule VOCABULARIES (taste: cliche/stage-business/on-the-nose/thesis/
personal-cost phrase patterns, cliche replacement pairs, banned phrases) live
in JSON at nodes/story_rules/<source_bank_id>.json; this module is pure
BEHAVIOR: strict fail-loud loading + validation + compilation. Zero content
literals. Stdlib only, the _otr_visual_styles.py conventions.

Law (Stage 4, kibitz r1-r4 + Sonnet 3-lens fan-out converged): JSON owns rule
vocabularies; Python owns rule ENGINES (matching, severity, flow) and GLOBAL
POLICY -- SFW profanity (DEFAULT_PROFANITY_TERMS) is deliberately NOT
pack-tunable. Unknown id = hard error; no fallbacks.

LAZY: zero import-time I/O; the sweep runs on first resolve, then caches
(the Stage-3 lazy-singleton judgment inherited unchanged). `_clear_caches()`
is the test hook.

Sweep contract (kibitz r2 M5 / r3 S3 / r4):
- only <registered_bank_id>.json files under nodes/story_rules/ (dirs and
  stray files = hard error);
- rules_id == filename stem == a REGISTERED source_bank_id (banks.json);
- science_news.json REQUIRED (the production lane always has rules);
- a MISSING pack is legal ONLY for a runnable:false bank; explicit
  resolve_story_rules() on a bank without a pack raises
  UnknownStoryRulesError (unreachable in production while the run gate
  holds);
- duplicate JSON keys rejected (object_pairs_hook, the story-pack loader's
  established convention);
- every regex source: length <= 200, NO control characters (kills the JSON
  single-backslash `\\b` -> backspace-byte trap at LOAD time), compiles
  fail-loud with uniform re.IGNORECASE (no per-pattern flag override in v1);
- every cliche replacement TEMPLATE is validated fail-loud against its
  compiled pattern via a re.sub probe (bad backrefs die at load, never in
  the repair path's swallow).

JSON escape rule (load-bearing): files carry the regex SOURCE with DOUBLED
backslashes (`\\\\b` in-file -> `\\b` to re.compile). science_news.json is
GENERATED from the live Python constants, so fidelity holds by construction;
the extraction test pins the round-trip and the corpus test pins compiled
behavior.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_STORY_RULES_ROOT = Path(__file__).resolve().parent / "story_rules"

DEFAULT_RULES_ID = "science_news"

KNOWN_RULES_SCHEMA_VERSIONS = frozenset({"v1"})

_RULES_ID_RE = re.compile(r"^[a-z0-9_]+$")

_MAX_PATTERN_LEN = 200

_PATTERN_LIST_FIELDS = (
    "cliche_patterns",
    "stage_business_patterns",
    "on_the_nose_patterns",
    "banned_thesis_patterns",
    "personal_cost_patterns",
)

_REQUIRED_FIELDS = (
    "rules_id", "label", "schema_version",
    *_PATTERN_LIST_FIELDS,
    "cliche_replacements", "banned_phrases",
)


class StoryRulesError(Exception):
    """Base: any fail-loud story-rules problem."""


class UnknownStoryRulesError(StoryRulesError):
    """No rules pack exists for the requested source_bank_id."""


class StoryRulesValidationError(StoryRulesError):
    """A rules pack violates the v1 schema / lint / path contract."""


@dataclass(frozen=True)
class StoryRules:
    """One loaded, validated, COMPILED rules pack (attribute access only)."""
    rules_id: str
    label: str
    schema_version: str
    cliche: "tuple[re.Pattern, ...]"
    stage_business: "tuple[re.Pattern, ...]"
    on_the_nose: "tuple[re.Pattern, ...]"
    banned_thesis: "tuple[re.Pattern, ...]"
    personal_cost: "tuple[re.Pattern, ...]"
    cliche_replacements: "tuple[tuple[re.Pattern, str], ...]"
    banned_phrases: "tuple[str, ...]"


_RULES: "dict[str, StoryRules] | None" = None


def _reject_dup_keys(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise StoryRulesValidationError(
                f"duplicate JSON key {k!r} in a story-rules pack")
        seen[k] = v
    return seen


def _lint_pattern_source(src: object, where: str) -> str:
    if not isinstance(src, str) or not src:
        raise StoryRulesValidationError(
            f"{where}: pattern source must be a non-empty string, "
            f"got {src!r}")
    if len(src) > _MAX_PATTERN_LEN:
        raise StoryRulesValidationError(
            f"{where}: pattern source exceeds {_MAX_PATTERN_LEN} chars")
    for ch in src:
        if ord(ch) < 0x20:
            raise StoryRulesValidationError(
                f"{where}: pattern source contains a control character "
                f"(0x{ord(ch):02x}) -- almost certainly a single-backslash "
                f"JSON escape (write \\\\b, not \\b: a bare \\b in JSON is "
                f"a BACKSPACE byte that compiles fine and matches nothing)")
    return src


def _compile_set(raw: object, where: str) -> "tuple[re.Pattern, ...]":
    if not isinstance(raw, list):
        raise StoryRulesValidationError(f"{where}: must be a list")
    out = []
    for i, src in enumerate(raw):
        s = _lint_pattern_source(src, f"{where}[{i}]")
        try:
            out.append(re.compile(s, re.IGNORECASE))
        except re.error as exc:
            raise StoryRulesValidationError(
                f"{where}[{i}]: invalid regex {s!r}: {exc}") from exc
    return tuple(out)


def _compile_replacements(raw: object, where: str):
    if not isinstance(raw, list):
        raise StoryRulesValidationError(f"{where}: must be a list")
    out = []
    for i, pair in enumerate(raw):
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(isinstance(x, str) for x in pair)):
            raise StoryRulesValidationError(
                f"{where}[{i}]: must be an exact two-string "
                f"[pattern, replacement] pair, got {pair!r}")
        src = _lint_pattern_source(pair[0], f"{where}[{i}][0]")
        try:
            rx = re.compile(src, re.IGNORECASE)
        except re.error as exc:
            raise StoryRulesValidationError(
                f"{where}[{i}]: invalid regex {src!r}: {exc}") from exc
        repl = pair[1]
        # r4 M2: validate the replacement TEMPLATE (backrefs) fail-loud at
        # load via a sub probe -- the runtime repair path swallows errors.
        try:
            rx.sub(repl, "probe text for template validation", count=1)
        except (re.error, IndexError) as exc:
            raise StoryRulesValidationError(
                f"{where}[{i}]: invalid replacement template {repl!r} "
                f"for pattern {src!r}: {exc}") from exc
        out.append((rx, repl))
    return tuple(out)


def _validate_row(raw: dict, path: Path) -> StoryRules:
    unknown = sorted(set(raw) - set(_REQUIRED_FIELDS))
    if unknown:
        raise StoryRulesValidationError(
            f"story rules {path}: unknown key(s) {unknown!r} -- the v1 "
            f"schema is exact")
    missing = sorted(set(_REQUIRED_FIELDS) - set(raw))
    if missing:
        raise StoryRulesValidationError(
            f"story rules {path}: missing required key(s) {missing!r}")
    rules_id = raw["rules_id"]
    if not isinstance(rules_id, str) or not _RULES_ID_RE.match(rules_id):
        raise StoryRulesValidationError(
            f"story rules {path}: rules_id {rules_id!r} must match "
            f"{_RULES_ID_RE.pattern}")
    if rules_id != path.stem:
        raise StoryRulesValidationError(
            f"story rules {path}: rules_id {rules_id!r} does not match the "
            f"filename {path.stem!r} -- the path IS the coordinate")
    if not isinstance(raw["label"], str) or not raw["label"]:
        raise StoryRulesValidationError(
            f"story rules {path}: label must be a non-empty string")
    if raw["schema_version"] not in KNOWN_RULES_SCHEMA_VERSIONS:
        raise StoryRulesValidationError(
            f"story rules {path}: unknown schema_version "
            f"{raw['schema_version']!r}; known: "
            f"{sorted(KNOWN_RULES_SCHEMA_VERSIONS)}")
    phrases_raw = raw["banned_phrases"]
    if not isinstance(phrases_raw, list) or not all(
            isinstance(p, str) and p.strip() for p in phrases_raw):
        raise StoryRulesValidationError(
            f"story rules {path}: banned_phrases must be a list of "
            f"non-empty strings")
    return StoryRules(
        rules_id=rules_id,
        label=raw["label"],
        schema_version=raw["schema_version"],
        cliche=_compile_set(raw["cliche_patterns"],
                            f"{path.name}:cliche_patterns"),
        stage_business=_compile_set(raw["stage_business_patterns"],
                                    f"{path.name}:stage_business_patterns"),
        on_the_nose=_compile_set(raw["on_the_nose_patterns"],
                                 f"{path.name}:on_the_nose_patterns"),
        banned_thesis=_compile_set(raw["banned_thesis_patterns"],
                                   f"{path.name}:banned_thesis_patterns"),
        personal_cost=_compile_set(raw["personal_cost_patterns"],
                                   f"{path.name}:personal_cost_patterns"),
        cliche_replacements=_compile_replacements(
            raw["cliche_replacements"], f"{path.name}:cliche_replacements"),
        banned_phrases=tuple(phrases_raw),
    )


def _registered_bank_ids() -> "tuple[str, ...]":
    from ._otr_story_routing import list_bank_ids
    return list_bank_ids()


def _runnable_bank_ids() -> "frozenset[str]":
    from ._otr_story_routing import get_bank, list_bank_ids
    return frozenset(
        b for b in list_bank_ids() if get_bank(b).runnable)


def _load_all() -> "dict[str, StoryRules]":
    if not _STORY_RULES_ROOT.is_dir():
        raise StoryRulesValidationError(
            f"story rules directory missing: {_STORY_RULES_ROOT}")
    bank_ids = set(_registered_bank_ids())
    rules: "dict[str, StoryRules]" = {}
    for entry in sorted(_STORY_RULES_ROOT.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            raise StoryRulesValidationError(
                f"unexpected subdirectory under story_rules/: {entry}")
        if entry.suffix != ".json":
            raise StoryRulesValidationError(
                f"unexpected non-pack file under story_rules/: {entry}")
        if entry.stem not in bank_ids:
            raise StoryRulesValidationError(
                f"story rules {entry}: stem {entry.stem!r} is not a "
                f"registered source_bank_id ({sorted(bank_ids)})")
        try:
            raw = json.loads(entry.read_text(encoding="utf-8"),
                             object_pairs_hook=_reject_dup_keys)
        except StoryRulesValidationError:
            raise
        except (OSError, ValueError) as exc:
            raise StoryRulesValidationError(
                f"story rules {entry}: unreadable/malformed JSON: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise StoryRulesValidationError(
                f"story rules {entry}: top level must be an object")
        rules[entry.stem] = _validate_row(raw, entry)
    if DEFAULT_RULES_ID not in rules:
        raise StoryRulesValidationError(
            f"the default rules pack {DEFAULT_RULES_ID!r}.json is missing "
            f"from {_STORY_RULES_ROOT} -- the production lane must have "
            f"rules")
    # A RUNNABLE bank without a rules pack is a contract violation (a
    # runnable lane's gates would silently have no vocabulary source).
    from ._otr_bank_variants import base_source_bank_id
    missing_runnable = sorted(
        {base_source_bank_id(b) for b in _runnable_bank_ids()} - set(rules)
    )
    if missing_runnable:
        raise StoryRulesValidationError(
            f"runnable bank(s) missing a story_rules pack: "
            f"{missing_runnable}")
    return rules


def _ensure_loaded() -> "dict[str, StoryRules]":
    global _RULES
    if _RULES is None:
        _RULES = _load_all()
    return _RULES


def list_rules_ids() -> "tuple[str, ...]":
    """Registered rules ids (deterministic filename order)."""
    return tuple(_ensure_loaded())


def resolve_story_rules(source_bank_id: str) -> StoryRules:
    """bank id -> compiled StoryRules. Missing pack = hard error (legal only
    for runnable:false banks, which the run gate keeps out of production)."""
    rules = _ensure_loaded()
    from ._otr_bank_variants import base_source_bank_id
    try:
        return rules[base_source_bank_id(source_bank_id)]
    except KeyError:
        raise UnknownStoryRulesError(
            f"no story_rules pack for source_bank {source_bank_id!r}; "
            f"packs exist for: {sorted(rules)} (a runnable lane must have "
            f"one; authoring it is a lane-enablement item)") from None


def get_story_rules(meta: object) -> StoryRules:
    """meta["source_bank"] (absent/empty => science_news) -> StoryRules."""
    m = meta if isinstance(meta, dict) else {}
    raw = m.get("source_bank")
    bank = str(raw).strip() if raw is not None else ""
    return resolve_story_rules(bank or DEFAULT_RULES_ID)


def _clear_caches() -> None:
    """Test hook (mirrors _otr_visual_styles)."""
    global _RULES
    _RULES = None


__all__ = [
    "DEFAULT_RULES_ID",
    "KNOWN_RULES_SCHEMA_VERSIONS",
    "StoryRules",
    "StoryRulesError",
    "StoryRulesValidationError",
    "UnknownStoryRulesError",
    "get_story_rules",
    "list_rules_ids",
    "resolve_story_rules",
]
