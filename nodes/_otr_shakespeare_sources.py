"""Shakespeare/Folger source-bank helpers.

V1 is local-manifest only: no network calls, no import-time I/O, and no bundled
large Folger corpus. The sample scene is deliberately small and stamped with
Folger's noncommercial terms so downstream ledgers can distinguish it from
public-domain-safe source banks.
"""
from __future__ import annotations

import html
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

from pydantic import BaseModel, Field

try:
    from . import _otr_source_payload as _osp
except ImportError:  # pragma: no cover -- flat import harnesses
    import _otr_source_payload as _osp  # type: ignore

try:
    from . import _otr_source_document as _osd
except ImportError:  # pragma: no cover -- flat import harnesses
    import _otr_source_document as _osd  # type: ignore

try:
    from . import _otr_roster_gender as _roster_gender
except ImportError:  # pragma: no cover -- flat import harnesses
    import _otr_roster_gender as _roster_gender  # type: ignore

try:
    from ._otr_structured_call import StructuredCallFailedError, structured_call
except ImportError:  # pragma: no cover -- flat import harnesses
    from _otr_structured_call import StructuredCallFailedError, structured_call  # type: ignore

MANIFEST_SCHEMA_VERSION = "v1"
PROMPT_VERSION = "shakespeare_interpreter_v2"
SCHEMA_VERSION = "shakespeare_briefs_v1"


_TOP_KEYS = frozenset({"schema_version", "scenes"})
_SCENE_KEYS = frozenset({
    "source_ref",
    "play_code",
    "play_title",
    "act",
    "scene",
    "scene_label",
    "synopsis",
    "year",
    "source_label",
    "source_url",
    "license_label",
    "license_url",
    "commercial_use_allowed",
    "adapter_type",
    "recommended_word_budget",
    "cast_hints",
    "text_path",
})
_ADAPTER_TYPES = frozenset({"curated_scene_text", "folger_txt", "folger_xml"})
_WS_RE = re.compile(r"\s+")


class ShakespeareSourceError(RuntimeError):
    """Base class for Shakespeare source-bank failures."""


class ShakespeareManifestError(ShakespeareSourceError):
    """A curated-scenes manifest violates the source-bank contract."""


class ShakespeareSourceRefError(ShakespeareSourceError):
    """A Shakespeare source_ref did not resolve to a manifest scene."""


class ShakespeareInterpreterError(ShakespeareSourceError):
    """Raised when the Shakespeare source brain cannot produce briefs."""

    def __init__(self, *, attempts: int, reason: str) -> None:
        self.attempts = attempts
        self.reason = reason
        super().__init__(
            f"shakespeare interpreter failed after {attempts} attempt(s): {reason}"
        )


@dataclass(frozen=True)
class ShakespeareScene:
    """Resolved curated Shakespeare scene."""

    scene: dict[str, Any]

    @property
    def source_ref(self) -> str:
        return self.scene["source_ref"]


@dataclass(frozen=True)
class FolgerScene:
    """Small parsed-scene surface for Folger XML/TEI snippets."""

    play_code: str
    act: int
    scene: int
    speakers: tuple[str, ...]
    stage_directions: tuple[str, ...]
    text: str


def _check_unknown_keys(obj: dict[str, Any], allowed: frozenset[str], origin: str) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ShakespeareManifestError(
            f"{origin}: unknown key(s) {unknown}; known: {sorted(allowed)}"
        )


def _require_str(obj: dict[str, Any], key: str, origin: str, *, allow_empty: bool = False) -> str:
    val = obj.get(key)
    if not isinstance(val, str) or (not allow_empty and not val.strip()):
        tail = "a string" if allow_empty else "a non-empty string"
        raise ShakespeareManifestError(f"{origin}: {key} must be {tail}")
    return val


def _require_bool(obj: dict[str, Any], key: str, origin: str) -> bool:
    val = obj.get(key)
    if not isinstance(val, bool):
        raise ShakespeareManifestError(f"{origin}: {key} must be boolean")
    return bool(val)


def _require_int(obj: dict[str, Any], key: str, origin: str, *, min_value: int, max_value: int | None = None) -> int:
    val = obj.get(key)
    if not isinstance(val, int) or isinstance(val, bool):
        raise ShakespeareManifestError(f"{origin}: {key} must be an integer")
    if val < min_value:
        raise ShakespeareManifestError(
            f"{origin}: {key} must be at least {min_value}"
        )
    if max_value is not None and val > max_value:
        raise ShakespeareManifestError(
            f"{origin}: {key} must be between {min_value} and {max_value}"
        )
    return val


def _require_str_list(obj: dict[str, Any], key: str, origin: str) -> list[str]:
    val = obj.get(key)
    if not isinstance(val, list) or any(not isinstance(v, str) or not v.strip() for v in val):
        raise ShakespeareManifestError(
            f"{origin}: {key} must be a list of non-empty strings"
        )
    return list(val)


def _validate_scene(raw: Any, origin: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ShakespeareManifestError(f"{origin}: scene must be an object")
    _check_unknown_keys(raw, _SCENE_KEYS, origin)
    scene = {
        "source_ref": _require_str(raw, "source_ref", origin),
        "play_code": _require_str(raw, "play_code", origin),
        "play_title": _require_str(raw, "play_title", origin),
        "act": _require_int(raw, "act", origin, min_value=1, max_value=5),
        "scene": _require_int(raw, "scene", origin, min_value=1, max_value=20),
        "scene_label": _require_str(raw, "scene_label", origin),
        "synopsis": _require_str(raw, "synopsis", origin),
        "year": _require_str(raw, "year", origin, allow_empty=True),
        "source_label": _require_str(raw, "source_label", origin),
        "source_url": _require_str(raw, "source_url", origin),
        "license_label": _require_str(raw, "license_label", origin),
        "license_url": _require_str(raw, "license_url", origin),
        "commercial_use_allowed": _require_bool(raw, "commercial_use_allowed", origin),
        "adapter_type": _require_str(raw, "adapter_type", origin),
        "recommended_word_budget": _require_int(
            # No upper bound: this is a RECOMMENDATION the operator may
            # exceed, and the word target is a request rather than a gate.
            # A 320-word ceiling pinned these episodes at roughly two
            # minutes, too short for the scenes worth vendoring. The real
            # limit is structural, not declarative -- the beat topology
            # tops out near 1,520 spoken words -- and a request beyond it
            # simply delivers the closest performable episode.
            raw, "recommended_word_budget", origin, min_value=30
        ),
        "cast_hints": _require_str_list(raw, "cast_hints", origin),
        "text_path": _require_str(raw, "text_path", origin),
    }
    if scene["adapter_type"] not in _ADAPTER_TYPES:
        raise ShakespeareManifestError(
            f"{origin}: adapter_type {scene['adapter_type']!r} is not in "
            f"{sorted(_ADAPTER_TYPES)}"
        )
    if Path(scene["text_path"]).is_absolute() or ".." in Path(scene["text_path"]).parts:
        raise ShakespeareManifestError(
            f"{origin}: text_path must be a relative manifest-local path"
        )
    return scene


def validate_shakespeare_manifest(data: Any, *, origin: str = "manifest") -> dict[str, Any]:
    """Validate and normalize a curated Shakespeare scene manifest."""
    if not isinstance(data, dict):
        raise ShakespeareManifestError(f"{origin}: top level must be an object")
    _check_unknown_keys(data, _TOP_KEYS, origin)
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ShakespeareManifestError(
            f"{origin}: schema_version must be {MANIFEST_SCHEMA_VERSION!r}"
        )
    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ShakespeareManifestError(f"{origin}: scenes must be a non-empty list")
    scenes: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for idx, raw_scene in enumerate(raw_scenes):
        scene = _validate_scene(raw_scene, f"{origin}.scenes[{idx}]")
        if scene["source_ref"] in seen_refs:
            raise ShakespeareManifestError(
                f"{origin}: duplicate source_ref {scene['source_ref']!r}"
            )
        seen_refs.add(scene["source_ref"])
        scenes.append(scene)
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "scenes": scenes}


def load_shakespeare_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a curated-scenes manifest from disk."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ShakespeareManifestError(f"manifest not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ShakespeareManifestError(f"manifest {p}: invalid JSON: {exc}") from exc
    return validate_shakespeare_manifest(data, origin=str(p))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_relative_path(raw: str, *, key: str) -> Path:
    val = str(raw or "").strip()
    if not val:
        raise ShakespeareManifestError(
            f"shakespeare bank defaults must declare {key}"
        )
    path = Path(val)
    if path.is_absolute():
        return path
    if ".." in path.parts:
        raise ShakespeareManifestError(
            f"shakespeare bank default {key} must not contain '..': {val!r}"
        )
    return _repo_root() / path


def resolve_shakespeare_scene(manifest: dict[str, Any], source_ref: str) -> ShakespeareScene:
    """Resolve a curated scene source_ref into a manifest scene."""
    ref = str(source_ref or "").strip()
    if not ref:
        raise ShakespeareSourceRefError("source_ref is required for shakespeare")
    checked = validate_shakespeare_manifest(manifest)
    for scene in checked["scenes"]:
        if scene["source_ref"] == ref:
            return ShakespeareScene(scene=scene)
    raise ShakespeareSourceRefError(f"unknown shakespeare source_ref {ref!r}")


def select_shakespeare_scene_ref(manifest: dict[str, Any], *, rng: Any | None = None) -> str:
    """Pick a curated scene from a validated manifest for blank source_ref runs."""
    checked = validate_shakespeare_manifest(manifest)
    scenes = checked["scenes"]
    chooser = rng if rng is not None else random.SystemRandom()
    try:
        scene = chooser.choice(scenes)
    except AttributeError:  # pragma: no cover -- defensive for minimal RNG shims
        scene = scenes[chooser.randrange(len(scenes))]
    return str(scene["source_ref"])


# The LEGACY seven-key payload's body cap -- a PROJECTION for that contract,
# not the source of truth. The complete scene lives in the SourceDocument
# built from normalize_shakespeare_body below.
INTERPRETER_TEXT_WINDOW: int = 12000


def normalize_shakespeare_body(text: Any) -> str:
    """Canonicalize a Shakespeare scene WITHOUT truncating it.

    The normalization owner, mirroring the public-domain lane: this result is
    the coordinate system spans and hashes index into, so changing it
    invalidates stored offsets.
    """
    raw = html.unescape(str(text or "")).replace("\ufeff", "")
    cleaned = _WS_RE.sub(" ", raw).strip()
    if not cleaned:
        raise ShakespeareSourceRefError("shakespeare source text is empty")
    return cleaned


def canonicalize_shakespeare_text(text: Any, *, max_chars: int = 12000) -> str:
    """Legacy payload projection: the canonical scene, capped at ``max_chars``.

    Returns a PREFIX by design. Callers needing the whole scene use
    ``normalize_shakespeare_body`` or the SourceDocument built from it.
    """
    cleaned = normalize_shakespeare_body(text)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip() or cleaned[:max_chars]
    if not cleaned:
        raise ShakespeareSourceRefError("shakespeare source text is empty")
    return cleaned


def source_document_from_text(
    text: Any, *, source_ref: str = "",
) -> "_osd.SourceDocument":
    """Build the transient uncapped document for a Shakespeare scene."""
    return _osd.build_source_document(
        normalize_shakespeare_body(text), source_ref=source_ref)


# Authentic Folger plain text marks a speech in one of two ways, and never with a
# colon -- that was the shape of the hand-written stub fixtures only:
#   verse -- the name alone on its line, speech beneath:  ORLANDO\nHang there, my verse
#   prose -- the name inline, two spaces, then speech:    TOBY  Come thy ways, Signior
# Either may carry a stage qualifier: ROSALIND, [as Ganymede] / BENEDICK, [aside]
# Recognizing only the colon form returned NO speakers from real Folger text, so
# payload_from_scene silently fell back to the curated cast_hints -- the list whose
# ordering dropped Orlando from the scene he is named in and let a writer substitute
# Romeo from another play. The people in a scene are a fact OF the scene.
_FOLGER_SPEECH_RE = re.compile(
    r"^(?P<name>[A-Z][A-Z'’.\-]*(?: [A-Z][A-Z'’.\-]*)*)"
    r"(?:,\s*\[[^\]]*\])?"
    r"(?:\s*$|\s{2,}(?=\S))"
)

# All-caps shapes that are structure or performance direction, never speakers.
_NON_SPEAKER_TOKENS = frozenset({
    "ACT", "SCENE", "FINIS", "THE END", "EPILOGUE", "PROLOGUE", "EXIT", "EXEUNT",
})


def _speaker_from_line(line: str) -> str | None:
    """The speaking character named by this line, or None.

    Accepts both authentic Folger layouts and the legacy ``NAME:`` form so a
    colon-style fixture still parses.
    """
    stripped = str(line or "").rstrip()
    if not stripped or stripped != stripped.lstrip():
        # Indented lines are continuations of a speech, never a prefix.
        return None
    if ":" in stripped:
        candidate = stripped.split(":", 1)[0].strip()
    else:
        match = _FOLGER_SPEECH_RE.match(stripped)
        if match is None:
            return None
        candidate = match.group("name").strip()
    if not candidate or len(candidate) > 40:
        return None
    if candidate.upper() != candidate:
        return None
    if candidate in _NON_SPEAKER_TOKENS:
        return None
    return candidate


def _speakers_from_text(text: str) -> list[str]:
    """Every speaking character, in first-appearance order."""
    seen: list[str] = []
    for line in str(text or "").splitlines():
        speaker = _speaker_from_line(line)
        if speaker and speaker not in seen:
            seen.append(speaker)
    return seen


def cast_presence_from_text(text: str) -> dict[str, dict[str, Any]]:
    """Per-speaker line count and first attested speech, from the RAW
    (pre-normalization) scene text.

    Exists so a cast member whose composed dialogue turned up empty (a tight
    beat budget can under-serve a slot even when the source gives them real
    lines -- PBUG-20260802-02) can be repaired with the SOURCE's own words
    rather than a freshly invented line. `first_speech` is exactly the prose
    a mechanical VERBATIM slice would carry: the speaker's opening line plus
    any immediate continuation lines, stopping at the next speaker prefix, a
    stage direction in brackets, or a blank line. Reuses `_speaker_from_line`
    -- one speaker-line classifier, not a second one that could drift from
    `_speakers_from_text`.
    """
    lines = str(text or "").splitlines()
    out: dict[str, dict[str, Any]] = {}
    i = 0
    while i < len(lines):
        speaker = _speaker_from_line(lines[i])
        if not speaker:
            i += 1
            continue
        stripped = lines[i].rstrip()
        if ":" in stripped:
            first = stripped.split(":", 1)[1].strip()
        else:
            match = _FOLGER_SPEECH_RE.match(stripped)
            first = stripped[match.end():].strip() if match else ""
        speech_parts = [first] if first else []
        j = i + 1
        while j < len(lines):
            candidate = lines[j]
            body = candidate.strip()
            if not body or body.startswith("[") or _speaker_from_line(candidate):
                break
            speech_parts.append(body)
            j += 1
        entry = out.setdefault(speaker, {"line_count": 0, "first_speech": ""})
        entry["line_count"] += 1
        if not entry["first_speech"] and speech_parts:
            entry["first_speech"] = " ".join(speech_parts).strip()
        i = j if j > i + 1 else i + 1
    return out


def payload_from_scene(
    resolved: ShakespeareScene,
    *,
    text: str,
    excerpt_chars: int = 1200,
) -> dict[str, str]:
    """Build the legacy source payload for a curated Shakespeare scene."""
    scene = resolved.scene
    full_text = canonicalize_shakespeare_text(text)
    excerpt = full_text[:excerpt_chars].rsplit(" ", 1)[0].rstrip() or full_text[:excerpt_chars]
    speakers = ", ".join(_speakers_from_text(text)[:6])
    payload = {
        "headline": f"{scene['play_title']}, Act {scene['act']}, Scene {scene['scene']}",
        "summary": scene["synopsis"],
        "full_text": full_text,
        "source": scene["source_label"],
        "date": f"{scene['year']} | {scene['license_label']}".strip(" |"),
        "link": scene["source_url"],
        "seed_text": (
            f"{scene['play_title']}, Act {scene['act']}, Scene {scene['scene']}\n"
            f"Scene: {scene['scene_label']}\n"
            f"Synopsis: {scene['synopsis']}\n"
            f"Speakers: {speakers or ', '.join(scene['cast_hints'])}\n"
            f"Excerpt: {excerpt}"
        ),
    }
    return _osp.validate_source_payload(payload, origin=f"shakespeare {resolved.source_ref}")


def source_rights_from_scene(resolved: ShakespeareScene) -> dict[str, Any]:
    """Rights sidecar for the selected scene."""
    scene = resolved.scene
    return {
        "license_label": scene["license_label"],
        "license_url": scene["license_url"],
        "source_url": scene["source_url"],
        "source_label": scene["source_label"],
        "commercial_use_allowed": scene["commercial_use_allowed"],
    }


def source_meta_from_scene(
    resolved: ShakespeareScene, *, text_path: "Path | None" = None,
    text: str | None = None,
) -> dict[str, Any]:
    """Metadata sidecar for the selected scene.

    When ``text_path`` is supplied, the provenance sidecar's ``characters``
    roster travels with it. source_meta is the only channel that reaches the
    writer and is copied wholesale into the durable ledger, so the roster becomes
    both an input to the gender pin and an auditable receipt for it.

    The key is added only when the roster is non-empty -- an absent key is honest
    absence, where an empty list would read as "the source has no characters".

    When ``text`` is supplied (the pre-normalization scene body the caller
    already holds), ``cast_hints_presence`` carries each hinted name's real
    line count and first attested speech -- the data a repair pass needs to
    ground a silent cast member (PBUG-20260802-02) in the source's own words
    rather than free invention. cast_hints itself is UNCHANGED: this is
    additional evidence about the hints, never a filter on them.
    """
    scene = resolved.scene
    meta: dict[str, Any] = {
        "source_ref": resolved.source_ref,
        "play_code": scene["play_code"],
        "play_title": scene["play_title"],
        "act": scene["act"],
        "scene": scene["scene"],
        "scene_label": scene["scene_label"],
        "year": scene["year"],
        "recommended_word_budget": scene["recommended_word_budget"],
        "cast_hints": list(scene["cast_hints"]),
    }
    if text_path is not None:
        characters = _roster_gender.load_roster_characters(text_path)
        if characters:
            meta["characters"] = [dict(row) for row in characters]
    if text:
        presence = cast_presence_from_text(text)
        if presence:
            meta["cast_hints_presence"] = presence
    return meta


def fetch_shakespeare_scene(*, bank: Any, source_ref: str = "") -> "_osp.SourceFetchResult":
    """Load a manifest-local curated Shakespeare scene.

    Explicit source_ref pins a scene. Blank source_ref defaults to a random
    manifest scene so the Shakespeare pack behaves like a story deck.
    """
    defaults = getattr(bank, "defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ShakespeareManifestError("shakespeare bank defaults must be a dict")
    manifest_path = _resolve_repo_relative_path(
        str(defaults.get("manifest_path", "")),
        key="manifest_path",
    )
    manifest = load_shakespeare_manifest(manifest_path)
    effective_ref = str(source_ref or "").strip()
    selection_mode = str(defaults.get("selection_mode", "random") or "random").strip().lower()
    if not effective_ref:
        if selection_mode in {"random", "random_scene", "shuffle"}:
            effective_ref = select_shakespeare_scene_ref(manifest)
        elif selection_mode in {"fixed", "default", "pinned"}:
            effective_ref = str(defaults.get("source_ref", "") or "").strip()
        else:
            bank_id = getattr(bank, "source_bank_id", "shakespeare")
            raise ShakespeareManifestError(
                f"source_bank {bank_id!r} has unsupported Shakespeare "
                f"selection_mode {selection_mode!r}; expected random or fixed"
            )
    if not effective_ref:
        bank_id = getattr(bank, "source_bank_id", "shakespeare")
        raise ShakespeareSourceRefError(
            f"source_bank {bank_id!r} fixed selection requires defaults.source_ref"
        )

    resolved = resolve_shakespeare_scene(manifest, effective_ref)
    text_path = manifest_path.parent / resolved.scene["text_path"]
    try:
        text = text_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ShakespeareSourceRefError(
            f"shakespeare source text not found for {resolved.source_ref}: {text_path}"
        ) from exc
    return _osp.SourceFetchResult(
        payload=payload_from_scene(resolved, text=text),
        source_meta=source_meta_from_scene(resolved, text_path=text_path, text=text),
        source_rights=source_rights_from_scene(resolved),
        # Transient: the COMPLETE scene. shakespeare is a style_pool_class
        # "adaptation" bank, so it is gated into source-derived grounding
        # exactly like public_domain -- without this it silently kept the
        # cast-seed drawn palette while the code claimed to have fixed
        # "the adaptation lanes".
        source_document=source_document_from_text(
            text, source_ref=resolved.source_ref),
    )


def parse_folger_scene(xml_text: str, play_code: str, act: int, scene: int) -> FolgerScene:
    """Parse a small Folger XML/TEI scene snippet into speaker/stage text.

    This helper is intentionally namespace-tolerant and conservative. V1 does
    not fetch Folger XML; tests use snippets so later import code has a rooted
    parser instead of ad hoc string splitting.
    """
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise ShakespeareManifestError(f"invalid Folger XML: {exc}") from exc

    def _tag(el) -> str:
        return str(el.tag).rsplit("}", 1)[-1].lower()

    speakers: list[str] = []
    stage: list[str] = []
    parts: list[str] = []
    for el in root.iter():
        tag = _tag(el)
        text = _WS_RE.sub(" ", "".join(el.itertext())).strip()
        if not text:
            continue
        if tag in {"speaker", "spkr"}:
            speaker = text.upper()
            if speaker not in speakers:
                speakers.append(speaker)
            parts.append(f"{speaker}:")
        elif tag in {"stage", "stagedir"}:
            stage.append(text)
            parts.append(f"[{text}]")
        elif tag in {"l", "p", "ab"}:
            parts.append(text)
    return FolgerScene(
        play_code=str(play_code),
        act=int(act),
        scene=int(scene),
        speakers=tuple(speakers),
        stage_directions=tuple(stage),
        text=canonicalize_shakespeare_text(" ".join(parts)),
    )


class ShakespeareBriefs(BaseModel):
    """Briefs contract consumed by the source adapter and writer."""

    casting_brief: str
    script_brief: str
    news_close_brief: str
    key_terms: list[str] = Field(default_factory=list)

    source_hash: str = ""
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    model_id: str = ""
    attempts: int = 0


def _source_hash(payload: dict[str, str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in ("headline", "summary", "full_text", "source", "date", "link"):
        h.update(str(payload.get(key, "")).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _build_interpreter_prompt(payload: dict[str, str]) -> list[dict[str, str]]:
    source_block = "\n".join(
        part for part in (
            f"Scene: {payload.get('headline', '').strip()}",
            f"Source: {payload.get('source', '').strip()}",
            # Date/Rights STAYS. The prompt asks this pass for a
            # "Folger/noncommercial source note", so the licence is INPUT TO A
            # REQUESTED OUTPUT -- strip it and the model is asked for a note it
            # has no facts to write. `tests/test_shakespeare_interpreter.py`
            # pins it deliberately: "RIGHTS terms stay -- they are a licensing
            # fact about the source."
            f"Date/Rights: {payload.get('date', '').strip()}",
            # THE URL IS NOT SHOWN (2026-08-05, PROMPT_VERSION v2). Unlike the
            # rights string it is input to NOTHING -- grounding is the scene text
            # below, and no instruction in this prompt references the link -- yet
            # the model echoed the whole block back, and shipped episodes read
            # "Source: Folger Shakespeare. Date/Rights: c. 1606 | CC BY-NC 3.0.
            # URL: https://www.folger.ed..." ALOUD while the captions burned it
            # into the video. `link` still rides the payload and `source_rights`
            # for the PRINTED credits, which is where a URL belongs.
            f"Synopsis: {payload.get('summary', '').strip()}",
            "Scene text:",
        # The interpreter is the ONE pass that reads the source, and it is
        # instructed to preserve the ending -- so a window shorter than the
        # canonicalized body means it is asked to preserve an ending it
        # cannot see. The Wells arrival unit is 11,410 bytes against a
        # 5,000-char window: the press ridicule, "Story be damned!" and the
        # closing image of listeners' faces in the dark all fell outside it.
        # canonicalize_* already caps the body at INTERPRETER_TEXT_WINDOW,
        # so this reads whatever survived that cap and truncates nothing further.
            str(payload.get("full_text", ""))[:INTERPRETER_TEXT_WINDOW],
        )
        if part
    )
    instruction = (
        "You are the source brain for a compact old-time-radio adaptation of "
        "a Shakespeare scene.\n\n"
        "Turn the scene below into JSON briefs for a short radio drama. "
        "Preserve the named characters, the scene's dramatic pressure, its "
        "major turn, and the play-world stakes. Compression is allowed; do "
        "not replace the scene with a modern mystery or unrelated framing "
        "story.\n\n"
        # The SFW clause that used to sit here was DELETED 2026-08-05 (operator
        # directive). It told the model to avoid "guns/knives/weapons" while we
        # handed it MACBETH -- so "Is this a dagger which I see before me" was
        # being discouraged at the prompt, rewritten if it survived, and finally
        # rejected at the G9 freeze gate. On a fidelity lane the author's own
        # language is carried AS WRITTEN; that is the whole point of the lane.
        "Make stage directions audible through spoken implication or concrete "
        "radio business.\n\n"
        "Return ONE JSON object only with exactly these keys:\n"
        "{\n"
        "  \"casting_brief\": \"source-grounded roles and voices\",\n"
        "  \"script_brief\": \"compact scene adaptation brief\",\n"
        "  \"news_close_brief\": \"Folger/noncommercial source note\",\n"
        "  \"key_terms\": [\"optional concise scene/adaptation terms\"]\n"
        "}\n\n"
        f"{source_block}"
    )
    return [{"role": "user", "content": instruction}]


def build_shakespeare_briefs(
    *,
    technical_fn: Callable[..., str],
    payload: dict[str, str],
    model_id: str = "",
    max_attempts: int = 3,
    base_temperature: float = 0.35,
    max_new_tokens: int = 520,
) -> ShakespeareBriefs:
    """Run the Shakespeare source brain through the structured JSON ladder."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    slot_calls = 0

    def _counting_slot_fn(msgs, *, temperature, max_new_tokens):
        nonlocal slot_calls
        slot_calls += 1
        # LLM slot: technical -- structured source-brief extraction for the
        # Shakespeare lane, routed through the writer's technical slot.
        return technical_fn(msgs, temperature=temperature, max_new_tokens=max_new_tokens)

    def _content_validator(brief: ShakespeareBriefs) -> str | None:
        # The brief-level safety rejection that used to sit here was DELETED
        # 2026-08-05 (operator directive). It re-rolled the brief whenever
        # Shakespeare's own vocabulary showed up in it -- a draft burned for
        # being faithful. Nothing replaces it; the validator is kept as a seam
        # so a future NON-content brief check has somewhere to live.
        del brief
        return None

    try:
        # LLM slot: technical -- Shakespeare source-brief structured pass.
        brief = structured_call(
            prompt=_build_interpreter_prompt(payload),
            schema=ShakespeareBriefs,
            slot_fn=_counting_slot_fn,
            base_temperature=float(base_temperature),
            structural_retry_temperature=float(base_temperature) / 2.0,
            post_validator=_content_validator,
            max_new_tokens=int(max_new_tokens),
            max_attempts=int(max_attempts),
            helper_name="build_shakespeare_briefs",
        )
    except StructuredCallFailedError as exc:
        raise ShakespeareInterpreterError(
            attempts=exc.attempts,
            reason=(
                f"{type(exc.last_error).__name__}: {exc.last_error}"
                if exc.last_error is not None
                else "no error captured"
            ),
        ) from exc

    brief.source_hash = _source_hash(payload)
    brief.model_id = model_id
    brief.attempts = slot_calls
    return brief


__all__ = [
    "FolgerScene",
    "MANIFEST_SCHEMA_VERSION",
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "ShakespeareBriefs",
    "ShakespeareInterpreterError",
    "ShakespeareManifestError",
    "ShakespeareScene",
    "ShakespeareSourceError",
    "ShakespeareSourceRefError",
    "build_shakespeare_briefs",
    "canonicalize_shakespeare_text",
    "fetch_shakespeare_scene",
    "load_shakespeare_manifest",
    "parse_folger_scene",
    "payload_from_scene",
    "resolve_shakespeare_scene",
    "select_shakespeare_scene_ref",
    "source_meta_from_scene",
    "source_rights_from_scene",
    "validate_shakespeare_manifest",
]
