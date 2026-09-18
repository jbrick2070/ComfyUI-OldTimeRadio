"""Episode language registry -- the multilingual one-switch.

Plan: ``docs/multilingual-one-switch.md``. Build contract:
``docs/2026-09-18-multilingual-notebooklm/coder_prompt_all_kokoro_day1.md``.

ONE dropdown, ``episode_language`` on ``OTR_LedgerScriptWriter``, appended
after ``replay_from``. All eight Kokoro rows are admitted day 1 (operator
override of the oval's Spanish-first staging). Kokoro is the dance leader and
the only engine on a non-English row.

``Off`` is resolver state, not a row. A missing/blank label resolves to the
English row (feature on). An unknown nonempty label fails closed.

The writer resolves the widget ONCE and stamps the ledger beside
``source_bank``; every downstream painter (line composer, credits roll,
captions, Kokoro adapter, CastLock) reads the LEDGER through
:func:`row_from_meta`, never the widget. An unstamped ledger -- ``Off``, or a
graph saved before the widget existed -- resolves to English, which is
today's behaviour byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, NamedTuple, Optional

__all__ = [
    "OFF_LABEL",
    "WIDGET_NAME",
    "EpisodeLanguageError",
    "LanguageRow",
    "LanguageResolution",
    "REGISTRY_PATH",
    "check_source_bank_admission",
    "credits_text",
    "dropdown_choices",
    "iso_from_meta",
    "kokoro_config",
    "lead_system",
    "load_registry",
    "native_authoring_instruction",
    "reload_registry",
    "replay_language_check",
    "resolve_label",
    "resolve_ledger",
    "row_by_iso",
    "row_by_label",
    "row_from_meta",
    "spoken_text",
    "title_language_instruction",
    "title_instruction",
    "validate_registry",
    "writer_language_instruction",
    "assert_readiness_extras",
    "readiness_extra_ok",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_REPO_ROOT, "config", "episode_languages.json")

OFF_LABEL = "Off"
WIDGET_NAME = "episode_language"
REGISTRY_ID = "episode_languages"
_SCHEMA_VERSION = 1
ENGLISH_LABEL = "English"
ENGLISH_ISO = "en"

_REQUIRED_ROW_KEYS = (
    "iso", "label", "admitted", "sort_order", "native_header", "row_revision",
    "authoring", "spoken", "credits", "captions", "engines", "admission",
)
_REQUIRED_AUTHORING = (
    "spoken_name", "writer_instruction", "visual_prompt_iso", "title_instruction",
)
_REQUIRED_SPOKEN = (
    "reserved_announcer_name", "sign_on_greeting", "station_id_open",
    "tonight_label", "station_id_close", "sign_off_greeting", "work_line_prefix",
    "open_on_prefix",
)
_REQUIRED_CREDITS = (
    "models_header", "production_ledger_header", "cast_voices_header",
    "story_spine_header", "premise_label", "subject_label",
    "classified_transcript_header", "system_header", "writer_llm_header",
    # origin_hud / more_hud are reserved for a HUD drawer that does not
    # exist yet. Painting them on today's card would add English pixels
    # the live roll never showed. credits_text still owns the strings so
    # a later drawer does not invent a second translation.
    "origin_hud", "more_hud",
)
_REQUIRED_CAPTIONS = ("font_policy", "wrap_policy", "cps_policy")
_REQUIRED_ADMISSION = (
    "source_bank_exclusions", "readiness_extras", "min_voice_count",
)


class EpisodeLanguageError(ValueError):
    """Registry or label resolution failed. Fail closed."""


class LanguageRow(NamedTuple):
    iso: str
    label: str
    admitted: bool
    sort_order: int
    native_header: str
    row_revision: int
    authoring: dict
    spoken: dict
    credits: dict
    captions: dict
    engines: dict
    admission: dict
    raw: dict


class LanguageResolution(NamedTuple):
    """Resolved widget / legacy label.

    ``kind`` is ``off`` (stamp nothing), ``row`` (stamp this row), or
    never mixed. ``row`` is None only for ``off``.
    """
    kind: str
    row: Optional[LanguageRow]
    stamp: bool


_CACHE: Optional[tuple] = None  # (mtime_ns, path, rows_tuple, by_label, by_iso)


def _nonempty_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EpisodeLanguageError("%s must be a non-empty string" % path)
    return value


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _require_str_map(obj: Any, path: str, keys: tuple) -> dict:
    if not isinstance(obj, dict):
        raise EpisodeLanguageError("%s must be an object" % path)
    out = {}
    for key in keys:
        if key not in obj:
            raise EpisodeLanguageError("%s missing required key %r" % (path, key))
        out[key] = _nonempty_str(obj[key], "%s.%s" % (path, key))
    return out


def _validate_row(raw: dict, index: int) -> LanguageRow:
    path = "rows[%d]" % index
    if not isinstance(raw, dict):
        raise EpisodeLanguageError("%s must be an object" % path)
    for key in _REQUIRED_ROW_KEYS:
        if key not in raw:
            raise EpisodeLanguageError("%s missing required key %r" % (path, key))
    iso = _nonempty_str(raw["iso"], path + ".iso")
    label = _nonempty_str(raw["label"], path + ".label")
    if label == OFF_LABEL:
        raise EpisodeLanguageError(
            "%s.label cannot be %r -- Off is resolver state, not a row" % (path, OFF_LABEL))
    if not isinstance(raw["admitted"], bool):
        raise EpisodeLanguageError("%s.admitted must be bool" % path)
    if not isinstance(raw["sort_order"], int) or isinstance(raw["sort_order"], bool):
        raise EpisodeLanguageError("%s.sort_order must be int" % path)
    if not isinstance(raw["row_revision"], int) or isinstance(raw["row_revision"], bool):
        raise EpisodeLanguageError("%s.row_revision must be int" % path)
    if raw["row_revision"] < 1:
        raise EpisodeLanguageError("%s.row_revision must be >= 1" % path)
    native_header = _nonempty_str(raw["native_header"], path + ".native_header")
    authoring = _require_str_map(raw["authoring"], path + ".authoring", _REQUIRED_AUTHORING)
    spoken = _require_str_map(raw["spoken"], path + ".spoken", _REQUIRED_SPOKEN)
    credits = _require_str_map(raw["credits"], path + ".credits", _REQUIRED_CREDITS)
    captions = _require_str_map(raw["captions"], path + ".captions", _REQUIRED_CAPTIONS)
    engines = raw["engines"]
    if not isinstance(engines, dict) or not engines:
        raise EpisodeLanguageError("%s.engines must be a non-empty object" % path)
    if "kokoro" not in engines:
        raise EpisodeLanguageError("%s.engines must include kokoro (dance leader)" % path)
    kokoro = engines["kokoro"]
    if not isinstance(kokoro, dict):
        raise EpisodeLanguageError("%s.engines.kokoro must be an object" % path)
    _nonempty_str(kokoro.get("lang_code"), path + ".engines.kokoro.lang_code")
    voices = kokoro.get("voices")
    if not isinstance(voices, list) or not all(isinstance(v, str) for v in voices):
        raise EpisodeLanguageError("%s.engines.kokoro.voices must be a list[str]" % path)
    admission = raw["admission"]
    if not isinstance(admission, dict):
        raise EpisodeLanguageError("%s.admission must be an object" % path)
    for key in _REQUIRED_ADMISSION:
        if key not in admission:
            raise EpisodeLanguageError("%s.admission missing %r" % (path, key))
    if not _is_str_list(admission["source_bank_exclusions"]):
        raise EpisodeLanguageError("%s.admission.source_bank_exclusions must be list[str]" % path)
    if not _is_str_list(admission["readiness_extras"]):
        raise EpisodeLanguageError("%s.admission.readiness_extras must be list[str]" % path)
    min_voices = admission["min_voice_count"]
    if not isinstance(min_voices, int) or isinstance(min_voices, bool) or min_voices < 1:
        raise EpisodeLanguageError("%s.admission.min_voice_count must be int >= 1" % path)
    if bool(raw["admitted"]) and min_voices > len(voices):
        raise EpisodeLanguageError(
            "%s.admission.min_voice_count %d exceeds the %d Kokoro voices on the row"
            % (path, min_voices, len(voices)))
    return LanguageRow(
        iso=iso,
        label=label,
        admitted=bool(raw["admitted"]),
        sort_order=int(raw["sort_order"]),
        native_header=native_header,
        row_revision=int(raw["row_revision"]),
        authoring=dict(authoring),
        spoken=dict(spoken),
        credits=dict(credits),
        captions=dict(captions),
        engines=dict(engines),
        admission=dict(admission),
        raw=dict(raw),
    )


def validate_registry(payload: Any) -> tuple:
    """Return (rows, by_label, by_iso). Fail closed on shape / uniqueness."""
    if not isinstance(payload, dict):
        raise EpisodeLanguageError("episode_languages root must be an object")
    version = payload.get("schema_version")
    if version != _SCHEMA_VERSION:
        raise EpisodeLanguageError(
            "episode_languages.schema_version must be %d (got %r)" % (_SCHEMA_VERSION, version))
    rows_raw = payload.get("rows")
    if not isinstance(rows_raw, list) or not rows_raw:
        raise EpisodeLanguageError("episode_languages.rows must be a non-empty list")
    rows = []
    by_label = {}
    by_iso = {}
    by_kokoro_lang_code = {}
    for i, raw in enumerate(rows_raw):
        row = _validate_row(raw, i)
        if row.label in by_label:
            raise EpisodeLanguageError("duplicate language label %r" % row.label)
        if row.iso in by_iso:
            raise EpisodeLanguageError("duplicate language iso %r" % row.iso)
        lang_code = row.engines["kokoro"]["lang_code"]
        if lang_code in by_kokoro_lang_code:
            raise EpisodeLanguageError(
                "duplicate Kokoro lang_code %r on rows %r and %r"
                % (lang_code, by_kokoro_lang_code[lang_code], row.label))
        by_kokoro_lang_code[lang_code] = row.label
        by_label[row.label] = row
        by_iso[row.iso] = row
        rows.append(row)
    english = by_label.get("English")
    if english is None or english.iso != "en" or not english.admitted:
        raise EpisodeLanguageError(
            "registry must admit an English row (label='English', iso='en')")
    # Every admitted row must carry the same spoken/credits/caption keys as
    # English so Wave 1 admission tests stay mechanical.
    en_spoken = set(english.spoken)
    en_credits = set(english.credits)
    en_captions = set(english.captions)
    for row in rows:
        if not row.admitted:
            continue
        if set(row.spoken) != en_spoken:
            raise EpisodeLanguageError(
                "admitted row %r spoken keys must match English" % row.label)
        if set(row.credits) != en_credits:
            raise EpisodeLanguageError(
                "admitted row %r credits keys must match English" % row.label)
        if set(row.captions) != en_captions:
            raise EpisodeLanguageError(
                "admitted row %r captions keys must match English" % row.label)
        for key, value in row.spoken.items():
            if not str(value).strip():
                raise EpisodeLanguageError(
                    "admitted row %r spoken.%s is empty" % (row.label, key))
        for key, value in row.credits.items():
            if not str(value).strip():
                raise EpisodeLanguageError(
                    "admitted row %r credits.%s is empty" % (row.label, key))
    return tuple(rows), by_label, by_iso


def _read_payload(path: str = None) -> dict:
    target = path or REGISTRY_PATH
    try:
        with open(target, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise EpisodeLanguageError("cannot read %s: %s" % (target, exc)) from exc
    if text.startswith("\ufeff"):
        raise EpisodeLanguageError("%s must be UTF-8 without BOM" % target)
    try:
        return json.loads(text)
    except ValueError as exc:
        raise EpisodeLanguageError("%s is not valid JSON: %s" % (target, exc)) from exc


def load_registry(*, path: str = None, force: bool = False) -> tuple:
    """Cached (rows, by_label, by_iso) for the on-disk registry."""
    global _CACHE
    target = path or REGISTRY_PATH
    try:
        mtime_ns = os.stat(target).st_mtime_ns
    except OSError as exc:
        raise EpisodeLanguageError("cannot stat %s: %s" % (target, exc)) from exc
    if (not force and _CACHE is not None and _CACHE[0] == mtime_ns
            and _CACHE[1] == target):
        return _CACHE[2], _CACHE[3], _CACHE[4]
    rows, by_label, by_iso = validate_registry(_read_payload(target))
    _CACHE = (mtime_ns, target, rows, by_label, by_iso)
    return rows, by_label, by_iso


def reload_registry(*, path: str = None) -> tuple:
    return load_registry(path=path, force=True)


def dropdown_choices(*, path: str = None) -> list:
    """COMBO values: Off first, then admitted labels by sort_order, then label."""
    rows, _by_label, _by_iso = load_registry(path=path)
    admitted = [r for r in rows if r.admitted]
    admitted.sort(key=lambda r: (r.sort_order, r.label))
    return [OFF_LABEL] + [r.label for r in admitted]


def row_by_label(label: str, *, path: str = None) -> Optional[LanguageRow]:
    _rows, by_label, _by_iso = load_registry(path=path)
    return by_label.get(str(label or ""))


def row_by_iso(iso: str, *, path: str = None) -> Optional[LanguageRow]:
    _rows, _by_label, by_iso = load_registry(path=path)
    return by_iso.get(str(iso or ""))


def resolve_label(label, *, path: str = None) -> LanguageResolution:
    """Map a widget / legacy value to stamp behaviour.

    * ``Off`` (exact) -> stamp nothing
    * blank / None -> English row (legacy missing widget = feature on)
    * known label -> that row (must be admitted)
    * anything else -> EpisodeLanguageError
    """
    if label is None:
        text = ""
    elif not isinstance(label, str):
        raise EpisodeLanguageError(
            "episode_language must be a string (got %s)" % type(label).__name__)
    else:
        text = label.strip()
    if text == OFF_LABEL:
        return LanguageResolution(kind="off", row=None, stamp=False)
    rows, by_label, _by_iso = load_registry(path=path)
    if not text:
        english = by_label["English"]
        return LanguageResolution(kind="row", row=english, stamp=True)
    row = by_label.get(text)
    if row is None:
        raise EpisodeLanguageError("unknown episode_language %r" % text)
    if not row.admitted:
        raise EpisodeLanguageError(
            "episode_language %r is not admitted in this pack version" % text)
    return LanguageResolution(kind="row", row=row, stamp=True)


def _row_sha256(row: LanguageRow) -> str:
    blob = json.dumps(row.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def resolve_ledger(label, *, path: str = None) -> Optional[dict]:
    """Fields to stamp beside ``source_bank``, or None when Off.

    Never hand ``Off`` into Lemmy / cameo helpers -- callers must check
    ``stamp`` / None before treating the result as an iso.
    """
    resolved = resolve_label(label, path=path)
    if not resolved.stamp or resolved.row is None:
        return None
    row = resolved.row
    return {
        "episode_language": row.iso,
        "language_header": row.native_header,
        "episode_language_receipt": {
            "registry_id": REGISTRY_ID,
            "schema_version": _SCHEMA_VERSION,
            "row_revision": row.row_revision,
            "row_sha256": _row_sha256(row),
            "label": row.label,
        },
    }


# --------------------------------------------------------------------------- #
# Downstream reader -- every painter resolves the LEDGER, never the widget
# --------------------------------------------------------------------------- #


def iso_from_meta(meta, *, path: str = None) -> str:
    """The episode iso recorded on a ledger's ``meta``.

    An absent / blank stamp is English: that covers ``Off`` (stamp nothing),
    a graph saved before the widget existed, and every ledger frozen before
    this feature. Fails closed on a stamped iso the registry does not carry --
    a ledger naming a row this pack version cannot paint is not paintable.
    """
    if not isinstance(meta, dict):
        return ENGLISH_ISO
    iso = str(meta.get("episode_language") or "").strip()
    if not iso:
        return ENGLISH_ISO
    if row_by_iso(iso, path=path) is None:
        raise EpisodeLanguageError(
            "ledger episode_language %r has no row in this pack version" % iso)
    return iso


def row_from_meta(meta, *, path: str = None) -> LanguageRow:
    """The language row a ledger paints in. Never None -- English is the floor."""
    return row_by_iso(iso_from_meta(meta, path=path), path=path)


def spoken_text(meta, key: str, *, path: str = None) -> str:
    """A Python-authored on-air string in the ledger's language."""
    row = row_from_meta(meta, path=path)
    try:
        return row.spoken[key]
    except KeyError:
        raise EpisodeLanguageError(
            "row %r has no spoken.%s" % (row.label, key)) from None


def credits_text(meta, key: str, *, path: str = None) -> str:
    """An audience-facing credits chrome string in the ledger's language."""
    row = row_from_meta(meta, path=path)
    try:
        return row.credits[key]
    except KeyError:
        raise EpisodeLanguageError(
            "row %r has no credits.%s" % (row.label, key)) from None


def kokoro_config(row_or_meta, *, path: str = None) -> dict:
    """``engines["kokoro"]`` for a row or a ledger meta.

    The shared contract carries no ``kokoro_lang_code``; the adapter asks the
    row. Fails closed when the episode's language does not list Kokoro --
    Kokoro is the dance leader and the only engine admitted day 1.
    """
    row = (row_or_meta if isinstance(row_or_meta, LanguageRow)
           else row_from_meta(row_or_meta, path=path))
    config = (row.engines or {}).get("kokoro")
    if not isinstance(config, dict):
        raise EpisodeLanguageError(
            "row %r does not list the kokoro engine" % row.label)
    return config


def readiness_extra_ok(token: str) -> bool:
    """True when a CastLock readiness extra fully imports on THIS box.

    ``misaki[ja]`` / ``misaki[zh]`` are never an English-install tax: English
    rows list no extras, so this is not called on the default path.

    A module spec is not readiness. ``misaki.ja`` can have a discoverable file
    while importing it raises because ``pyopenjtalk`` is absent. Import the
    selected adapter so its transitive contract is exercised before TTS.
    """
    extra = str(token or "").strip()
    if not extra:
        return True
    if extra.startswith("misaki[") and extra.endswith("]"):
        sub = extra[7:-1].strip()
        if not sub:
            return False
        import importlib
        try:
            importlib.import_module("misaki.%s" % sub)
            return True
        except Exception:
            return False
    return False


def assert_readiness_extras(row: LanguageRow) -> None:
    """Fail closed when this box cannot serve the row's extras."""
    for extra in row.admission.get("readiness_extras") or []:
        if not readiness_extra_ok(extra):
            raise EpisodeLanguageError(
                "language %s needs readiness extra %s on this box "
                "(CastLock extra, never an English-install tax). Install it "
                "outside the render with this ComfyUI Python: "
                "python -m pip install %r"
                % (row.label, extra, extra))


# --------------------------------------------------------------------------- #
# Writer-side gates
# --------------------------------------------------------------------------- #


def check_source_bank_admission(row: Optional[LanguageRow], source_bank_id) -> None:
    """Fail loud when a fidelity bank meets a language that cannot carry it.

    The verbatim lanes (shakespeare, public_domain) perform the author's own
    words; a verbatim lane cannot also be a translation lane. Called at the
    writer BEFORE any LLM call, so the refusal costs nothing.

    ``None`` (``Off``) admits everything -- Off is today's path.
    """
    if row is None:
        return
    bank_id = str(source_bank_id or "").strip()
    if not bank_id:
        return
    excluded = row.admission.get("source_bank_exclusions") or []
    if bank_id in excluded:
        raise EpisodeLanguageError(
            "source_bank %r cannot be authored in %s (%s): that lane performs "
            "the author's own words, and a verbatim lane cannot also be a "
            "translation. Pick English, or pick another source bank."
            % (bank_id, row.label, row.iso))


def writer_language_instruction(row: Optional[LanguageRow]) -> str:
    """The native authoring instruction for the writer prompt.

    Authored, never translated: the model is asked to WRITE in the episode
    language, and is never handed an English draft to convert.
    """
    if row is None:
        return ""
    return str(row.authoring.get("writer_instruction") or "")


def native_authoring_instruction(meta, *, path: str = None) -> str:
    """Row-owned native authoring instruction, or "" for English/legacy.

    Strict by design: a stamped unknown language is structural corruption.
    Audience-facing callers that promise a fail-soft fallback catch around
    this leaf at their existing boundary.
    """
    row = row_from_meta(meta, path=path)
    if row.iso == ENGLISH_ISO:
        return ""
    return writer_language_instruction(row)


def lead_system(system: str, language_instruction: str) -> str:
    """Put the native instruction ahead of a system prompt.

    An empty instruction returns ``system`` unchanged, so every English and
    Off prompt stays byte-identical.
    """
    language_instruction = str(language_instruction or "").strip()
    if not language_instruction:
        return system
    return language_instruction + "\n\n" + system


def title_instruction(row: Optional[LanguageRow]) -> str:
    """The native title rule for the title-regen pass."""
    if row is None:
        return ""
    return str(row.authoring.get("title_instruction") or "")


def title_language_instruction(meta, *, path: str = None) -> str:
    """Native authoring plus title mechanics, or "" for English/legacy."""
    row = row_from_meta(meta, path=path)
    if row.iso == ENGLISH_ISO:
        return ""
    parts = (
        writer_language_instruction(row),
        title_instruction(row),
    )
    return " ".join(part.strip() for part in parts if part.strip())


# --------------------------------------------------------------------------- #
# Replay -- the ledger's iso wins, and drift is named
# --------------------------------------------------------------------------- #


class ReplayLanguageCheck(NamedTuple):
    """Outcome of comparing a replayed ledger against the live widget.

    ``row_drift`` is a receipt dict when the recorded row revision / hash no
    longer matches the row on disk, else None. Drift WARNS and continues on
    the current row; it never silently claims identity.
    """
    iso: str
    row: LanguageRow
    row_drift: Optional[dict]


def replay_language_check(meta, resolution: LanguageResolution,
                          *, path: str = None) -> ReplayLanguageCheck:
    """Validate a replay's recorded language against the widget.

    The LEDGER's iso wins -- a replay re-renders THAT episode. A widget that
    names a different language fails here, at the writer, BEFORE the cast
    block, so a Spanish widget can never re-cast an English Lemmy ledger.

    ``Off`` never compares: it stamps nothing and asks for today's path, so a
    replay under ``Off`` re-renders whatever the bundle recorded.
    """
    iso = iso_from_meta(meta, path=path)
    row = row_by_iso(iso, path=path)
    if resolution.stamp and resolution.row is not None:
        if resolution.row.iso != iso:
            raise EpisodeLanguageError(
                "replay language mismatch: the frozen ledger is %s (%s) and the "
                "episode_language widget asks for %s (%s). A replay re-renders "
                "the recorded episode -- set the widget to %r or to %r."
                % (row.label, iso, resolution.row.label, resolution.row.iso,
                   row.label, OFF_LABEL))
    recorded = (meta or {}).get("episode_language_receipt")
    drift = None
    if isinstance(recorded, dict):
        current_sha = _row_sha256(row)
        recorded_sha = str(recorded.get("row_sha256") or "")
        recorded_rev = recorded.get("row_revision")
        if recorded_sha and recorded_sha != current_sha:
            drift = {
                "registry_id": REGISTRY_ID,
                "iso": iso,
                "label": row.label,
                "recorded_row_revision": recorded_rev,
                "current_row_revision": row.row_revision,
                "recorded_row_sha256": recorded_sha,
                "current_row_sha256": current_sha,
                "resolution": "continued on the current row",
            }
    return ReplayLanguageCheck(iso=iso, row=row, row_drift=drift)
