"""The Scene Producer's direction: its schema, its names and its paths.

One module owns them (plan: ``planning/scene-producer/PLAN.md``, "One shared
module") so the Workflow Validator, Shot Lock, the image dispatcher, Batch and
the freezer all read a direction the same way. Radio Play Studio cannot import
the pack; it mirrors the schema below and checks its version.

WHAT A DIRECTION IS. Shot Lock's optional ``direction_json`` widget. Blank -- the
shipped value, and every graph saved before the widget existed -- means an
ordinary run, and nothing in this module refuses or changes anything for it.
Filled, it asks for one Scene Producer stage:

    {"version": 1,
     "project_id": "a1b2c3d4e5f6",          # lowercase letters, digits, hyphens
     "attempt_id": "0f8e...-uuid",           # unique per submitted run
     "stage": "stills" | "video" | "publish",
     "out": "03-video",                      # the bundle this run writes, numbered one
                                             # past the latest; none for publish
     "source": "<sha256 of the previous bundle's manifest.json>",  # none on the first run
     "scenes": {"<scene_id>": {"note": "..."}},
     "cards":  {"<shot_id>": {"picture_text": "...", "motion_text": "..."}},
     "takes":  {"characters": {"<char_id>": 1}, "stills": {"<shot_id>": 2},
                "clips": {"<shot_id>": 1}},
     "kept":   {"stills": ["<shot_id>"], "clips": ["<shot_id>"]}}

A WIDGET VALUE IS UNTRUSTED INPUT: it arrives in the body of ComfyUI's
unauthenticated ``/prompt``. Every name that becomes part of a path is matched
against a strict pattern here, and every path is confined to the producer's own
folder, before anything touches the disk.

THE IMAGE-ENGINE STAMP (chunk 3). A producer stills run records the per-slot
image engines it painted with on the durable ledger (meta
``producer_image_engines``); every producer freeze copies them into the bundle
manifest's ``producer`` section, so a video bundle still names the engines its
stills came from. A later run must resolve the same engines from its own video
director, or :func:`admit_prompt_direction` refuses it: a repaint with another
engine would mix two looks in one episode.

WHAT IS NOT HERE YET, and which build step adds it beside its caller (a check
with nothing to compare against is the defect the repo's own rules forbid):
``cards.json`` and the unpainted-edit check (chunk 3, with the repaint they
protect), the clip store path (chunk 4). Until then :func:`not_yet_supported`
refuses what a direction cannot get yet.

UTF-8, no BOM. Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ._otr_paths import (OtrPathContractError, confine_to_output_tree,
                             confine_to_read_roots, otr_episodes_root)
except ImportError:  # pragma: no cover -- flat (sys.path) load
    from _otr_paths import (OtrPathContractError, confine_to_output_tree,  # type: ignore
                            confine_to_read_roots, otr_episodes_root)

DIRECTION_VERSION = 1
STAGES = ("stills", "video", "publish")
WIDGET_NAME = "direction_json"
SHOT_LOCK_CLASS = "OTR_ShotLock"
WRITER_CLASS = "OTR_LedgerScriptWriter"
DIRECTOR_CLASS = "OTR_VideoDirector"
#: The three image slots the video director resolves, in its policy's order.
IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")
#: What the image director puts in a slot the video director left empty
#: (``otr_image_director.direct``).
DEFAULT_IMAGE_ENGINE = "flux_gen1"
#: The durable-ledger meta key a producer stills run writes its engines to.
META_IMAGE_ENGINES = "producer_image_engines"
#: The bundle manifest's producer section, and its version.
MANIFEST_PRODUCER_KEY = "producer"
MANIFEST_PRODUCER_VERSION = 1
MANIFEST_NAME = "manifest.json"

#: EVERY NAME IS CHECKED WITH ``fullmatch`` (Codex QA, 2026-10-22): ``$`` under
#: ``re.match`` also accepts a trailing newline, so ``"01-stills\n"`` would have
#: become a directory name. Digits are spelled ``[0-9]``, never ``\d``, which
#: also accepts non-ASCII digits.
#:
#: A project id names a folder: short, so bundle paths stay under Windows'
#: 260-character limit on a long install root.
PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{3,31}$")
#: ``NN-stills`` or ``NN-video``. Nothing else, which also keeps a bundle name
#: from ever colliding with the project's clip store.
OUT_RE = re.compile(r"^([0-9]{2})-(stills|video)$")
#: The one bundle name a project's first run may write.
FIRST_OUT = "01-stills"
ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
SOURCE_RE = re.compile(r"^[0-9a-f]{64}$")
#: Shot, scene and character ids as the ledger mints them: ``shot_shot_001_b2``,
#: ``shot_b000_music_open``, ``s01``, ``c03``. A letter or digit first, then
#: letters, digits, ``_`` or ``-`` -- no dot, colon or leading underscore, so
#: ``.``, ``..``, ``a:b`` and ``__proto__`` are refused before later build steps
#: put ids into file names (Sonnet QA, 2026-10-22).
ITEM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
#: Names Windows reserves for devices, refused as ids for the same reason. The
#: pattern above allows no extension, so a bare name is the whole case.
_DEVICE_NAMES = frozenset(["con", "prn", "aux", "nul"]
                          + ["com%d" % n for n in range(10)]
                          + ["lpt%d" % n for n in range(10)])
#: How much of a refused value a refusal quotes back.
_ECHO_CHARS = 80

MAX_DIRECTION_CHARS = 1_000_000
MAX_NOTE_CHARS = 2_000
MAX_CARD_TEXT_CHARS = 4_000
MAX_TAKE = 9_999

_TOP_KEYS = frozenset({"version", "project_id", "attempt_id", "stage", "out", "source",
                       "scenes", "cards", "takes", "kept"})
_CARD_KEYS = frozenset({"picture_text", "motion_text"})
_SCENE_KEYS = frozenset({"note"})
_TAKE_GROUPS = ("characters", "stills", "clips")
_KEPT_GROUPS = ("stills", "clips")

__all__ = [
    "DIRECTION_VERSION", "STAGES", "WIDGET_NAME", "DirectionError", "is_blank",
    "parse_direction", "check_stage_rules", "bundle_root", "project_dir", "FIRST_OUT",
    "project_bundle", "not_yet_supported", "RUNTIME_KEY", "attach_runtime",
    "runtime_direction", "freeze_stage", "execution_blocker",
    "bundle_path", "manifest_digest", "admit_prompt_direction",
    "IMAGE_SLOTS", "META_IMAGE_ENGINES", "MANIFEST_PRODUCER_KEY",
    "policy_image_engines", "prompt_image_engines", "manifest_producer_fields",
    "PAYLOAD_KEY", "RAW_TEXTS_KEY", "TAKES_KEY", "object_shot_id", "initial_raw_texts",
    "merged_raw_texts", "compose_picture", "painted_takes", "SUPERSEDED_KEY",
    "producer_replay_needs",
]


class DirectionError(ValueError):
    """A Scene Producer direction that cannot be run, and the plain reason why."""


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def is_blank(raw: Any) -> bool:
    """True for the ordinary-run value: absent, empty, or only whitespace."""
    return raw is None or (isinstance(raw, str) and not raw.strip())


def _shown(value: Any) -> str:
    """``repr(value)``, cut short: a refusal names the bad value without echoing
    a megabyte of it back (Sonnet QA, 2026-10-22: a 900,000-character project
    id produced a 900,068-character refusal)."""
    text = repr(value)
    return text if len(text) <= _ECHO_CHARS else text[:_ECHO_CHARS - 3] + "..."


def _require_dict(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise DirectionError("%s must be an object" % where)
    return value


def _refuse_unknown(value: dict, allowed: frozenset, where: str) -> None:
    unknown = sorted(str(k) for k in value if k not in allowed)
    if unknown:
        shown = ", ".join(_shown(k) for k in unknown[:5])
        more = " and %d more" % (len(unknown) - 5) if len(unknown) > 5 else ""
        raise DirectionError("%s has unknown field(s): %s%s" % (where, shown, more))


def _item_id(value: Any, where: str) -> str:
    if (not isinstance(value, str) or not ITEM_ID_RE.fullmatch(value)
            or value.lower() in _DEVICE_NAMES):
        raise DirectionError("%s is not a valid id: %s" % (where, _shown(value)))
    return value


def _no_duplicate_keys(pairs: list) -> dict:
    """``object_pairs_hook`` for the direction: a key named twice is refused.
    Plain ``json.loads`` keeps the last one silently, so a direction naming
    ``project_id`` twice ran on whichever came second (Sonnet QA, 2026-10-22)."""
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise DirectionError("direction_json names %s twice" % _shown(key))
        seen[key] = value
    return seen


def _text(value: Any, limit: int, where: str) -> str:
    if not isinstance(value, str):
        raise DirectionError("%s must be text" % where)
    if len(value) > limit:
        raise DirectionError("%s is longer than %d characters" % (where, limit))
    return value


def _take(value: Any, where: str) -> int:
    # bool is an int in Python; a take of True is a malformed direction, not 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise DirectionError("%s must be a whole number" % where)
    if not 0 <= value <= MAX_TAKE:
        raise DirectionError("%s must be between 0 and %d" % (where, MAX_TAKE))
    return value


def parse_direction(raw: Any) -> Optional[Dict[str, Any]]:
    """The validated direction, or ``None`` for a blank widget.

    Returns a normalized copy with every optional section present. Raises
    :class:`DirectionError` for anything that is not a well-formed version-1
    direction: unknown fields, bad names, out-of-range takes, text that is not
    text. Rules that need the ledger (unknown shot ids, motion on a card with no
    editable action) are applied where the ledger is, not here.
    """
    if is_blank(raw):
        return None
    if not isinstance(raw, str):
        raise DirectionError("direction_json must be literal text, not a link "
                             "from another node")
    if len(raw) > MAX_DIRECTION_CHARS:
        raise DirectionError("direction_json is larger than %d characters"
                             % MAX_DIRECTION_CHARS)
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except DirectionError:
        raise
    except RecursionError:
        # Arrays nested a few thousand deep fit inside the size limit and
        # exhaust the decoder's stack; refused as a direction, not raised as a
        # crash past the Workflow Validator (Sonnet QA, 2026-10-22).
        raise DirectionError("direction_json is nested too deeply to be a direction") \
            from None
    except ValueError as exc:
        raise DirectionError("direction_json is not valid JSON: %s" % exc) from None
    data = _require_dict(data, "direction_json")
    _refuse_unknown(data, _TOP_KEYS, "direction_json")

    version = data.get("version")
    if type(version) is not int or version != DIRECTION_VERSION:
        raise DirectionError("direction_json version must be %d, got %s"
                             % (DIRECTION_VERSION, _shown(version)))
    project_id = data.get("project_id")
    if not isinstance(project_id, str) or not PROJECT_ID_RE.fullmatch(project_id):
        raise DirectionError("project_id must be 4-32 lowercase letters, digits or "
                             "hyphens, got %s" % _shown(project_id))
    attempt_id = data.get("attempt_id")
    if not isinstance(attempt_id, str) or not ATTEMPT_ID_RE.fullmatch(attempt_id):
        raise DirectionError("attempt_id must be 8-64 letters, digits or hyphens, "
                             "got %s" % _shown(attempt_id))
    stage = data.get("stage")
    if not isinstance(stage, str) or stage not in STAGES:
        raise DirectionError("stage must be one of %s, got %s"
                             % (", ".join(STAGES), _shown(stage)))

    out = data.get("out")
    if stage == "publish":
        if out is not None:
            raise DirectionError("a publish run writes no bundle, so it takes no out")
    else:
        match = OUT_RE.fullmatch(out) if isinstance(out, str) else None
        if not match:
            raise DirectionError("out must look like 03-%s, got %s" % (stage, _shown(out)))
        if match.group(2) != stage:
            raise DirectionError("out %r does not match stage %r" % (out, stage))

    source = data.get("source")
    if source is not None and (not isinstance(source, str) or not SOURCE_RE.fullmatch(source)):
        raise DirectionError("source must be a lowercase sha256, got %s" % _shown(source))
    if source is None and stage != "stills":
        raise DirectionError("a %s run replays an earlier bundle, so it needs a source"
                             % stage)
    if source is None and out != FIRST_OUT:
        # A run with no source starts a project. Only one bundle name can do
        # that, so a source-less direction cannot start a second, disconnected
        # episode inside a project that already has one (Codex QA, 2026-10-22).
        raise DirectionError("a project's first run writes %s, got %r" % (FIRST_OUT, out))

    scenes: Dict[str, Dict[str, str]] = {}
    for sid, body in _require_dict(data.get("scenes", {}), "scenes").items():
        sid = _item_id(sid, "scene id")
        body = _require_dict(body, "scene %s" % sid)
        _refuse_unknown(body, _SCENE_KEYS, "scene %s" % sid)
        scenes[sid] = {"note": _text(body.get("note", ""), MAX_NOTE_CHARS,
                                     "scene %s note" % sid)}

    cards: Dict[str, Dict[str, str]] = {}
    for shot, body in _require_dict(data.get("cards", {}), "cards").items():
        shot = _item_id(shot, "card id")
        body = _require_dict(body, "card %s" % shot)
        _refuse_unknown(body, _CARD_KEYS, "card %s" % shot)
        cards[shot] = {key: _text(body[key], MAX_CARD_TEXT_CHARS, "card %s %s" % (shot, key))
                       for key in sorted(_CARD_KEYS) if key in body}
        if "picture_text" in cards[shot] and not cards[shot]["picture_text"].strip():
            # An empty ACTION is a valid edit (the shot simply has none); an
            # empty picture would hand the image model nothing to draw.
            raise DirectionError("card %s picture_text is empty; a still needs a "
                                 "description" % shot)

    takes_in = _require_dict(data.get("takes", {}), "takes")
    _refuse_unknown(takes_in, frozenset(_TAKE_GROUPS), "takes")
    takes: Dict[str, Dict[str, int]] = {}
    for group in _TAKE_GROUPS:
        takes[group] = {_item_id(k, "%s take id" % group): _take(v, "%s take for %s" % (group, k))
                        for k, v in _require_dict(takes_in.get(group, {}),
                                                  "takes.%s" % group).items()}

    kept_in = _require_dict(data.get("kept", {}), "kept")
    _refuse_unknown(kept_in, frozenset(_KEPT_GROUPS), "kept")
    kept: Dict[str, list] = {}
    for group in _KEPT_GROUPS:
        ids = kept_in.get(group, [])
        if not isinstance(ids, list):
            raise DirectionError("kept.%s must be a list" % group)
        seen = [_item_id(i, "kept.%s id" % group) for i in ids]
        if len(set(seen)) != len(seen):
            raise DirectionError("kept.%s lists an id twice" % group)
        kept[group] = seen

    if source is None and (scenes or cards or any(takes.values())
                           or any(kept.values())):
        # The first run writes the episode the cards are made from, so there
        # is nothing to edit, re-take or keep yet.
        raise DirectionError("a project's first run has no cards yet, so it carries "
                             "no edits, takes or keeps")
    return {"version": DIRECTION_VERSION, "project_id": project_id,
            "attempt_id": attempt_id, "stage": stage, "out": out, "source": source,
            "scenes": scenes, "cards": cards, "takes": takes, "kept": kept}


def check_stage_rules(direction: Dict[str, Any], replay_from: str) -> None:
    """The rules that tie a direction to the writer's ``replay_from``.

    The first Stills run is a fresh episode: no ``source`` and a blank
    ``replay_from``. Every later run replays the previous bundle, so it needs
    both. (Which bundle ``source`` must name is checked against the bundle
    itself, in :func:`admit_prompt_direction`.)
    """
    replay_from = str(replay_from or "").strip()
    if direction["source"] is None:
        if replay_from:
            raise DirectionError(
                "the first stills run writes and voices a new episode, so the writer's "
                "replay_from must be empty (it is %s)" % _shown(replay_from))
    elif not replay_from:
        raise DirectionError(
            "this %s run continues an earlier bundle, so the writer's replay_from must "
            "name it" % direction["stage"])


# --------------------------------------------------------------------------- #
# names and paths
# --------------------------------------------------------------------------- #

def bundle_root() -> Path:
    """Every producer bundle lives under ``<episodes>/_replay/p/``."""
    return Path(otr_episodes_root()) / "_replay" / "p"


def project_dir(project_id: str) -> Path:
    """One project's folder, confined to :func:`bundle_root`."""
    if not isinstance(project_id, str) or not PROJECT_ID_RE.fullmatch(project_id):
        raise DirectionError("not a valid project id: %r" % (project_id,))
    root = bundle_root()
    return Path(confine_to_output_tree(str(root / project_id), "producer project",
                                       roots=[str(root)]))


def bundle_path(project_id: str, out: str) -> Path:
    """Where the bundle named ``out`` of ``project_id`` lives (or will)."""
    if not isinstance(out, str) or not OUT_RE.fullmatch(out):
        raise DirectionError("not a valid bundle name: %r" % (out,))
    root = bundle_root()
    return Path(confine_to_output_tree(str(project_dir(project_id) / out), "producer bundle",
                                       roots=[str(root)]))


def project_bundle(project_id: str, replay_from: str) -> Path:
    """``replay_from`` resolved as the writer resolves it, and proven to be one of
    THIS project's bundles: a direct child of :func:`project_dir` whose name is
    a bundle name, by its typed path and by its real one.

    Without this a direction for one project could continue another project's
    bundle, or any ordinary replay bundle under the read roots (Codex QA,
    2026-10-22).
    """
    try:
        resolved = Path(confine_to_read_roots(replay_from, "replay_from"))
        project_dir(project_id)                   # the id, and its folder's confinement
    except OtrPathContractError as exc:
        raise DirectionError("replay_from is outside the permitted folders: %s" % exc) \
            from None
    # REAL PATHS ONLY (Sonnet QA, 2026-10-22). Comparing the typed spelling
    # refused the real-path spelling of the very same folder when the output
    # root sits behind a junction (the Comfy Desktop layout); checking only the
    # parent's real path admitted a leaf junction (``A/02-video`` leading to
    # ``B/01-stills``) and a project folder aliased to another project. So the
    # bundle's real parent must be the project's folder under the REAL bundle
    # root -- never wherever the project folder itself might lead -- and the
    # leaf must be that folder's own child under its own name.
    real = os.path.realpath(str(resolved))
    expected_parent = os.path.join(os.path.realpath(str(bundle_root())), project_id)
    if (os.path.normcase(os.path.dirname(real)) != os.path.normcase(expected_parent)
            or os.path.basename(real) != resolved.name
            or not OUT_RE.fullmatch(resolved.name)):
        raise DirectionError(
            "replay_from %s is not one of project %s's bundles; a producer run continues "
            "its own project's latest bundle" % (_shown(replay_from), project_id))
    return resolved


def _latest_bundle(pdir: Path, project_id: str) -> Optional[str]:
    """The name of the project's highest-numbered bundle, or ``None`` while it
    has none: no folder yet, or only a folder a refused or crashed freeze left
    behind (its ``.NN-stage.building`` sibling is not a bundle). Two bundles
    sharing the top number are refused, because which is latest would be a
    guess."""
    if not os.path.lexists(str(pdir)):
        return None
    try:
        names = sorted(entry.name for entry in os.scandir(str(pdir))
                       if OUT_RE.fullmatch(entry.name))
    except OSError as exc:
        raise DirectionError("project %s's folder cannot be read: %s"
                             % (project_id, exc)) from None
    if not names:
        return None
    top = max(int(name[:2]) for name in names)
    tops = [name for name in names if int(name[:2]) == top]
    if len(tops) > 1:
        raise DirectionError("project %s has two bundles numbered %02d (%s), so which one "
                             "is latest cannot be told" % (project_id, top, ", ".join(tops)))
    return tops[0]


def manifest_digest(bundle_dir: str) -> str:
    """SHA-256 of a bundle's ``manifest.json`` bytes: the bundle's identity.

    ``bundle_dir`` is resolved exactly as the writer resolves ``replay_from``
    (relative to ComfyUI's output folder) and must sit inside the read roots.
    """
    bundle = confine_to_read_roots(bundle_dir, "replay_from")
    path = os.path.join(bundle, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise DirectionError("no %s under %s" % (MANIFEST_NAME, bundle))
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# --------------------------------------------------------------------------- #
# the image-engine stamp and its check
# --------------------------------------------------------------------------- #

def policy_image_engines(image_policy: Any) -> Dict[str, str]:
    """The per-slot engine ids an image policy (the image director's output,
    which the dispatcher paints with) names: what a producer stills run
    stamps. A slot entry is ``{"engine_id": ...}`` or a bare id."""
    models = (image_policy or {}).get("image_models") if isinstance(image_policy, dict) else None
    models = models if isinstance(models, dict) else {}
    out = {}
    for slot in IMAGE_SLOTS:
        entry = models.get(slot)
        out[slot] = str((entry.get("engine_id") if isinstance(entry, dict) else entry) or "")
    return out


def prompt_image_engines(prompt: dict) -> Dict[str, str]:
    """The per-slot image engines a queued prompt will paint with: the video
    director's three picks, resolved exactly as it resolves them
    (``OTRVideoDirector._image_picks``: a "+ Add Custom" pick takes its
    ``custom_models_json`` entry), then given the image director's default for
    an empty slot. A producer run types these, like its direction; a pick
    wired from another node cannot be checked, so it is refused."""
    directors = _nodes_of(prompt, DIRECTOR_CLASS)
    if len(directors) != 1:
        raise DirectionError("a Scene Producer run needs exactly one %s in the graph, "
                             "found %d" % (DIRECTOR_CLASS, len(directors)))
    node_id, node = directors[0]
    inputs = node.get("inputs") or {}
    for name in IMAGE_SLOTS + ("custom_models_json",):
        if isinstance(inputs.get(name), list):
            raise DirectionError("%s node %s takes %s from another node; a producer run "
                                 "must type it" % (DIRECTOR_CLASS, node_id, name))
    try:
        from .otr_video_director import OTRVideoDirector
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from otr_video_director import OTRVideoDirector  # type: ignore
    picked = OTRVideoDirector._image_picks(
        inputs.get("custom_models_json", ""),
        {slot: inputs.get(slot, "") for slot in IMAGE_SLOTS})
    return {slot: (picked[slot].strip() if isinstance(picked.get(slot), str)
                   and picked[slot].strip() else DEFAULT_IMAGE_ENGINE)
            for slot in IMAGE_SLOTS}


def manifest_producer_fields(direction: Dict[str, Any]) -> Dict[str, Any]:
    """The direction's identity as a producer freeze records it in the bundle
    manifest. The freezer adds the image-engine stamp from the ledger."""
    return {"version": MANIFEST_PRODUCER_VERSION, "project_id": direction["project_id"],
            "attempt_id": direction["attempt_id"], "stage": direction["stage"],
            "out": direction["out"], "source": direction["source"]}


def _check_image_engines(prompt: dict, manifest: dict, bundle_name: str) -> None:
    """A later run paints, or replays stills painted, with the bundle's own
    engines -- refused otherwise, before any model loads."""
    producer = manifest.get(MANIFEST_PRODUCER_KEY)
    stamped = producer.get("image_engines") if isinstance(producer, dict) else None
    if (not isinstance(stamped, dict) or set(stamped) != set(IMAGE_SLOTS)
            or not all(isinstance(v, str) for v in stamped.values())):
        raise DirectionError(
            "bundle %s carries no image-engine stamp, so a producer run cannot continue "
            "it; it was not frozen by a producer stills run" % bundle_name)
    queued = prompt_image_engines(prompt)
    differ = ["%s %s (the bundle used %s)" % (slot, _shown(queued[slot]), _shown(stamped[slot]))
              for slot in IMAGE_SLOTS if queued[slot] != stamped[slot]]
    if differ:
        raise DirectionError(
            "the video director's image engines differ from bundle %s's: %s; a producer "
            "run keeps the engines its stills were painted with" % (bundle_name,
                                                                   "; ".join(differ)))


# --------------------------------------------------------------------------- #
# what a producer stills run paints from: raw texts, takes, keeps
# --------------------------------------------------------------------------- #

#: Kept on the ledger's images section by a producer stills run: the payload
#: node 89 handed the first one, the raw texts every later one composes from,
#: and the takes it painted with.
PAYLOAD_KEY = "producer_prompt_payload"
RAW_TEXTS_KEY = "producer_raw_texts"
TAKES_KEY = "producer_takes"
#: The earlier stills of each object, kept beside the one current row per
#: object so an undone edit is a cache hit on the still it restores.
SUPERSEDED_KEY = "producer_superseded"


def object_shot_id(obj: Any) -> Optional[str]:
    """The card a still object answers to: the shot of the beat whose own
    scene still it is (``still_<beat>`` -> ``shot_<beat>``, as Shot Lock names
    shots), or None. Portraits have no card; a 3D lane's mesh and plate objects
    and the dispatcher's jump-segment clones are not the card's picture."""
    if not isinstance(obj, dict):
        return None
    beat = str(obj.get("beat_id") or "")
    if not beat or str(obj.get("object_id") or "") != "still_%s" % beat:
        return None
    return "shot_%s" % beat


def initial_raw_texts(payload: Any) -> Dict[str, Dict[str, str]]:
    """The raw picture text of every card in a first payload -- the text node
    89 wrote for its still -- and no scene notes yet."""
    cards: Dict[str, str] = {}
    objects = (payload or {}).get("objects") if isinstance(payload, dict) else None
    for obj in objects or []:
        shot = object_shot_id(obj)
        if shot and shot not in cards:
            cards[shot] = str(obj.get("prompt") or "")
    return {"cards": cards, "scenes": {}}


def merged_raw_texts(previous: Any, direction: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """The raw texts a run paints from: the previous bundle's, with each picture
    text and scene note the direction names put in its place. What the
    direction leaves out keeps its previous text; an empty note it sends clears
    the old one."""
    prev = previous if isinstance(previous, dict) else {}
    cards = {str(k): str(v) for k, v in dict(prev.get("cards") or {}).items()}
    scenes = {str(k): str(v) for k, v in dict(prev.get("scenes") or {}).items()}
    for shot, body in (direction.get("cards") or {}).items():
        if "picture_text" in body:
            cards[shot] = body["picture_text"]
    for scene, body in (direction.get("scenes") or {}).items():
        scenes[scene] = body.get("note", "")
    return {"cards": cards, "scenes": scenes}


def compose_picture(picture_text: Any, note: Any) -> str:
    """A still's effective prompt: its picture text, then its scene's note.
    Composed afresh from the two raw texts on every run, so a replaced or
    cleared note never leaves its old words behind."""
    picture = str(picture_text or "").strip()
    note = str(note or "").strip()
    if picture and note:
        return "%s, %s" % (picture, note)
    return picture or note


def painted_takes(direction: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """The still takes a stills run paints with (characters and stills; clip
    takes belong to the video stage), as the ledger keeps them."""
    takes = direction.get("takes") or {}
    return {"characters": dict(takes.get("characters") or {}),
            "stills": dict(takes.get("stills") or {})}


# --------------------------------------------------------------------------- #
# the queue-time admission the Workflow Validator runs
# --------------------------------------------------------------------------- #

def _nodes_of(prompt: dict, class_type: str) -> list:
    return [(str(nid), node) for nid, node in prompt.items()
            if isinstance(node, dict) and node.get("class_type") == class_type]


def admit_prompt_direction(prompt: Any) -> Optional[Dict[str, Any]]:
    """Check a queued prompt's Scene Producer direction before anything runs.

    Returns ``None`` for an ordinary run -- no Shot Lock, or every Shot Lock's
    widget blank or absent -- and refuses nothing then. Otherwise returns
    ``{"direction": ..., "replay_from": ..., "manifest": ... or None}`` or raises
    :class:`DirectionError` naming the problem: more than one Shot Lock or
    writer, a direction or ``replay_from`` wired from another node instead of
    typed, a malformed direction, an ``out`` that already exists (the same run
    submitted twice; execution is sequential, so the second copy reaches this
    check only after the first has frozen its bundle), a first run for a
    project that already has a bundle, or a later run that does not continue
    the project's LATEST bundle -- verified file by file and matching
    ``source`` -- into the next bundle number.
    """
    if not isinstance(prompt, dict):
        return None
    locks = _nodes_of(prompt, SHOT_LOCK_CLASS)
    raws = [(nid, (node.get("inputs") or {}).get(WIDGET_NAME)) for nid, node in locks]
    if all(is_blank(raw) for _nid, raw in raws):
        return None
    if len(locks) != 1:
        raise DirectionError("a Scene Producer run needs exactly one %s in the graph, "
                             "found %d" % (SHOT_LOCK_CLASS, len(locks)))
    lock_id, raw = raws[0]
    if isinstance(raw, list):
        raise DirectionError("%s node %s takes direction_json from another node; a "
                             "producer run must type it into the widget"
                             % (SHOT_LOCK_CLASS, lock_id))
    direction = parse_direction(raw)

    writers = _nodes_of(prompt, WRITER_CLASS)
    if len(writers) != 1:
        raise DirectionError("a Scene Producer run needs exactly one %s in the graph, "
                             "found %d" % (WRITER_CLASS, len(writers)))
    writer_id, writer = writers[0]
    replay_from = (writer.get("inputs") or {}).get("replay_from", "")
    if isinstance(replay_from, list):
        raise DirectionError("%s node %s takes replay_from from another node; a producer "
                             "run must type it into the widget" % (WRITER_CLASS, writer_id))
    replay_from = str(replay_from or "").strip()
    check_stage_rules(direction, replay_from)

    manifest = None
    project_id = direction["project_id"]
    try:
        pdir = project_dir(project_id)
        target = (bundle_path(project_id, direction["out"])
                  if direction["out"] is not None else None)
    except OtrPathContractError as exc:
        raise DirectionError("the project folder is outside the producer folder: %s"
                             % exc) from None
    if target is not None and os.path.lexists(str(target)):
        raise DirectionError(
            "bundle %s of project %s already exists: this run was already submitted, "
            "or the direction reuses a bundle name" % (direction["out"], project_id))
    latest = _latest_bundle(pdir, project_id)
    if direction["source"] is None:
        # A PROJECT STARTS WITH ITS FIRST BUNDLE, NOT WITH ITS FOLDER (Sonnet
        # QA, 2026-10-22). A refused first freeze, or a crash in the middle of
        # one, can leave the folder behind with no bundle in it, and the plan
        # promises that run a retry.
        if latest is not None:
            raise DirectionError(
                "project %s already has bundle %s: continue it with source and "
                "replay_from, or start a new project" % (project_id, latest))
    else:
        bundle = project_bundle(project_id, replay_from)
        if latest is None:
            raise DirectionError("project %s has no bundle yet, so there is nothing to "
                                 "continue; its first run carries no source" % project_id)
        if bundle.name != latest:
            # The plan's stage table: every later run replays the LATEST bundle.
            # A stale submit that continued an older one would fork the
            # project's history, and "latest" would stop meaning anything.
            raise DirectionError(
                "replay_from names bundle %s, but project %s's latest bundle is %s; a "
                "producer run continues the latest one" % (bundle.name, project_id, latest))
        try:
            from .production_ledger import ReplayBundleError, load_replay_manifest
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from production_ledger import (ReplayBundleError,  # type: ignore
                                           load_replay_manifest)
        try:
            manifest = load_replay_manifest(str(bundle))
            digest = manifest_digest(str(bundle))
        except (ReplayBundleError, OtrPathContractError) as exc:
            raise DirectionError("the bundle in replay_from cannot be trusted: %s" % exc) \
                from None
        if digest != direction["source"]:
            raise DirectionError(
                "replay_from %s is not the bundle this direction continues (its manifest "
                "is %s, the direction names %s)" % (_shown(replay_from), digest[:12],
                                                    direction["source"][:12]))
        _check_image_engines(prompt, manifest, bundle.name)
        if direction["out"] is not None:
            # Bundles are numbered in the order they were made, one apart, so
            # "latest" is always the highest number and no number is skipped.
            following = int(latest[:2]) + 1
            if following > 99:
                raise DirectionError("project %s has used every bundle number up to 99; "
                                     "start a new project" % project_id)
            if int(direction["out"][:2]) != following:
                raise DirectionError(
                    "out %s does not follow project %s's latest bundle %s; this run "
                    "writes %02d-%s" % (direction["out"], project_id, latest, following,
                                        direction["stage"]))
    return {"direction": direction, "replay_from": replay_from, "manifest": manifest}


# --------------------------------------------------------------------------- #
# readiness: what a producer replay must have ready
# --------------------------------------------------------------------------- #

def producer_replay_needs(prompt: Any) -> Optional[Dict[str, Any]]:
    """What a queued Scene Producer REPLAY renders with, for node 63's three
    readiness gates -- the cloud-slug check, the wallet estimate and the weight
    downloads -- or ``None`` when the prompt is not one: no Shot Lock, a blank
    direction, or a project's first run, which plans as the ordinary live run
    it is.

    An ordinary replay renders video only, so those gates read the video
    director's picks on a replay. A producer replay is narrower and more exact:

    * a STILLS replay paints, with the image engines its bundle's stills were
      actually painted with, and renders no video;
    * a VIDEO replay films its frozen plan's lanes, whatever the live
      dropdowns say;
    * PUBLISH renders nothing.

    Both lists come from the bundle manifest's producer section, which node 63's
    admission has verified file by file before any gate runs. No writer, voice
    or music engine is needed by any replay.
    """
    if not isinstance(prompt, dict):
        return None
    locks = _nodes_of(prompt, SHOT_LOCK_CLASS)
    raws = [(nid, (node.get("inputs") or {}).get(WIDGET_NAME)) for nid, node in locks]
    if all(is_blank(raw) for _nid, raw in raws):
        return None
    if len(locks) != 1 or isinstance(raws[0][1], list):
        raise DirectionError("a Scene Producer run needs exactly one %s with a typed "
                             "direction" % SHOT_LOCK_CLASS)
    direction = parse_direction(raws[0][1])
    if direction["source"] is None:
        return None
    writers = _nodes_of(prompt, WRITER_CLASS)
    replay_from = (writers[0][1].get("inputs") or {}).get("replay_from", "") if writers else ""
    try:
        bundle = confine_to_read_roots(str(replay_from or "").strip(), "replay_from")
        with open(os.path.join(bundle, MANIFEST_NAME), "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OtrPathContractError, OSError, ValueError) as exc:
        raise DirectionError("the bundle in replay_from cannot be read: %s" % exc) from None
    producer = manifest.get(MANIFEST_PRODUCER_KEY) if isinstance(manifest, dict) else None
    producer = producer if isinstance(producer, dict) else {}
    stage = direction["stage"]
    return {
        "stage": stage,
        "image_engines": (sorted(str(e) for e in producer.get("image_engines_used") or [])
                          if stage == "stills" else []),
        "video_engines": (sorted(str(e) for e in producer.get("video_engines") or [])
                          if stage == "video" else []),
    }


# --------------------------------------------------------------------------- #
# what this build can run
# --------------------------------------------------------------------------- #

def not_yet_supported(direction: Dict[str, Any]) -> Optional[str]:
    """What a direction asks for that this build cannot honour yet, in plain
    words, or ``None``. A direction is refused rather than run with part of it
    silently ignored: an edit that does nothing would look like a broken edit.

    Each arrives with the build step that makes it real (plan, Build order):
    Publish needs clip reuse, or it would film every shot again; picture edits,
    scene notes, takes and keeps need the repaint. ACTION edits
    (``motion_text``) are real: Shot Lock recomposes them on replay.
    """
    if direction["stage"] == "publish":
        return ("Publish is not available in this build yet: without clip reuse it "
                "would film every shot again")
    asked = [label for label, present in (
        ("scene notes", bool(direction["scenes"])),
        ("picture edits", any("picture_text" in body
                              for body in direction["cards"].values())),
        ("takes", any(direction["takes"].values())),
        ("kept cards", any(direction["kept"].values())),
    ) if present]
    if asked:
        return ("%s are not available in this build yet; this build runs the stills "
                "and video stages with action edits only" % ", ".join(asked))
    return None


# --------------------------------------------------------------------------- #
# the runtime key: the direction on the wire, never on disk
# --------------------------------------------------------------------------- #

#: Shot Lock puts the parsed direction on the ledger it passes downstream under
#: this top-level key; the image prompt node, the dispatcher and Batch read it
#: from their wire input. The durable save copies named sections only, so the
#: key never reaches disk, and a freeze reads the disk ledger, so no bundle
#: carries it. A blank widget means no key at all.
RUNTIME_KEY = "_otr_producer"


def attach_runtime(ledger: dict, direction: Optional[Dict[str, Any]]) -> dict:
    """Put ``direction`` on the wire ledger, or remove any key when it is None."""
    if direction is None:
        ledger.pop(RUNTIME_KEY, None)
    else:
        ledger[RUNTIME_KEY] = direction
    return ledger


def runtime_direction(ledger: Any) -> Optional[Dict[str, Any]]:
    """The direction Shot Lock put on this wire ledger, or ``None``."""
    if not isinstance(ledger, dict):
        return None
    direction = ledger.get(RUNTIME_KEY)
    if (not isinstance(direction, dict) or direction.get("version") != DIRECTION_VERSION
            or direction.get("stage") not in STAGES):
        return None
    return direction


def freeze_stage(direction: Dict[str, Any], episode_id: str, *,
                 include_render_outputs: bool) -> Path:
    """Freeze the finished stage's workspace into the bundle the direction
    names. The names are re-validated here (``bundle_path``), because this runs
    in a node and the wire ledger is only as trustworthy as its sender. The
    source commit is ``unknown``: inside the server there is no honest way to
    know it, and a package version is not a commit."""
    try:
        from ._otr_paths import otr_audio_dir
        from ._otr_replay_freezer import freeze
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from _otr_paths import otr_audio_dir  # type: ignore
        from _otr_replay_freezer import freeze  # type: ignore
    ep_dir = Path(otr_audio_dir(str(episode_id or ""))).parent
    dest = bundle_path(direction["project_id"], direction["out"])
    return freeze(ep_dir, None, dest=dest, source_commit="unknown",
                  include_render_outputs=include_render_outputs,
                  producer=manifest_producer_fields(direction))


def execution_blocker() -> Any:
    """ComfyUI's silent blocker: every node downstream of the one returning it
    is skipped, and the prompt still ends as a success. Always given ``None``;
    a message would turn the skip into an execution error. Imported lazily so
    this module loads outside ComfyUI."""
    from comfy_execution.graph_utils import ExecutionBlocker  # type: ignore
    return ExecutionBlocker(None)
