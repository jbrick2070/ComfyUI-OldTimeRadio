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

CARDS.JSON AND UNPAINTED EDITS (chunk 3). Every producer freeze writes
``cards.json``, the page's view of its bundle, and each card still records the
digest of what it was painted from. A replay's direction may name only the
cards, scenes and cast that file shows, and only the edits each card can take
(:func:`check_direction_ids`). A Video or Publish run whose direction asks for
a still its card does not show -- an edited picture or note, a raised take --
is refused (:func:`unpainted_shots`) unless the card is kept, and so is one
that changes a character's take before it is painted
(:func:`unpainted_characters`).

CLIP REUSE (chunk 4a) lives beside this module, in ``_otr_producer_clips``: a
Video run films only the shots whose causal key changed. Since chunk 4b a clip
take films a shot again, a kept clip -- or the floor a shot fell to -- is used
as it is, and ``cards.json`` shows each card's clip and its state.

PUBLISH (chunk 4c) films nothing: node 63 refuses one while a card is not
filmed as it stands (:func:`unfilmed_shots`) unless it is kept, and the render
refuses a shot it finds no clip for before any weights load. Every part of a
direction is now built, so none is refused for being unbuilt.

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
    from ._otr_shared import still_receipt as _still_receipt
except ImportError:  # pragma: no cover -- flat (sys.path) load
    from _otr_paths import (OtrPathContractError, confine_to_output_tree,  # type: ignore
                            confine_to_read_roots, otr_episodes_root)
    from _otr_shared import still_receipt as _still_receipt  # type: ignore

DIRECTION_VERSION = 1
STAGES = ("stills", "video", "publish")
WIDGET_NAME = "direction_json"
SHOT_LOCK_CLASS = "OTR_ShotLock"
WRITER_CLASS = "OTR_LedgerScriptWriter"
DIRECTOR_CLASS = "OTR_VideoDirector"
#: The three image slots the video director resolves, in its policy's order.
IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")
#: The durable-ledger meta key a producer stills run writes its engines to.
META_IMAGE_ENGINES = "producer_image_engines"
#: ...and the image seed configuration, locked for the project like the
#: engines: every still's seed (and so its cache key) derives from it.
META_IMAGE_SEED = "producer_image_seed"
IMAGE_DIRECTOR_CLASS = "OTR_ImageDirector"
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
    "project_bundle", "unfilmed_shots", "RUNTIME_KEY", "attach_runtime",
    "runtime_direction", "freeze_stage", "execution_blocker",
    "bundle_path", "manifest_digest", "admit_prompt_direction",
    "IMAGE_SLOTS", "META_IMAGE_ENGINES", "MANIFEST_PRODUCER_KEY",
    "policy_image_engines", "prompt_image_engines", "manifest_producer_fields",
    "PAYLOAD_KEY", "RAW_TEXTS_KEY", "TAKES_KEY", "object_shot_id", "initial_raw_texts",
    "merged_raw_texts", "compose_picture", "painted_takes", "SUPERSEDED_KEY",
    "producer_replay_needs", "CARDS_NAME", "ROW_INPUT_HASH", "payload_object_hash",
    "still_input_digest", "card_still_digest", "build_cards", "unpainted_shots",
    "META_IMAGE_SEED", "policy_image_seed", "prompt_image_seed", "check_policy_matches_stamp",
    "action_is_editable", "check_direction_ids", "unpainted_characters",
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


def _typed_inputs(prompt: dict, class_type: str, names: tuple) -> Dict[str, Any]:
    """The one ``class_type`` node's inputs, refused when the graph has none or
    several, or when one of ``names`` is wired from another node: a producer
    run types every value its checks read."""
    nodes = _nodes_of(prompt, class_type)
    if len(nodes) != 1:
        raise DirectionError("a Scene Producer run needs exactly one %s in the graph, "
                             "found %d" % (class_type, len(nodes)))
    node_id, node = nodes[0]
    inputs = node.get("inputs") or {}
    for name in names:
        if isinstance(inputs.get(name), list):
            raise DirectionError("%s node %s takes %s from another node; a producer run "
                                 "must type it" % (class_type, node_id, name))
    return inputs


def prompt_image_engines(prompt: dict) -> Dict[str, str]:
    """The per-slot image engines a queued prompt will paint with, resolved by
    the two directors' own code in the order they run: the video director's
    three picks (``OTRVideoDirector._image_picks``: a "+ Add Custom" pick takes
    its ``custom_models_json`` entry), then the image director's
    (``OTRImageDirector.resolve_image_models``: the gen-1 engine for an empty
    slot, a sentinel still unresolved from its own ``custom_models_json``, and
    the role check).

    BOTH DIRECTORS, NOT ONE (Sonnet QA, 2026-10-22): resolved by the video
    director alone, a custom pick declared only on the image director came out
    here as the sentinel itself while the stills were painted with the declared
    engine, so the project's own stamp refused every later run."""
    inputs = _typed_inputs(prompt, DIRECTOR_CLASS, IMAGE_SLOTS + ("custom_models_json",))
    image_inputs = _typed_inputs(prompt, IMAGE_DIRECTOR_CLASS, ("custom_models_json",))
    try:
        from .otr_image_director import OTRImageDirector, _registry_descriptors
        from .otr_video_director import OTRVideoDirector
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from otr_image_director import (OTRImageDirector,  # type: ignore
                                        _registry_descriptors)
        from otr_video_director import OTRVideoDirector  # type: ignore
    picked = OTRVideoDirector._image_picks(
        inputs.get("custom_models_json", ""),
        {slot: inputs.get(slot, "") for slot in IMAGE_SLOTS})
    custom = OTRImageDirector._parse_json_obj(
        image_inputs.get("custom_models_json", "{}"), "custom_models_json", [])
    try:
        resolved = OTRImageDirector.resolve_image_models(
            picked, custom, _registry_descriptors(), [])
    except ValueError as exc:
        raise DirectionError(str(exc)) from None
    return {slot: str((resolved.get(slot) or {}).get("engine_id") or "")
            for slot in IMAGE_SLOTS}


def _seed_config(mode: Any, request_seed: Any) -> Dict[str, Any]:
    """A seed configuration as the dispatcher reads one
    (``resolve_seed_and_mode``): the mode, ``request_hash`` when empty, the
    request seed as an integer, 0 when empty -- and the two environment
    settings that also move a still's seed, read by the dispatcher's own
    reader (``_otr_shared.still_seed_settings``; Codex QA, 2026-10-22). This
    process paints and checks alike, so a setting changed between two runs
    of a project is a changed seed, and refused like one."""
    try:
        from ._otr_shared import still_seed_settings as _settings
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from _otr_shared import still_seed_settings as _settings  # type: ignore
    try:
        seed = int(request_seed or 0)
    except (TypeError, ValueError):
        seed = 0
    return {"mode": str(mode or "request_hash"), "request_seed": seed,
            "bookend_seed": _settings.bookend_seed(),
            "identity_seed": _settings.identity_seed_enabled()}


def policy_image_seed(image_policy: Any) -> Dict[str, Any]:
    """The seed configuration an image policy paints with (its ``seed``
    section, which the image director writes): what a producer stills run
    stamps beside its engines."""
    seed = (image_policy or {}).get("seed") if isinstance(image_policy, dict) else None
    seed = seed if isinstance(seed, dict) else {}
    return _seed_config(seed.get("mode"), seed.get("request_seed"))


def prompt_image_seed(prompt: dict) -> Dict[str, Any]:
    """The seed configuration a queued prompt will paint with: the image
    director's ``seed_mode`` and ``request_seed`` widgets, typed, as it writes
    them into the policy."""
    inputs = _typed_inputs(prompt, IMAGE_DIRECTOR_CLASS, ("seed_mode", "request_seed"))
    return _seed_config(inputs.get("seed_mode"), inputs.get("request_seed"))


def manifest_producer_fields(direction: Dict[str, Any]) -> Dict[str, Any]:
    """The direction's identity as a producer freeze records it in the bundle
    manifest. The freezer adds the image-engine stamp from the ledger."""
    return {"version": MANIFEST_PRODUCER_VERSION, "project_id": direction["project_id"],
            "attempt_id": direction["attempt_id"], "stage": direction["stage"],
            "out": direction["out"], "source": direction["source"],
            "kept": {"stills": list(direction["kept"]["stills"]),
                     "clips": list(direction["kept"]["clips"])}}


def _read_cards(bundle: Path) -> Dict[str, Any]:
    """The bundle's ``cards.json`` -- a member the manifest lists, so admission
    has already hashed it -- or a refusal when a producer bundle carries none."""
    path = os.path.join(str(bundle), CARDS_NAME)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cards = json.load(fh)
    except (OSError, ValueError) as exc:
        raise DirectionError("bundle %s has no readable %s (%s); it was not frozen by a "
                             "producer run" % (bundle.name, CARDS_NAME, exc)) from None
    if not isinstance(cards, dict) or cards.get("version") != CARDS_VERSION:
        raise DirectionError("bundle %s's %s is not version %d"
                             % (bundle.name, CARDS_NAME, CARDS_VERSION))
    return cards


def _check_image_engines(prompt: dict, manifest: dict, bundle_name: str) -> None:
    """A later run paints, or replays stills painted, with the bundle's own
    engines and seed configuration -- refused otherwise, before any model
    loads. (The dispatcher repeats the check against the policy it actually
    receives: :func:`check_policy_matches_stamp`.)"""
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
    # THE SEED CONFIGURATION IS LOCKED THE SAME WAY (Codex QA, 2026-10-22):
    # every still's seed, and so its cache key, derives from it, so a changed
    # widget would repaint every still with no edit at all.
    stamped_seed = producer.get("image_seed")
    if not isinstance(stamped_seed, dict):
        raise DirectionError(
            "bundle %s carries no image-seed stamp, so a producer run cannot continue it; "
            "it was not frozen by a producer stills run" % bundle_name)
    queued_seed = prompt_image_seed(prompt)
    # Compared with the stamp AS STAMPED: rebuilt here, its environment half
    # would be read from this run's environment and always agree.
    if queued_seed != stamped_seed:
        raise DirectionError(
            "the image director's seed (%s) differs from the one bundle %s was painted "
            "with (%s); a producer run keeps its stills' seed" % (
                _shown(queued_seed), bundle_name, _shown(stamped_seed)))


def check_policy_matches_stamp(image_policy: Any, meta: Any) -> Optional[str]:
    """THE RUN-TIME HALF OF THE CHECK (Codex QA, 2026-10-22): the policy a
    producer stills REPLAY actually paints with, against the engines and seed
    its ledger was stamped with. Node 63 reads the video and image directors'
    widgets at queue time, before anything loads; this catches a graph whose
    dispatcher is wired to any other policy. Returns the difference in plain
    words, or None."""
    meta = meta if isinstance(meta, dict) else {}
    engines = meta.get(META_IMAGE_ENGINES)
    seed = meta.get(META_IMAGE_SEED)
    if not isinstance(engines, dict) or not isinstance(seed, dict):
        return "the replayed ledger carries no image-engine and seed stamp"
    painting = policy_image_engines(image_policy)
    differ = ["%s %s (stamped %s)" % (slot, _shown(painting[slot]), _shown(engines.get(slot)))
              for slot in IMAGE_SLOTS if painting[slot] != engines.get(slot)]
    seed_now = policy_image_seed(image_policy)
    if seed_now != seed:
        differ.append("seed %s (stamped %s)" % (_shown(seed_now), _shown(seed)))
    return "; ".join(differ) or None


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


#: The video section's maps a producer Video run writes beside the shot rows
#: (``_otr_producer_clips``): the record of every clip it delivered, and the
#: payload of every shot it floored. Paths in both are relative to the
#: episode folder.
CLIP_MAP_KEY = "producer_clip_reuse"
FLOOR_MAP_KEY = "producer_floor_payloads"


# --------------------------------------------------------------------------- #
# cards.json, and the still-input digest it records per card
# --------------------------------------------------------------------------- #

CARDS_NAME = "cards.json"
CARDS_VERSION = 1
#: Stamped on each card still's ledger row when it is painted; a kept row
#: carries the digest of the paint it keeps, never of the texts since.
ROW_INPUT_HASH = "producer_input_hash"
#: Stamped on each portrait row a producer stills run paints: the character
#: take it was painted with. A refused repaint leaves the old row, and its old
#: take, live -- what was painted, never what was asked for (Codex QA,
#: 2026-10-22).
ROW_IDENTITY_TAKE = "producer_identity_take"


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def payload_object_hash(obj: Any) -> str:
    """What a still is painted from besides its prompt -- identity, references,
    size, kind: the payload object without ``prompt`` and ``prompt_hash``, which
    its raw picture text and note stand for instead."""
    if not isinstance(obj, dict):
        return ""
    return _hash_json({k: v for k, v in obj.items() if k not in ("prompt", "prompt_hash")})


def still_input_digest(payload_hash: str, picture_text: str, note: str, char_id: str,
                       identity_take: int, still_take: int, engine_id: str) -> str:
    """Everything that decides a card still's pixels, as one digest: the payload
    object, the PROMPT its picture text and scene note compose to, the character
    and its take, the still's take and the image engine. A Stills run records it
    per card; node 63 recomputes it for a Video or Publish run. An ACTION never
    enters it, so a motion edit never asks for a repaint.

    THE COMPOSED PROMPT, NOT THE RAW PARTS (Codex QA, 2026-10-22): the painter
    sees only ``compose_picture(picture, note)``, so a whitespace-only edit, or
    one text moved between the picture and the note, paints the same pixels and
    must not be refused as unpainted. The seed configuration is not here: it is
    locked for the whole project (:func:`_check_image_engines`), like the
    engines, so it cannot differ between a paint and a check.

    KNOWINGLY NOT THE LAST WORD: the dispatcher's word substitutions (the
    banana route) run after this, so two texts they turn into one prompt key
    differently here. A Video run straight after such an edit is refused, and
    the Stills run that clears it paints nothing -- its still is a cache hit
    (Codex QA, 2026-10-22, weighed and left: the substitutions need the
    bundle's ledger, which node 63 does not load)."""
    return _hash_json({"v": 2, "payload": str(payload_hash or ""),
                       "prompt": compose_picture(picture_text, note),
                       "char": str(char_id or ""), "identity_take": int(identity_take or 0),
                       "still_take": int(still_take or 0), "engine": str(engine_id or "")})


def card_still_digest(obj: Any, engine_id: str, raw_texts: Dict[str, Any],
                      takes: Dict[str, Any], scene_of_beat: Dict[str, str]) -> str:
    """The digest of a card still as a run paints it: its payload object, its
    raw picture text and its scene's note from ``raw_texts``, its OWN
    character (the still object's, which an announcer beat's still has none
    of) and that character's take, its own take and its engine. The
    dispatcher stamps it on the row it paints; node 63 recomputes the same."""
    shot = object_shot_id(obj)
    char_id = str(obj.get("char_id") or "")
    scene = scene_of_beat.get(str(obj.get("beat_id") or ""), "")
    return still_input_digest(
        payload_object_hash(obj),
        (raw_texts.get("cards") or {}).get(shot, obj.get("prompt") or ""),
        (raw_texts.get("scenes") or {}).get(scene, ""), char_id,
        int((takes.get("characters") or {}).get(char_id, 0) or 0) if char_id else 0,
        int((takes.get("stills") or {}).get(shot, 0) or 0), str(engine_id or ""))


def _scene_of_beat(ledger: dict) -> Dict[str, str]:
    return {str(b.get("beat_id")): str(b.get("scene_id") or "")
            for b in (ledger.get("beats") or []) if isinstance(b, dict)}


def action_is_editable(shot: Any) -> bool:
    """Whether a planned shot has an action a person can edit: a character
    beat whose ``creative`` carries a ``motion`` leaf. Bookends and music have
    no ``creative``; a Ghost shot composes from its own finalized prompt
    (``ghost_prompt``), so editing ``motion`` there would change nothing. One
    rule for a card's Motion box, node 63's refusal and Shot Lock's."""
    creative = shot.get("creative") if isinstance(shot, dict) else None
    return (isinstance(creative, dict) and "motion" in creative
            and not shot.get("ghost_prompt"))


def build_cards(ledger: dict, producer: Dict[str, Any], files: list) -> Dict[str, Any]:
    """The small versioned view a producer freeze writes beside the manifest:
    the one contract between the nodes and Radio Play Studio's page. Media
    paths are bundle-relative and name only files the manifest lists. Each
    card's ``still_input_hash`` is the digest recorded on its row when its
    still was painted, so a kept still keeps the digest of the paint it is."""
    images = ledger.get("images") if isinstance(ledger.get("images"), dict) else {}
    raw = images.get(RAW_TEXTS_KEY) if isinstance(images.get(RAW_TEXTS_KEY), dict) else {}
    takes = images.get(TAKES_KEY) if isinstance(images.get(TAKES_KEY), dict) else {}
    payload = images.get(PAYLOAD_KEY) if isinstance(images.get(PAYLOAD_KEY), dict) else {}
    objects = {str(o.get("object_id")): o for o in payload.get("objects") or []
               if isinstance(o, dict)}
    rows = {str(r.get("object_id")): r for r in images.get("images") or []
            if isinstance(r, dict) and r.get("object_id")}
    # A row's member is its own folder and name: the dispatcher paints every
    # row into the episode's stills folder, and an older episode kept its
    # portraits in a portraits folder, so a name the two folders share can
    # never point a card at the other's file (Sonnet QA, 2026-10-22).
    listed = {str(entry.get("path") or "") for entry in files}

    def media(row):
        path = str((row or {}).get("path") or "")
        if not path:
            return ""
        rel = "%s/%s" % (os.path.basename(os.path.dirname(path)), os.path.basename(path))
        return rel if rel in listed else ""

    def filed(record):
        """A video-map record's clip, bundle-relative, when the bundle has it."""
        item = (((record or {}).get("files") or {}).get("clip")
                if isinstance(record, dict) else None)
        rel = str((item or {}).get("path") or "") if isinstance(item, dict) else ""
        return rel if rel in listed else ""

    lines = {str(ln.get("line_id")): ln for ln in ledger.get("lines") or []
             if isinstance(ln, dict)}
    scene_of = _scene_of_beat(ledger)
    kept = set((producer.get("kept") or {}).get("stills") or [])
    kept_clips = set((producer.get("kept") or {}).get("clips") or [])
    video = ledger.get("video") if isinstance(ledger.get("video"), dict) else {}
    filmed = video.get(CLIP_MAP_KEY) if isinstance(video.get(CLIP_MAP_KEY), dict) else {}
    floored = video.get(FLOOR_MAP_KEY) if isinstance(video.get(FLOOR_MAP_KEY), dict) else {}
    characters = takes.get("characters") or {}
    stills = takes.get("stills") or {}
    # Beats whose still the image model REFUSED, read from the receipt as the
    # render reads it (render_driver.sanctioned_gap_beat_ids). A repaint that
    # was refused leaves the old row in place, so the row alone would show a
    # painted still the render will skip as a gap.
    refused_beats = {str(row.get("beat_id")) for row in images.get("required_scene_targets") or []
                     if _still_receipt.is_sanctioned_gap(row) and row.get("beat_id")}
    scenes: Dict[str, Dict[str, Any]] = {}
    for shot in video.get("shots") or []:
        if not isinstance(shot, dict) or not str(shot.get("shot_id") or "").startswith("shot_"):
            continue
        shot_id = str(shot["shot_id"])
        beat = shot_id[len("shot_"):]
        obj = objects.get("still_%s" % beat)
        row = rows.get("still_%s" % beat)
        line = lines.get(beat) or {}
        creative = shot.get("creative") if isinstance(shot.get("creative"), dict) else {}
        editable = action_is_editable(shot)
        scene_id = scene_of.get(beat, "")
        still_char = str((obj or {}).get("char_id") or "")
        still = media(row)
        refused = beat in refused_beats
        still_state = ("failed" if refused else
                       "kept" if shot_id in kept else
                       "painted" if still else
                       "failed" if obj is not None else "")
        # THE CLIP (chunk 4b): the record of the clip the last Video run
        # delivered, or the payload of the floor it fell to -- a failed card
        # until it is filmed again or kept as it is.
        record = filmed.get(shot_id) if isinstance(filmed.get(shot_id), dict) else None
        if record is not None and not filed(record):
            # Filmed only with its clip in this bundle: the bundle is what a
            # replay of it delivers, and the clip store only a shortcut.
            record = None
        floor = floored.get(shot_id) if isinstance(floored.get(shot_id), dict) else None
        clip = filed(record) or filed(floor)
        clip_state = ("kept" if shot_id in kept_clips and (record or floor) else
                      "filmed" if record is not None else
                      "failed" if floor is not None else "")
        marks = (floor or {}).get("marks") if isinstance((floor or {}).get("marks"), dict) else {}
        state = ("failed" if "failed" in (still_state, clip_state) else
                 "kept" if "kept" in (still_state, clip_state) else
                 "filmed" if clip_state == "filmed" else still_state)
        card = {
            "shot_id": shot_id, "char_id": str(shot.get("char_id") or ""),
            "line": str(line.get("text") or ""),
            "start_s": float(line.get("start_s") or shot.get("start_s") or 0.0),
            "dur_s": float(line.get("dur_s") or shot.get("dur_s") or 0.0),
            "picture_text": ((raw.get("cards") or {}).get(shot_id, obj.get("prompt") or "")
                             if obj else ""),
            "motion_text": str(creative.get("motion") or "") if editable else "",
            "motion_editable": bool(editable), "picture_editable": obj is not None,
            "still": still, "clip": clip or None,
            "still_take": int(stills.get(shot_id, 0) or 0),
            "clip_take": int((record or {}).get("clip_take") or 0),
            "still_state": still_state, "clip_state": clip_state,
            # Why the clip is a floor -- the scene still, or black -- when it is.
            "clip_floor": str(marks.get("cloud_floor")
                              or ("budget" if marks.get("budget_floor") else "")
                              or marks.get("status") or ""),
            "clip_input_hash": str((record or {}).get("key") or ""),
            # A refused still has no painted digest: its baseline is the texts
            # and takes it was last asked to paint (unpainted_shots).
            "still_input_hash": "" if refused else str((row or {}).get(ROW_INPUT_HASH) or ""),
            "payload_hash": payload_object_hash(obj) if obj else "",
            "still_char_id": still_char,
            # The character take this card was last asked to paint with: with
            # its texts and still take, a failed card's baseline.
            "still_char_take": int(characters.get(still_char, 0) or 0) if still_char else 0,
            "engine_id": str((row or {}).get("engine_id") or ""),
            "still_dirty": False, "clip_dirty": False, "state": state,
        }
        group = scenes.setdefault(scene_id, {
            "scene_id": scene_id,
            # Shots in no scene share a group the page shows, but no note can
            # name it: a direction's scene ids are never empty (Codex QA,
            # 2026-10-22).
            "note_editable": bool(scene_id),
            "note": (raw.get("scenes") or {}).get(scene_id, ""), "shots": []})
        group["shots"].append(card)
    cast = []
    for member in ledger.get("cast") or []:
        if not isinstance(member, dict) or not member.get("char_id"):
            continue
        char_id = str(member["char_id"])
        portrait = rows.get(char_id)
        picture = media(portrait) or next(
            (card["still"] for group in scenes.values() for card in group["shots"]
             if card["char_id"] == char_id and card["still"]), "")
        cast.append({
            "char_id": char_id, "name": str(member.get("name") or ""), "picture": picture,
            # The take the character's portrait SHOWS, as stamped on its row
            # when painted -- a refused repaint leaves the old one live -- and,
            # with no portrait, the take their stills were asked for, which each
            # card's own digest carries (Codex QA, 2026-10-22).
            "has_portrait": portrait is not None,
            "take": (int(portrait.get(ROW_IDENTITY_TAKE) or 0) if portrait is not None
                     else int(characters.get(char_id, 0) or 0))})
    return {"version": CARDS_VERSION, "project_id": producer.get("project_id"),
            "attempt_id": producer.get("attempt_id"), "stage": producer.get("stage"),
            "bundle": producer.get("out"), "source": producer.get("source"),
            "cast": cast, "scenes": list(scenes.values())}


def _card_digest(card: Dict[str, Any], picture: Any, note: Any, char_take: Any,
                 still_take: Any) -> str:
    """:func:`still_input_digest` for a card's still with these texts and takes."""
    char_id = str(card.get("still_char_id") or "")
    return still_input_digest(str(card.get("payload_hash") or ""), picture, note, char_id,
                              int(char_take or 0) if char_id else 0, int(still_take or 0),
                              str(card.get("engine_id") or ""))


def unpainted_shots(cards: Dict[str, Any], direction: Dict[str, Any]) -> list:
    """The cards whose stills do not show what the direction asks for: an
    edited picture or note, a raised take -- any change to a still's inputs
    since it was painted -- unless the card is kept. A Video or Publish run
    with any is refused: it would film the old stills under the new edits.

    A card whose still FAILED has no painted digest. Its baseline is the texts
    and takes it was last asked to paint (Sonnet QA, 2026-10-22), so an edit
    made to retry it is caught like any other, and an untouched failed card is
    left to the video stage, which floors it."""
    kept = set(direction["kept"]["stills"])
    characters = direction["takes"]["characters"]
    stills = direction["takes"]["stills"]
    pending = []
    for group in cards.get("scenes") or []:
        recorded_note = group.get("note", "")
        scene_note = direction["scenes"].get(group.get("scene_id"), {}).get(
            "note", recorded_note)
        for card in group.get("shots") or []:
            shot_id = card.get("shot_id")
            if not card.get("picture_editable") or shot_id in kept:
                continue
            char_id = str(card.get("still_char_id") or "")
            asked = _card_digest(
                card, direction["cards"].get(shot_id, {}).get(
                    "picture_text", card.get("picture_text", "")),
                scene_note, characters.get(char_id, 0), stills.get(shot_id, 0))
            shown = card.get("still_input_hash") or _card_digest(
                card, card.get("picture_text", ""), recorded_note,
                card.get("still_char_take", 0), card.get("still_take", 0))
            if asked != shown:
                pending.append(shot_id)
    return pending


def unpainted_characters(cards: Dict[str, Any], direction: Dict[str, Any]) -> list:
    """The cast whose portrait does not show the take the direction asks for.
    A character's take repaints the portrait, which their stills and the frames
    they share are drawn against, so a Video or Publish run that changes one is
    refused until a Stills run has painted it -- whether or not any card's own
    digest names the character (Sonnet QA, 2026-10-22: a cast member with no
    still of their own would have been filmed with the old face). The take is
    the one stamped on the portrait row as painted, so a refused repaint is not
    taken for a painted one (Codex QA, the same day). A character with no
    portrait shows the take only in their stills, whose digests carry it."""
    asked = direction["takes"]["characters"]
    return [str(m.get("char_id")) for m in cards.get("cast") or []
            if isinstance(m, dict) and m.get("has_portrait")
            and int(asked.get(str(m.get("char_id")), 0)) != int(m.get("take") or 0)]


def unfilmed_shots(cards: Dict[str, Any], direction: Dict[str, Any]) -> list:
    """The cards a Publish run cannot show as filmed, each with why: a still
    the image model refused, no clip yet, a floor -- or a refused-still gap --
    that is not kept, or an action or clip take changed since its clip was
    filmed. A kept card publishes as it is, a kept floor visibly degraded, a
    kept clip on a refused still as the clip it shows.

    A still repainted since its clip was filmed is not seen here: cards.json
    does not say which still a clip was filmed from. The Publish render refuses
    it instead, before any weights load, because the clip's key no longer
    matches what the shot shows."""
    kept = set(direction["kept"]["clips"])
    takes = direction["takes"]["clips"]
    pending = []
    for group in cards.get("scenes") or []:
        for card in group.get("shots") or []:
            sid = str(card.get("shot_id"))
            if sid in kept:
                continue
            body = direction["cards"].get(sid, {})
            if card.get("still_state") == "failed":
                # The render skips a shot whose still was refused, as a gap,
                # and refuses that gap in a Publish its card does not keep.
                pending.append("%s (its still was refused)" % sid)
            elif card.get("clip_floor"):
                pending.append("%s (failed: %s)" % (sid, card["clip_floor"]))
            elif card.get("clip_state") not in ("filmed", "kept"):
                pending.append("%s (not filmed)" % sid)
            elif ("motion_text" in body and str(body["motion_text"]).strip()
                  != str(card.get("motion_text") or "").strip()):
                pending.append("%s (its action changed)" % sid)
            elif int(takes.get(sid, 0) or 0) != int(card.get("clip_take") or 0):
                pending.append("%s (its clip take changed)" % sid)
    return pending


def check_direction_ids(cards: Dict[str, Any], direction: Dict[str, Any]) -> None:
    """Refuse a replay's direction that names what its bundle's ``cards.json``
    does not show, or asks a card for what it cannot take: a card, scene or
    character that is not there, a picture text or still take on a card with
    no still, an action on a card with none to edit, a kept still on a card
    with no still to keep, a take on a character with no picture to take again.

    Each would otherwise do nothing at all, and an edit that does nothing looks
    like a broken edit (plan, "Direction JSON": unknown ids and motion edits on
    cards without an editable action are refused). Shot Lock refuses an unknown
    or uneditable action again at run time, as a backstop. Takes are sent in
    full, so a take of 0 -- no take -- is never refused for having nothing to
    take again."""
    shots = {str(card.get("shot_id")): card for group in cards.get("scenes") or []
             for card in group.get("shots") or [] if isinstance(card, dict)}
    scenes = {str(group.get("scene_id")) for group in cards.get("scenes") or []}
    cast = {str(m.get("char_id")): m for m in cards.get("cast") or [] if isinstance(m, dict)}
    drawn = {str(card.get("still_char_id")) for card in shots.values()
             if card.get("still_char_id")}
    problems = []
    for label, ids, known in (
            ("card", list(direction["cards"]) + list(direction["takes"]["stills"])
             + list(direction["takes"]["clips"]) + list(direction["kept"]["stills"])
             + list(direction["kept"]["clips"]), shots),
            ("scene", list(direction["scenes"]), scenes),
            ("character", list(direction["takes"]["characters"]), cast)):
        problems.extend("no %s %s" % (label, i) for i in sorted(set(ids) - set(known)))

    def refuse(ids, holds, words):
        problems.extend("card %s %s" % (i, words) for i in sorted(ids)
                        if i in shots and not holds(shots[i]))

    refuse([s for s, body in direction["cards"].items() if "picture_text" in body],
           lambda card: card.get("picture_editable"), "has no still to describe")
    refuse([s for s, take in direction["takes"]["stills"].items() if take],
           lambda card: card.get("picture_editable"), "has no still to take again")
    refuse([s for s, body in direction["cards"].items() if "motion_text" in body],
           lambda card: card.get("motion_editable"), "has no action to edit")
    refuse(direction["kept"]["stills"], lambda card: card.get("still"),
           "has no still to keep")
    # A clip to keep is one a Video run filmed, or the floor it fell to.
    refuse(direction["kept"]["clips"], lambda card: card.get("clip_state") in
           ("filmed", "failed", "kept"), "has no clip to keep")
    problems.extend("character %s has no picture to take again" % c
                    for c, take in sorted(direction["takes"]["characters"].items())
                    if take and c in cast and not cast[c].get("has_portrait")
                    and c not in drawn)
    if problems:
        raise DirectionError("the direction does not fit bundle %s: %s"
                             % (cards.get("bundle"), "; ".join(problems[:8])))


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
        cards = _read_cards(bundle)
        check_direction_ids(cards, direction)
        if direction["stage"] in ("video", "publish"):
            # UNPAINTED EDITS (plan, "Validation"): a Video or Publish run films
            # the stills it has, so an edit the stills do not show yet -- a
            # picture, a note, a take -- must be painted first, or the card kept.
            pending = unpainted_shots(cards, direction)
            if pending:
                raise DirectionError(
                    "%d card(s) carry edits their stills do not show yet: %s; paint the "
                    "stills first, or keep those cards as they are"
                    % (len(pending), ", ".join(pending[:8])))
            recast = unpainted_characters(cards, direction)
            if recast:
                raise DirectionError(
                    "the take of %s differs from the one their pictures were painted "
                    "with; paint the stills first" % ", ".join(recast[:8]))
        if direction["stage"] == "publish":
            # UNFILMED (plan, "Publish"): Publish films nothing, so every card
            # must show a clip filmed as it stands, or be kept as it is.
            unfilmed = unfilmed_shots(cards, direction)
            if unfilmed:
                raise DirectionError(
                    "%d card(s) are not filmed as they stand: %s; film them in a Video "
                    "run first, or keep them as they are"
                    % (len(unfilmed), "; ".join(unfilmed[:8])))
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
