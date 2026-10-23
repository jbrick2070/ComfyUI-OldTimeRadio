"""The Scene Producer in Radio Play Studio: a project kept on disk, one stage
run at a time, and the shot editor's view of each bundle the stages freeze
(planning/scene-producer/PLAN.md in the pack, "Studio").

A PROJECT is ``<out_dir>/projects/<project_id>/project.json``, written whole
and swapped in, so a reader never sees half of one: the first run's workflow
and form choices, every bundle in order, the view of the latest one, the
person's draft and its revision, and at most one stage in flight. Every change
to it is made under one lock (:func:`update_project`), so a draft saved while
a stage runs and the stage's own bookkeeping never write over each other.

A STAGE RUN submits the shipped graph with two values set by node type: Shot
Lock's ``direction_json`` and, after the first run, the writer's
``replay_from`` -- with the story fields every replay leaves empty. Before
anything is sent, the attempt is saved as the project's ``pending``, with the
exact graph beside it, and the job's id the moment the server answers, so a
restart can find it.

A BUNDLE is read through ComfyUI's own ``/view`` (:func:`read_output`): its
``manifest.json``, then ``cards.json``, whose sha256 the manifest lists, then
every picture and clip the cards name, each checked against the manifest and
copied to ``<project>/media/<sha256>.<ext>``. The page reads only those
copies, so a deleted bundle never blanks it. Bundles live under ComfyUI's
output folder at ``otr/episodes/_replay/p/<project>/<NN-stage>``;
``replay_from`` names one by that relative path, which the pack anchors at
the same output folder.

Standard library only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

from .form import FIELDS, FIELDS_BY_KEY, MY_STORY, SHOT_LOCK, WRITER
from .graph import ServerUnreachable

PROJECT_VERSION = 1
#: The cards.json version this page reads (the pack's CARDS_VERSION).
CARDS_VERSION = 1
PROJECTS_DIR = "projects"
MEDIA_DIR = "media"
ATTEMPTS_DIR = "attempts"
PROJECT_FILE = "project.json"
STAGES = ("stills", "video", "publish")
#: The pack's own rules for what a direction carries (``_otr_producer_direction``):
#: names, and the limits on texts and takes. Checked here too, so a draft the
#: pack would refuse never reaches the disk.
PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{3,31}$")
OUT_RE = re.compile(r"^([0-9]{2})-(stills|video)$")
ITEM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_NOTE_CHARS = 2_000
MAX_CARD_TEXT_CHARS = 4_000
MAX_TAKE = 9_999
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
#: Where a project's bundles sit under ComfyUI's output folder.
REPLAY_ROOT = "otr/episodes/_replay/p"
#: What a replay sets the story bank to: the writer's replay check accepts it,
#: and the bundle, not the bank, decides the story.
ROLL_BANK = "roll (any eligible bank)"
#: The files a card may show.
MEDIA_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov")

ReadFile = Callable[[str, str], bytes]

#: One lock for every change to a project file in this process: the page's
#: draft saves and a stage run's bookkeeping arrive on different threads.
_LOCK = threading.Lock()


class ProjectError(ValueError):
    """A project, a draft or a bundle Studio cannot use, and why."""


class NoProject(ProjectError):
    """No project goes by that id."""


class StaleDraft(ProjectError):
    """A draft saved over a revision the project has moved past."""


class StageInFlight(ProjectError):
    """A stage of this project is already running, or not yet settled."""


class BundleMissing(ProjectError):
    """A file a bundle should hold is not in ComfyUI's output folder (yet)."""


# --------------------------------------------------------------------------- #
# the project file
# --------------------------------------------------------------------------- #

def project_dir(out_dir: Path, project_id: str) -> Path:
    if not isinstance(project_id, str) or not PROJECT_ID_RE.fullmatch(project_id):
        raise NoProject(f"not a project id: {project_id!r}")
    return Path(out_dir) / PROJECTS_DIR / project_id


def new_project(out_dir: Path, *, workflow: str, choices: dict) -> dict:
    """A new, empty project for these first-run choices, saved."""
    project = {"version": PROJECT_VERSION, "project_id": uuid.uuid4().hex[:12],
               "created": time.time(), "workflow": str(workflow),
               "choices": copy.deepcopy(choices or {}), "bundles": [], "last_number": 0,
               "view": None, "draft": empty_draft(), "pending": None, "published": None,
               "last_error": None}
    with _LOCK:
        save_project(out_dir, project)
    return project


def empty_draft() -> dict:
    return {"revision": 0, "scenes": {}, "cards": {},
            "takes": {"characters": {}, "stills": {}, "clips": {}},
            "kept": {"stills": [], "clips": []}}


def load_project(out_dir: Path, project_id: str) -> dict:
    path = project_dir(out_dir, project_id) / PROJECT_FILE
    try:
        project = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NoProject(f"no project {project_id}") from None
    except (OSError, ValueError) as exc:
        raise ProjectError(f"project {project_id} cannot be read: {exc}") from None
    if not isinstance(project, dict) or project.get("version") != PROJECT_VERSION:
        raise ProjectError(f"project {project_id} is not a version-{PROJECT_VERSION} project")
    return project


def save_project(out_dir: Path, project: dict) -> None:
    """Write the project whole and swap it in: a reader never sees half.
    Callers that change a saved project go through :func:`update_project`."""
    folder = project_dir(out_dir, project["project_id"])
    folder.mkdir(parents=True, exist_ok=True)
    side = folder / f".{PROJECT_FILE}.{uuid.uuid4().hex[:8]}.tmp"
    side.write_text(json.dumps(project, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")
    os.replace(side, folder / PROJECT_FILE)


def update_project(out_dir: Path, project_id: str, change: Callable[[dict], Any]) -> dict:
    """Load a project, apply ``change`` to it and save it, under the one lock,
    and return it. ``change`` raises to leave the project as it was."""
    with _LOCK:
        project = load_project(out_dir, project_id)
        change(project)
        save_project(out_dir, project)
        return project


def list_projects(out_dir: Path) -> list:
    """Every project on disk, newest first."""
    found = []
    for path in sorted((Path(out_dir) / PROJECTS_DIR).glob(f"*/{PROJECT_FILE}")):
        try:
            project = load_project(out_dir, path.parent.name)
        except ProjectError:
            continue
        found.append({"project_id": project["project_id"], "created": project["created"],
                      "bundles": [b["out"] for b in project["bundles"]],
                      "busy": project["pending"] is not None,
                      "published": project["published"] is not None})
    return sorted(found, key=lambda p: p["created"], reverse=True)


# --------------------------------------------------------------------------- #
# the draft
# --------------------------------------------------------------------------- #

def _item_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not ITEM_ID_RE.fullmatch(value):
        raise ProjectError(f"{where} names {value!r}, which is not an id")
    return value


def clean_draft(draft: Any) -> dict:
    """The draft the page sent, in the shape a direction is built from, or
    ProjectError. The pack checks every id and limit again; this keeps a
    draft it would refuse off the disk. A blank picture text is a draft in
    progress and is kept: only a stage run refuses one."""
    if not isinstance(draft, dict):
        raise ProjectError("the draft must be an object")

    def texts(value: Any, limits: dict, where: str) -> dict:
        if not isinstance(value, dict):
            raise ProjectError(f"{where}s must be an object")
        out = {}
        for item, body in value.items():
            item = _item_id(item, where)
            if not isinstance(body, dict) or set(body) - set(limits):
                raise ProjectError(f"{where} {item} must be an object of "
                                   f"{', '.join(sorted(limits))}")
            for key, text in body.items():
                if not isinstance(text, str):
                    raise ProjectError(f"{where} {item} {key} must be text")
                if len(text) > limits[key]:
                    raise ProjectError(f"{where} {item} {key} is longer than "
                                       f"{limits[key]} characters")
            out[item] = dict(body)
        return out

    takes_in = draft.get("takes", {})
    kept_in = draft.get("kept", {})
    if not isinstance(takes_in, dict) or not isinstance(kept_in, dict):
        raise ProjectError("takes and kept must be objects")
    takes = {}
    for group in ("characters", "stills", "clips"):
        given = takes_in.get(group, {})
        if not isinstance(given, dict) or not all(
                isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= MAX_TAKE
                for v in given.values()):
            raise ProjectError(f"takes.{group} holds whole numbers from 0 to {MAX_TAKE}")
        takes[group] = {_item_id(k, f"takes.{group}"): int(v) for k, v in given.items()}
    kept = {}
    for group in ("stills", "clips"):
        given = kept_in.get(group, [])
        if not isinstance(given, list):
            raise ProjectError(f"kept.{group} is a list of shot ids")
        kept[group] = sorted({_item_id(v, f"kept.{group}") for v in given})
    return {"scenes": texts(draft.get("scenes", {}), {"note": MAX_NOTE_CHARS}, "scene"),
            "cards": texts(draft.get("cards", {}), {"picture_text": MAX_CARD_TEXT_CHARS,
                                                    "motion_text": MAX_CARD_TEXT_CHARS},
                           "card"),
            "takes": takes, "kept": kept}


def save_draft(out_dir: Path, project_id: str, draft: Any, revision: Any) -> dict:
    """Save the person's draft over ``revision``, the one their page last
    read, and return the project. A stale revision is refused, so two pages
    can never write over each other's edits unseen. Drafts are saved while a
    stage runs too: the run carries the revision it was started from."""
    cleaned = clean_draft(draft)

    def change(project: dict) -> None:
        if revision != project["draft"]["revision"]:
            raise StaleDraft("this project was edited elsewhere since the page read it; "
                             "reload to see the newer edits")
        project["draft"] = dict(cleaned, revision=int(revision) + 1)

    return update_project(out_dir, project_id, change)


# --------------------------------------------------------------------------- #
# a stage's direction and graph
# --------------------------------------------------------------------------- #

def latest_bundle(project: dict) -> Optional[dict]:
    return project["bundles"][-1] if project["bundles"] else None


def next_out(project: dict, stage: str) -> Optional[str]:
    """The bundle a stage writes, numbered past every number this project has
    ever sent -- an attempt that failed may have written its bundle all the
    same, and a bundle is never written over. None for Publish."""
    if stage == "publish":
        return None
    number = int(project.get("last_number") or 0) + 1
    if number > 99:
        raise ProjectError("this project has used every bundle number; start a new one")
    return "%02d-%s" % (number, stage)


def bundle_subfolder(project_id: str, out: str) -> str:
    """A bundle's place under ComfyUI's output folder, as `/view` and the
    writer's ``replay_from`` both name it."""
    if not OUT_RE.fullmatch(str(out or "")):
        raise ProjectError(f"not a bundle name: {out!r}")
    project_dir(Path("."), project_id)            # checks the id
    return f"{REPLAY_ROOT}/{project_id}/{out}"


def build_direction(project: dict, stage: str) -> dict:
    """The direction a stage run carries: the project's identity and the
    latest bundle's digest, the draft's edits that differ from what the
    latest bundle shows, and its takes and keeps in full."""
    if stage not in STAGES:
        raise ProjectError(f"not a stage: {stage!r}")
    latest = latest_bundle(project)
    direction = {"version": 1, "project_id": project["project_id"],
                 "attempt_id": uuid.uuid4().hex, "stage": stage}
    out = next_out(project, stage)
    if out is not None:
        direction["out"] = out
    if latest is None:
        if stage != "stills":
            raise ProjectError("a project starts with its stills")
        return direction
    if stage == "publish" and latest["stage"] != "video":
        raise ProjectError("Publish puts together the shots a Video run filmed: film them first")
    direction["source"] = latest["digest"]
    view = project.get("view") or {}
    shown = {card["shot_id"]: card for scene in view.get("scenes", []) for card in scene["shots"]}
    notes = {scene["scene_id"]: scene.get("note", "") for scene in view.get("scenes", [])}
    draft = project["draft"]
    cards = {}
    for shot_id, body in draft["cards"].items():
        card = shown.get(shot_id)
        if card is None:
            continue
        edit = {}
        if ("picture_text" in body and card.get("picture_editable")
                and body["picture_text"] != card.get("picture_text", "")):
            if not body["picture_text"].strip():
                raise ProjectError(f"card {shot_id} has no picture text; a still needs "
                                   "a description")
            edit["picture_text"] = body["picture_text"]
        if ("motion_text" in body and card.get("motion_editable")
                and body["motion_text"] != card.get("motion_text", "")):
            edit["motion_text"] = body["motion_text"]
        if edit:
            cards[shot_id] = edit
    scenes = {scene_id: {"note": body.get("note", "")}
              for scene_id, body in draft["scenes"].items()
              if scene_id in notes and body.get("note", "") != notes[scene_id]}
    direction.update(cards=cards, scenes=scenes, takes=copy.deepcopy(draft["takes"]),
                     kept=copy.deepcopy(draft["kept"]))
    return direction


def patch_graph(graph: dict, node_ids: dict, direction: dict,
                replay_from: str = "") -> dict:
    """Set a stage run's values on the shipped graph, by node type, in place:
    Shot Lock's ``direction_json``; and on a replay the writer's
    ``replay_from``, the story bank to the roll, and ``source_ref`` and every
    My Story box empty -- the bundle decides the story. A value wired from
    another node in this workflow cannot be set, and is refused."""
    lock = graph.get(node_ids.get(SHOT_LOCK) or "")
    writer = graph.get(node_ids.get(WRITER) or "")
    if lock is None or writer is None:
        raise ProjectError(f"the workflow needs one {SHOT_LOCK} and one {WRITER} to direct shots")
    values = [(lock, "direction_json", json.dumps(direction, ensure_ascii=False))]
    if replay_from:
        values.append((writer, "replay_from", replay_from))
        values.append((writer, FIELDS_BY_KEY["bank"].input, ROLL_BANK))
        values.append((writer, FIELDS_BY_KEY["source_ref"].input, ""))
        values += [(writer, f.input, "") for f in FIELDS
                   if f.group == MY_STORY and f.node_type == WRITER]
    for node, name, value in values:
        if isinstance(node["inputs"].get(name), list):
            raise ProjectError(f"{name} is wired to another node in this workflow; "
                               "it cannot be set here")
        node["inputs"][name] = value
    return graph


# --------------------------------------------------------------------------- #
# one attempt, from before it is sent to its end
# --------------------------------------------------------------------------- #

def begin_attempt(out_dir: Path, project_id: str, direction: dict, graph: dict,
                  node_ids: dict, revision: int) -> dict:
    """Save an attempt as the project's ``pending`` before anything is sent:
    its id, stage, bundle and source, the draft revision it carries, and --
    in ``attempts/<attempt_id>.json`` beside the project -- the exact graph
    it submits. Refused when another attempt is pending."""
    attempt_id = direction["attempt_id"]

    def change(project: dict) -> None:
        if project["pending"] is not None:
            raise StageInFlight("this project already has a stage in flight")
        folder = project_dir(out_dir, project_id) / ATTEMPTS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{attempt_id}.json").write_text(
            json.dumps({"direction": direction, "graph": graph, "node_ids": node_ids},
                       ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        out = direction.get("out")
        if out:
            project["last_number"] = max(int(project.get("last_number") or 0), int(out[:2]))
        project["pending"] = {"attempt_id": attempt_id, "stage": direction["stage"],
                              "out": out, "source": direction.get("source"),
                              "revision": int(revision), "started": time.time(),
                              "job_id": None, "state": "submitting", "note": ""}

    return update_project(out_dir, project_id, change)


def _pending_change(attempt_id: str, apply: Callable[[dict], None]) -> Callable[[dict], None]:
    """A change to the pending attempt ``attempt_id``, and to nothing else: a
    late word about an attempt the project has moved past is refused."""
    def change(project: dict) -> None:
        pending = project["pending"]
        if pending is None or pending["attempt_id"] != attempt_id:
            raise ProjectError(f"attempt {attempt_id} is not this project's pending one")
        apply(project)
    return change


def record_job(out_dir: Path, project_id: str, attempt_id: str, job_id: str) -> dict:
    """The server took the attempt: its job id, saved at once."""
    def apply(project: dict) -> None:
        project["pending"].update(job_id=str(job_id), state="running")
    return update_project(out_dir, project_id, _pending_change(attempt_id, apply))


def hold_attempt(out_dir: Path, project_id: str, attempt_id: str, state: str,
                 note: str) -> dict:
    """Keep an attempt pending, saying why: the server may or may not have
    taken it (``uncertain``), or it is done and its result is not yet saved
    here (``finishing``). Nothing is submitted for this project until it is
    settled."""
    def apply(project: dict) -> None:
        project["pending"].update(state=state, note=str(note))
    return update_project(out_dir, project_id, _pending_change(attempt_id, apply))


def abandon_attempt(out_dir: Path, project_id: str, attempt_id: str, message: str) -> dict:
    """An attempt that ended with nothing to keep: the project may try again."""
    def apply(project: dict) -> None:
        pending = project["pending"]
        project["pending"] = None
        project["last_error"] = {"stage": pending["stage"], "attempt_id": attempt_id,
                                 "message": str(message), "at": time.time()}
    return update_project(out_dir, project_id, _pending_change(attempt_id, apply))


def finish_bundle(out_dir: Path, project_id: str, attempt_id: str, view: dict) -> dict:
    """A Stills or Video attempt whose bundle is adopted: the bundle joins the
    project and its view becomes the one the draft is read against. The draft
    is left as it is, so an edit made while the stage ran stays an edit."""
    def apply(project: dict) -> None:
        pending = project["pending"]
        project["bundles"].append({"out": pending["out"], "stage": pending["stage"],
                                   "digest": view["digest"], "attempt_id": attempt_id,
                                   "source": pending["source"], "finished": time.time()})
        project["view"] = view
        project["pending"] = None
        project["last_error"] = None
    return update_project(out_dir, project_id, _pending_change(attempt_id, apply))


def finish_publish(out_dir: Path, project_id: str, attempt_id: str, run: str,
                   file: str) -> dict:
    """A Publish attempt whose episode is saved here."""
    def apply(project: dict) -> None:
        pending = project["pending"]
        project["published"] = {"run": run, "file": file, "attempt_id": attempt_id,
                                "source": pending["source"], "finished": time.time()}
        project["pending"] = None
        project["last_error"] = None
    return update_project(out_dir, project_id, _pending_change(attempt_id, apply))


# --------------------------------------------------------------------------- #
# reading a bundle
# --------------------------------------------------------------------------- #

def read_output(comfyui_url: str, subfolder: str, name: str, timeout: float = 120.0) -> bytes:
    """A file from ComfyUI's output folder, through its own ``/view``.
    BundleMissing when ComfyUI has no such file, ServerUnreachable when
    ComfyUI cannot be asked."""
    query = urllib.parse.urlencode({"filename": name, "subfolder": subfolder, "type": "output"})
    url = comfyui_url.rstrip("/") + "/view?" + query
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise BundleMissing(f"ComfyUI's output folder has no {subfolder}/{name}") from None
        raise ServerUnreachable(f"ComfyUI answered {exc.code} for {url}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise ServerUnreachable(f"could not read {url} ({exc})") from exc


def adopt_bundle(out_dir: Path, project: dict, out: str, read_file: ReadFile,
                 *, attempt_id: str, source: Optional[str]) -> dict:
    """Read a finished stage's bundle and return its page view, its media
    copied into the project. ``read_file(subfolder, name)`` returns a file's
    bytes from ComfyUI's output folder.

    Only the intended bundle is adopted: its cards must name this project,
    the attempt that made it and the bundle it continued, and every file is
    checked against the manifest's sha256 before it is kept."""
    subfolder = bundle_subfolder(project["project_id"], out)
    raw_manifest = read_file(subfolder, "manifest.json")
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProjectError(f"bundle {out}'s manifest cannot be read: {exc}") from None
    if not isinstance(manifest, dict):
        raise ProjectError(f"bundle {out}'s manifest is not an object")
    listed = {str(row.get("path")): str(row.get("sha256") or "")
              for row in (manifest.get("files") or []) if isinstance(row, dict)}

    def fetch(rel: str) -> bytes:
        digest = listed.get(rel)
        if not digest or not _SHA_RE.fullmatch(digest):
            raise ProjectError(f"bundle {out} does not list {rel}")
        path = PurePosixPath(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ProjectError(f"bundle {out} names {rel}, outside itself")
        folder = "/".join((subfolder,) + path.parts[:-1])
        data = read_file(folder, path.name)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ProjectError(f"bundle {out}'s {rel} is not the file its manifest lists")
        return data

    try:
        cards = json.loads(fetch("cards.json").decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProjectError(f"bundle {out}'s cards cannot be read: {exc}") from None
    if not isinstance(cards, dict) or cards.get("version") != CARDS_VERSION:
        raise ProjectError(f"bundle {out}'s cards are not version {CARDS_VERSION}")
    if (cards.get("project_id"), cards.get("attempt_id"), cards.get("bundle"),
            cards.get("source")) != (project["project_id"], attempt_id, out, source):
        raise ProjectError(f"bundle {out} is not the one this run made")

    media = project_dir(out_dir, project["project_id"]) / MEDIA_DIR
    base = f"/episodes/{PROJECTS_DIR}/{project['project_id']}/{MEDIA_DIR}"

    def keep(rel: Any) -> Optional[str]:
        """A media file the cards name, copied in once by its hash; its URL."""
        if not rel:
            return None
        rel = str(rel)
        suffix = PurePosixPath(rel).suffix.lower()
        if suffix not in MEDIA_SUFFIXES:
            raise ProjectError(f"bundle {out} names {rel}, which is not a picture or a clip")
        if not _SHA_RE.fullmatch(listed.get(rel, "")):
            raise ProjectError(f"bundle {out} does not list {rel}")
        name = listed[rel] + suffix
        target = media / name
        if not target.is_file():
            media.mkdir(parents=True, exist_ok=True)
            side = media / f".{name}.{uuid.uuid4().hex[:8]}.part"
            side.write_bytes(fetch(rel))
            os.replace(side, target)
        return f"{base}/{name}"

    view = copy.deepcopy(cards)
    for member in view.get("cast") or []:
        member["picture"] = keep(member.get("picture"))
    for scene in view.get("scenes") or []:
        for card in scene.get("shots") or []:
            card["still"] = keep(card.get("still"))
            card["clip"] = keep(card.get("clip"))
    view["digest"] = hashlib.sha256(raw_manifest).hexdigest()
    return view
