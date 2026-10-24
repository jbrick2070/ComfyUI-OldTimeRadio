"""The handful of choices Radio Play Studio puts in front of a person, and how
each lands on the pack's graph.

A field names a node TYPE and an input, never a node id: the workflow's ids
change whenever it is re-saved, its types do not. Everything a field may be set
to is read live from the server (`/object_info`), so the form always offers the
story banks, styles and languages the installed pack really has.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .graph import GraphError, coerce_value, describe_input, has_input, node_id_for_type

WRITER = "OTR_LedgerScriptWriter"
IMAGE_DIRECTOR = "OTR_ImageDirector"
#: The node a Scene Producer stage directs (its ``direction_json``).
SHOT_LOCK = "OTR_ShotLock"

#: The story bank that reads the four My Story boxes, and their group on the page.
MY_STORY_BANK = "my_story"
MY_STORY = "My story"
#: The writer's own group: a closed panel on the page, sent only when changed.
WRITER_GROUP = "Writer"
#: The cloud writers the page offers, each with the slot-A model picker it reads.
#: Slot A is the creative passes' slot and slot B the technical passes'; the page
#: makes the writer one choice and leaves the technical passes on the workflow's
#: own model (Codex, agy and Fable, 2026-10-23; operator: "A/B combined, one
#: choice"). Comfy credits ride on the key the page holds (the SDK sends it as
#: extra_data.api_key_comfy_org).
CLOUD_WRITERS = {"openrouter:slot-a": "openrouter_slot_a_model",
                 "google_api:slot-a": "google_api_slot_a_model",
                 "comfy:slot-a": "comfy_slot_a_model"}


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    node_type: str
    input: str
    help: str
    group: str
    #: Choices the pack lists but cannot run (a dropdown row that fails loud).
    exclude: tuple = ()
    #: More inputs on the same node that take the same value (one picker, several
    #: widgets), each checked against its own dropdown.
    also: tuple = ()
    #: Never offer a choice that starts with one of these.
    exclude_prefixes: tuple = ()
    #: An input an older pack may not have yet: left off the form there, and
    #: refused only if a run actually tries to set it.
    optional: bool = False


#: In the order they appear on the page. Language leads: it is the one choice
#: every story makes (operator, 2026-10-23: "put language first").
FIELDS: tuple = (
    Field("language", "Language of the play", WRITER, "episode_language",
          "The language the play is written and spoken in. The page's own language is "
          "the menu at the top right.", "The show"),
    Field("bank", "Story bank", WRITER, "source_bank",
          "Where the story comes from. Pick shakespeare or public_domain to adapt a "
          "classic, my_story to use your own details below, or roll to be surprised.",
          "The show", exclude=("custom_source_bank",)),
    Field("source_ref", "Source to adapt", WRITER, "source_ref",
          "Optional. Name the classic to adapt (needs a named bank, not the roll).",
          "The show"),
    Field("acts", "Acts", WRITER, "act_count",
          "More acts, a longer play.", "The show"),
    Field("cast_size", "Cast size", WRITER, "num_characters",
          "Speaking parts.", "The show"),
    Field("style", "Visual style", WRITER, "visual_style",
          "Restyles the still and video prompts, so it shows once a video lane draws from "
          "stills. On the audio-reactive lanes it changes the file name, not the picture.",
          "The show"),
    Field("stills", "Still pictures", "OTR_VideoDirector", "character_image_model",
          "Paints every still. Open models run here; paid ones run in the cloud and need a "
          "key. Character photos only work with FLUX 3 Klein.",
          "The show", also=("announcer_image_model", "music_image_model"),
          # Mix and match (operator, 2026-10-23: "people can mix and match whatever
          # they want"): every engine but the custom-model row, which opens a node
          # editor this page does not have.
          exclude=("+ Add Custom Model",)),
    Field("writer", "Writer", WRITER, "creative_writing_model",
          "The model that writes the play. Local models run on this machine for free; "
          "OpenRouter, Google and Comfy are paid cloud models and need a key.",
          WRITER_GROUP, exclude=("openrouter:slot-b", "google_api:slot-b", "comfy:slot-b"),
          optional=True),
    Field("writer_openrouter", "OpenRouter model", WRITER, "openrouter_slot_a_model",
          "Used when the writer is OpenRouter. Needs OPENROUTER_API_KEY where ComfyUI runs.",
          WRITER_GROUP, optional=True),
    Field("writer_google", "Google model", WRITER, "google_api_slot_a_model",
          "Used when the writer is Google. Needs GEMINI_API_KEY (or GOOGLE_API_KEY) "
          "where ComfyUI runs.", WRITER_GROUP, optional=True),
    Field("writer_comfy", "Comfy model", WRITER, "comfy_slot_a_model",
          "Used when the writer is Comfy. Bills Comfy credits through the key on this page.",
          WRITER_GROUP, optional=True),
    Field("my_characters", "Who is in it", WRITER, "story_characters",
          "Names, how they know each other, what they are like. Put (she) or (he) "
          "right after a name and that character is voiced and drawn that way; "
          "anyone unmarked is cast to fit the part. Leave it empty and the cast is "
          "invented for you.", MY_STORY),
    Field("my_photos", "Character photos", IMAGE_DIRECTOR, "character_photos",
          "Optional. One line per character, Name = file, naming a picture in "
          "ComfyUI's input folder. The character's portrait is painted from it, "
          "in the episode's style, and every still follows the portrait. The "
          "cast cards fill this in when you add a photo.",
          MY_STORY, optional=True),
    Field("my_plot", "What happens", WRITER, "story_plot",
          "Events, the trouble, a twist, the ending you want. Rough notes are fine: "
          "what you ask for outranks anything the writer would rather invent, and "
          "what you leave out is invented to fit.", MY_STORY),
    Field("my_setting", "Where and when", WRITER, "story_setting",
          "Place, era, weather, mood. This is the story's world, not its look: the "
          "Visual style choice sets the look.", MY_STORY),
    Field("my_author", "Written by", WRITER, "story_author",
          "Your name, a pen name, or whoever you want it credited to. The announcer "
          "says it and the closing credits print it; leave it empty and the episode "
          "credits one of our listeners.", MY_STORY),
)

FIELDS_BY_KEY: dict = {f.key: f for f in FIELDS}

#: The node whose output is the finished episode.
EPISODE_NODE_TYPE = "OTR_MasterAudioMux"


def locate_nodes(workflow: dict) -> dict:
    """`{node_type: node_id}` for every node type the form or the run needs.

    A type only optional fields use may be missing from a workflow (an
    audio-only graph has no still pictures to give a photo to); it maps to None
    and those rows are left off. So does Shot Lock, which only a Scene
    Producer stage sets."""
    needed = {f.node_type for f in FIELDS if not f.optional} | {EPISODE_NODE_TYPE}
    ids = {t: node_id_for_type(workflow, t) for t in sorted(needed)}
    for t in sorted(({f.node_type for f in FIELDS} - needed) | {SHOT_LOCK}):
        present = [n for n in workflow.get("nodes") or [] if n.get("type") == t]
        ids[t] = node_id_for_type(workflow, t) if present else None
    return ids


def _offered(f: Field, schemas: dict, node_ids: dict | None) -> bool:
    """False for an optional field this server or this workflow cannot take."""
    if not f.optional:
        return True
    if node_ids is not None and node_ids.get(f.node_type) is None:
        return False
    return has_input(schemas, f.node_type, f.input)


def describe_form(schemas: dict, api_prompt: dict | None = None, node_ids: dict | None = None) -> list:
    """The form as the page draws it, with the server's live choices and limits.

    Given the converted graph, each row starts on the value the workflow itself
    ships with, so an untouched form runs the show exactly as the pack's own app
    would; without it, a row starts on the node's default."""
    rows = []
    for f in FIELDS:
        if not _offered(f, schemas, node_ids):
            continue
        info = _usable(describe_input(schemas, f.node_type, f.input), f)
        default = info["default"]
        if api_prompt is not None and node_ids is not None:
            saved = api_prompt.get(node_ids[f.node_type], {}).get("inputs", {}).get(f.input)
            if saved is not None and not isinstance(saved, list):
                default = saved
        choices = info.get("choices")
        if info["kind"] == "choice" and choices and default not in choices:
            # A workflow can save a choice this form never offers (a cloud still
            # engine on a cloud graph). Start on the first one the page lists, so
            # what it shows is what it sends.
            default = choices[0]
        rows.append({
            "key": f.key, "label": f.label, "group": f.group, "help": f.help,
            "kind": info["kind"], "default": default,
            "choices": info.get("choices"), "min": info.get("min"),
            "max": info.get("max"), "multiline": info.get("multiline", False),
            "placeholder": info.get("placeholder", ""),
        })
    return rows


def apply_choices(api_prompt: dict, schemas: dict, node_ids: dict, choices: dict) -> dict:
    """Write `choices` onto a copy-free graph in place and return it.

    Anything not in `choices` keeps the workflow's own value. A key that is not a
    field, a value the server would refuse, or an input that is wired to another
    node (so a typed value would be ignored) raises, never passes quietly."""
    unknown = sorted(set(choices) - set(FIELDS_BY_KEY))
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
    for key, value in choices.items():
        f = FIELDS_BY_KEY[key]
        if not _offered(f, schemas, node_ids):
            if value in (None, ""):
                continue
            if node_ids.get(f.node_type) is None:
                raise GraphError(f"{f.label}: this workflow has no {f.node_type}, so it "
                                 "draws no stills to use them in")
            raise GraphError(f"{f.label} needs a newer Old-Time Radio pack: this ComfyUI's "
                             f"{f.node_type} has no {f.input!r} input")
        node = api_prompt.get(node_ids[f.node_type])
        if node is None:
            raise GraphError(f"the graph has no {f.node_type} node")
        if isinstance(node["inputs"].get(f.input), list):
            raise GraphError(f"{f.label} is wired to another node in this workflow; it cannot be set here")
        for name in (f.input, *f.also):
            if isinstance(node["inputs"].get(name), list):
                raise GraphError(f"{f.label} is wired to another node in this workflow; "
                                 "it cannot be set here")
            info = _usable(describe_input(schemas, f.node_type, name), f)
            node["inputs"][name] = coerce_value(info, _blank_to_text(value, info), f.label)
    _check_my_story(api_prompt, node_ids)
    _check_writer(api_prompt, node_ids)
    return api_prompt


def _check_writer(api_prompt: dict, node_ids: dict) -> None:
    """A cloud writer whose model picker is still on its "(enable ...)" or
    "(select ...)" line has no model to call: said before submitting, not when
    the writer fails minutes in."""
    inputs = api_prompt.get(node_ids.get(WRITER) or "", {}).get("inputs", {})
    slot = CLOUD_WRITERS.get(str(inputs.get("creative_writing_model") or ""))
    if slot is None:
        return
    model = inputs.get(slot)
    if not isinstance(model, str) or not model.strip() or model.startswith("("):
        raise GraphError(f"Writer: {inputs['creative_writing_model']} has no model picked; "
                         "choose one in its model box, or pick a local writer")


def _check_my_story(api_prompt: dict, node_ids: dict) -> None:
    """Three runs the pack stops, said before submitting instead of minutes in:
    a story bank picked by hand as anything but my_story while a My Story box
    holds text (a roll reads the boxes if it lands on My Story and ignores them
    otherwise); my_story with a classic named in Source to adapt, which that bank
    has no way to fetch (both nodes/_otr_story_input.py); and a real roll with a
    classic named, which belongs to one bank (nodes/_otr_rolls.py)."""
    writer = api_prompt.get(node_ids.get(WRITER, ""), {}).get("inputs", {})
    bank = writer.get(FIELDS_BY_KEY["bank"].input)
    if bank == MY_STORY_BANK:
        ref = writer.get(FIELDS_BY_KEY["source_ref"].input)
        if isinstance(ref, str) and ref.strip():
            raise ValueError(
                f"{MY_STORY_BANK} tells your own story, so it has no classic to fetch, "
                f"but Source to adapt says {ref.strip()!r}. Clear it, or pick a bank "
                "that adapts a classic (shakespeare, public_domain).")
        return
    if isinstance(bank, str) and bank.startswith("roll"):
        ref = writer.get(FIELDS_BY_KEY["source_ref"].input)
        if isinstance(ref, str) and ref.strip() and len(_pool_names(writer.get("bank_roll_pool"))) != 1:
            raise ValueError(
                f"Source to adapt says {ref.strip()!r}, but the story bank is the roll. A "
                "classic belongs to one bank: pick it (shakespeare, public_domain), or clear "
                "Source to adapt to roll.")
        return
    if not isinstance(bank, str):
        return
    filled = [f.label for f in FIELDS if f.group == MY_STORY
              and isinstance(writer.get(f.input), str) and writer[f.input].strip()]
    if filled:
        raise ValueError(
            f"the story bank is {bank}, but My story has text in: {', '.join(filled)}. "
            f"Pick the {MY_STORY_BANK} bank to tell your own story, or clear those boxes.")


def _pool_names(raw: Any) -> list:
    """The bank names in a roll pool, as the pack reads one (_otr_rolls.py
    parse_roll_pool): commas, semicolons or new lines between names, or a list.
    One name is a pick, not a roll, so a classic may go with it."""
    if isinstance(raw, dict) and "__value__" in raw:
        raw = raw["__value__"]
    if isinstance(raw, (list, tuple)):
        names = (str(x).strip() for x in raw)
    elif isinstance(raw, str):
        names = (x.strip(" \t'\"[]") for x in re.split(r"[,;\n]", raw))
    else:
        return []
    # One bank however it is spelled or repeated: "Public-Domain, public_domain".
    seen: dict = {}
    for name in names:
        key = re.sub(r"[\s_-]+", "_", name.strip()).strip("_").casefold()   # the pack's _pool_key
        if key:
            seen.setdefault(key, name)
    return list(seen.values())


def _usable(info: dict, f: Field) -> dict:
    """`info` without the choices this field must not offer or accept."""
    if (f.exclude or f.exclude_prefixes) and info["kind"] == "choice":
        info = dict(info, choices=[
            c for c in info["choices"]
            if c not in f.exclude and not any(str(c).startswith(x) for x in f.exclude_prefixes)])
    return info


def _blank_to_text(value: Any, info: dict) -> Any:
    """A page sends an empty box as null; a text input wants an empty string."""
    return "" if value is None and info["kind"] == "text" else value
