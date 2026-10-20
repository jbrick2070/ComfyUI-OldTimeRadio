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

from .graph import GraphError, coerce_value, describe_input, node_id_for_type

WRITER = "OTR_LedgerScriptWriter"

#: The story bank that reads the four My Story boxes, and their group on the page.
MY_STORY_BANK = "my_story"
MY_STORY = "My story"


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


#: In the order they appear on the page.
FIELDS: tuple = (
    Field("bank", "Story bank", WRITER, "source_bank",
          "Where the story comes from. Pick shakespeare or public_domain to adapt a "
          "classic, my_story to use your own details below, or roll to be surprised.",
          "The show", exclude=("custom_source_bank",)),
    Field("source_ref", "Source to adapt", WRITER, "source_ref",
          "Optional. Name the classic to adapt (needs a named bank, not the roll).",
          "The show"),
    Field("language", "Language", WRITER, "episode_language",
          "The language of the play.", "The show"),
    Field("acts", "Acts", WRITER, "act_count",
          "More acts, a longer play.", "The show"),
    Field("cast_size", "Cast size", WRITER, "num_characters",
          "Speaking parts.", "The show"),
    Field("style", "Visual style", WRITER, "visual_style",
          "Restyles the still and video prompts, so it shows once a video lane draws from "
          "stills. On the audio-reactive lanes it changes the file name, not the picture.",
          "The show"),
    Field("my_characters", "Who is in it", WRITER, "story_characters",
          "Names, how they know each other, what they are like. Put (she) or (he) "
          "right after a name and that character is voiced and drawn that way; "
          "anyone unmarked is cast to fit the part. Leave it empty and the cast is "
          "invented for you.", MY_STORY),
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
    """`{node_type: node_id}` for every node type the form or the run needs."""
    types = {f.node_type for f in FIELDS} | {EPISODE_NODE_TYPE}
    return {t: node_id_for_type(workflow, t) for t in sorted(types)}


def describe_form(schemas: dict, api_prompt: dict | None = None, node_ids: dict | None = None) -> list:
    """The form as the page draws it, with the server's live choices and limits.

    Given the converted graph, each row starts on the value the workflow itself
    ships with, so an untouched form runs the show exactly as the pack's own app
    would; without it, a row starts on the node's default."""
    rows = []
    for f in FIELDS:
        info = _usable(describe_input(schemas, f.node_type, f.input), f)
        default = info["default"]
        if api_prompt is not None and node_ids is not None:
            saved = api_prompt.get(node_ids[f.node_type], {}).get("inputs", {}).get(f.input)
            if saved is not None and not isinstance(saved, list):
                default = saved
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
        node = api_prompt.get(node_ids[f.node_type])
        if node is None:
            raise GraphError(f"the graph has no {f.node_type} node")
        if isinstance(node["inputs"].get(f.input), list):
            raise GraphError(f"{f.label} is wired to another node in this workflow; it cannot be set here")
        info = _usable(describe_input(schemas, f.node_type, f.input), f)
        node["inputs"][f.input] = coerce_value(info, _blank_to_text(value, info), f.label)
    _check_my_story(api_prompt, node_ids)
    return api_prompt


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
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        names = (x.strip(" \t'\"[]") for x in re.split(r"[,;\n]", raw))
        return [x for x in names if x]
    return []


def _usable(info: dict, f: Field) -> dict:
    """`info` without the choices this field must not offer or accept."""
    if f.exclude and info["kind"] == "choice":
        info = dict(info, choices=[c for c in info["choices"] if c not in f.exclude])
    return info


def _blank_to_text(value: Any, info: dict) -> Any:
    """A page sends an empty box as null; a text input wants an empty string."""
    return "" if value is None and info["kind"] == "text" else value
