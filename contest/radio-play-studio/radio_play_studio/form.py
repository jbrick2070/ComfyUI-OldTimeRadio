"""The handful of choices Radio Play Studio puts in front of a person, and how
each lands on the pack's graph.

A field names a node TYPE and an input, never a node id: the workflow's ids
change whenever it is re-saved, its types do not. Everything a field may be set
to is read live from the server (`/object_info`), so the form always offers the
story banks, styles and languages the installed pack really has.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .graph import GraphError, coerce_value, describe_input, node_id_for_type

WRITER = "OTR_LedgerScriptWriter"


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
    Field("my_characters", "Characters", WRITER, "story_characters",
          "Used when the story bank is my_story.", "My story"),
    Field("my_plot", "Plot", WRITER, "story_plot",
          "Used when the story bank is my_story.", "My story"),
    Field("my_setting", "Setting", WRITER, "story_setting",
          "Used when the story bank is my_story.", "My story"),
    Field("my_author", "In the style of", WRITER, "story_author",
          "Used when the story bank is my_story.", "My story"),
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
    return api_prompt


def _usable(info: dict, f: Field) -> dict:
    """`info` without the choices this field must not offer or accept."""
    if f.exclude and info["kind"] == "choice":
        info = dict(info, choices=[c for c in info["choices"] if c not in f.exclude])
    return info


def _blank_to_text(value: Any, info: dict) -> Any:
    """A page sends an empty box as null; a text input wants an empty string."""
    return "" if value is None and info["kind"] == "text" else value
