"""Single source of truth for the v2 audio/casting NODE registrations
(plan piece 6).

The exact ComfyUI mapping key, module path, class name, display name, and
CATEGORY for every NEW node introduced by the audio + voice-casting overhaul
live here -- in ONE place -- so the node modules, the top-level package
``__init__`` registration, the opt-in workflow builder (Wave 2b), and the tests
all agree on the surface.

Wiring note: the top-level ``__init__._NODE_MODULES`` merges
``new_node_modules_table()`` only once each node module actually exists (added
per node in Wave 1a-c / 2a). Registering a node whose module is not yet on disk
would make the isolated loader log a phantom "Skipped" line and flip the
all-nodes-loaded banner, so this module stays a passive declaration until then
(C-6: box-fresh must still Queue clean).

Import-time is side-effect-free (C-5) -- no IO, no node import here.
"""
from __future__ import annotations

from dataclasses import dataclass

# CATEGORY for the v2 audio lane. Sits beside OTR_LedgerFreezeCascade's
# "OldTimeRadio/v2" so the new nodes group together in the ComfyUI add-node menu.
V2_AUDIO_CATEGORY = "OldTimeRadio/v2/audio"


@dataclass(frozen=True)
class NewNodeSpec:
    """One new node's full registration surface.

    ``key`` is the permanent public NODE_CLASS_MAPPINGS id (never rename).
    ``module_path`` is the package-relative import path used by the top-level
    ``__init__`` loader; ``class_name`` is the class within that module;
    ``display_name`` is the NODE_DISPLAY_NAME_MAPPINGS value (leading space +
    Title Case, matching the existing OTR nodes); ``category`` is the ComfyUI
    CATEGORY the node class declares.
    """

    key: str
    module_path: str
    class_name: str
    display_name: str
    category: str


# Order = build order (Wave 1a, 1b, 1c, then Wave 2a CastLock).
NEW_AUDIO_NODES = (
    NewNodeSpec(
        "OTR_BatchCharacterVoices",
        ".nodes.batch_character_voices",
        "BatchCharacterVoices",
        " Batch Character Voices (v2)",
        V2_AUDIO_CATEGORY,
    ),
    NewNodeSpec(
        "OTR_AnnouncerVoice",
        ".nodes.announcer_voice",
        "AnnouncerVoice",
        " Announcer Voice (v2)",
        V2_AUDIO_CATEGORY,
    ),
    NewNodeSpec(
        "OTR_StableAudioTheme",
        ".nodes.stable_audio_theme",
        "StableAudioTheme",
        " Stable Audio Theme (v2)",
        V2_AUDIO_CATEGORY,
    ),
    NewNodeSpec(
        "OTR_CastLock",
        ".nodes.cast_lock",
        "CastLock",
        " Cast Lock (v2 ledger authority)",
        V2_AUDIO_CATEGORY,
    ),
)

NEW_NODE_SPECS = {spec.key: spec for spec in NEW_AUDIO_NODES}


def new_node_modules_table() -> dict:
    """The ``{key: (module_path, class_name, display_name)}`` mapping in the
    exact shape the top-level ``__init__._NODE_MODULES`` consumes.

    The package ``__init__`` merges these as each module lands so the registry
    here stays the single declaration the loader reads from.
    """
    return {
        spec.key: (spec.module_path, spec.class_name, spec.display_name)
        for spec in NEW_AUDIO_NODES
    }
