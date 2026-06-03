"""OTR_BatchCharacterVoices -- model-agnostic character-voice node (Wave 1 / 1a).

The generic character-voice surface for the opt-in v2 audio lane. It picks its
engine from the shared audio-engine registry and dispatches FAIL-CLOSED via the
shared voice-node base:

  * ``bark`` (byte-identical legacy default, ``batch`` interface) -> RAW
    verbatim delegation to OTR_BatchBarkGenerator with the exact upstream
    ``script_json`` string plus the frozen widget tuple from
    ``config/legacy_invocation_manifest.json`` (zero transform; I-1, I-3).
  * ``chatterbox`` / ``indextts2`` (opt-in, flag-gated, ``per_line``) -> one
    frozen ``ResolvedVoiceRequest`` per character line, prepared text, the
    adapter call, then ``pack_audio_batch`` into the Bark AUDIO-batch contract
    (C-4).

All dispatch, teardown-before-done (I-7), the gate chain, the never-empty batch,
and the C-5 INPUT_TYPES live in ``_otr_voice_node_common`` and are shared with
OTR_AnnouncerVoice. Engine libraries are lazy-imported inside ``generate``.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

from ._otr_voice_node_common import OTRVoiceNodeBase, voice_input_types


class BatchCharacterVoices(OTRVoiceNodeBase):
    """Generic character-voice node. Registered as ``OTR_BatchCharacterVoices``.

    Engine order: bark (legacy byte-identical default) > chatterbox > indextts2.
    """

    ROLE = "char_voice"
    LINE_ROLES = ("character",)
    DONE_PREFIX = "char_voice"
    LEGACY_FIRST_FALLBACK = ("bark", "chatterbox", "indextts2")

    CATEGORY = "OldTimeRadio/v2/audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("character_audio", "render_log", "done")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return voice_input_types(cls.ROLE, cls.LEGACY_FIRST_FALLBACK)
