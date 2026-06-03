"""OTR_AnnouncerVoice -- model-agnostic announcer-voice node (Wave 1 / 1b).

The generic announcer-voice surface for the opt-in v2 audio lane. Shares the
voice-node dispatch core with OTR_BatchCharacterVoices:

  * ``kokoro`` (byte-identical legacy default, ``batch`` interface) -> RAW
    verbatim delegation to OTR_KokoroAnnouncer with the exact upstream
    ``script_json`` string plus the frozen widget tuple from
    ``config/legacy_invocation_manifest.json`` (zero transform; I-1, I-3).
  * ``chatterbox`` (opt-in, flag-gated, ``per_line``) -> one frozen
    ``ResolvedVoiceRequest`` per announcer line, prepared text, the adapter
    call, then ``pack_audio_batch`` into the Bark AUDIO-batch contract (C-4).

Only the announcer lines (``speaker_role == "announcer"``) are routed here; the
character bus stays on OTR_BatchCharacterVoices. Teardown-before-done (I-7), the
gate chain, the never-empty batch, and the C-5 INPUT_TYPES come from
``_otr_voice_node_common``. Engine libraries are lazy-imported inside
``generate``. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

from ._otr_voice_node_common import OTRVoiceNodeBase, voice_input_types


class AnnouncerVoice(OTRVoiceNodeBase):
    """Generic announcer-voice node. Registered as ``OTR_AnnouncerVoice``.

    Engine order: kokoro (legacy byte-identical default) > chatterbox.
    """

    ROLE = "announcer_voice"
    LINE_ROLES = ("announcer",)
    DONE_PREFIX = "announcer"
    LEGACY_FIRST_FALLBACK = ("kokoro", "chatterbox")

    CATEGORY = "OldTimeRadio/v2/audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("announcer_audio", "render_log", "done")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return voice_input_types(cls.ROLE, cls.LEGACY_FIRST_FALLBACK)
