"""OTR_AnnouncerVoice -- model-agnostic announcer-voice node (Wave 1 / 1b).

The generic announcer-voice surface for the opt-in v2 audio lane. Shares the
voice-node dispatch core with OTR_BatchCharacterVoices:

  * ``kokoro`` (default, ``per_line``) -> one announcer line per call via the
    self-contained Kokoro engine (audio clean-break 1b). One British voice is
    chosen per episode (seeded from episode_seed); the voice .pt is verified on
    local disk in a C-7 preflight (never fetched during execute).
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
