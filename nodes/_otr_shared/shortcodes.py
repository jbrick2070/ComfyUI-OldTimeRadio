"""``shortcodes`` -- four-character names for every dimension a filename carries.

WHY THIS FILE EXISTS (operator ruling, 2026-09-07). The published episode name
concatenates the choices that produced it:

    <title>_<ts>__<style>__<video>__<image>__<tts>__<bank>_final.mp4

MEASURED, and one correction worth keeping. An earlier estimate put that name at
249 of a 250-unit path budget -- one unit from failing -- but that figure was
built from DROPDOWN LABELS (``ltx098_low_video``, ``shakespeare_folger``) when
the name is actually built from ledger values (``ltx_8gb``, ``shakespeare``).
The real numbers for the episode that failed, both measured:

    old, 5 fields spelled in full   name 130/150   path 221/250
    new, 7 fields as short codes    name 106/150   path 197/250

So the published name was NOT about to overflow MAX_PATH; the path overflow was
in the ``episodes/`` tree, where the id appears twice (see
``_otr_shared/pathbudget.py``). What the codes actually buy here is different
and still worth having: ``_OBS_NAME_MAX`` caps this name at 150 and TRIMS THE
TITLE to fit, so every character the components spend is one the operator's
episode title loses in the folder he watches daily. Coding them buys 24
characters of title back AND makes room for the writer and music dimensions,
which were previously invisible -- two episodes differing only by writer were
indistinguishable.

That mistake has a lesson in it, and it is the same one that nearly broke this
table: there are TWO vocabularies for several dimensions, and only the ledger's
reaches a filename. Measure with the values the code actually writes.

    OPERATOR RULING: four characters, with one deliberate exception -- the
    writer LLM may use five, so a row can carry both family and parameter
    count (`q354b` = Qwen 3.5 4B). See MAX_CODE_LEN_BY_DIMENSION.

CODES ARE UNIQUE WITHIN A DIMENSION, NOT GLOBALLY. The name has fixed positional
slots, so ``csd2`` may mean cloud_seedance_2 in the video slot and
cloud_seedream_2 in the image slot without ambiguity -- they can never appear in
the same position. Enforcing global uniqueness across 85 values would force
unreadable codes for no benefit. ``tests/test_shortcodes.py`` asserts
per-dimension uniqueness and the four-character cap, and -- more importantly --
asserts that every live dropdown value HAS a code, so a new engine cannot ship
without one.

THIS IS A NAMING LAYER, NOT AN IDENTITY LAYER. The ledger, the manifests and
every widget keep the full names; only generated FILENAMES use the codes. A code
is not a key: never resolve a code back to an engine to decide behaviour, and
never store one where the full name belongs. It exists so a path fits.
"""
from __future__ import annotations

#: Sentinels that are UI affordances rather than engine choices. They never
#: reach a filename, so they deliberately have no code.
SENTINELS = ("+ Add Custom Model",)

LLM = {
    "Qwen/Qwen3.5-4B": "q354b",
    "unsloth/Llama-3.2-3B-Instruct": "lla3",
    "mistralai/Mistral-Nemo-Instruct-2407": "nemo",
    "google/gemma-4-E2B-it": "g4e2",
    "google/gemma-4-E4B-it": "g4e4",
    "google/gemma-4-12b-it": "g412",
    "google/gemma-2-2b-it": "g22b",
}

SOURCE_BANK = {
    "roll (any eligible bank)": "roll",
    "media_archive": "marc",
    "original": "orig",
    "scifi_news_pro": "news",
    "public_domain": "pubd",
    "shakespeare": "sspr",
    "custom_source_bank": "cust",
}

VISUAL_STYLE = {
    "roll (any style)": "roll",
    "anime": "anim",
    "archival_documentary": "arch",
    "cartoon": "cart",
    "paper_origami": "pori",
    "recur_frac": "rfrc",
    "sci_fi_radio": "scif",
    "shakespeare_stage_realism": "shst",
    "storybook_engraving": "sbke",
    "video_art": "vart",
    "visual_storybased": "vstb",
}

#: KEYED ON ENGINE IDS, NOT DROPDOWN LABELS -- and the difference is not
#: cosmetic. The video dimension is the one place the pack carries TWO
#: vocabularies: the operator picks ``ltx098_low_video (16:9)`` from the
#: dropdown, but the ledger records ``engine_id`` and the published name is
#: built from that, which is why every obs episode on disk reads ``ltx_8gb``.
#: A table keyed on the labels would have spelled every single video lane
#: ``unk``. The ids below are ``_otr_video_engines.registry.CAPABILITIES``, and
#: the completeness test reads that registry rather than the dropdown for this
#: one dimension.
#:
#: The dropdown labels are deliberately NOT aliased in here. Mapping label to
#: engine is someone else's job and guessing it would put a wrong-but-plausible
#: code in a filename, which is worse than the ``unk`` that would announce it.
VIDEO_LANE = {
    "animatediff15_v3_haunted_video": "adhv",
    "animatediff15_v3_stillin_lab_video": "adsl",
    "cloud_kling_avatar": "ckla",
    "cloud_seedance_2": "csd2",
    "cloud_vidu_q2_pro_fast_720p": "cvdu",
    "cloud_wan_i2v": "cwan",
    "cloud_wan_i2v_audio": "cwna",
    "fastwan_8gb": "fw8g",
    "google_omni_video": "gomn",
    "google_veo_video": "gveo",
    "humo": "humo",
    "humo_1.7B": "h17",
    "humo_1.7B_169": "h17w",
    "humo_14B_169": "h14w",
    "ltx25_foley_plus": "l25f",
    "ltx25_mime": "l25m",
    "ltx25_video": "l25v",
    "ltx_8gb": "lx8g",
    "ltx_audio_in": "lxai",
    "ltx_video": "lxvd",
    "mesh_stage": "mesh",
    "minimax_h3_audio_in": "mh3a",
    "minimax_h3_video": "mh3v",
    "still_flat": "stfl",
    "still_motion": "stmo",
    "still_pan": "stpa",
    "still_word": "stwo",
    "viz_camera": "vcam",
    "viz_green": "vgrn",
    "viz_mxc_cpu": "vmcp",
    "viz_mxc_mandala": "vmmn",
    "wan_ti2v": "wti2",
    "word_razzle": "wraz",
}

IMAGE_GEN = {
    "cloud_flux_pro": "cflx",
    "cloud_krea_2_turbo": "ckr2",
    "cloud_luma_photon_flash": "clum",
    "cloud_nano_banana_2": "cnb2",
    "cloud_seedream_2": "csd2",
    "flux2_klein": "fkln",
    "flux_gen1": "flg1",
    "google_image": "gimg",
    "ideo": "ideo",
    "ideogram4_local": "idg4",
    "lumina_image": "lumi",
    "z_image_turbo": "zimg",
}

#: The UNION of `char_voice` and `announcer_voice` from
#: `_otr_engine_profiles._LEGACY_FIRST_ENGINES`, not the announcer dropdown.
#: The published name is built from `meta.char_voice_engine`, and the char_voice
#: set carries `indextts2` -- which the announcer dropdown does not offer, so a
#: table checked only against that dropdown looked complete while spelling every
#: indextts2 episode `unk`. It was caught by an existing test fixture, not by
#: the completeness check, which is why that check now reads the profiles.
TTS = {
    "kokoro": "koko",
    "chatterbox": "chat",
    "dia": "dia",
    "elevenlabs": "elev",
    "google_tts": "gtts",
    "bark": "bark",
    "indextts2": "idx2",
}

MUSIC_GEN = {
    "stable_audio_3": "sa3",
    "musicgen": "mgen",
    "stable_audio_music": "sam",
    "sonilo": "soni",
    "google_lyria": "lyra",
}

UPSCALER = {
    "off": "off",
    "spandrel_esrgan": "esrg",
}

#: Every dimension, in the order the published name spells them. The name is
#: positional, which is what lets codes repeat across dimensions.
DIMENSIONS = {
    "llm": LLM,
    "source_bank": SOURCE_BANK,
    "visual_style": VISUAL_STYLE,
    "video_lane": VIDEO_LANE,
    "image_gen": IMAGE_GEN,
    "tts": TTS,
    "music_gen": MUSIC_GEN,
    "upscaler": UPSCALER,
}

#: Default cap. Operator ruling 2026-09-07: four characters, and the arithmetic
#: in the module docstring is computed against it.
MAX_CODE_LEN = 4

#: Per-dimension exceptions, granted deliberately and one at a time.
#:
#: ``llm`` is 5 because a writer row has to carry BOTH a family and a parameter
#: count to be readable -- `q354b` says Qwen 3.5 4B, where a four-character
#: `q354` drops the B and reads like a version number. The LLM appears exactly
#: once in a name, so the extra character costs one unit, and the measured name
#: sits at 215 of 250 with it. Do not widen this to a habit: every additional
#: character is spent on every episode forever.
MAX_CODE_LEN_BY_DIMENSION = {"llm": 5}


def max_code_len(dimension: str) -> int:
    """The cap that applies to ``dimension``."""
    return MAX_CODE_LEN_BY_DIMENSION.get(dimension, MAX_CODE_LEN)


def _bare(value: str) -> str:
    """A dropdown label reduced to the key this table is written against.

    Live labels carry decorations the table deliberately does not: a size badge
    on the writer rows (``Qwen/Qwen3.5-4B (4.3 GB)``), an aspect tag on the
    video rows (``ltx098_low_video (16:9)``) and a trailing note on the
    audio-reactive ones. All of them start at the first ``" ("``.
    """
    text = (value or "").strip()
    cut = text.find(" (")
    return (text[:cut] if cut > 0 else text).strip()


def code_for(dimension: str, value: str, default: str = "unk") -> str:
    """The four-character code for ``value`` in ``dimension``.

    Returns ``default`` for an unknown value rather than raising: a filename is
    not the place to fail a finished render, and an operator who typed a custom
    model still gets a nameable episode. The test suite is what keeps the table
    complete; this fallback is for the custom-model case the dropdown allows.
    """
    table = DIMENSIONS.get(dimension) or {}
    bare = _bare(value)
    return table.get(bare) or table.get(value) or default
