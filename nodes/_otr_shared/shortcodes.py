"""``shortcodes`` -- four-character names for every dimension a filename carries.

WHY THIS FILE EXISTS (operator ruling, 2026-09-07). The published episode name
concatenates the choices that produced it:

    <title>_<ts>__<style>__<video>__<image>__<tts>__<bank>_final.mp4

Measured on the 4060 with a 65-character episode id, that name reached **249 of
the 250-unit budget** -- one unit of headroom -- because the components are
spelled in full: ``archival_documentary`` is 20 characters, ``shakespeare_folger``
18, ``ltx098_low_video`` 16, ``z_image_turbo`` 13. The operator also wants three
MORE dimensions in the name (the writer LLM, the music engine and the upscaler),
which the current spelling cannot afford at any title length.

With four-character codes the same name measures 200. That is the difference
between a name that fits every title and one that is already over the line for
anything longer than the episode that failed.

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

VIDEO_LANE = {
    "animatediff15_v3_haunted_video": "adhv",
    "animatediff15_v3_stillin_lab_video": "adsl",
    "cloud_kling_avatar": "ckla",
    "cloud_seedance_2": "csd2",
    "cloud_vidu_q2_pro_fast_720p": "cvdu",
    "cloud_wan_i2v": "cwan",
    "cloud_wan_i2v_audio": "cwna",
    "wan22_high_fast": "w22f",
    "wan22_high_video": "w22v",
    "google_omni_video": "gomn",
    "google_veo_video": "gveo",
    "humo14_high_audio_in_portrait": "h14p",
    "humo14_high_audio_in_wide": "h14w",
    "humo17_high_audio_in_portrait": "h17p",
    "humo17_high_audio_in_wide": "h17w",
    "ltx25_high_foley_plus": "l25f",
    "ltx25_high_mime": "l25m",
    "ltx25_high_video": "l25v",
    "ltx098_low_video": "l098",
    "ltx23_low_audio_in": "l23a",
    "ltx23_high_video": "l23v",
    "mesh_stage": "mesh",
    "h3_low_audio_in": "h3la",
    "h3_low_video": "h3lv",
    "still_flat": "stfl",
    "still_motion": "stmo",
    "still_pan": "stpa",
    "still_word": "stwo",
    "viz_camera": "vcam",
    "viz_green": "vgrn",
    "viz_mxc_cpu": "vmcp",
    "viz_mxc_mandala": "vmmn",
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

TTS = {
    "kokoro": "koko",
    "chatterbox": "chat",
    "dia": "dia",
    "elevenlabs": "elev",
    "google_tts": "gtts",
    "bark": "bark",
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
