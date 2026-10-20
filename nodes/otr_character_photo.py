"""OTR_CharacterPhoto -- give a character a photo on the canvas.

A ComfyUI-native picker for the character-photo feature: core's image picker
(upload button, paste, preview -- declared exactly as core's LoadImage does)
next to the character's name. The photo becomes the REFERENCE that character's
portrait is painted from, in the episode's style; every still of them, and of
anyone in a shot with them, then follows that portrait. The photo itself never
appears in the episode.

ONE picture per node, chained: the canvas only gives the first required picker
its upload button, and an optional picker would quietly default to the first
file in the folder (Codex design review, 2026-10-20). For an amalgamation --
several photos of one face -- chain more nodes with the same name, up to three.
The chain ends in OTR_ImageDirector's `character_photos_in` socket.

BLANK IS A PASS-THROUGH. The canonical workflow ships one of these wired and
empty (no name, no picture), so the socket is there on every machine and a
saved graph never names a file another machine lacks. A name without a picture,
or a picture without a name, is refused before the run starts.

Typed `character_photos` lines on OTR_ImageDirector (the Radio Play Studio app
and the command line write them) still work; for a name both give, the typed
lines win.

Cold-import clean: stdlib only at module scope; folder_paths is read lazily.
"""
from __future__ import annotations

import os

try:
    from ._otr_shared import character_photos as _photos
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import character_photos as _photos  # type: ignore

#: The chain's type: ``{"version": 1, "photos": {name: [file, ...]}}``.
CHARACTER_PHOTOS = "OTR_CHARACTER_PHOTOS"


def _input_images() -> list:
    """The pictures in ComfyUI's input folder, as core's LoadImage lists them."""
    try:
        import folder_paths  # type: ignore
        root = folder_paths.get_input_directory()
        files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        return sorted(folder_paths.filter_files_content_types(files, ["image"]))
    except Exception:  # noqa: BLE001 -- headless or test: no picker choices
        return []


def _chain(value) -> dict:
    photos = (value or {}).get("photos") if isinstance(value, dict) else None
    return {k: _photos.photo_list(v) for k, v in (photos or {}).items()}


class OTRCharacterPhoto:
    """Registered as ``OTR_CharacterPhoto``."""

    DESCRIPTION = (
        "A photo for one character. Their portrait is painted from it, in the "
        "episode's style, and every still of them follows that portrait -- the "
        "photo itself is never shown. Chain one node per photo (up to three "
        "for one face) into Image Director's character_photos_in. Leave the "
        "name and picture empty to pass the chain through unchanged."
    )
    CATEGORY = "OldTimeRadio/v2/image"
    FUNCTION = "add"
    RETURN_TYPES = (CHARACTER_PHOTOS,)
    RETURN_NAMES = ("character_photos",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "",
                    "tooltip": ("The character's name as the story uses it, e.g. "
                                "Jeffrey. Empty (with no picture) passes the "
                                "chain through."),
                }),
                "image": (_input_images(), {
                    "image_upload": True,
                    "tooltip": ("Their photo: pick, upload or paste one. A clear "
                                "face works best."),
                }),
            },
            "optional": {
                "photos_in": (CHARACTER_PHOTOS, {
                    "tooltip": "The previous Character Photo node, to chain them.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, name="", image=""):
        # Declaring `image` here takes over core's "value is in the list"
        # check, which is what lets the shipped blank node pass.
        name, image = " ".join(str(name or "").split()), str(image or "").strip()
        if not name and not image:
            return True
        if not name:
            return "Character Photo: give the character's name for this photo"
        if not image:
            return f"Character Photo: pick a photo for {name}, or clear the name"
        if not _photos.resolve_photo(image):
            return f"Character Photo: {image!r} is not a picture in ComfyUI's folders"
        return True

    @classmethod
    def IS_CHANGED(cls, name="", image="", **_):
        # The picture's bytes, as core's LoadImage does: a new photo saved
        # under the same file name must re-run the image phase.
        image = str(image or "").strip()
        return _photos.photos_fingerprint({name or "": [image]}) if image else ""

    def add(self, name="", image="", photos_in=None):
        photos = _chain(photos_in)
        name, image = " ".join(str(name or "").split()), str(image or "").strip()
        if name and image:
            photos = _photos.add_photo(photos, name, image)
        return ({"version": 1, "photos": photos},)

