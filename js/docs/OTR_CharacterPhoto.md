<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Character Photo

A photo for one character, for a My Story episode. The character's portrait is painted from it in the episode's style, and every still of them follows that portrait.

## What it does

A photo for one character. Their portrait is painted from it, in the episode's style, and every still of them follows that portrait -- the photo itself is never shown. Chain one node per photo (up to three for one face) into Image Director's character_photos_in. Leave the name and picture empty to pass the chain through unchanged.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**`name`** -- text; starts as empty.

The character's name as the story uses it, e.g. Jeffrey. Empty (with no picture) passes the chain through.

**`image`** -- a choice from the dropdown.

Their photo: pick, upload or paste one. A clear face works best.

### Optional

**`photos_in`** -- a connection from another node (`OTR_CHARACTER_PHOTOS`).

The previous Character Photo node, to chain them.

## Outputs

- `character_photos` (`OTR_CHARACTER_PHOTOS`)

## Read more

- [IMAGE_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/IMAGE_MODELS.md)

Class `OTR_CharacterPhoto`, category `OldTimeRadio/v2/image`.
