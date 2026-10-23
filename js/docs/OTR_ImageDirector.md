<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# ImageDirector (image policy)

Decides how the still pictures are made: which image model draws each role, and the character photos that portraits are painted from.

## What it does

Defines the still-image policy for the visual pipeline upstream of video rendering: forwards the image models picked on the video director, and sets the seed rule and the cloud still estimate. Which stills render follows the episode itself -- one portrait per character, the scenes its beats need.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**`fresh_cap`** -- a whole number from 1 to 512; starts as `15`.

Recorded on the image policy and used by the cloud cost estimate before a paid run: the estimate assumes this many stills per image role, never fewer than the default. It does NOT limit how many stills render -- the still count follows the episode's beats (docs review, 2026-10-14).

**`seed_mode`** -- a choice from the dropdown; starts as `request_hash`.

request_hash: seeds derive deterministically from each request (same episode re-renders byte-identical stills). fixed: every still uses request_seed verbatim -- for A/B comparisons.

**`request_seed`** -- a whole number from 0 to 4294967295; starts as `42`.

Base seed for the stills. Each shot derives its own seed from this and its request, so the same number reproduces the same pictures.

**`video_policy_json`** -- a connection from another node (`STRING`).

OTR_VideoDirector policy (REQUIRED; wire the provider before this consumer). Carries the frozen route, device policy and per-role aspects; empty/malformed FAILS CLOSED.

### Optional

**`custom_models_json`** -- text, as many lines as you like; starts as `{}`.

Declare custom image engines on OTR_VideoDirector's custom_models_json instead -- that one is resolved before this node and is what the queue check reads. This map is used only for a '+ Add Custom Model' pick that arrives here unresolved (a policy built without OTR_VideoDirector), e.g. {"music_image_model":"z_image"}.

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal (opaque STRING).

**`dtype_policy`** -- a choice from the dropdown; starts as `fp8_ok`.

Recorded on the episode receipt only. No image engine currently reads it to choose a precision.

**Character photos (Name = file)** (`character_photos`) -- text, as many lines as you like; starts as empty.

Optional. One line per character, Name = file (or Name = a.png, b.png for up to three photos of one face), naming pictures in ComfyUI's input folder. That character's portrait is painted FROM the photo, in the episode's style, and every still of them follows the portrait. Needs a still engine that takes a reference image (flux3_klein_ref); with any other the photo is reported and not used. A name that matches no one in the cast is reported too.

**`character_photos_in`** -- a connection from another node (`OTR_CHARACTER_PHOTOS`).

Character Photo nodes, chained: each one is a photo (image picker with upload and preview) for one character. Same use as the typed lines above.

## Outputs

- `image_policy_json` (`STRING`)

## Read more

- [IMAGE_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/IMAGE_MODELS.md)
- [STYLES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/STYLES.md)

Class `OTR_ImageDirector`, category `OldTimeRadio/v2/image`.
