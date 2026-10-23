<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Image Gen Dispatcher (cache + ledger + image_done)

Paints the stills: the character portraits and the scene pictures, with the models Image Director chose, and records each one in the ledger so a repeat run paints only what changed.

## What it does

Coordinates still image generation for character portraits and scene backgrounds against a content-addressed disk cache. Reuses existing stills on cache hits, renders missing images using the selected image model, records image paths in the ledger, and emits the image-done signal. Connect script, image policy, and prompt wires or supply ordering gates.

## Inputs

The shipped workflows set these; most people never change them.

**`script_json`** -- text, as many lines as you like; starts as `{}`.

Frozen ledger JSON; the dispatcher stamps ledger['images'] into it.

**`image_policy_json`** -- text, as many lines as you like; starts as `{}`.

OTR_ImageDirector policy (engine per role + fresh cap + seed).

**`image_prompts_json`** -- text, as many lines as you like; starts as `{}`.

OTR_MetaBriefImagePromptGen output: the versioned {"objects":[...]} payload (portraits + scene stills).

### Optional

**`gate_in`** -- text, as many lines as you like; starts as empty.

Optional ordering signal (e.g. audio_done); opaque STRING.

**`episode_id`** -- text; starts as empty.

Episode id (still-spine ST-3/DS-3; wired in the saved json). Every still materializes into episodes/<episode_id>/stills/ + stills_manifest.json. Falls back to the ledger's episode_id; an unkeyed dispatch is LOUD.

## Outputs

- `patched_ledger_json` (`STRING`)
- `image_done` (`STRING`)
- `report` (`STRING`)

## Read more

- [IMAGE_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/IMAGE_MODELS.md)

Class `OTR_ImageGenDispatcher`, category `OldTimeRadio/v2/image`.
