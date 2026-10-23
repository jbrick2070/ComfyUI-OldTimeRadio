<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Meta-Brief Image Prompt Gen

Writes the prompts the image model paints from, one per character and scene, from the episode's creative brief and visual style.

## What it does

Composes descriptive still-image prompts for characters and scenes using the propagating creative brief stamped in the ledger. Formats visual prompts and aspect ratios to match downstream video engines with optional language-model prompt refinement. Wire script and image policy JSON inputs or route through an optional execution gate.

## Inputs

The shipped workflows set these; most people never change them.

**`script_json`** -- a connection from another node (`STRING`).

Frozen ledger JSON (cast + meta brief). Image prompts derive from here.

### Optional

**`image_policy_json`** -- a connection from another node (`STRING`).

OTR_ImageDirector policy: seed + per-role still 'aspects' (so character stills match the selected video engine: portrait 832x1216 vs 16:9 832x480).

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal (opaque STRING).

## Outputs

- `image_prompts_json` (`STRING`)
- `report` (`STRING`)

## Read more

- [STYLES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/STYLES.md)
- [IMAGE_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/IMAGE_MODELS.md)

Class `OTR_MetaBriefImagePromptGen`, category `OldTimeRadio/v2/image`.
