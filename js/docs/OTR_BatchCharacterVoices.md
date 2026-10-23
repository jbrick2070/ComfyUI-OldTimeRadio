<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Batch Character Voices (v2)

Speaks every character line in the script with the cast that Cast Lock fixed, one audio clip per line.

## What it does

Synthesizes spoken dialogue for all dramatic characters in the script using voice assignments from the production ledger. Dispatches character lines to the chosen speech engine and packs the resulting audio into a batch for sequencing. The engine is chosen on OTR_CastLock (char_voice_engine), which casts every voice before this node runs -- IndexTTS2, Chatterbox, Bark, Kokoro or a cloud voice.

## Inputs

The shipped workflows set these; most people never change them.

**`script_json`** -- a connection from another node (`STRING`).

Frozen v2 ledger JSON from OTR_LedgerFreezeCascade (its script_json output). Passed VERBATIM to the legacy engine on the byte-identical batch path.

### Optional

**`ledger_json`** -- a connection from another node (`STRING`).

Cast-locked ledger from OTR_CastLock (carries voice_ref_id / episode_seed). Read on the per-line path; the batch path ignores it and delegates the raw script_json. No episode_seed widget (I-4).

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal. Wire an upstream 'done' here to force this node to run after it. Optional on the first chain node; the audio chain binds it node->node.

## Outputs

- `character_audio` (`AUDIO`)
- `render_log` (`STRING`)
- `done` (`STRING`)

## Read more

- [VOICES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VOICES.md)

Class `OTR_BatchCharacterVoices`, category `OldTimeRadio/v2/audio`.
