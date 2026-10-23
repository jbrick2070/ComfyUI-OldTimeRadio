<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Announcer Voice (v2)

Speaks the announcer's lines: the opening, the title, the bookends and the closing. The engine comes from Cast Lock.

## What it does

Synthesizes narration, opening introductions, and outro bookends for the episode using the selected voice engine. Dispatches each announcer line from the script to generate an audio batch for the scene sequencer. The engine is chosen on OTR_CastLock (announcer_voice_engine), which casts the announcer before this node runs -- Kokoro, Chatterbox, Bark or a cloud voice.

## Inputs

The shipped workflows set these; most people never change them.

**`script_json`** -- a connection from another node (`STRING`).

Frozen v2 ledger JSON from OTR_LedgerFreezeCascade (its first output). Passed VERBATIM to the legacy engine on the byte-identical batch path.

### Optional

**`ledger_json`** -- a connection from another node (`STRING`).

Cast-locked ledger from OTR_CastLock (carries voice_ref_id / episode_seed). Read on the per-line path; the batch path ignores it and delegates the raw script_json. No episode_seed widget (I-4).

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal. Wire an upstream 'done' here to force this node to run after it. Optional on the first chain node; the audio chain binds it node->node.

## Outputs

- `announcer_audio` (`AUDIO`)
- `render_log` (`STRING`)
- `done` (`STRING`)

## Read more

- [VOICES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VOICES.md)

Class `OTR_AnnouncerVoice`, category `OldTimeRadio/v2/audio`.
