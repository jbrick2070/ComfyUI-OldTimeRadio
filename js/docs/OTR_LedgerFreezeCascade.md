<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# LFC Ledger Freeze Cascade (v2.0)

Checks the script ledger the writer produced and freezes it, so every node after it performs one fixed script. Nothing here is a creative choice; it refuses a ledger that would break downstream.

## What it does

Validates ledger structure, runs readiness checks, and freezes the production ledger before media rendering begins. Unloads resident story-writing language models from memory to free VRAM for downstream audio and video generation. Toggle individual diagnostic audit phase flags or inspect the freeze verdict to ensure story and cast integrity.

## Inputs

The shipped workflows set these; most people never change them.

**`script_text`** -- a connection from another node (`STRING`).

Passthrough from OTR_LedgerScriptWriter. The cascade reads the production ledger directly via peek_ledger(); this socket exists so the graph wires the writer and cascade in line.

### Optional

**`script_json`** -- a connection from another node (`STRING`).

Forwarded from the writer's script_json slot. The cascade re-serializes this from the post-freeze ledger so downstream consumers see the final state.

**`news_used`** -- a connection from another node (`STRING`).

Passthrough of the writer's news_used slot. Not touched by the cascade.

**`estimated_minutes`** -- a connection from another node (`INT`).

Passthrough of the writer's estimated_minutes INT slot. Not touched by the cascade.

**`technical_model`** -- a connection from another node (`STRING`).

Technical model ID from the writer. Required for compatibility on a normal current-ledger run; freeze does not acquire or generate with this model. Validated via _otr_model_inputs.require_model -- an unwired or blank socket raises MissingModelInputError loud.

**`enable_phase_7_audio_readiness`** -- on or off; starts as on.

LFC Phase 7 -- audio readiness. Expands abbreviations (Dr. -> Doctor), symbols (& -> and), and numbers (42 -> forty-two) so TTS produces pronounceable output. Default ON (deterministic + cheap).

**`enable_phase_8_video_readiness`** -- on or off; starts as on.

LFC Phase 8 -- video readiness audit. Checks cast portraits + voiced-line visual coverage. Mutates nothing; stamps meta.video_readiness. Default ON.

## Outputs

- `script_text` (`STRING`)
- `script_json` (`STRING`)
- `news_used` (`STRING`)
- `estimated_minutes` (`INT`)
- `freeze_verdict` (`STRING`)
- `episode_seed` (`INT`)
- `v2_ledger_json` (`STRING`)

This is an output node: the workflow runs to reach it.

## Read more

- [ARCHITECTURE.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/ARCHITECTURE.md)

Class `OTR_LedgerFreezeCascade`, category `OldTimeRadio/v2`.
