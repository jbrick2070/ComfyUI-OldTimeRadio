<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Stable Audio Theme (v2)

Composes the opening and closing theme cues to fit the episode's setting and mood, with the composer engine chosen here. My Story episodes can ask for their music in words.

## What it does

Generates opening and closing musical theme cues tailored to the episode's setting, mood, and genre from the creative brief. Dispatches cue generation to the selected audio model and outputs a batched audio tensor with a manifest for the episode assembler. Change the music engine dropdown, specify a custom music style prompt, or connect execution gates.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**`script_json`** -- text, as many lines as you like; starts as `{}`.

Frozen v2 ledger JSON from OTR_LedgerFreezeCascade. Passed VERBATIM to the legacy engine on the batch path; read for cue mood on the clip path.

**Music - composer** (`engine`) -- a choice from the dropdown; starts as `stable_audio_3`.

Theme-music engine. `stable_audio_3` is the shipped default and is ungated; `musicgen` is selectable and is noncommercial; `stable_audio_music` needs a Hugging Face licence acceptance. Unusable selections fail closed with a named error.

### Optional

**`ledger_json`** -- text, as many lines as you like; starts as empty.

Cast-locked ledger from OTR_CastLock (carries episode_seed). Read on the clip path for the per-cue seed; the batch path delegates the raw script_json.

**`gate_in`** -- text, as many lines as you like; starts as empty.

Optional ordering signal. Wire an upstream 'done' here to force this node to run after it.

**My Story - music in words** (`music_style`) -- text; starts as empty.

Your own music, in your own words -- 'gamelan orchestra', 'surf rock', 'solo cello'. Applies to every generated cue in the episode -- opening, act-break interstitials and closing -- and only on the listener-authored My Story bank. Blank uses the bank's own genre: sci-fi news is Detroit techno, media archive a jazz quartet, the LLM-written Original Radio Drama bank salsa, public domain Chicago house, Shakespeare an Elizabethan consort.

## Outputs

- `cue_audio_clips` (`AUDIO`)
- `cue_manifest_json` (`STRING`)
- `render_log` (`STRING`)
- `done` (`STRING`)

## Read more

- [MUSIC.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/MUSIC.md)

Class `OTR_StableAudioTheme`, category `OldTimeRadio/v2/audio`.
