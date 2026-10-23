<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Cast Lock (v2 ledger authority)

Gives each character and the announcer a voice, and locks that cast for the whole episode. The voice engines are chosen here; the voices themselves follow the character's stated gender and the bank's pools.

## What it does

Locks character and announcer voice assignments between the script freeze cascade and audio synthesis. Maps cast members to reference audio from the voice bank, enforces language compatibility, and stamps the canonical ledger consumed by downstream audio nodes. Adjust the casting policy, toggle voice reuse when casts exceed the bank, or set default voice engines for characters and the announcer.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**`script_json`** -- text, as many lines as you like; starts as `{}`.

Frozen v2 ledger JSON from OTR_LedgerFreezeCascade (node 62 slot 1). CastLock rewrites it into the canonical ledger_json; the legacy raw path keeps reading this untouched string.

### Optional

**`cast_voice_policy`** -- a choice from the dropdown; starts as `auto_registry`.

Who picks the voices. auto_registry casts them from the bank, the same way every time; preserve_ledger keeps whatever the writer already assigned.

**`allow_voice_reuse`** -- on or off; starts as on.

Let two characters share a voice when the bank runs out of distinct ones. Off asks for a distinct voice per character wherever the bank has them; a language with only a few voices, or keeping a character's gender right, can still reuse one rather than stop.

**Characters - voices** (`char_voice_engine`) -- a choice from the dropdown; starts as `auto`.

Which engine speaks the characters. auto keeps the shipped Kokoro voice. Character Voices inherits this stamp -- there is no second engine dropdown.

**Announcer - voice** (`announcer_voice_engine`) -- a choice from the dropdown; starts as `auto`.

Which engine reads the announcer. auto keeps the shipped Kokoro voice. Announcer Voice inherits this stamp -- there is no second engine dropdown.

**`gate_in`** -- text, as many lines as you like; starts as empty.

Optional ordering signal (wire an upstream done).

**`voice_device`** -- a choice from the dropdown; starts as `default`.

Device for local voice and music engines. 'default' asks ComfyUI what this machine has and RECORDS what it chose; an explicit device is never second-guessed and still fails loud if it is not there.

**Randomize audio (non-cloud)** (`roll_audio_engines`) -- on or off; starts as off.

Yes: one local voice engine, drawn at random, voices the whole cast and the announcer, and one local music engine, drawn at random, scores the theme. Cloud engines are never drawn, nor one this machine cannot run or this episode's language does not allow. With preserve_ledger the writer's voices are kept, so only the music is drawn. Suggested for 16 GB cards and up.

## Outputs

- `ledger_json` (`STRING`)
- `cast_lock_revision` (`INT`)
- `cast_report` (`STRING`)
- `done` (`STRING`)

## Read more

- [VOICES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VOICES.md)
- [MULTILINGUAL.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/MULTILINGUAL.md)

Class `OTR_CastLock`, category `OldTimeRadio/v2/audio`.
