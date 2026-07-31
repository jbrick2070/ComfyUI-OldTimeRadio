# NEWBUG 2026-07-31 -- SceneSequencer advertises two music sockets its own method cannot accept

**Status: STATIC finding, NOT a PBUG.** Found by the Claude anchor pass of the
SFX kibitz r1 arc while grounding a panel claim about the music bus. It has NOT
been reproduced on a live leg, so per the admission rule it does not enter
`PROD_BUG_LOG.md` and is not Bible-eligible yet. This document exists so a
cross-check window can compare it against past PBUGs and the Bible before anyone
decides what it is.

**Found at HEAD** `c8cc3e0f` on `v2.0-alpha`. Nothing was changed in response.

## The defect

`nodes/scene_sequencer.py` declares two music inputs on the `SceneSequencer`
node class:

```python
# nodes/scene_sequencer.py:741
"music_cue_audio": ("AUDIO", {
    "tooltip": "Padded AUDIO batch of every music cue from "
               "OTR_StableAudioTheme. Sliced per manifest "
               "sample_count; never the padded tail."
}),
# nodes/scene_sequencer.py:746
"music_cue_manifest_json": ("STRING", { ... }),
```

Its execution method accepts neither, and has no `**kwargs`:

```python
# nodes/scene_sequencer.py:793-798
def sequence(self, script_json,
             tts_audio_clips=None,
             announcer_audio_clips=None,
             start_line=0, end_line=999, output_dir=DEFAULT_OUT,
             dialogue_offset_ms=0.0,
             ):
```

ComfyUI passes declared inputs as keyword arguments, so wiring EITHER socket
raises `TypeError: sequence() got an unexpected keyword argument
'music_cue_audio'` at execution time. The sockets are visible and connectable in
the UI and are unreachable in practice.

## Why it is latent today

The canonical workflow never wires them. Node 3 `OTR_SceneSequencer` has exactly
three inbound links -- `239` character voices, `240` announcer audio, `277`
CastLock `ledger_json` -> `script_json`. The music bus goes somewhere else
entirely: node 83 `OTR_StableAudioTheme` -> node 7 `OTR_EpisodeAssembler` via
links `282` (`music_cue_audio`) and `283` (`music_cue_manifest_json`).

The WORKING implementation of that music bus lives in `EpisodeAssembler`, whose
signature does accept both (`scene_sequencer.py:1230`), and which is also the
only caller of `_reconcile_active_music_manifest` (`:1252`).

So the defect is invisible until somebody believes the UI.

## Why it matters now

It was found because the SFX plan proposed mixing generated SFX into the
SceneSequencer scene bus. Any plan that reads those declared sockets as evidence
that SceneSequencer owns a music/SFX bus is reading a promise the code does not
keep -- and would wire exactly the connection that raises.

## Two readings, and the reason this is not being fixed on the spot

1. **Copy-paste leftover.** The declaration was copied from `EpisodeAssembler`
   and the method was never updated. Fix = delete the two `INPUT_TYPES` entries.
2. **Abandoned half-migration.** Someone intended to move the music bus into
   SceneSequencer and stopped after the declaration. Fix = implement the
   parameters, which is a real behaviour change to the scene bus.

These have opposite fixes and opposite blast radii, and the git history around
the "720-bakeoff C3" comment at `:735-740` is where the answer lives. Guessing
would be the shim this repo bans. It needs one grounded decision, not a patch.

## Repro sketch (offline, no GPU)

Instantiate the node class and call `sequence()` with `music_cue_audio=None`, or
add a link from node 83 output 0 to node 3 input `music_cue_audio` in a scratch
copy of the graph and submit it. Either produces the `TypeError`. A test that
asserts every declared `INPUT_TYPES` key is accepted by the node's `FUNCTION`
signature would catch this class across the whole node registry, and no such
test exists today.

## Suggested disposition

Cheap, high-value, and independent of the SFX build: add a registry-wide
signature-parity test (every declared input name is a parameter of the node's
`FUNCTION`, or the function takes `**kwargs`), then let it name every node that
has drifted. Fix this one according to whichever reading the history supports.
