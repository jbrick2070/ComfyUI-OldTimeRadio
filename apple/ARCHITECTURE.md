# How an episode becomes a file

A one-page map of `workflows/otr_canonical.json`, for someone reading the code
for the first time. It describes the graph as it ships; the per-engine detail
lives in the guides listed at the end.

## The one idea

Everything downstream reads **the ledger**, not the node before it. The story
writer produces a ledger -- one JSON file per episode, kept in the episode's
folder under `otr/episodes/` -- and every later stage reads the fields it needs
from that file and writes its own fields back. A stage does not guess what an
earlier stage "meant". That is why a replay can freeze the ledger and re-run
only the picture (`--replay-from`), and why a stage that is ripped out must
still fill every field it used to own.

## The graph, in running order

| Stage | Node | What it does |
|---|---|---|
| Preflight | `OTR_WorkflowValidator` | Reads the workflow JSON from disk and runs the same contract check CI runs, so drift is caught at queue time, before anything is spent |
| 1 | `OTR_LedgerScriptWriter` | Writes the script and opens the ledger. The writer model, act count, source bank, visual style and `episode_language` are widgets here |
| 2 | `OTR_LedgerFreezeCascade` | Freezes the script; later stages read it as fixed |
| 3 | `OTR_CastLock` | Casts every speaking role to a voice engine and voice, and stamps the voice device |
| 4a / 4b / 4c | `OTR_BatchCharacterVoices`, `OTR_AnnouncerVoice`, `OTR_StableAudioTheme` | Speak the lines and score the theme cues through the engine each role was cast to |
| 5 | `OTR_SceneSequencer` | Lays voices and music on one timeline |
| 6 | `OTR_AudioEnhance` | Mix polish |
| 7 | `OTR_EpisodeAssembler` | Writes the 48 kHz master WAV and emits `audio_done`, the signal every picture stage waits on |
| 8 | `OTR_SignalLostVideo` | Draws a procedural base video from the audio; it is the floor under every episode |
| 9a-9d | `OTR_ShotLock`, `OTR_MetaBriefImagePromptGen`, `OTR_ImageGenDispatcher`, `OTR_VideoRenderBatch` | Plan the shots, write the image prompts, mint the stills, render a clip per beat and emit a clip manifest |
| 10 | `OTR_SilentComposite` | Puts the clips on the floor at the canvas size (1920x1080, 25 fps), optionally through an upscale engine |
| 12 | `OTR_CaptionBurn` | Burns captions and the title card |
| 13 | `OTR_CreditsRoll` | Rolls the credits over the final frame |
| 14 | `OTR_MasterAudioMux` | Copies the master WAV onto the picture and publishes to `otr/obs/` |

Two settings nodes (`OTR_VideoDirector`, `OTR_ImageDirector`) hold the per-role
engine choices. `OTR_ComfyCredential` receives the queue's Comfy API key and is
the only node that sees it; it reports only "present" or "absent", so the key
cannot appear in an error report. API keys never live in the workflow JSON.

## Engines are pluggable

Voice, music, image, video and upscale engines each have their own registry
under `nodes/_otr_*_engines/`: an adapter class decorated with `@register`, a
capabilities row (which devices it runs on, which vendor, which weights), and
nothing imported at module scope that is heavy. The dropdowns are built from
the registries, so adding an engine adds a menu entry. A hosted engine is never
chosen automatically; a local one fetches its own weights on first use.

## Presets are generated, not hand-edited

`workflows/otr_canonical.json` is the source. The 25 presets for specific
machines are produced from it by `scripts/build_variants.py` using the rows in
`config/workflow_matrix.json` and `config/machine_classes.json`;
`python scripts/build_variants.py --check` confirms they are in sync. Widget
values are positional, so a new widget is only ever appended at the end of a
node's list.

## Where things land

- Per-episode working files: `otr/episodes/<episode>/` (the ledger, the master
  audio, the clips, the intermediate videos).
- The finished episode: `otr/obs/`. An episode that does not reach `otr/obs/`
  did not finish, whatever the logs say.

## Reading further

- [RUN.md](RUN.md) -- the widgets and what to change
- [WRITERS.md](WRITERS.md), [VOICES.md](VOICES.md), [MUSIC.md](MUSIC.md),
  [IMAGE_MODELS.md](IMAGE_MODELS.md), [VIDEO_MODELS.md](VIDEO_MODELS.md),
  [UPSCALERS.md](UPSCALERS.md) -- the engine guides
- [MULTILINGUAL.md](MULTILINGUAL.md) -- the eight languages
- [EXTENDING.md](EXTENDING.md) -- adding an engine or a lane
