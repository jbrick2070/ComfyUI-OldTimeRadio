# Radio Play Studio

Press one button and your own ComfyUI writes, casts, performs, scores and films an
old-time radio play. Radio Play Studio is a small local web app built on the
[Comfy SDK](https://docs.comfy.org/development/api-development/sdks): it reads the
[Old-Time Radio](https://github.com/jbrick2070/ComfyUI-OldTimeRadio) pack's 16 GB
foley workflow, turns it into an API graph against your server's live node definitions,
sets the few things a person chooses, runs it through the SDK and plays the
finished episode.

Everything that renders is open weights on your own GPU. The app makes no Comfy
Router or Comfy Cloud calls and holds no API key.

```
browser  ->  Radio Play Studio  ->  Comfy SDK  ->  comfy-api-proxy  ->  your ComfyUI + Old-Time Radio pack
 (form,        (this repo:          (v2 API,        (bridge, :8189)      (:8188; writes, casts, voices,
  player)       graph + events)      jobs, events)                         scores and films the episode)
```

## What you need

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running on your machine, with the
  **Old-Time Radio** pack installed (ComfyUI Manager: `comfyui-old-time-radio`, or clone it into
  `custom_nodes/`), on ComfyUI v0.41.0 or newer (the FLUX 3 Klein stills need it).
- By default, an NVIDIA card with 16 GB. Other cards run one of the pack's other workflows
  (see "Pick the workflow for your GPU" below). The first run downloads the open weights it
  needs (tens of GB), with no Hugging Face token.
- Python 3.10 or newer for this app.

## Set up

```
cd contest/radio-play-studio
pip install -r requirements.txt     # comfy-sdk and comfy-api-proxy
comfy-api-proxy                     # leave running: serves the Comfy API v2 on :8189 for your ComfyUI on :8188
```

Tell the app where the pack is, by any one of:

```
set OTR_PACK_DIR=C:\path\to\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio     (or export on macOS/Linux)
set COMFYUI_DIR=C:\path\to\ComfyUI
python -m radio_play_studio ... --workflow C:\path\to\otr_16gb_foley.json
```

(Run from a checkout of the pack and it finds `workflows/otr_16gb_foley.json` itself.)

## Run

```
python -m radio_play_studio doctor     # checks ComfyUI, the pack, the SDK and the bridge; says what is missing
python -m radio_play_studio serve      # open http://127.0.0.1:8765
```

Choose **A classic, or surprise me** or **Tell my own story** (see below), press
**Make the episode**, and watch the progress bar. The form starts on the workflow's own
settings, which for the 16 GB foley workflow is three acts; a one-act episode took about
an hour on an RTX 5080 Laptop (16 GB), most of it filming the beats, and more acts take
longer. The first run's bar counts the graph's stages; after one
finished run the app knows how long each stage takes on your machine, and the bar follows
the clock and says about how many minutes are left. The finished video is saved under
`./episodes/<id>/` and plays in the page; ComfyUI also publishes it to its own
`output/otr/obs/` folder.

From the command line:

```
python -m radio_play_studio run --bank shakespeare --acts 1
python -m radio_play_studio export-graph graph_api.json     # the exact graph the app would submit
```

### Pick the workflow for your GPU

`--workflow` takes any of the pack's shipped graphs, so the same app runs on the
machine each one is tuned for, for example `workflows/otr_8gb_ltx25_foley.json` on an
8 GB card, `workflows/otr_16gb_video.json` for silent LTX video on 16 GB, or
`workflows/otr_app.json` for the pack's any-machine graph. The default is
`otr_16gb_foley.json`: LTX 2.5 films every beat with its own sound, mixed under the voices.

## Tell your own story

Press **Tell my own story** at the top of the page. Four boxes come first, each
showing an example, and every one of them is optional:

| Box | What to write | Example |
|---|---|---|
| Who is in it | Names, how they know each other, what they are like. Put `(she)` or `(he)` right after a name and that character is voiced and drawn that way. | `Ada (she), the lighthouse keeper. Tom (he), her brother, who does not believe her.` |
| What happens | Events, the trouble, a twist, the ending you want. Rough notes are fine. | `A ship's bell rings from the fog every night, but no ship has come in thirty years. Ada rows out to find it.` |
| Where and when | Place, era, weather, mood: the story's world, not its look. | `A rock lighthouse off Maine, winter 1921.` |
| Written by | Your name or a pen name. The announcer says it and the credits print it. | `Your Name` |

What you write outranks anything the writer would rather invent; what you leave out
is invented to fit. Leave all four empty and the pack writes its own standing premise.
Then set the language, acts, cast size and visual style as usual and press **Make the
episode**.

From the command line, the same boxes are flags:

```
python -m radio_play_studio run --bank my_story --acts 1 ^
  --my-characters "Ada (she), the lighthouse keeper. Tom (he), her brother." ^
  --my-plot "A ship's bell rings from the fog every night. Ada rows out to find it." ^
  --my-setting "A rock lighthouse off Maine, winter 1921." --my-author "Your Name"
```

(`^` continues a line in the Windows command prompt; use `\` on macOS and Linux.) The
boxes are read only with the `my_story` bank: text in them with another bank picked by
hand is refused before anything is sent, because the pack would stop that run.

## What a person can choose

Story bank, the classic to adapt, language, acts, cast size, visual style, and the four
My Story boxes above. Every dropdown is read live from your ComfyUI, so it offers exactly
the banks and styles your installed pack has. Everything else (models, voices, music,
the video lane) keeps the workflow's own setting.

## How it stays honest

- **Live node definitions.** A saved workflow's widget values mean nothing without the node's input order, so the
  graph is built against your server's `/object_info`. A pack and a workflow from different versions are refused
  with a message naming the node, never sent.
- **Never the cloud by accident.** The Comfy SDK talks to Comfy Cloud when `COMFY_BASE_URL` is unset. The app
  always sets it to your local bridge first.
- **A finished job is not a finished episode.** The run ends in success only when the video file is on disk.
- **Local only.** The web app listens on loopback, refuses other Host headers and only accepts JSON posts.

## Known limits

- There is no cancel button. Cancel a run from ComfyUI's own queue.
- Restarting the app drops its memory of a run. The render keeps going on your ComfyUI and its episode still
  lands in `otr/obs/`, but the page cannot re-attach to it.
- The app listens only on this machine and has no login, so it refuses `--host` values beyond loopback.

## Tests

```
cd contest/radio-play-studio
python -m pytest tests          # no ComfyUI, GPU or pack needed
```

From the pack's repo root, `tests/test_contest_app_converter_parity.py` pins the app's
standard-library graph converter to the pack's own on every shipped workflow, and checks
the form against the pack's real node definitions.

## License

MIT, see [LICENSE](LICENSE).
