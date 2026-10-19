# Radio Play Studio

Press one button and your own ComfyUI writes, casts, performs, scores and films an
old-time radio play. Radio Play Studio is a small local web app built on the
[Comfy SDK](https://docs.comfy.org/development/api-development/sdks): it reads the
[Old-Time Radio](https://github.com/jbrick2070/ComfyUI-OldTimeRadio) pack's app
workflow, turns it into an API graph against your server's live node definitions,
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
  `custom_nodes/`). Its README lists the GPU it needs; 8 GB NVIDIA cards work with the 8 GB workflows.
- Python 3.10 or newer for this app.

## Set up

```
cd contest/radio-play-studio
pip install -r requirements.txt     # comfy-sdk and comfy-api-proxy
comfy-api-proxy                     # leave running: serves the Comfy API v2 on :8189 for your ComfyUI on :8188
```

Tell the app where the pack's `otr_app.json` is, by any one of:

```
set OTR_PACK_DIR=C:\path\to\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio     (or export on macOS/Linux)
set COMFYUI_DIR=C:\path\to\ComfyUI
python -m radio_play_studio ... --workflow C:\path\to\otr_app.json
```

(Run from a checkout of the pack and it finds `workflows/otr_app.json` itself.)

## Run

```
python -m radio_play_studio doctor     # checks ComfyUI, the pack, the SDK and the bridge; says what is missing
python -m radio_play_studio serve      # open http://127.0.0.1:8765
```

Pick a story bank (or leave it on roll), press **Make the episode**, and watch the
progress bar. A full episode takes minutes on a 16 GB card and hours on an 8 GB card;
the bar moves in steps. The finished video is saved under `./episodes/<id>/` and plays
in the page.

From the command line:

```
python -m radio_play_studio run --bank shakespeare --acts 1
python -m radio_play_studio export-graph graph_api.json     # the exact graph the app would submit
```

### Pick the workflow for your GPU

`--workflow` takes any of the pack's shipped graphs, so the same app runs on the
machine each one is tuned for, for example `workflows/otr_8gb_video.json` on an
8 GB card or `workflows/otr_16gb_video.json` on 16 GB. The default is `otr_app.json`.

## What a person can choose

Story bank, the classic to adapt, language, acts, cast size, visual style, and
the "My story" details (characters, plot, setting, in the style of). Every
dropdown is read live from your ComfyUI, so it offers exactly the banks and
styles your installed pack has. Everything else (models, voices, music, the
video lane) keeps the workflow's own setting.

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
