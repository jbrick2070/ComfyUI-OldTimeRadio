# Radio Play Studio

**Press one button. Your own GPU writes, casts, voices, scores and films a whole radio play.**

Built for the Comfy [Open Source Higgsfield](https://comfy.org/open-source-higgsfield) challenge.
An open-source take on Higgsfield's Cinema Studio: where Cinema Studio directs a concept into
shots, Radio Play Studio writes the whole story, casts it, and directs every beat. Every character
keeps the same face and voice from shot to shot, and every shot is filmed with LTX on your own
card, with voices, music and foley mixed under the picture. Open weights only, nothing to pay
per episode; the app is MIT licensed, and each model keeps its own license (see License).

> **Demo video:** to be recorded before submission.

```
browser  ->  Radio Play Studio  ->  Comfy SDK  ->  comfy-api-proxy  ->  your ComfyUI + Old-Time Radio pack
 (form,        (this repo:          (v2 API,        (bridge, :8189)      (:8188; writes, casts, voices,
  player)       graph + events)      jobs, events)                         scores and films the episode)
```

## What it does

One press runs the [Old-Time Radio](https://github.com/jbrick2070/ComfyUI-OldTimeRadio) node pack
(on the Comfy registry as `comfyui-old-time-radio`) end to end on your machine:

| Stage | Open-weights model (default 16 GB foley workflow) |
|---|---|
| Story, cast and every line | Gemma 4 12B, run by ComfyUI itself (int8) |
| A still for every beat | MOSS-Image; FLUX 3 Klein as an option (the **Still pictures** picker) |
| Voices, one per character | Kokoro |
| Theme music | Stable Audio 3 |
| Video, with its own sound | LTX 2.5 (16 GB foley lane: each beat filmed and given foley, mixed under the voices) |

The app reads the pack's shipped workflow, turns it into an API graph against your server's live
node definitions, sets what a person chose, submits it through the Comfy SDK, follows it with a
progress bar that knows how long each stage takes on *your* machine, and plays the finished
episode. The app makes no Comfy Router or Comfy Cloud calls and holds no API key.

## What you need

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) v0.42.0 or newer running on your machine
  (MOSS-Image, the default still model, needs v0.42.0), with the **Old-Time Radio** pack installed
  (ComfyUI Manager: `comfyui-old-time-radio`, or clone it into `custom_nodes/`). One node pack;
  every model downloads itself from public, ungated repos with no Hugging Face token -- the
  writer and still models at pinned revisions, the rest checked against their published SHA-256.
- By default, an NVIDIA card with 16 GB. Other cards run one of the pack's other workflows (see
  "Pick the workflow for your GPU"). The first run downloads tens of GB of open weights.
- Python 3.10 or newer for this app (its only packages are the Comfy SDK and the bridge). Built and run on Windows.

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
python -m radio_play_studio ... --workflow C:\path\to\otr_16gb_foley_open.json
```

(Run from a checkout of the pack and it finds `workflows/otr_16gb_foley_open.json` itself.)

## Run

```
python -m radio_play_studio doctor     # checks ComfyUI, the pack, the SDK and the bridge; says what is missing
python -m radio_play_studio serve      # open http://127.0.0.1:8765
```

Choose **A classic, or surprise me** or **Tell my own story**, press **Make the episode**, and watch
the bar. The form starts on the workflow's own settings (three acts on the 16 GB foley workflow).
The finished video plays in the page, is saved under `./episodes/<id>/`, and ComfyUI also
publishes it to its own `output/otr/obs/` folder.

- **Reload-proof.** A reload, or a second tab, picks the episode back up where it is.
- **One at a time.** A second episode while one is being made is refused, not queued behind it.
- **The GPU goes back.** Within seconds of the last tab closing (two and a half minutes if the
  browser vanishes without saying goodbye), with nothing being made, the app asks ComfyUI to
  free the models its episodes loaded.

From the command line:

```
python -m radio_play_studio run --bank shakespeare --source-ref "folger-midsummer:act3-scene1-bottom-titania" --acts 1
python -m radio_play_studio export-graph graph_api.json     # the exact graph the app would submit
```

### Pick the workflow for your GPU

`--workflow` takes any of the pack's shipped graphs, so the same app runs on the machine each
one is tuned for: for example `workflows/otr_8gb_ltx25_foley.json` on an 8 GB card,
`workflows/otr_16gb_video.json` for silent LTX video on 16 GB, `workflows/otr_16gb_still.json`
for a still with motion (minutes, not an hour), or `workflows/otr_app.json` for the pack's
any-machine graph. The default is `otr_16gb_foley_open.json`: the 16 GB foley lane with every
model open weights (FLUX 3 Klein stills; the 1080p step is ffmpeg, no model). `otr_16gb_foley.json` is the
same lane with NVIDIA RTX Video Super Resolution, a closed SDK.

## Tell your own story

Press **Tell my own story** at the top of the page. Everything here is optional.

**Who is in it** is a card per character (**Add a character**):

| On the card | What to write |
|---|---|
| Name | What the story calls them. |
| Voice | he or she (the pack voices every character as a man or a woman), or leave it to the writer. |
| Who they are | Their part in the story and what they are like. |
| How they look | In fixed terms: age, build, face, hair, the clothes they always wear. Fixed terms are what keep a face the same from shot to shot. |
| Photos (optional) | Up to three pictures of them (one face, blended). Their portrait is painted **from** it, in the episode's own style, and every still of them follows that portrait. The photo itself never appears in the episode. |

**Load the example** fills the cards and the boxes below with a ready story: the author's own
Jeffrey and MagnoliaRex, from his novel *Mime Lung Music Festival*.

Then the three boxes:

| Box | What to write | Example |
|---|---|---|
| What happens | Events, the trouble, a twist, the ending you want. Rough notes are fine. | `A ship's bell rings from the fog every night, but no ship has come in thirty years. Ada rows out to find it.` |
| Where and when | Place, era, weather, mood: the story's world, not its look. | `A rock lighthouse off Maine, winter 1921.` |
| Written by | Your name or a pen name. The announcer says it and the credits print it. | `Your Name` |

What you write outranks anything the writer would rather invent; what you leave out is invented
to fit. Leave everything empty and the pack writes its own standing premise. Then set the
language, acts, cast size, visual style and still model as usual and press **Make the episode**.

### Same faces in every shot

The default workflow keeps every model open weights, and its stills paint each character from
the written description on their cast card (plus a portrait-derived seed), so describe looks in
fixed terms. For faces drawn **from** pictures, run the pack's reference workflow instead,
`--workflow workflows/otr_16gb_foley_flux3_ref.json`: every character gets a portrait (or the
photos on their card) and each still is painted from the portraits of everyone in the frame, so a
face does not change from beat to beat. Its still model, FLUX 3 Klein, is open weights under the
FLUX Kommunity License. On the default workflow a card's photo is reported as unused.

From the command line, the same boxes are flags. A card is one line of `--my-characters`, and a
photo is a `Name = file` line naming a picture in ComfyUI's input folder:

```
python -m radio_play_studio run --bank my_story --acts 1 --stills flux3_klein ^
  --my-characters "Ada (she): the lighthouse keeper. Looks: 40, weathered, a grey wool coat." ^
  --my-photos "Ada = ada.png" ^
  --my-plot "A ship's bell rings from the fog every night. Ada rows out to find it." ^
  --my-setting "A rock lighthouse off Maine, winter 1921." --my-author "Your Name"
```

(`^` continues a line in the Windows command prompt; use `\` on macOS and Linux.) The boxes are
read only with the `my_story` bank: text in them with another bank picked by hand is refused
before anything is sent, because the pack would stop that run.

## What a person can choose

Story bank, the classic to adapt, language, acts, cast size, visual style, the still model
(**Still pictures**: local engines only, MOSS-Image by default), the cast cards and the My Story boxes.
Every dropdown is read live from your ComfyUI, so it offers exactly what your installed pack has.
Everything else (voices, music, the video lane) keeps the workflow's own setting.

## Measured timings (RTX 5080 Laptop, 16 GB, ComfyUI 0.42.1, through the SDK and proxy)

| Job | Time |
|---|---|
| One-act episode, 16 GB foley workflow (LTX 2.5 with sound), five runs | 57, 64, 64, 68 and 76 min |
| One-act episode, 16 GB LTX video workflow (silent picture) | 65 min |
| One-act episode, 16 GB stills-with-motion workflow | 5 min |
| Where a foley hour goes (averaged over the runs) | video about 95% (60 of 64 min); the writer about 1 min; all stills about 2 min |
| Progress bar against the wall clock, a 68-minute episode | within 4 points throughout |

Run length follows the story: the number and length of the beats decide how many frames LTX films.

## Recipes that took real renders to find

- **ComfyUI keeps one websocket per client id.** A second listener on a running job (a debugging
  `curl` on the proxy's event stream) silently takes its messages, and the first stream goes quiet
  with no error. The app now refuses to learn timings from a stream that never saw the graph settle.
- **The video node's own bar was the inner sampler's.** One LTX clip runs three inner progress bars
  in a row, so the node's bar restarted at 0 several times a clip. The pack now quiets inner bars on
  the render thread and draws "frames rendered of the episode"; the app trusts a node's bar only
  while it moves forward under one total.
- **Counting nodes does not track time.** One node of 22 is about 95% of the hour, so a node-count
  bar sat near 80% for most of the run. Time-weighted bars still trailed by up to 17 points,
  because stories differ in length; only the video node's frame count fixed it.
- **The 12B writer doubled a quote before a JSON key** (`",""presence":`) on every retry. The pack's
  JSON repair now drops the stray quote, and touches only text strict JSON already rejects.
- **`/object_info` takes up to 40 s while ComfyUI is rendering.** The app waits up to 120 s and
  says what it is waiting for, and a reload picks a run back up without waiting for it at all.
- **FLUX 3 Klein needs ComfyUI v0.41.0** (its CLIPLoader type); an older core is refused before the
  10 GB download, with the update message, not after it.

## How it stays honest

- **Live node definitions.** A saved workflow's widget values mean nothing without the node's input
  order, so the graph is built against your server's `/object_info`. A pack and a workflow from
  different versions are refused with a message naming the node, never sent.
- **Never the cloud by accident.** The Comfy SDK talks to Comfy Cloud when `COMFY_BASE_URL` is unset.
  The app always sets it to your local bridge first, and never offers a cloud model in its pickers.
- **A finished job is not a finished episode.** The run ends in success only when the video file is
  on disk.
- **Local only.** The web app listens on loopback, refuses other Host headers and only accepts JSON
  posts.

## Known limits

- There is no cancel button. Cancel a run from ComfyUI's own queue.
- Restarting the app drops its memory of a run. The render keeps going on your ComfyUI and its
  episode still lands in `otr/obs/`, but the page cannot re-attach to it.
- No per-shot camera control: the pack's director picks each beat's camera move.
- The app listens only on this machine and has no login, so it refuses `--host` values beyond
  loopback.

## Tests

```
cd contest/radio-play-studio
python -m pytest tests          # no ComfyUI, GPU or pack needed
```

From the pack's repo root, `tests/test_contest_app_converter_parity.py` pins the app's
standard-library graph converter to the pack's own on every shipped workflow, and checks the form
against the pack's real node definitions.

## License

The app is MIT, see [LICENSE](LICENSE). Model weights carry their own licenses, documented in the
pack's `apple/IMAGE_MODELS.md` with their sources in `nodes/_otr_visual_assets.py`. Note that the
default still model, MOSS-Image, is Apache-2.0. FLUX 3 Klein (the other Still pictures choice) is
under the FLUX Kommunity License: free for non-commercial use, with commercial use of its outputs
limited to qualifying users.
