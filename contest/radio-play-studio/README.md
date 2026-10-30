# Radio Play Studio

**Press one button. Your own GPU writes, casts, voices, scores and films a whole radio play.**

Built for the Comfy [Open Source Higgsfield](https://comfy.org/open-source-higgsfield) challenge.
An open-source take on Higgsfield's Cinema Studio: where Cinema Studio directs a concept into
shots, Radio Play Studio writes the whole story, casts it, and directs every beat. Every character
keeps the same face and voice from shot to shot, and by default every shot is filmed with LTX
on your own card, with voices, music and foley mixed under the picture. By default it is open weights only,
with nothing to pay per episode; cloud models are there if you pick them. The app is MIT licensed,
and each model keeps its own license (see License).

> **Demo video (1:53):** https://youtu.be/UeQQeVMaMVs

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
| A still for every beat | FLUX 3 Klein; Tell my own story paints each character from their portrait or photos (FLUX 3 Klein with references) |
| Voices, one per character | Kokoro |
| Theme music | Stable Audio 3 |
| Video, with its own sound | LTX 2.5 (16 GB foley lane: each beat filmed and given foley, mixed under the voices) |
| 1080p | ffmpeg (Lanczos + sharpen; no model) |

The app reads the pack's shipped workflow, turns it into an API graph against your server's live
node definitions, sets what a person chose, submits it through the Comfy SDK, follows it with a
progress bar that knows how long each stage takes on *your* machine, and plays the finished
episode. Left on its defaults it makes no cloud calls and needs no API key; a cloud machine, a
cloud still model or a cloud writer is always something a person picked (see "Keys").

## What you need

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) v0.42.0 or newer running on your machine
  (FLUX 3 Klein's text encoder needs a recent core), with the **Old-Time Radio** pack installed
  (`git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio` into `custom_nodes/`; the app is
  built and tested against `main`, and it ships with the clone, not in the registry's zip). One node pack;
  every model downloads itself from public, ungated repos with no Hugging Face token -- the
  writer and still models at pinned revisions, the rest checked against their published SHA-256.
- By default, an NVIDIA card with 16 GB. Other cards run one of the pack's other workflows (see
  "Your machine, and how it's filmed"). The first run downloads tens of GB of open weights.
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

(Run from inside the pack, a clone or a Manager install, and it finds
`workflows/otr_16gb_foley_open.json` itself. `COMFYUI_DIR` looks for the pack under either folder
name: `custom_nodes/ComfyUI-OldTimeRadio` for a clone, `custom_nodes/comfyui-old-time-radio` for a
Manager install.)

## Run

```
python -m radio_play_studio doctor     # checks ComfyUI, the pack, the SDK and the bridge; says what is missing
python -m radio_play_studio serve      # open http://127.0.0.1:8765
```

Pick your machine and how it's filmed (it starts on the card your ComfyUI reports), choose
**A classic, or surprise me** or **Tell my own story**, press **Make the episode**, and watch the
bar. The form starts on the chosen workflow's own settings (three acts on the 16 GB foley workflow).
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

### Your machine, and how it's filmed

The page starts by asking two things, and fills in the rest of the form from the answer:

- **Your machine:** NVIDIA 8 GB, NVIDIA 16 GB or more, a 16 GB Mac, AMD (ROCm), or no GPU at all
  through Comfy Cloud or Google Cloud. It starts on the card your ComfyUI reports.
- **How it's filmed:** every video engine that machine ships: **TTS + Foley video** (voices, moving
  pictures and their own sound effects), **TTS w/ video** (voices and moving pictures), **TTS w/
  stills** (voices and still pictures with slow motion), **TTS w/ AnimateDiff**, **Mime video**
  (the video's own sound carries the scenes), **Audio-in video** (pictures made from the voices and
  music) and **TTS w/ visualiser** (no video model at all). No foley or mime on a Mac, stills only on
  AMD; a way with no published episode yet is marked *not yet proven*.
  Each card names what that workflow runs (video, stills, voices, music, writer), read from the
  workflow itself, and how long the acts you set took on our test machine. The cloud ways are
  offered but marked "not yet proven" until an episode from them is published (`UNPROVEN` in
  `radio_play_studio/config.py`).

Every choice is a workflow the pack ships (the table is `MACHINES` in `radio_play_studio/config.py`),
so the page never builds a graph of its own. Switching keeps what you typed and picked; anything you
left alone takes the new workflow's own value. **Tell my own story** still uses a workflow's
reference twin when it has one.

### Pick the workflow by hand

`--workflow` takes any of the pack's shipped graphs, so the same app runs on the machine each
one is tuned for: for example `workflows/otr_8gb_ltx25_foley.json` on an 8 GB card,
`workflows/otr_16gb_video.json` for silent LTX video on 16 GB, `workflows/otr_16gb_still.json`
for a still with motion (minutes, not an hour), or `workflows/otr_app.json` for the pack's
any-machine graph. A graph on the menu above keeps the menu; any other runs by itself with no
menu. The default is `otr_16gb_foley_open.json`: the 16 GB foley lane with every
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
| Photos (optional) | Up to three pictures of them (one face, blended). Their portrait is painted **from** it, in the episode's own style, and every still of them follows that portrait. The photo itself never appears in the episode. If the **Still pictures** choice cannot paint from a picture, a line under it says the photos will not be used. |

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

With its default models everything is open weights. **Tell my own story** runs the chosen
workflow's reference twin when one ships beside it -- for the default, `workflows/otr_16gb_foley_open_flux3_ref.json` -- by itself: every character gets a portrait (or
the photos on their card) and each still is painted from the portraits of everyone in the frame,
so a face does not change from beat to beat. Only the 16 GB foley ways ship a twin today; on any
other way, and for the other stories, the chosen workflow runs as named and paints each character
from the written description on their cast card (the page says when photos would go unused), so
describe looks in fixed terms. The still model, FLUX 3 Klein, is open weights under the FLUX Kommunity License.

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

## Direct the shots

On the **Tell my own story** path a second, quieter button sits beside **Make the episode**:
**Direct the shots first**. The play is written, cast and voiced as usual, then it stops after
the stills. The page becomes a board at once and fills in when the stills are done, so you
can change any picture before a frame is filmed.

Three stages, each a lamp with its state in plain words (down the left on a wide screen, across the top on a phone): **Stills** (how many
painted, how many to paint), **Video** (how many of the shots are filmed), **Publish** (after
every shot is filmed). Under them:

- **Cast:** one picture per character, with **Paint again**. A new portrait repaints every shot
  that character is in, except the kept ones, so try the shots first.
- **Scenes**, each holding its shot cards. A card shows the still, who speaks and the line, a
  **Picture** box (the words the still is painted from) and a **Motion** box (the words the
  clip is filmed from -- shown only where the shot's lane lets you set it; elsewhere the card
  says the motion is set by its lane), and three buttons: **Paint again** (another draw of the
  same picture), **Film again** (keep the still, film a new take) and **Keep** (freeze the card
  as it is). A badge names the card's state: changed and not painted yet, the action changed and
  not filmed yet, painted and not filmed, to film again, filmed, kept, failed.
- One bar at the foot: **Paint the stills (N changed)**, **Film the shots (N to film)**,
  **Publish the episode**, with one line beside them saying why a button is off.

Only what changed is made again. A Stills run paints the changed stills and keeps the rest; a
Video run films the shots that are not filmed yet, or that you changed, and takes every other
clip from what the project already filmed; Publish mixes the voices and music under the filmed clips
and loads no generation weights. Measured on 2026-10-23 on the author's 16 GB laptop, on a
22-shot play: one picture edit repainted one still, and the Video run after it re-filmed one
clip and reused 21; one motion edit re-filmed one clip and reused 21.

A directed project survives a restart. Its draft and its one pending run live in a project
file; start the app again and it looks at each project it left pending, re-attaches to a run
that is still going, finds a submit whose answer was lost, saves a result that finished while
it was away, and -- when ComfyUI no longer has the run -- gives it up so you can press again.
A Video run after a crash reuses every shot that finished before it; only the rest are filmed.

Where things land: each Stills or Video run's bundle under ComfyUI's output folder at
`otr/episodes/_replay/p/<project>/NN-stage/`, the clip store beside them, the page's own copies
under `./episodes/projects/<project>/`, and the published episode in `otr/obs/` like any other.

## What a person can choose

Your machine and how it's filmed (above), the language of the play, the story bank, acts, cast
size, visual style, the still model, the writer, the cast cards and the My Story boxes. Every
dropdown is read live from your ComfyUI, so it offers exactly what your installed pack has.
Everything else (voices, music) keeps the workflow's own setting.

- **Still pictures** and **Writer** offer what suits the machine you picked: a model the machines
  table (`apple/MACHINES.md`) marks "not offered", OOM, "no" or too slow there is left off, and a
  machine with no GPU writes with a cloud writer. Mix and match within that: the open models run on
  your card, the cloud ones are marked paid. The list comes from `radio_play_studio/machine_choices.json`,
  which `scripts/otr_dropdown_matrix.py` writes from the same cells as that table.
- **Writer** is a closed panel; its title always says which writer will run. Untouched, the run
  keeps the workflow's own. Inside: the local models, and OpenRouter, Google and Comfy as paid cloud
  writers, each with its model picker. The writer is the creative passes' model; the technical
  passes that check the script's structure stay on the workflow's own model. A cloud writer with
  no model picked is refused before anything is sent.
- **Source to adapt** is not on the page (your own story is My Story); the command line still has
  `--source-ref`.

## Keys

A cloud pick needs its key, and the page lists only the keys the current picks need:

- **Comfy** (Comfy Cloud, the `cloud_` and `ideo` still models, the Comfy writer): type it on the
  page, or start the Studio with `OTR_COMFY_API_KEY` set. The server holds it in memory and sends it
  with each run the way ComfyUI expects (`extra_data.api_key_comfy_org`); the app never writes it
  to disk, logs it, or sends it back to a page. ComfyUI itself keeps it with each job in its own
  queue and history, exactly as it does for its own sign-in, so treat your ComfyUI as you would
  any program that holds the key.
- **The press bar names a missing Comfy key** and jumps to the box, which sits just above it. It
  stays quiet about Google and OpenRouter: their keys may sit in ComfyUI's own environment, which
  the page cannot see, so their rows say where a key goes and whether a key file is there.
- **Google** (Google Cloud, `google_image`, the Google writer) and **OpenRouter** (the OpenRouter
  writer): ComfyUI reads these itself, so the page says where to put them -- the first line of
  `google.secret` or `openrouter.secret` in the pack folder, or `GEMINI_API_KEY` /
  `OPENROUTER_API_KEY` where ComfyUI runs -- and whether a key file is there.

## From a Stream Deck +

[`streamdeck/`](streamdeck/README.md) is a Stream Deck + plugin for this app: dials turn
through the language, story bank, visual style, acts and machine; one key makes the episode
and fills in as it goes, then plays it. It talks to the same local server the page does,
over the same routes, and the server did not change for it (`streamdeck/API-CONTRACT.md`). It is
packed ready to install: double-click `streamdeck/dist/com.jbrick2070.otr.streamDeckPlugin`, and
keys work too, for a deck without dials ([install steps](streamdeck/README.md#install)).

## The page speaks nine languages

English, Español, Português, Italiano, Français, हिन्दी, 日本語, 简体中文 and 繁體中文: the eight languages the
pack writes episodes in, with Mandarin given both scripts. The page opens in your browser's language (English
if it is none of these; Taiwan, Hong Kong and Macau browsers get 繁體中文, other Chinese browsers 简体中文)
and the picker at the top right switches it at once, without losing anything you have typed; the choice is
remembered. The story bank and episode language dropdowns show translated names, while the values sent
to the pack stay its own. Model and style names, the pack's own examples, and error text from ComfyUI stay
as they are.

Each language is one file, `radio_play_studio/static/i18n/<code>.json` (`es`; Chinese also carries its script,
`zh-hant`), and adding a file adds a language. A browser picks a file by its primary language subtag, and by
script or region only for Chinese.
English (`en.json`) is the fallback and carries the form's own words from `form.py`. The other eight are
machine translations that no native speaker has reviewed yet, so corrections are welcome.
`tests/test_rps_i18n.py` checks that every file has every key with the same `{placeholders}`.

## Measured timings (RTX 5080 Laptop, 16 GB, ComfyUI 0.42.1, through the SDK and proxy)

Two sets, measured on different days and different graphs, and both are real. The first was
measured through the app on 2026-10-19, the day it was added, on the episodes those runs happened
to write. The second is the pack's regression round of 2026-10-23 (`planning/regression/ROUNDS.md`),
run by a harness with whatever else was open; by then the 16 GB video and stills workflows paint
with MOSS-Image, which takes longer per still, and those episodes ran about three minutes. The
time on each filming card is the second set, scaled by the acts you set.

| Job | 2026-10-19, through the app | 2026-10-23 regression round |
|---|---|---|
| One-act episode, 16 GB foley | 57, 64, 64, 68 and 76 min (five runs, `otr_16gb_foley_open`) | 202 min, a 186 s episode (`otr_16gb_foley`, the closed-SDK sibling) |
| One-act episode, 16 GB LTX video (silent picture) | 65 min | 147 min, a 173 s episode |
| One-act episode, 16 GB stills with motion | 5 min | 26 min, a 182 s episode |
| Where a foley hour goes (averaged over the 10-19 runs) | video about 95% (60 of 64 min); the writer about 1 min; all stills about 2 min | |
| Progress bar against the wall clock, a 68-minute episode | within 4 points throughout | |

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
- **A lost progress stream used to freeze the bar.** On 2026-10-23 a foley episode's bar sat at 78%
  for three hours while ComfyUI filmed shot after shot: the SDK reconnects a dropped connection,
  but its own status check timed out against the busy server. The app now opens the stream again
  while ComfyUI's queue still holds the job, and says so on the page.

## How it stays honest

- **Live node definitions.** A saved workflow's widget values mean nothing without the node's input
  order, so the graph is built against your server's `/object_info`. A pack and a workflow from
  different versions are refused with a message naming the node, never sent.
- **Never the cloud by accident.** The Comfy SDK talks to Comfy Cloud when `COMFY_BASE_URL` is unset.
  The app always sets it to your local bridge first, starts on local models, and marks every cloud
  choice as paid.
- **Only proven ways, said plainly.** The filming cards name what each workflow runs, read from the
  workflow itself; a way with no published episode is marked "not yet proven"; and each machine's
  dropdowns come from the same cells as the pack's machines table.
- **A finished job is not a finished episode.** The run ends in success only when the video file is
  on disk.
- **Local only.** The web app listens on loopback, refuses other Host headers and only accepts JSON
  posts.

## Known limits

- There is no cancel button. Cancel a run from ComfyUI's own queue.
- Restarting the app drops its memory of an ordinary **Make the episode** run. The render keeps
  going on your ComfyUI and its episode still lands in `otr/obs/`, but the page cannot re-attach
  to it. A directed project (**Direct the shots first**) is the exception: it is re-attached.
- Per-shot control is the Picture and Motion text on a card. Where a card says the motion is
  set by its lane, that beat's motion is set by the pack.
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
default still model, FLUX 3 Klein, is under the FLUX Kommunity License: free for non-commercial
use, with commercial use of its outputs limited to qualifying users. The other open still models
keep their own licenses (MOSS-Image is Apache-2.0), and the cloud ones are their providers' paid
services.
