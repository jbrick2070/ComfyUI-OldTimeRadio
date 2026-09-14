# Video engines -- how a beat becomes moving picture

The episode is written, cast and performed as audio first. The **video engine**
is what turns each beat of that audio into something to look at.

There is not one engine. There are three dropdowns on the
**OTR_VideoDirector** node, one per kind of beat, and they are set
independently:

| Dropdown | The beats it covers |
|---|---|
| `announcer_video_model` | the announcer -- the open, the close, the links between acts |
| `music_video_model` | the music beats |
| `character_video_model` | the scenes, where the characters speak |

Every engine in the list is legal in all three, so you can give the announcer a
scope, the music a mandala and the scenes a diffusion model, or set all three
the same. Nothing here changes the script, the voices or the music. It changes
the picture -- except for two lanes, which change the sound as well, and those
are marked.

---

## What you already have

The canonical graph ships with all three set to **procedural** lanes --
`viz_mxc_cpu`, `viz_green` and `viz_camera`. They draw their own frames out of
the audio in plain Python and ffmpeg. No weights, no download, no GPU, and they
run on every machine including a CPU-only one.

That is why a first run downloads no video model at all -- and no image model
either. The canonical's image dropdowns name one, but a procedural lane consumes
no still, so those weights sit dormant and are never fetched. Switch a role to a
lane that does consume a still and the image weights come down then.

Everything below is a step up from there, and most of the steps cost a download.

---

## Reading the dropdown

An entry looks like this:

```
humo14_high_audio_in_portrait (portrait)
viz_green (16:9) (audio-reactive, no scene image)
```

The **name** is the engine. Where the name carries `low` or `high`, that is a
measured VRAM bucket, not a quality rating -- `low` lanes were measured cheaper
to run than `high` ones. Where it carries `audio_in`, the lane is conditioned on
the beat's own audio.

The **suffixes are generated from the engine itself**, so they cannot drift:

| Suffix | What it means |
|---|---|
| `(16:9)` | the lane renders landscape -- almost everything |
| `(portrait)` | the lane renders tall. The finished episode is still 16:9, so a portrait clip is pillarboxed into it |
| `(audio-reactive, no scene image)` | the lane invents its picture from the sound. It mints no still, so the image dropdown for that role does nothing |

`+ Add Custom Model` at the bottom is the escape hatch for an engine you have
written yourself -- see [EXTENDING.md](EXTENDING.md).

---

## The families

### Procedural -- nothing to download

| Dropdown | What it makes |
|---|---|
| `viz_mxc_cpu` | a radio-dial spectrum sweep in a muted rainbow, on a dark field |
| `viz_green` | green and amber CRT scopes -- frequency ring, waveform, bars |
| `viz_camera` | warm gold noir: spinning reels, projector beam, mandala rings |
| `viz_mxc_mandala` | a vector tuning-eye mandala on radio bronze. Needs `pycairo` |
| `still_flat` | the beat's still image, held dead flat. Nothing moves, and nothing is cropped |
| `still_pan` | the same still, given a slow pan and zoom across the frame |
| `still_motion` | the same slow pan. These two do the same thing today |
| `still_word` | a flat hold like `still_flat`, but the still is a word card built from the spoken line |

The four `viz_` lanes react to audio and mint no image. The four `still_` lanes
do the opposite: they draw nothing themselves and display the picture your
**image** dropdown mints, so the image engine is what decides how they look.
All eight are CPU and ffmpeg only.

`still_pan` and `still_motion` are not a real choice -- they run the identical
pan. Pick either.

### AnimateDiff -- the haunted lanes

| Dropdown | What it makes | Needs |
|---|---|---|
| `animatediff15_v3_haunted_video` | the degraded-transmission look: SD 1.5 through the official v3 motion module plus a domain adapter | manual weights + ComfyUI-AnimateDiff-Evolved |
| `animatediff15_lightning_video` | the same graph on a distilled 8-step module -- much faster, and the only AnimateDiff lane proven on Apple Silicon. Marked experimental | manual weights + ComfyUI-AnimateDiff-Evolved |
| `animatediff15_v3_stillin_lab_video` | the haunted lane started from a plate instead of empty noise. A lab variant | manual weights + ComfyUI-AnimateDiff-Evolved |

These three write their own picture from text. Like the visualizers they mint no
still, so **your image dropdown is idle when one of them is selected** -- they
just do not carry the label, which is reserved for the audio-reactive family.

### LTX

| Dropdown | What it makes | Needs |
|---|---|---|
| `ltx098_low_video` | animates the beat's still. The cheapest local video lane, and the only one that downloads itself | automatic, about 16 GB |
| `ltx23_low_audio_in` | animates the still, conditioned on the beat's audio | manual weights + ComfyUI-GGUF + ComfyUI-LTXVideo |
| `ltx23_high_video` | the LTX 2.3 22B silent lane | manual weights + ComfyUI-GGUF + ComfyUI-LTXVideo |
| `ltx25_high_video` | LTX 2.5, two-stage: rendered small, then refined to a large decode. Silent | licence + manual weights + ComfyUI-GGUF |
| `ltx25_high_foley_plus` | the same picture, and it **keeps the model's own sound** | licence + manual weights + ComfyUI-GGUF |
| `ltx25_high_mime` | the same picture, and the model's sound **replaces** the episode audio over those beats | licence + manual weights + ComfyUI-GGUF |

`ltx098_low_video` is the one lane in this whole page that needs no manual step.
Pick it and queue, and the weights arrive. It is what the 8 GB and Mac graphs
ship with.

Read the foley and mime rows twice before picking them -- see **Two lanes that
change the sound**, below.

### Wan

| Dropdown | What it makes | Needs |
|---|---|---|
| `wan22_high_video` | animates the beat's still on Wan 2.2 TI2V 5B. The standard Wan lane | manual weights + ComfyUI-GGUF |
| `wan22_high_fast` | the same motion at the same size for the same memory, roughly 2.7x sooner | manual weights + ComfyUI-GGUF |

`wan22_high_fast` is a throughput option, not a quality one. Neither fits an
8 GB card.

### HuMo -- the mouth-moving lanes

| Dropdown | What it makes | Needs |
|---|---|---|
| `humo14_high_audio_in_portrait` | a character's portrait animated in sync with their own speech, tall | manual weights |
| `humo14_high_audio_in_wide` | the same, landscape | manual weights |
| `humo17_high_audio_in_portrait` | the smaller checkpoint, tall. Handles longer beats | manual weights |
| `humo17_high_audio_in_wide` | the smaller checkpoint, landscape | manual weights |

This is the talking-head path: it takes a portrait and the beat's voice track
and moves the mouth to it. All four are large downloads and all four want a
16 GB card.

### MiniMax H3

| Dropdown | What it makes | Needs |
|---|---|---|
| `h3_low_video` | a 33B model with a still pinned as the first frame. No audio anywhere in its graph | manual weights |
| `h3_low_audio_in` | the same stack conditioned on a reference portrait plus the beat's own audio | manual weights |

The two largest downloads on this page, both 16 GB territory, and the slowest
local lanes in the pack by a wide margin. Neither emits audio.

### The odd one

`mesh_stage` turns a character portrait into an actual 3D mesh and renders a
turntable of it in a headless Blender. Manual: it needs both the mesh weights
and a portable Blender, and its model carries a community licence rather than an
open one.

### Hosted -- no weights, but you supply a key

`cloud_kling_avatar`, `cloud_seedance_2`, `cloud_wan_i2v`, `cloud_wan_i2v_audio`,
`cloud_vidu_q2_pro_fast_720p` and `word_razzle` render on Comfy's partner
services and bill your Comfy credits. `google_veo_video` and
`google_omni_video` go straight to Google and bill your Google key.

They download nothing and use no VRAM, so they run on any machine. Selecting one
in the dropdown is the whole switch -- there is no enable flag. Without a
credential the render stops and says so. Set `OTR_COMFY_API_KEY` (or be logged
into Comfy) for the first group, `OTR_GOOGLE_API_KEY` for the Google pair.
Never put a key in a workflow widget.

---

## Which one suits your machine

The shipped graphs already answer this, and opening one is easier than setting
six dropdowns by hand:

| Machine | Graph | The video lane it sets |
|---|---|---|
| Anything, first run | `otr_canonical.json` | the three procedural visualizers |
| 8 GB NVIDIA | `variants/otr_8gb_video.json` | `ltx098_low_video` |
| 16 GB+ NVIDIA | `variants/otr_16gb_video.json` | `ltx25_high_video` |
| Mac 16 GB | `variants/otr_mac16_video.json` | `ltx098_low_video` |
| CPU only | `variants/otr_cpu_low.json` | the visualizers |

Choosing by hand instead, the short version: **on 8 GB, start at
`ltx098_low_video`** -- it is what the 8 GB graph ships and the safe local
diffusion pick. The Wan pair, all four HuMo lanes, both H3 lanes and
`ltx23_low_audio_in` run out of memory on 8 GB. **On 16 GB the LTX 2.5 lanes,
both Wan lanes, all four HuMo lanes and `h3_low_video` all fit.** On a Mac,
`ltx098_low_video` and `animatediff15_lightning_video` are the two proven
diffusion lanes; the still and visualizer lanes work there too, and the rest of
the local list is CUDA only.

One lane is in the list and fits neither card: **`ltx23_high_video` runs out of
memory on 8 GB and on 16 GB both.** It is selectable because the menu shows
every registered engine rather than hiding the ones that do not fit -- being in
the list is not a recommendation.

Exact download sizes, the per-machine grid and the repository each file comes
from are in [MACHINES.md](MACHINES.md), sections 2 and 3.

---

## Two lanes that change the sound

`ltx25_high_foley_plus` and `ltx25_high_mime` are the same picture as
`ltx25_high_video`. The difference is that they keep the audio the model
generated alongside it.

- **foley_plus** mixes that audio with the episode master at an even split. It
  reaches the whole episode mix, music included.
- **mime** replaces the episode audio entirely over those beats. Picking it for
  a role makes **every beat of that role a silent performance** -- the voices
  and music are still generated for those beats and then thrown away.

These are mix decisions wearing a picture dropdown, which is why they are the
two rows worth reading twice.

**And there is a reason the prompts for those lanes look bare.** On these lanes
the picture and the sound are decoded from one latent, off one piece of text --
so the model can and does *say the prompt out loud*. It was caught doing exactly
that: a beat rendered a woman speaking a character's title, because that title
was in her description. Names, faces and descriptions are therefore kept out of
that text on purpose; the identity comes from the still instead. If you are
writing your own prompts anywhere near these lanes, do the same.

---

## When something goes wrong

**The render stops naming a node class you have never heard of.** Two different
causes look the same on screen, and the error text is what tells them apart:

- **Somebody else's node pack.** There are three: **ComfyUI-AnimateDiff-Evolved**
  for the three AnimateDiff lanes, **ComfyUI-GGUF** for the LTX 2.3, LTX 2.5 and
  Wan lanes, and **ComfyUI-LTXVideo** as well, specifically for the two LTX 2.3
  lanes (`ltx23_low_audio_in`, `ltx23_high_video`). The error names the pack and
  its URL. Install it into `custom_nodes/`, restart, queue again.
- **ComfyUI itself.** The three LTX 2.5 lanes and the two MiniMax H3 lanes use
  node classes that ship inside ComfyUI's own code rather than a separate pack.
  If your ComfyUI predates them, the error says to update ComfyUI -- and, on
  rare occasions, that an older ComfyUI-GGUF needs a small patch -- instead of
  naming anything to install. Update ComfyUI, restart, queue again.

**The render stops naming a missing file.** The lane needs weights that do not
fetch themselves. Every lane on this page except `ltx098_low_video` and the
hosted ones is in that position. [MACHINES.md](MACHINES.md) section 3 says which
repository and which folder. The error names the file and never quietly
substitutes another one.

**It says the weights are gated.** The three LTX 2.5 lanes need you to accept a
licence on the model's page while signed in to Hugging Face, then log in
locally. [INSTALL.md](INSTALL.md) section 7.

**It runs out of memory.** The lane is bigger than your card. Check your row in
[MACHINES.md](MACHINES.md) section 2 and step down -- on 8 GB that almost always
means `ltx098_low_video`, and from there the still and visualizer lanes always
work.

**Your image model is being ignored.** You picked a lane that mints no still:
any `viz_` lane, or any AnimateDiff lane. That is correct behaviour, not a
fault. Switch to a `still_` lane or a diffusion lane if you want the image
dropdown to matter.

**You set `canvas_w` and `canvas_h` and the clip came out a different size.**
Several lanes render at a size of their own and say so -- the LTX 2.5 lanes, for
instance, render small and then refine to a larger decode. The canvas widgets
are what the plan budgets with; an engine that declares its own native size
wins at render.

**The picture is pillarboxed down the sides.** You picked a `(portrait)` lane.
The episode is 16:9, so a tall clip is pillarboxed into it rather than cropped
or stretched. Pick the `_wide` twin of the same engine to fill the frame.

**An old saved graph refuses with "is retired and is no longer selectable".**
That engine was removed. The refusal names it on purpose -- it means the graph
was once valid, not that your install is broken. Pick a current lane from the
dropdown and save the graph again. Engines that were merely *renamed* do not do
this; an old name still resolves to its current engine silently.

**Two engines in the same episode look like different shows.** They are
different models. The engine is set per role, so the announcer, the music and
the scenes can legitimately look nothing alike. Set all three the same if you
want one look.

---

Neighbouring pages: [MACHINES.md](MACHINES.md) for what fits your hardware and
where every file comes from, [RUN.md](RUN.md) for the run itself, and
[EXTENDING.md](EXTENDING.md) for adding an engine of your own.
