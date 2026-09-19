# Image engines -- what draws the picture

Every beat of the episode shows something. The **image engine** is the model that
draws that still picture; the video lane then takes the still and makes it move.

Three engines are one-click: pick them and they fetch themselves. The other two
local ones want files you download by hand. That is the only part of this page
most people need.

And before any of it: **on the shipped canonical graph the image pick does
nothing at all.** That is not a fault, and the section after next explains it.

---

## Where the control is

Three dropdowns, all on **OTR_VideoDirector**:

| Widget | The still it draws |
|---|---|
| `announcer_image_model` | The announcer beats -- the voice that opens and closes the show |
| `music_image_model` | The music beats |
| `character_image_model` | The character beats -- the shots that carry a face |

`OTR_ImageDirector` has no image-model dropdowns of its own. It reads these
three, so there is one place to set them, not two. If you go looking on the
image node for a model dropdown, that is why you cannot find one.

---

## First check whether a picture is being drawn at all

Seven of the video lanes draw their own frames and never touch the still. If the
role you care about is on one of them, its image engine sits idle, nothing
downloads, and changing the pick changes nothing:

`viz_camera`, `viz_green`, `viz_mxc_cpu`, `viz_mxc_mandala`,
`animatediff15_lightning_video`, `animatediff15_v3_haunted_video`,
`animatediff15_v3_stillin_lab_video`

The four `viz_` lanes say so in the dropdown itself -- the canonical's saved
value reads `viz_mxc_cpu (16:9) (audio-reactive, no scene image)`. The three
AnimateDiff lanes animate from their own checkpoint instead.

**The canonical graph has all three video roles on `viz_` lanes.** It also has
all three image slots set to `z_image_turbo`, and that engine is dormant there:
no weights are fetched, because nothing will ask it for a picture. Switch one
video role to a lane that does consume a still -- `still_flat`, `still_motion`,
`still_pan`, `still_word`, any of the LTX or Wan or HuMo lanes, or a hosted
video lane -- and the image engine for *that role* wakes up and its download
happens then.

This is also the answer to "why did my first run take 20 GB". A `still_` lane
paired with `z_image_turbo` fetches about 19 GiB before it can draw anything.

---

## The engines you can pick

Twelve, plus an escape hatch. The dropdown lists them by their plain id, which
is exactly what you see below.

### Local -- runs on your own machine

| Dropdown | Download | Fetches itself? | What it is |
|---|---|---|---|
| `sd15` | 2.0 GiB | **yes** | Stable Diffusion 1.5. The small one, and the only local engine proven on a 16 GB Mac. Native 512; the pack fits every request down to 768 on the long side, because past that it starts drawing two heads. |
| `z_image_turbo` | 19.3 GiB | **yes** | The canonical graph's saved pick, and the AMD graph's -- the one with a Radeon receipt. Eight steps a still, so it is quick once the weights are down -- but they are the largest download of any image engine here. |
| `lumina_image` | 9.7 GiB | **yes** | Lumina-Image 2.0. **The default in the 16 GB NVIDIA graphs** (`otr_16gb_still`, `otr_16gb_video`). Fetches the ungated Comfy-Org split set on first use, same path as `sd15` -- a 5.2 GB model plus a 5.2 GB text encoder. |
| `flux_gen1` | 13.0 GiB | no | FLUX.1-dev, the first engine this pack ever had. Its licence is non-commercial. |
| `ideogram4_local` | 17.3 GiB | no | Typography specialist -- built for the `still_word` card, where the script's own words go on screen. Measured at about 95 seconds a card against `z_image_turbo`'s 12, and its licence is non-commercial. |

### Hosted -- no weights, you supply an account

| Dropdown | Provider |
|---|---|
| `cloud_flux_pro` | Comfy partner |
| `cloud_krea_2_turbo` | Comfy partner |
| `cloud_luma_photon_flash` | Comfy partner |
| `cloud_nano_banana_2` | Comfy partner |
| `cloud_seedream_2` | Comfy partner |
| `ideo` | Comfy partner (Ideogram) |
| `google_image` | Google directly, with your own API key |

These download nothing and use no video memory. They cost money instead, per
picture. [CLOUD.md](CLOUD.md) covers the keys and how to set them.

All seven hosted rows check credentials at the same point in the pipeline: per
still, while the dispatcher renders each shot -- which is after the script is
written and the voices are recorded. Picking `google_image` over a Comfy
partner row does not buy you an earlier warning; set the key you need before
you queue, whichever hosted row you use.

### `+ Add Custom Model`

The last entry in the dropdown is not an engine. It is the door to naming one of
your own -- but declare it on **`OTR_ImageDirector`**, not `OTR_VideoDirector`.
Both nodes carry a widget called `custom_models_json`, and they are not the same
box: `OTR_VideoDirector`'s copy maps the three *video* roles, so typing your
image engine there does nothing. The one that resolves an image role's sentinel
lives on `OTR_ImageDirector` -- map the role key to your engine id there, e.g.
`{"music_image_model": "my_engine"}`.

Pick the sentinel and leave that role out of the mapping, and nothing refuses on
the spot. That role's picture is skipped while the rest of the episode keeps
rendering, and the run only fails at the end, when the completion check finds a
still missing.

---

## Which ones download themselves, and which do not

**`sd15`**, **`z_image_turbo`** and **`lumina_image`**. Pick any of the three and
a node inside the graph fetches it at queue time -- no account, no token, no
script to run. `lumina_image` joined them on 2026-09-16 and is what the 16 GB
NVIDIA graphs ship, so the default 16 GB path now downloads itself.

The other two local engines (`flux_gen1`, `ideogram4_local`) stop the render and tell you the exact filename they want and the folder it belongs in. **That refusal is the install instruction.** It never quietly substitutes another model. They ship no provisioner manifest; the refusal message at queue time is the only place the filename and folder show up.

[MACHINES.md](MACHINES.md) section 2 has the per-machine grid -- which of these
has actually been run on 8 GB, on 16 GB, on a Mac, on CPU, and which will run out
of memory.

---

## Three roles, three engines

The three slots are independent. Nothing makes you set them the same, and every
shipped graph setting them the same is a convenience, not a rule.

Where that earns its keep: put `ideogram4_local` on the one role whose lane is
`still_word` and leave the other two on something quicker, and you pay the slow
typography engine only for the cards that need lettering. Or run the character
role on a bigger local engine and the announcer and music beats on `sd15`, which
is a fraction of the download and a fraction of the time.

Each role mints on the model picked for that role. A slot that is present but
blank fails loudly -- it never borrows the engine from the slot next to it.

---

## When something goes wrong

In the order it actually happens.

**Nothing downloaded and the picture never changes when I change the engine.**
Your role's video lane draws its own frames. See the second section -- this is
correct behaviour, not a miss.

**It stops and names a file.** You picked one of the two manual local engines,
`flux_gen1` or `ideogram4_local`. The refusal names the exact filename and the
folder it belongs in, and for these two that message is the only place that
information appears -- section 3 ships no manifest for them, by design.

ComfyUI-GGUF. Install the pack, restart ComfyUI, queue again.

**A hosted pick failed partway through.** No credentials. Every hosted row --
`google_image` included -- discovers this at the same point, when the dispatcher
tries to draw that still. [CLOUD.md](CLOUD.md).

**Two heads, mirrored bodies, duplicate limbs -- on `sd15`.** That is SD 1.5
past its native resolution. The pack already holds it down to 768 on the long
side for exactly this reason, so if you are seeing it, something is asking for a
bigger canvas than the engine can hold.

**The announcer looks identical in every beat.** That is the granularity
setting, not the engine. `OTR_ImageDirector` ships all three roles on
`per_object`, which draws one picture per character and reuses it. `per_beat`
draws a fresh one every time -- more variety, and considerably more rendering.

---

Adding an image engine of your own is a supported path --
see [EXTENDING.md](EXTENDING.md).
