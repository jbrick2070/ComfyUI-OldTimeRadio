# ComfyUI-OldTimeRadio

<p align="center">
  <img src="https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/assets/otr_icon.gif" alt="Old Time Radio" width="400">
</p>

> *"Good evening. This is SIGNAL LOST."*

That is how every episode opens. What follows is a radio drama nobody has heard
before: a script written on your own machine from tonight's news, a public-domain
story, a scene of Shakespeare, or nothing at all; a cast of neural voices and an
announcer; a theme in the bank's own idiom; pictures to watch while you listen;
burned captions and a credit roll. One workflow, one press of **Queue**, and a
finished `.mp4` lands in your output folder.

It runs entirely locally -- no account, no API key, no cloud service -- on NVIDIA
cards and on Apple Silicon. Paid lanes exist for people who want them, and every
one of them stays off until you turn it on.

---

## Make an episode

**1. Install the pack.** In ComfyUI Manager, search for **Old Time Radio** (registry
id `comfyui-old-time-radio`, publisher `fluxus`). Or clone it into `custom_nodes/`:

```bash
git clone -b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio
python -m pip install -r ComfyUI-OldTimeRadio/requirements.txt
```

The branch matters: `v2.0-alpha` is the default and the only current one. `main`
is a stale v1.7 merge thousands of commits behind. Use ComfyUI's own Python for
the `pip install`, not a system one.

**2. Put `ffmpeg` and `ffprobe` on your PATH.** Both. Every episode is mixed,
captioned and muxed through them, and a missing `ffprobe` fails at the very end,
after everything expensive has already run. `winget install Gyan.FFmpeg`,
`brew install ffmpeg`, or `apt install ffmpeg` all supply the pair. On Linux also
install one monospace font (`fonts-dejavu-core` is enough) for the captions.

**3. Restart ComfyUI fully** and look for `[OldTimeRadio]` in the console and an
**OldTimeRadio** category in the node menu.

**4. Load the show.** **Workflow → Browse Templates → EXTENSIONS →
comfyui-old-time-radio**. There is exactly one entry, **`otr_canonical`**. Open
it and press **Queue**. You do not have to change anything: every dropdown already
holds a working value, and the ones set to *roll* pick for themselves, so two runs
in a row give you two different shows.

**5. Wait, then look in `<your ComfyUI output folder>/otr/obs/`.** The first run
downloads about **12 GB** -- the writer, the music model and its text encoder, and
the Kokoro voices -- and then writes, casts, performs, scores and cuts an episode.
Later runs skip the download. On a 16 GB NVIDIA card a short episode is minutes;
on CPU it is a long wait, and that is the model working, not a hang.

That folder is the finish line. **If nothing is in `otr/obs/`, the run did not
finish**, however green the console looked.

The long form of all five steps, with the traps: [apple/INSTALL.md](apple/INSTALL.md)
and [apple/RUN.md](apple/RUN.md).

### The widgets worth touching

All on **OTR_LedgerScriptWriter**. Everything else has a considered default.

| Widget | What it does |
|---|---|
| `episode_title` | Blank, and the show titles itself. Anything you type becomes the title card. |
| `num_characters` | Speaking parts. Ships at 2. |
| `act_count` | `1` for a short show, `3` for a full one with act breaks. Ships at 1. |
| `custom_premise` | A sentence or two of your own. Blank means the source decides. |
| `source_bank` | Where the story comes from. Ships on *roll*, which picks any eligible bank. |
| `visual_style` | How it looks -- one of nine. Ships on *roll*. |
| `creativity` | `balanced` by default. |
| `lemmy_cameo` | Whether a character named Lemmy drops by. Ships on *roll*, about an 11% chance. |

Pin `source_bank` and `visual_style` when you want to compare two runs; a rolled
bank and a rolled style change more than anything else you could adjust.

**Your own story:** set `source_bank` to **`my_story`** (the dropdown lists bank
ids, so that is the value to pick) and put your idea in `custom_premise`. The
`story_characters`, `story_plot`, `story_setting` and `story_author` fields on the
same node take the rest. That bank exists to produce your idea rather than adapt
something, and it is the one bank the roll never lands on.

---

## The guides that ship with it

Everything under `apple/` is in the pack, whichever way you installed it.

| Everyone | |
|---|---|
| [apple/INSTALL.md](apple/INSTALL.md) | Getting the nodes loading: ffmpeg, Python versions, what downloads itself. |
| [apple/RUN.md](apple/RUN.md) | Your first episode, where it lands, and what to do when it does not. |
| [apple/MACHINES.md](apple/MACHINES.md) | Which graph to open for your card, what runs where, and where every hand-fetched weight comes from. |

| If it applies to you | |
|---|---|
| [apple/MAC.md](apple/MAC.md) | Apple Silicon, and the one warning that matters there. |
| [apple/RUNPOD.md](apple/RUNPOD.md) | Renting a GPU for the lanes your own card cannot hold. |
| [apple/ROCM.md](apple/ROCM.md) | AMD. Not in v2.0 -- cut, parked, and open to anyone who has the card. |
| [apple/CLOUD.md](apple/CLOUD.md) | The optional paid writer lanes. Off by default; you do not need them. |

| Adding to it | |
|---|---|
| [apple/EXTENDING.md](apple/EXTENDING.md) | Adding an engine, or a source bank of your own. |
| [apple/PREFLIGHT.md](apple/PREFLIGHT.md) | The checks that say whether what you built will actually work. |

The development record -- bug logs, measurements, design notes -- lives under
`docs/` in the
[GitHub tree](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/tree/v2.0-alpha/docs)
and is **not** part of a Manager install. Nothing in this file depends on it.

---

## What it makes

### The story

Five source banks roll automatically; a sixth takes your own premise. Each bank is
independent -- its own story pack, its own fetch -- and each fails closed: a bad
source, a context overflow or a broken contract stops the run rather than shipping
a degraded story. The language model writes every line of prose; Python validates,
it never rewrites.

| Bank | Where the story comes from |
|---|---|
| `scifi_news_pro` | A live science feed, turned into science-fiction radio. |
| `media_archive` | Media RSS and archive items, turned into restoration-adventure episodes. |
| `public_domain` | A faithful radio adaptation of a public-domain source text. |
| `shakespeare` | A Folger scene, adapted with the author's own language carried as written. The Folger texts are noncommercial (CC BY-NC), and an episode inherits that. |
| `original` | No source at all: original fiction seeded from an entropy draw. |
| `my_story` | Your idea, characters, plot and setting. Never rolled; you choose it. |

### The music each bank gets

Every bank has a fixed musical identity, and the composer leads every cue with it.

| Bank | Idiom |
|---|---|
| `media_archive` | small-group jazz quartet, relaxed swing -- brushed drums, walking upright bass, piano, tenor sax |
| `original` | salsa conjunto at 100 BPM, clave-driven -- congas and timbales, montuno, tumbao, brass |
| `public_domain` | Chicago house at 122 BPM, soulful and steady -- TR-707, rolling bass, warm piano chords |
| `scifi_news_pro` | Detroit techno at 128 BPM, hypnotic machine funk -- TR-909, sub bass, detuned stabs |
| `shakespeare` | Elizabethan consort music -- viols, recorders, lute |
| `my_story` | **Yours.** Type it into the `music_style` widget on **OTR_StableAudioTheme**: "gamelan orchestra", "surf rock", "solo cello". Blank means the house radio orchestra. |

`music_style` is honoured on `my_story` only. Every other bank keeps its idiom
whatever is typed, because a Shakespeare episode scored as surf rock is not a
feature.

### The look

Nine visual styles, rolled or pinned on `visual_style`: `sci_fi_radio` (the
production look), `anime`, `archival_documentary`, `cartoon`, `paper_origami`,
`recur_frac`, `shakespeare_stage_realism`, `storybook_engraving` and `video_art`.
Every style drives both the stills and the video.

### The engines

The video, image, voice and music layers are each a registry of swappable
engines, chosen per role from dropdowns. Whatever you pick is honoured exactly: a
missing or out-of-memory engine **stops the render with a named error** rather
than swapping in something you did not choose. There is no silent fallback.

What the canonical ships, and why:

- **Video:** three procedural, audio-reactive lanes -- `viz_mxc_cpu`, `viz_green`,
  `viz_camera` -- one per role. They draw their own frames, download nothing, and
  run on every machine including CPU. Real video diffusion (LTX, Wan, HuMo,
  AnimateDiff, MiniMax H3) and the `still_*` family are all one dropdown away.
- **Images:** `z_image_turbo`, sitting **dormant**. The three procedural video
  lanes consume no still, so the image weights are never fetched on a default run.
  Switch a video role to a still-consuming lane and they download then.
- **Voices:** `kokoro` on both slots. It is the only one-click voice on every
  platform, which is why it is the default. Six voice engines come with a Manager
  install; the seventh, the IndexTTS2 voice cloner, ships in the GitHub tree only.
- **Music:** `stable_audio_3`. Commercially clean and ungated. MusicGen remains
  selectable and is noncommercial.

### How it fits together

Audio is the source of truth. The script is written, cast and performed into one
frozen 48 kHz master, and that master defines the timeline; video is rendered to
fit it and muxed in last. The archival copy in `otr/episodes/` keeps that audio
byte-identical; the published copy in `otr/obs/` re-encodes it to AAC for players.

```
source bank -> LedgerScriptWriter -> LedgerFreezeCascade -> CastLock
    -> character voices + announcer + theme  -> SceneSequencer -> AudioEnhance
    -> EpisodeAssembler                                    ==> 48 kHz MASTER (frozen)
    -> VideoDirector / ShotLock -> image prompts -> VideoRenderBatch
    -> SilentComposite -> CaptionBurn -> CreditsRoll -> MasterAudioMux
                                                       ==> final .mp4 in otr/obs/
```

Renders are deterministic for a given request hash, and every episode carries a
ledger (`episode_canon.json`) recording what actually ran.

---

## Changing what renders it

The engine dropdowns live on **OTR_VideoDirector** (video and image roles),
**OTR_CastLock** (the two voice slots) and **OTR_StableAudioTheme** (music). The
writer dropdowns are on **OTR_LedgerScriptWriter**.

You never need all the weights in this workflow. One graph ships; the dropdowns
decide what it loads, and therefore what you have to download. The table below
prices each choice and says whether it runs on the three machines most people
have. It is generated from the code -- whether a machine is offered an engine
comes from that engine's own declaration, and sizes come from the real fetch
manifests -- so it cannot drift from what the pack does. The same table with AMD
and CPU-only columns, and a legend for every word in it, is
[apple/MACHINES.md](apple/MACHINES.md) section 2.

<!-- BEGIN GENERATED: dropdown-matrix -->

**Video -- procedural, nothing to download**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `still_flat` | nothing | -- | fits | **proven** | **proven** |
| `still_motion` | nothing | -- | **proven** | **proven** | **proven** |
| `still_pan` | nothing | -- | **proven** | **proven** | **proven** |
| `still_word` | nothing | -- | fits | measured | **proven** |
| `viz_camera` | nothing | -- | fits | **proven** | **proven** |
| `viz_green` | nothing | -- | fits | **proven** | **proven** |
| `viz_mxc_cpu` | nothing | -- | **proven** | **proven** | **proven** |
| `viz_mxc_mandala` | nothing | -- | fits | **proven** | fits |

**Video -- hosted, no weights but you supply the key**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `cloud_kling_avatar` | none, **but see below** | -- | key | key | key |
| `cloud_seedance_2` | none, **but see below** | -- | key | key | key |
| `cloud_vidu_q2_pro_fast_720p` | none, **but see below** | -- | key | key | key |
| `cloud_wan_i2v` | none | -- | key | key | key |
| `cloud_wan_i2v_audio` | none | -- | key | key | key |
| `google_omni_video` | none | -- | key | key | key |
| `google_veo_video` | none | -- | key | key | key |
| `word_razzle` | none | -- | key | key | key |

**Video -- local diffusion**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `animatediff15_lightning_video` | manual | 3.1 GiB | fits | fits | **proven** |
| `animatediff15_v3_haunted_video` | manual | 3.6 GiB | **proven** | **proven** | not offered |
| `animatediff15_v3_stillin_lab_video` | manual | 3.6 GiB | fits | fits | not offered |
| `mesh_stage` | manual | 4.6 GiB | fits | fits | not offered |
| `wan22_high_video` | manual | 9.4 GiB | **no** | **proven** | not offered |
| `wan22_high_fast` | manual | 10.0 GiB | **OOM** | fits | not offered |
| `humo17_high_audio_in_portrait` | manual | 12.6 GiB | **OOM** | **proven** | not offered |
| `humo17_high_audio_in_wide` | manual | 12.6 GiB | **OOM** | **proven** | not offered |
| `ltx23_high_video` | manual | 14.8 GiB | **OOM** | **OOM** | not offered |
| `ltx23_low_audio_in` | manual | 15.2 GiB | **OOM** | fits | not offered |
| `ltx098_low_video` | **auto** | 16.1 GiB | **proven** | **proven** | **proven** |
| `ltx25_high_foley_plus` | GATED + manual | 22.2 GiB | fits | **proven** | not offered |
| `ltx25_high_mime` | GATED + manual | 22.2 GiB | fits | **proven** | not offered |
| `ltx25_high_video` | GATED + manual | 22.2 GiB | **proven** | **proven** | not offered |
| `humo14_high_audio_in_portrait` | manual | 26.7 GiB | **OOM** | **proven** | not offered |
| `humo14_high_audio_in_wide` | manual | 26.7 GiB | **OOM** | **proven** | not offered |
| `h3_low_video` | manual | 41.9 GiB | **OOM** | **proven** | not offered |
| `h3_low_audio_in` | manual | 42.5 GiB | **OOM** | fits | not offered |

**Image -- local**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `sd15` | **auto** | 2.0 GiB | fits | **proven** | **proven** |
| `flux2_klein` | manual | 10.2 GiB | **proven** | **proven** | not offered |
| `lumina_image` | manual | 10.4 GiB | **OOM** | **proven** | not offered |
| `flux_gen1` | manual | 13.0 GiB | **OOM** | **proven** | not offered |
| `ideogram4_local` | manual | 17.3 GiB | **no** | **proven** | not offered |
| `z_image_turbo` | **auto** | 19.3 GiB | **proven** | **proven** | not offered |

**Image -- hosted**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `cloud_flux_pro` | none | -- | key | key | key |
| `cloud_krea_2_turbo` | none | -- | key | key | key |
| `cloud_luma_photon_flash` | none | -- | key | key | key |
| `cloud_nano_banana_2` | none | -- | key | key | key |
| `cloud_seedream_2` | none | -- | key | key | key |
| `google_image` | none | -- | key | key | key |
| `ideo` | none | -- | key | key | key |

**Voice and music -- local**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `kokoro` | **auto** | 0.3 GiB | **proven** | **proven** | **proven** |
| `musicgen` | **auto** | 2.2 GiB | **proven** | **proven** | measured |
| `chatterbox` | own installer (Windows) | 3.0 GiB | not offered | fits | not offered |
| `stable_audio_3` | **auto** | 3.5 GiB | **proven** | **proven** | **proven** |
| `bark` | **auto** | 4.2 GiB | **proven** | **proven** | **OOM** |
| `stable_audio_music` | GATED | 4.5 GiB | fits | fits | not offered |
| `dia` | own installer (Windows) | 6.0 GiB | not offered | fits | not offered |
| `indextts2` | own installer (Windows) | 11.1 GiB | not offered | **proven** | not offered |

**Voice and music -- hosted**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `elevenlabs` | none | -- | key | key | key |
| `google_lyria` | none | -- | key | key | key |
| `google_tts` | none | -- | key | key | key |
| `sonilo` | none | -- | key | key | key |

**Upscale**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `off` | nothing | -- | **proven** | **proven** | **proven** |
| `spandrel_esrgan` | manual | 0.1 GiB | fits | **proven** | measured |

**Writer (the LLM that writes the script)**

| dropdown | how you get it | size | 8 GB NVIDIA | 16 GB+ NVIDIA | Mac 16 GB |
|---|---|---|---|---|---|
| `google/gemma-2-2b-it` | GATED | 5.2 GiB | **proven** | fits | fits |
| `google/gemma-4-E2B-it` | **auto** | 6.0 GiB | **proven** | **proven** | **OOM** |
| `unsloth/Llama-3.2-3B-Instruct` | **auto** | 6.4 GiB | fits | fits | fits |
| `Qwen/Qwen3.5-4B` | **auto** | 8.7 GiB | **proven** | **proven** | **proven** |
| `google/gemma-4-E4B-it` | **auto** | 9.0 GiB | measured | **proven** | **tight** |
| `google/gemma-4-12b-it` | **auto** | 23.9 GiB | measured | **proven** | **no** |
| `mistralai/Mistral-Nemo-Instruct-2407` | **auto** | 24.0 GiB | **no** | **proven** | **no** |
**How you get the weights.** Two things do the fetching for an **auto** row, and
neither of them is a script you have to run: the engine's own library pulls it
through the Hugging Face cache, or `OTR_WorkflowValidator` -- a node inside the
graph -- downloads it at queue time. A **manual** row may still have a helper in
`scripts/`, but `scripts/` is not in the registry bundle, so from a normal
install it is a step you take by hand and it is labelled as one.

**auto** -- fetched on first use, no account and no
token; just pick it and run. **GATED** -- fetches itself, but only after you
accept a licence on the model page and set `HF_TOKEN`. **manual** -- you fetch
it yourself; `docs/MODEL_ASSET_INDEX.md` names the files and where they go.
**none** -- no weights at all. *no lane* -- the engine is registered but no
provisioning lane is declared for it, so nothing will fetch it for you.

**own installer** -- installs through its own script rather than the model
provisioner; **(Windows)** marks the three whose installer is PowerShell with no
`.sh` twin, so on Linux and macOS there is no install path today. That is
packaging, not hardware -- writing the shell installer is what clears it. **nothing** -- pure code; there is nothing to obtain.

Sizes are GiB, summed from the real artifact bytes in the fetch manifests where
a lane carries them, otherwise the figure the fetcher's own pick list states.

**What a machine cell means, and read this before you read one.** Each cell
answers TWO questions in order.

* **not offered** -- OTR will not put this engine in your dropdown on that
  machine, because its declaration does not list that backend. This is a
  statement about the code, **not about your hardware**: several of these have
  run on that hardware, and the declaration is a record of what has been
  PROVEN, not of what is possible. Making one available is a code change plus a
  receipt, not a purchase.
* **too slow** -- offered on a CPU-only box in principle, kept off it because it
  is not practical there.
* Otherwise the engine IS offered, and the word is the memory verdict:
  **proven** (a PUBLISHED EPISODE used it), measured (it ran on that hardware
  in a lab test and worked, but no episode has ever used it), fits (nothing
  blocks it and the arithmetic says it fits -- nobody has run it at all),
  **OOM** (expect to exhaust memory), **?** (offered, nobody has measured it).

**The proven/measured split IS the test plan.** "measured" is precisely the list
of engines to close next, and the distinction was earned: a first pass called
both states "proven", which put engines in the same column as ones that had
carried a whole episode. Note also that an episode's FILENAME records only its
dominant video lane, so counting receipts from filenames under-reports -- one
published episode here ran viz_camera, viz_mxc_cpu and viz_green together.

**The AMD column is the weakest one here, and it is weak by construction.** The
ROCm profiles declare `device_backend: "cuda"`, because that is how ROCm
presents itself to torch -- so every CUDA lane reads as offered there, and the
column is really answering "is this vendor-locked or sidecar-locked?" rather
than "has this been run on AMD?". Nothing in this repo has an AMD receipt. Treat
an AMD cell as the absence of a hard blocker, nothing more.

**On a Mac, OOM is a HARD MACHINE REBOOT, not a failed render** -- unified
memory has no separate pool to exhaust. That is why the Mac column is worth
reading before you pick, and why an unmeasured **?** there deserves more caution
than the same mark on a discrete card.
<!-- END GENERATED: dropdown-matrix -->

Two things to know before you change a dropdown:

- **The image trap.** With the procedural video lanes the `z_image_turbo` default
  costs nothing. Pick a `still_*` or `ltx098_low_video` lane and it wakes up: a
  19 GB download, and on a 16 GB Mac an out-of-memory. If you did not mean to
  spend that, set the three image dropdowns to `sd15` (2 GB, same job) at the
  same time. Both fetch themselves; the only thing that changes is which one
  you spend.
- **The writer is the biggest single download** and, on a Mac, the biggest single
  memory user. `Qwen/Qwen3.5-4B` is the default because it is ungated,
  Apache-2.0, and the smallest row that runs everywhere. The writer column in the
  table says which others fit your machine, and the pack refuses before
  downloading if you pick one that will not.

---

## A graph pre-set for your machine

`otr_canonical` names no vendor anywhere and resolves your device at run time,
so it is correct as shipped on NVIDIA, Apple Silicon and CPU. If you would rather
skip the dropdowns, the pack also ships **18 generated graphs in
`workflows/variants/`** -- one per machine class and episode kind, named
`otr_<machine>_<tier>.json`. Browse Templates lists only the canonical -- its
scanner looks one directory deep -- so these are files you **drag onto the
canvas** or open with Workflow → Open. Never hand-edit one; each is the canonical
with its dropdowns set, regenerated from it, and checked against it.

The tiers are named by what the episode is made of. **low** runs the procedural
visualiser lanes and needs no video or image weights at all. **still** generates
one image per beat and animates it (`still_motion`). **video** is real video
diffusion. **foley** is video that generates its own sound, mixed under the
voices; **mime** is the same render as a silent performance -- the video's
own sound carries its beats and the voices and music are muted there.
**animatediff** is SD 1.5 motion driven by the text prompt alone; it mints
no still. Kokoro voices, MusicGen music,
three acts and three characters on every graph; the upscaler is off.

<!-- BEGIN GENERATED: tier-matrix -->
### 8 GB NVIDIA

| tier | graph | writer | quant | lanes (announcer / music / character) | image | weights | also install | acts | chars | status |
|---|---|---|---|---|---|---|---|---|---|---|
| **low** | `otr_8gb_low` | Qwen3.5-4B | bnb_nf4 | viz_camera | none (dormant) | none | nothing | 3 | 3 | draft |
| **still** | `otr_8gb_still` | Qwen3.5-4B | bnb_nf4 | still_flat / viz_green / still_motion | sd15 | none | nothing | 3 | 3 | draft |
| **video** | `otr_8gb_video` | Qwen3.5-4B | bnb_nf4 | ltx_8gb | sd15 | auto | nothing | 3 | 3 | draft |
| **foley** | `otr_8gb_foley` | Qwen3.5-4B | bnb_nf4 | ltx25_foley_plus | sd15 | manual | ComfyUI-GGUF | 3 | 3 | draft |
| **mime** | `otr_8gb_mime` | Qwen3.5-4B | bnb_nf4 | ltx25_mime | sd15 | manual | ComfyUI-GGUF | 3 | 3 | draft |
| **animatediff** | `otr_8gb_animatediff` | Qwen3.5-4B | bnb_nf4 | animatediff15_v3_haunted_video | none (dormant) | manual | ComfyUI-AnimateDiff-Evolved | 3 | 3 | shipping |

### 16 GB NVIDIA

| tier | graph | writer | quant | lanes (announcer / music / character) | image | weights | also install | acts | chars | status |
|---|---|---|---|---|---|---|---|---|---|---|
| **low** | `otr_16gb_low` | gemma-4-12b-it | bnb_nf4 | viz_mxc_cpu / viz_mxc_mandala / viz_camera | none (dormant) | none | nothing | 3 | 3 | shipping |
| **still** | `otr_16gb_still` | gemma-4-12b-it | bnb_nf4 | viz_mxc_cpu / viz_mxc_cpu / still_motion | z_image_turbo | none | nothing | 3 | 3 | draft |
| **video** | `otr_16gb_video` | gemma-4-12b-it | bnb_nf4 | ltx25_high_video | z_image_turbo | manual | ComfyUI-GGUF | 3 | 3 | shipping |
| **foley** | `otr_16gb_foley` | gemma-4-12b-it | bnb_nf4 | ltx25_foley_plus | z_image_turbo | manual | ComfyUI-GGUF | 3 | 3 | shipping |
| **mime** | `otr_16gb_mime` | gemma-4-12b-it | bnb_nf4 | ltx25_mime | z_image_turbo | manual | ComfyUI-GGUF | 3 | 3 | shipping |
| **animatediff** | `otr_16gb_animatediff` | gemma-4-12b-it | bnb_nf4 | animatediff15_v3_haunted_video | none (dormant) | manual | ComfyUI-AnimateDiff-Evolved | 3 | 3 | draft |

### Apple Silicon, 16 GB

| tier | graph | writer | quant | lanes (announcer / music / character) | image | weights | also install | acts | chars | status |
|---|---|---|---|---|---|---|---|---|---|---|
| **low** | `otr_mac16_low` | Qwen3.5-4B | none | viz_mxc_cpu / viz_green / viz_camera | none (dormant) | none | nothing | 3 | 3 | shipping |
| **still** | `otr_mac16_still` | Qwen3.5-4B | none | still_motion | sd15 | none | nothing | 3 | 3 | shipping |
| **video** | `otr_mac16_video` | Qwen3.5-4B | none | ltx098_low_video | sd15 | auto | nothing | 3 | 3 | shipping |
| foley | _not built_ | | | | | | | | | |
| mime | _not built_ | | | | | | | | | |
| **animatediff** | `otr_mac16_animatediff` | Qwen3.5-4B | none | animatediff15_lightning_video | none (dormant) | manual | ComfyUI-AnimateDiff-Evolved | 3 | 3 | shipping |

### AMD ROCm (experimental -- no receipts)

| tier | graph | writer | quant | lanes (announcer / music / character) | image | weights | also install | acts | chars | status |
|---|---|---|---|---|---|---|---|---|---|---|
| low | _not built_ | | | | | | | | | |
| **still** | `otr_amd_still` | Qwen3.5-4B | none | viz_mxc_cpu / viz_mxc_cpu / still_motion | z_image_turbo | none | nothing | 3 | 3 | draft |
| video | _not built_ | | | | | | | | | |
| foley | _not built_ | | | | | | | | | |
| mime | _not built_ | | | | | | | | | |
| animatediff | _not built_ | | | | | | | | | |

### CPU only

| tier | graph | writer | quant | lanes (announcer / music / character) | image | weights | also install | acts | chars | status |
|---|---|---|---|---|---|---|---|---|---|---|
| **low** | `otr_cpu_low` | Qwen3.5-4B | none | viz_mxc_cpu / viz_green / viz_camera | none (dormant) | none | nothing | 3 | 3 | draft |
| still | _not built_ | | | | | | | | | |
| video | _not built_ | | | | | | | | | |
| foley | _not built_ | | | | | | | | | |
| mime | _not built_ | | | | | | | | | |
| animatediff | _not built_ | | | | | | | | | |
<!-- END GENERATED: tier-matrix -->

Weights marked **auto** download themselves the first time you queue; **manual**
means the launch recipe beside the graph (`<graph>.launch.md`) lists what to
fetch and where; **none** means those lanes need no video weights. A `draft`
status is a graph cut from the same canonical that has not yet been proven on
that hardware; the AMD graph has no receipts at all. What each machine class
runs, engine by engine, is in [apple/MACHINES.md](apple/MACHINES.md).

**On a Mac, read [apple/MAC.md](apple/MAC.md) before picking anything heavier
than the defaults.** An out-of-memory on unified memory can reboot the machine,
not the render, which is why the Mac column is conservative.

The rows below are generated from the machine classes the pack ships, and say
what each class runs for writer, video, voice, music and image.

<!-- BEGIN GENERATED: machine-matrix -->

## What works on what machine

**Read the `extra install` column before you pick a row.** A lane that needs a third-party ComfyUI node pack cannot say so in `requirements.txt` or `pyproject.toml`, because a node pack is not a pip distribution -- so the requirement is invisible until the render fails with a missing class. That has already cost one divergence between the PROVEN path and the DOCUMENTED path (PBUG-20260829-09), when the box that proved the lane had git-cloned the pack by hand. `gated` means the weights need a Hugging Face licence acceptance before they download.

| your machine | writer | video | voice | music | image | extra install | status |
|---|---|---|---|---|---|---|---|
| **8 GB NVIDIA (RTX 4060, 3070, 2080)** | gemma-4-E2B | animatediff15_v3_haunted_video | kokoro | musicgen | flux2_klein | ComfyUI-AnimateDiff-Evolved | **EPISODE PATH PROVEN** -- writer/video/voice/music on RTX 4060; image lane (Klein) proven 2026-09-02 on a Python 3.13 clean room |
| **16 GB or more NVIDIA (RTX 5080, 3090, 4090, A4500)** | gemma-4-12b | wan22_high_video | kokoro | musicgen | z_image_turbo | ComfyUI-GGUF | **COMPONENTS PROVEN** -- Wan on named Ampere/Blackwell hardware; exact row tuple and unlisted cards unproven |
| **10-15 GB NVIDIA (RTX 4070, 3080, 3080 Ti 12 GB)** | gemma-4-E2B | animatediff15_v3_haunted_video | kokoro | musicgen | flux2_klein | ComfyUI-AnimateDiff-Evolved | `draft`, unproven |
| **AMD / ROCm (Linux only)** | gemma-4-E2B | still_motion | kokoro | musicgen | flux2_klein | none | `draft`, unproven |

**Use the machine key, not an experimental profile name.** Run these with the exact Python executable that launches ComfyUI (shown as `<ComfyUI Python>`). Preview the install plan first, then run the same command without `--list` to install it.

* **8 GB NVIDIA (RTX 4060, 3070, 2080)** -> `<ComfyUI Python> scripts/otr_provision.py --machine 8gb --list`
* **16 GB or more NVIDIA (RTX 5080, 3090, 4090, A4500)** -> `<ComfyUI Python> scripts/otr_provision.py --machine 16gb --list`
* **10-15 GB NVIDIA (RTX 4070, 3080, 3080 Ti 12 GB)** -> `<ComfyUI Python> scripts/otr_provision.py --machine 12gb --list`
* **AMD / ROCm (Linux only)** -> `<ComfyUI Python> scripts/otr_provision.py --machine amd --list`

Provisioning installs and verifies artifacts; it does not rewrite the saved graph. To apply one row atomically to the real canonical workflow on a normal port-8188 ComfyUI server, run `<ComfyUI Python> scripts/otr_canonical_api_run.py --comfyui-url http://127.0.0.1:8188 --machine 8gb --act-count 1 --source-bank original --visual-style sci_fi_radio --timeout 0`, replacing only the exact machine key. To use an explicit profile instead, replace `--machine 8gb` with `--profile <exact-profile-id>`; the two selectors are intentionally exclusive. Every machine row selects the Kokoro voice. On the Python 3.13 that ComfyUI Desktop and the portable build ship it runs through kokoro-onnx on the CPU (the same voices, about six times faster than realtime); on Python 3.12 through the torch kokoro package. Python 3.14 has no kokoro backend packaged yet; there, run `--profile otr_4060_floor` for the bark route or switch the OTR_CastLock voice dropdowns to bark.

Apple Silicon is `otr_mac_mps`, PROVEN on a named physical system -- a Mac mini M4 / 16 GB published episodes to `otr/obs/` on 2026-09-07 and 2026-09-08, including local `sd15` stills and local `ltx_8gb` video diffusion. It is not promoted to a machine key: a machine key implies a measured VRAM tier, and one 16 GB Mac is one data point, not a tier. Read `docs/MAC_PORTABILITY_GUIDE.md` before starting. CPU-only is `cpu_floor`, still unproven -- no named system has published on it.

<!-- END GENERATED: machine-matrix -->

The `scripts/otr_provision.py` commands that table prints need the **git clone**:
`scripts/` is not in a Manager install. The saved graphs in `workflows/variants/`
need nothing but a drag, and the weights a graph selects download at queue time
whenever the pack can fetch them itself.

---

## Node packs some lanes need

A few engines build their graph out of another pack's nodes. Those are ComfyUI
node packs, not Python packages, so `pip` cannot supply them. Install them into
`custom_nodes/` and restart. Nothing the canonical selects needs any of these.

| If you select | Install |
|---|---|
| `animatediff15_v3_haunted_video`, `animatediff15_v3_stillin_lab_video`, `animatediff15_lightning_video` -- including the three `otr_*_animatediff` graphs | [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) |
| `flux2_klein`, `ltx23_*`, `ltx25_*`, `wan22_*` | [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) -- the LTX 2.5 lanes also want the one-file patch described in [patches/README.md](patches/README.md) |

If you pick one of these lanes without its pack, the render stops with an error
that names the pack and its URL. The AnimateDiff lanes are also the one place
where weights do not fetch themselves: the SD 1.5 checkpoint, the motion module
and the adapter are a hand fetch, and [apple/MACHINES.md](apple/MACHINES.md)
section 3 names each file, its repository and its folder.

---

## Weights, tokens and downloads

**Most engines fetch their own weights** the first time a dropdown selects them
-- either through the engine's own library and the Hugging Face cache, or through
`OTR_WorkflowValidator`, a node inside the graph that looks at what you actually
picked and pulls only that, before the writer runs. The cache lives inside your
ComfyUI install at `models/huggingface` unless `HF_HOME` was already set when
ComfyUI started; if that volume is short of room, set `HF_HOME` **before**
launching, because setting it later creates a second cache rather than moving the
first.

**You do not need a Hugging Face token to run OTR.** Everything the canonical
selects, and everything the 8 GB and Mac graphs select, is ungated. A token is
needed only for the handful of gated rows -- `google/gemma-2-2b-it`, the LTX 2.5
video weights, `stable_audio_music` -- and for those you accept the licence on
the model's page while signed in, then log in locally once:

```bash
hf auth login
```

**Never paste a token into a workflow widget.** No node here asks for one, and a
value saved into a graph travels with it -- into every workflow you share, every
queue entry, and the metadata of every image you generate. A token in the login
file, or in `HF_TOKEN` in the environment that launches ComfyUI, is the whole
setup. ComfyUI Desktop on Windows does not inherit user-scope environment
variables, which is one more reason to prefer the login file.

The weights that do **not** fetch themselves -- every hand-fetched file, its
repository, its size, and the folder under `models/` it goes in -- are listed in
[apple/MACHINES.md](apple/MACHINES.md) section 3. Where no manifest exists, the
engine refuses by name before anything else runs, and that refusal is the install
instruction.

---

## Two optional lanes

**A cloud writer.** If your card cannot hold a local writer, or you want to write
scripts on a machine with no usable GPU, the script step alone can run on
OpenRouter, Google, or Comfy Credits. Voices, music, images and video stay local.
It costs money, it is off by default, and it exists for hardware reasons rather
than prose quality. [apple/CLOUD.md](apple/CLOUD.md) has the three switches.

**The GGUF writer lane.** Nothing selects it today -- no `*-GGUF` writer appears
in the dropdown -- but the lane is wired for people who want to run a quantized
writer on a small card off NVIDIA. If you wire it up, pin the library:

```bash
pip install llama-cpp-python==0.3.33 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Do not take the latest. `0.3.35` dies with an illegal-instruction fault inside
`llama_init_from_model` before a single token, on the CPU backend, so no GPU
avoids it; an unpinned install resolves to it. That line is the Windows CUDA
recipe, the only one measured here; on other platforms use upstream's own build
flags with the same pin. And test it the way OTR does -- a bare `import llama_cpp`
fails even on a working install, because the pack preloads the CUDA DLLs first:

```python
from nodes._otr_gguf_backend import _import_llama_cpp; _import_llama_cpp()
```

---

## Where things land

Everything for an episode goes under your ComfyUI output folder:

- `otr/episodes/<episode>/` -- the working files: stems, frames, intermediate
  clips, and the ledger.
- `otr/obs/` -- the **finished, playable episodes**.

The published file is named after what produced it, so a folder of episodes reads
at a glance without opening any of them:

```
<title>_<timestamp>__<style>__<video>__<image>__<tts>__<bank>_final.mp4
```

A lane that renders no stills reports `none` in the image field. Point OBS, or any
player with a watch folder, at `otr/obs/` for a continuous broadcast -- new
episodes appear there as they render.

---

## When something goes wrong

**The whole OldTimeRadio category is missing.** That is never a missing library
-- each node loads in its own try/except, so a missing library costs one node and
prints `[OldTimeRadio] Skipped '<name>': <reason>`. A total absence means the pack
is in the wrong folder or ComfyUI crashed during startup before any node loaded.

**One node is missing.** Find that `Skipped` line in the console; it names the
library. Install it into ComfyUI's own interpreter.

**The render stops naming a missing class** (`ADE_*`, `UnetLoaderGGUF`, `LTXV*`).
The lane you picked needs a node pack from the table above; the message names it.

**The render stops naming a missing file.** The lane needs weights that do not
fetch themselves. The message names the file; it never quietly substitutes
another. [apple/MACHINES.md](apple/MACHINES.md) section 3 says where it comes from.

**It refuses before downloading, saying the writer will not fit.** You picked a
language model bigger than your card's ceiling. Pick a smaller one.

**No mp4, and the log mentions ffmpeg or ffprobe.** You need both binaries. The
`imageio-ffmpeg` wheel that comes down with the requirements ships only ffmpeg.
If yours lives somewhere unusual, set `OTR_FFMPEG` to the binary's full path.

**`neither kokoro backend is installed` at the first voice line.** Python 3.10 to
3.12 run Kokoro on torch; 3.13 runs the same voices through `kokoro-onnx` on the
CPU; 3.14 has no Kokoro build yet and is refused. The message names the exact pip
line. Or open **OTR_CastLock** and switch both voice engines to `bark`, which
installs everywhere -- except on a 16 GB Mac, where bark is a memory hazard;
stay on Python 3.12 or 3.13 there and keep Kokoro.

**A gated model returns HTTP 401.** The LTX 2.5 weights and `gemma-2-2b-it` need a
licence click on Hugging Face plus a login; every default weight is ungated.

**On a Mac, a 20 GB download starts the moment you queue.** The image dropdowns
still say `z_image_turbo` while the video lane you picked consumes a still. Set
all three to `sd15` first.

**It finished but nothing is in `otr/obs/`.** Read the console from the end
backwards for the first error. A run that ends without publishing did not
succeed, and if it has been quiet for more than five minutes after the downloads
finished, it is not going to.

---

## What it does not do well

**Some episodes hand a line to the wrong character.** You may hear a character
claim a job that belongs to someone else, or -- rarest and most obvious -- address
themselves by name. It is uncommon, it does not break a render, and the episode
still plays, but it is real and you should know before you run this.

It is not fixed because the fix did not pass. A post-story clean stage already
reads every spoken row with a model and rewrites anything that is not speech
(stage directions, sound cues) -- that works and is on by default. The stricter
pass that judges *who* should be speaking was built, measured, and found unstable
on a 12B-class model, which is the largest thing a 16 GB card holds. It ships
disabled rather than quietly making episodes worse. If you have a much larger
model and want to try it, the switch is `JUDGE_ATTRIBUTION` in
`nodes/_otr_ledger_clean.py`.

**AMD did not make v2.0, and that is a scope cut rather than a bug.** Nobody on
the project owns an AMD card, and we were not going to claim a platform we could
not put an episode through. The ROCm graphs are built from the same source as
every working profile and every engine they select is plain PyTorch, so on paper
they should work -- and on paper is exactly the problem.

It is parked, realistically for 2.5. But it is open source and it is fair game:
[apple/ROCM.md](apple/ROCM.md) has the two built profiles, a five-minute probe
that needs no model download, and the open questions written down. If you have
the card, none of it is waiting on us.

---

## Adding to it

Two things you can add: an **engine** -- a way of rendering video, images,
speech, music or an upscale -- and a **source bank**, a place stories come from.
An engine is a Python adapter in this repo; a bank can be a folder of your own
that this repo never sees. [apple/EXTENDING.md](apple/EXTENDING.md) is the recipe
and [apple/PREFLIGHT.md](apple/PREFLIGHT.md) is the checklist, and the rule
underneath both is the same one the rest of the pack lives by: **green tests are
not a lane.** The proof is one real render through `otr_canonical` that lands a
file in `otr/obs/`.

Adding a bank uses `scripts/otr_check.py`, so it wants the git clone.

Development runs under a sibling QA harness, the
[ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide):
a machine-readable bible of bugs distilled from this project's live incidents,
with a regression suite that runs against the pack after every change. Only bugs
that actually failed in a real run get in.

---

## Licence and credits

The pack is [MIT](LICENSE). It is built on ComfyUI and the open-weight LTX, Wan,
HuMo, AnimateDiff, Z-Image-Turbo, Lumina, FLUX.2 Klein, Kokoro, Stable Audio,
Qwen and Gemma ecosystems, plus the optional Chatterbox, Dia, Bark, MusicGen and
IndexTTS2 engines -- thanks to all of their authors.

**The shipped defaults are not a blanket commercial clearance.** A few optional,
off-by-default pieces carry restricted terms:

- `flux_gen1` (Flux.1-dev) -- BFL non-commercial licence.
- `ideogram4_local` -- non-commercial model agreement; the code ships, the weights do not.
- `h3_low_video` / `h3_low_audio_in` (MiniMax H3) -- a personal, non-transferable
  authorization the maintainer obtained directly from MiniMax. It does not carry to
  your install.
- The three AnimateDiff lanes are declared not commercially clean. The haunted
  lane's motion module publishes no licence grant at all; fine for personal use,
  not cleared for commercial redistribution.
- `musicgen` and `indextts2` carry non-commercial terms; `bark`'s are unconfirmed.
- Several heavier video lanes -- `ltx25_*`, the `humo*` family, `mesh_stage`,
  `wan22_high_fast` -- also declare themselves not commercially clean in their
  adapters. Read each model's own licence.
- The `shakespeare` bank adapts Folger Digital Texts, which are CC BY-NC.

Review the exact source, engine and weight licences before commercial use. A
successful render is not a licence receipt.

---

## The Lemmy easter egg

Every so often a character named **Lemmy** makes a cameo -- a small tribute
carried across the project's generations. Born of the machine, still loud on the
airwaves.
