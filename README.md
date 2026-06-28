# ComfyUI-OldTimeRadio (SIGNAL LOST)

Turn **real science news** into a finished **sci-fi radio-drama video** — script, voices,
music, and CRT-style video — fully automated inside ComfyUI. Drop it in, queue one workflow,
walk away, and a complete episode lands in your output folder.

**Pipeline:** real news → LLM script → character voices (IndexTTS2) + announcer (Kokoro) +
themes (Stable Audio) → 48 kHz master mix → model-agnostic video (HuMo / LTX / Wan / CRT
visualizer) → final MP4.

100% local by default. No API keys required. (An optional hosted-LLM path exists; it stays
off unless you turn it on.)

> **Branch note:** active development lives on the **`v2.0-alpha`** branch (the Open Video
> Model Platform below). Check out `v2.0-alpha` to get the current pipeline.

---

## New to ComfyUI? Start here

You need four things: ComfyUI, a GPU, the models, and this node pack. ~20 minutes start to finish.

### 1. Install ComfyUI

Use the official [ComfyUI Desktop installer](https://www.comfy.org/download) (easiest), or a
manual/portable ComfyUI install. Launch it once to confirm it opens in your browser.

### 2. Install this node pack

In **ComfyUI Manager → Install via Git URL**, paste:

```
https://github.com/jbrick2070/ComfyUI-OldTimeRadio
```

Then **check out the `v2.0-alpha` branch** (the active line). Restart ComfyUI so it loads the nodes.

*(Manual alternative: `git clone` into `ComfyUI/custom_nodes/`, `git checkout v2.0-alpha`, restart.)*

### 3. Install the models

OTR downloads most assets on first run, but the heavy diffusion checkpoints (HuMo, LTX, Wan,
Flux) and the TTS/music weights must be present in your ComfyUI models tree (`diffusion_models/`,
`vae/`, etc.). If a model is missing, the engine fails **loudly** and the pipeline falls back to
a guaranteed CRT floor — it never silently produces garbage. Watch the console on the first run;
it names any missing weight and where it expects it.

### 4. Run it

1. Drag **`workflows/otr_scifi_16gb_full.json`** into the ComfyUI canvas.
2. Hit **Queue Prompt**.
3. Walk away. Script, voices, music, mastering, and video all run automatically.
4. Find the finished episode in **`output/otr/obs/`**.

---

## Requirements

- **GPU:** an NVIDIA card with ~16 GB VRAM for the full 14B video tier. Lower-VRAM and
  CPU-only tiers exist (see Profiles) and degrade gracefully.
- **OS:** Windows or Linux. Tested heavily on Windows + RTX (Blackwell/sm_120).
- **Disk:** the model set is large (tens of GB). Episodes are a few dozen MB each.

---

## How it works

**Audio is the source of truth.** The writer produces a script, the voice/music engines render
it, and everything is assembled into a single **frozen 48 kHz master mix**. That master defines
the episode timeline and the per-beat clip budget. Video is rendered to fit the audio and is
**muxed in last, byte-identical** — the audio is never re-encoded or altered by the video stage.

```
News → LedgerScriptWriter (LLM) → FreezeCascade → CastLock
     → IndexTTS2 (characters) + Kokoro (announcer) + Stable Audio (themes)
     → SceneSequencer → AudioEnhance → EpisodeAssembler  ==> 48 kHz MASTER (frozen)
     → VideoDirector / ShotLock (per-role engine + per-beat prompts)
     → VideoRenderBatch (render each beat through its engine)
     → SilentComposite → CaptionBurn → MasterAudioMux  ==> final MP4 in otr/obs
```

---

## v2.0-alpha — the Open Video Model Platform

The video layer is **model-agnostic**: a registry of pluggable engine adapters, chosen
**per role**, with no single model treated as "primary." You pick the engine for each kind of
beat; every chain ends at a guaranteed CRT "radio-floor" clip, so a missing or OOMing engine
**degrades loudly** and never aborts the episode or touches the frozen audio.

**Roles** (each selectable in `OTR_VideoDirector`):

| Role | What it is | Default engine (16 GB profile) |
|------|------------|-------------------------------|
| `announcer_visual` | the announcer bookends | `humo_14B_169` (16:9 talking face) |
| `music_visual` | opening/closing theme bookends | `humo_14B_169` (radio-face still) |
| `character_video` | character dialogue beats | `humo_14B_169` (audio-driven face) |
| `scene_broll` | scene b-roll | `wan_ti2v` |
| `background_abstract` | text-only background | `ltx_video` |

**Engines available:** HuMo (audio-driven face, 14B + 1.7B tiers), LTX (text/image→video and
audio-in), Wan (TI2V / I2V), and the cheap CPU floors (CRT **visualizer**, Ken-Burns, flat
still, station card). Audio-driven engines are offered only where audio exists; everything is
single-resident under a 14.5 GB ceiling and request-hash deterministic.

### Profiles (per-tier presets)

Apply a capability profile to retarget every engine for your hardware in one step:

- **`16gb_full`** — the full 14B video tier (default above).
- **`8gb_lite`** — lighter engines for ~8 GB cards.
- **`cpu_floor`** — CPU/procedural only (the CRT visualizer + still floors).

Headless: `python scripts/queue_smoke.py --profile 16gb_full` queues one episode.

### Optional: hosted LLM via OpenRouter (off by default)

The writer runs locally (Mistral-Nemo) out of the box. You can optionally route the creative
and/or technical slot to a hosted frontier model via OpenRouter — it only activates when you
set both `OPENROUTER_API_KEY` and `OTR_ENABLE_OPENROUTER=1`, is cost-guarded, and fails closed.
Full walkthrough: [`docs/openrouter-setup.md`](docs/openrouter-setup.md).

---

## Output layout

Everything for an episode lands under your ComfyUI `output/otr/` tree:

- `output/otr/episodes/<episode>/` — working assets (audio, frames, intermediate clips).
- `output/otr/obs/` — the **finished, playable episodes** (what you watch / publish).

Point OBS (or any player) at `otr/obs/` for a continuous broadcast — new finished episodes
appear there as they render.

---

## Troubleshooting

- **"SERVER DID NOT COME UP" on headless boot** — set `PYTHONUTF8=1` and
  `PYTHONIOENCODING=utf-8`; a non-UTF-8 console crashes on the first emoji log line.
- **An engine "fails loudly" mid-render** — that's by design; check the log for the missing
  model/dependency. The beat falls back to a CRT floor so the episode still completes.
- **Out of VRAM on the 14B tier** — apply the `8gb_lite` or `cpu_floor` profile.
- **No audio under the end credits** — known limitation: the credits scroll can outlast the
  master mix's closing theme. Tracked for a fix.
- **Nodes don't appear after install** — restart ComfyUI; confirm you're on the `v2.0-alpha`
  branch.

---

## The LEMMY easter egg

Every so often a character named **Lemmy** makes a cameo — a small tribute carried across the
project's generations. Born of the machine, still raising hell on the airwaves. 🤘

---

## Changelog

The current line is **v2.0-alpha** (Open Video Model Platform; per-role engines; HuMo-14B
character/announcer/music promotion; frozen byte-identical audio master). Full per-version
history is in the git log and the GitHub Releases page.

## License & Credits

See [`LICENSE`](LICENSE). Built on ComfyUI and the open-source HuMo / LTX / Wan / Flux /
IndexTTS2 / Kokoro / Stable Audio ecosystems — thanks to all of their authors.
