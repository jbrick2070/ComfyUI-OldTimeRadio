# ComfyUI-OldTimeRadio v1.0

**SIGNAL LOST** — a fully automated AI radio drama factory inside ComfyUI.

Generates complete sci-fi anthology episodes from scratch: real science headlines become full scripts via a local Gemma 4 LLM, voiced by Bark TTS with emotional expression tags, spatially mastered in 48 kHz stereo, and rendered into CRT-aesthetic video — all without leaving ComfyUI.

Think *X Minus One* meets procedural generation. The show is called **"Transmission From Tomorrow."**

> **New to ComfyUI?** Jump to [Step 1: Install ComfyUI](#step-1-install-comfyui). Already running ComfyUI? Skip to [Step 2: Install This Node Pack](#step-2-install-this-node-pack).

---

## What It Does

Every time you queue the workflow, the pipeline:

1. Fetches today's real science headlines via RSS (Nature, Ars Technica, Phys.org, etc.)
2. Feeds them to **Gemma 4** running locally — it writes a multi-act radio drama script with dialogue, SFX cues, music cues, and act breaks
3. A second Gemma 4 pass acts as **Director** — assigning voices per character, planning SFX, setting pacing, choosing vintage filter intensity
4. **Bark TTS** voices every line using explicitly tracked character identities and expressive bracket tags: `[sighs]`, `[whispers]`, `[laughs nervously]`
5. Procedural **SFX** (theremin, radio tuning, room tone, explosions) are generated and placed
6. **Spatial Audio Enhance** masters everything to 48 kHz stereo with Haas effect and mid-side widening
7. **Episode Assembler** wraps scenes with opening/closing themes and crossfades
8. **Signal Lost Video** renders a CRT-aesthetic procedural MP4 — audio-reactive visuals, no VRAM required

```
Real Science News (RSS)
        │
        ▼
┌─────────────────────┐
│  Gemma 4 ScriptWriter│──→ Full radio drama script (JSON + text)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Gemma 4 Director    │──→ Production plan: voices, SFX, music, pacing
└─────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌──────────────┐          ┌──────────────┐
│  Batch Bark  │          │ SFX Generator│
│  Generator   │          │ (procedural) │
└──────────────┘          └──────────────┘
        │                          │
        ▼                          │
┌──────────────┐                   │
│Scene Sequencer│◄─────────────────┘
└──────────────┘
        │
        ▼
┌───────────────────────┐
│ Spatial Audio Enhance │──→ 48 kHz stereo master
└───────────────────────┘
        │
        ▼
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐
│ Episode Assembler│◄────│ Opening SFX  │     │ Closing SFX  │
│                  │◄────│ (radio tune) │     │ (theremin)   │
└──────────────────┘◄────└──────────────┘     └──────────────┘
        │
        ├───────────────────┐
        ▼                   ▼
  📻 Final .WAV      📺 Signal Lost .MP4
```

---

## Step 1: Install ComfyUI

If you already have ComfyUI running, skip to Step 2.

### Option A: ComfyUI Desktop App (recommended for Windows)

Download the installer from [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) releases page. The Desktop App bundles Python, PyTorch, and a GUI wrapper — no terminal needed.

Default install location: `C:\Users\<you>\AppData\Local\Programs\ComfyUI\`
Default workspace: `C:\Users\<you>\Documents\ComfyUI\`

### Option B: Manual / Portable Install

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py
```

> **Pro tip:** If you have an NVIDIA GPU, make sure you install the CUDA version of PyTorch. See the [ComfyUI README](https://github.com/comfyanonymous/ComfyUI#installing) for per-platform instructions.

---

## Step 2: Install This Node Pack

### Option A: ComfyUI Manager (easiest)

1. Open ComfyUI in your browser
2. Click **Manager** → **Install Custom Nodes**
3. Search for `ComfyUI-OldTimeRadio`
4. Click **Install** and restart ComfyUI

### Option B: Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio.git
cd ComfyUI-OldTimeRadio
```

Then install Python dependencies. **Use ComfyUI's Python, not your system Python:**

| Install Type | Command |
|---|---|
| Desktop App (Windows) | `& "C:\Users\<you>\Documents\ComfyUI\.venv\Scripts\python.exe" -m pip install -r requirements.txt`|
| Portable (Windows) | `..\..\python_embeded\python.exe -m pip install -r requirements.txt` |
| Standard / venv | `pip install -r requirements.txt` |

Restart ComfyUI after installing.

### Verify Installation

After restart, you should see in the ComfyUI console:

```
[OldTimeRadio] ✓ All 11 nodes loaded successfully
```

If any nodes fail, the console will tell you which ones and why. The rest still work — isolated loading means one broken dependency doesn't take down the pack.

---

## Step 3: Model Downloads

Models download automatically from HuggingFace on first use. Grab a coffee ☕ — the first run takes a while.

| Model | Size on Disk | VRAM (fp16) | Used By | Purpose |
|---|---|---|---|---|
| `google/gemma-4-12b-it` | ~24 GB | ~8 GB | Script Writer, Director | Writes scripts from news, plans production |
| `google/gemma-4-E4B-it` | ~5 GB | ~3 GB | Script Writer, Director | Lightweight alternative (default in Lite workflow) |
| `suno/bark` | ~5 GB | ~4 GB | Bark TTS, Batch Bark | Expressive character voices with emotional tags |
| `parler-tts/parler-tts-large-v1` | ~4 GB | ~3 GB | Parler-TTS | Text-described voice control (narrator presets) |

> **VRAM note:** Gemma 4 and Bark cannot run simultaneously on GPUs with less than 16 GB VRAM. The pipeline handles this automatically — Gemma generates the full script, unloads from VRAM, then Bark loads for TTS. You never need to manage this yourself.

Models are cached in `ComfyUI/models/huggingface/` (set by `prestartup_script.py`). If you already have these models elsewhere, set the `HF_HOME` environment variable before launching ComfyUI.

---

## Step 4: Run Your First Episode

1. Open ComfyUI in your browser
2. Click **Load** and navigate to:
   ```
   ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/workflows/
   ```
3. Choose a workflow:

| Workflow | File | What It Does |
|---|---|---|
| **Lite** (recommended first run) | `old_time_radio_scifi_lite.json` | Uses Gemma 4 E4B (smaller model), 5 acts, full pipeline including video |
| **Full** | `old_time_radio_scifi_full.json` | Uses Gemma 4 12B, longer scripts, all nodes, higher quality |

4. Set your episode title in the **Script Writer** node (or leave the default "The Last Frequency")
5. Hit **Queue Prompt**
6. Wait ~30–35 minutes for a full render

### Where's My Output?

| Output | Location |
|---|---|
| Episode WAV | `ComfyUI/output/old_time_radio/episode_001_*.wav` |
| Episode MP4 | `ComfyUI/output/old_time_radio/signal_lost_*.mp4` |
| Runtime log | `custom_nodes/ComfyUI-OldTimeRadio/otr_runtime.log` |
| Dashboard JSON | `custom_nodes/ComfyUI-OldTimeRadio/otr_dashboard.json` |

---

## Node Reference

All 11 nodes, in pipeline order.

### Script Generation

| Node | Display Name | What It Does |
|---|---|---|
| `OTR_Gemma4ScriptWriter` | 📻 Gemma 4 Script Writer | Fetches real science headlines via RSS, feeds them to Gemma 4 to write a complete radio drama script. Outputs `script_text` (human-readable), `script_json` (structured for pipeline), `news_used` (headlines that seeded the story), and `estimated_minutes`. |
| `OTR_Gemma4Director` | 🎬 Gemma 4 Director | Takes the script and generates a full production plan: which voice preset for each character, SFX generation prompts per scene, music cue placement, vintage filter intensity, and beat/pause pacing. |

**Script Writer widgets:**

| Widget | Default | What It Controls |
|---|---|---|
| `episode_title` | "The Last Frequency" | Episode title — used in output filenames and video overlay |
| `genre` | `hard_sci_fi` | Genre preset: `hard_sci_fi`, `cosmic_horror`, `noir_detective`, `space_opera` |
| `num_acts` | 5 | Number of acts in the script |
| `num_characters` | 4 | Number of speaking characters |
| `model_id` | `google/gemma-4-E4B-it` | HuggingFace model ID for Gemma 4 |
| `custom_premise` | (empty) | Optional custom premise — overrides news-based generation |
| `max_news_articles` | 3 | How many RSS headlines to fetch and weave into the story |
| `temperature` | 0.8 | LLM temperature: lower = more predictable, higher = more creative |
| `use_hallucination_guard` | false | When enabled, validates script against known factual constraints |

### TTS (Text-to-Speech)

| Node | Display Name | What It Does |
|---|---|---|
| `OTR_BarkTTS` | 🎙️ Bark TTS (Suno) | Single-line TTS with emotional expression. Supports bracket tags like `[sighs]`, `[whispers]`, `[laughs]`, `[gasps]`. 10 speaker presets. Best for dramatic character dialogue. |
| `OTR_ParlerTTS` | 🔊 Parler-TTS | Text-described voice control. Write a natural language description of the voice you want, or use one of 10 old-time radio presets. Best for narrators and announcers. |
| `OTR_BatchBarkGenerator` | ⚡ Batch Bark Generator | Processes an entire script through Bark in one pass — reads `script_json` + `production_plan_json`, generates all dialogue lines sequentially, outputs a batch of audio clips ready for the Scene Sequencer. This is the node used in the standard pipeline. |

**Parler-TTS voice presets:**

| Preset | Description |
|---|---|
| `announcer_male` | 1940s transatlantic accent, warm baritone |
| `announcer_female` | Mid-Atlantic accent, poised mezzo-soprano |
| `detective_noir` | Low, world-weary, slight rasp |
| `scientist_nervous` | Quick, breathless, educated |
| `commander_authority` | Deep, measured, military precision |
| `ingenue_young` | Bright, vulnerable, clear soprano |
| `villain_sinister` | Slow, dark amusement, theatrical |
| `alien_otherworldly` | Unusual cadence, mechanical |
| `narrator_documentary` | Rich baritone, educational |
| `operator_radio` | Clipped, nasal, 1950s operator |

### Audio Processing

| Node | Display Name | What It Does |
|---|---|---|
| `OTR_VintageRadioFilter` | 📡 Vintage Radio Filter | Full degradation chain: AM bandpass, tube saturation, 60 Hz hum, vinyl crackle, radio static, broadcast compression. 6 presets from subtle warmth to crystal-radio distortion. |
| `OTR_SFXGenerator` | 💥 SFX Generator | Procedural sound effects with no model downloads needed. Types: `radio_tuning`, `theremin`, `explosion`, `footsteps`, `heartbeat`, `door_knock`, `wind`, `siren`, `ticking_clock`, `room_tone`. |
| `OTR_AudioEnhance` | 🔊 Spatial Audio Enhance | 48 kHz stereo mastering: Haas effect for spatial width, mid-side stereo widening, gentle high-shelf EQ, loudness normalization. Turns mono TTS output into broadcast-quality stereo. |
| `OTR_AudioBatcher` | 🔀 Audio Batcher | Utility node that collects multiple AUDIO inputs into a single batch for downstream processing. |

**Vintage Radio Filter presets:**

| Preset | Character |
|---|---|
| `subtle` | Light touch — mostly clean with slight warmth |
| `authentic` | Sounds like a well-maintained 1950s broadcast (default) |
| `heavy_am` | Noisy AM reception from a distance |
| `war_era` | WWII-era broadcast quality — heavy static and compression |
| `crystal_radio` | Extreme degradation — like listening through a crystal set |

### Assembly & Output

| Node | Display Name | What It Does |
|---|---|---|
| `OTR_SceneSequencer` | 🎞️ Scene Sequencer | Renders the parsed script through the TTS pipeline with proper inter-line pauses (200 ms beats), scene-break silences (1.0 s), and SFX placement. Takes `script_json` + `production_plan_json` + `tts_audio_clips` and outputs a continuous scene audio stream. |
| `OTR_EpisodeAssembler` | 📼 Episode Assembler | Combines scene audio with opening/closing theme music and crossfades into a final episode WAV. Writes to `ComfyUI/output/old_time_radio/`. |
| `OTR_SignalLostVideo` | 📺 Signal Lost Video | Renders a procedural CRT-aesthetic MP4 synced to the episode audio. Pure CPU rendering (PIL + numpy) — zero VRAM. Audio-reactive visuals: circular frequency ring, orbiting particles, warping geometric grid, mirrored waveform, frequency bars, scan lines, vignette, and noise. Requires `ffmpeg` on PATH. |

---

## Vintage Filter Presets — Deep Dive

The Vintage Radio Filter node chains six DSP stages in series. Each preset dials different intensities:

| Stage | What It Does |
|---|---|
| AM Bandpass | Restricts frequency range to simulate AM radio bandwidth (~200 Hz – 5 kHz) |
| Tube Saturation | Soft-clips the signal to simulate vacuum tube warmth and compression |
| 60 Hz Hum | Adds power-line hum characteristic of old equipment |
| Vinyl Crackle | Random impulse noise simulating record surface noise |
| Radio Static | Additive Gaussian noise simulating RF interference |
| Broadcast Compression | Hard limiter + makeup gain for that "over-the-air" loudness |

---

## Requirements

### Python

Python 3.10 or later. ComfyUI manages its own Python environment — don't install packages into your system Python.

### Python Packages

From `requirements.txt`:

| Package | Version | Why |
|---|---|---|
| `transformers` | >=4.40, <6.0 | Gemma 4 + Bark model loading and inference |
| `soundfile` | >=0.12 | WAV file I/O |
| `numpy` | >=1.24 | Audio DSP and array operations |
| `feedparser` | >=6.0 | RSS headline fetching (required — no fallback) |
| `tokenizers` | >=0.15 | Required by transformers |
| `sentencepiece` | >=0.1.99 | Required by Gemma 4 tokenizer |

Optional: `parler-tts>=0.2,<1.0` if you want to use the Parler-TTS narrator node.

> **Do NOT pin torch.** ComfyUI manages its own PyTorch version. Pinning torch in a node pack's requirements will break things.

### System Dependencies

| Dependency | Required By | How to Install |
|---|---|---|
| `ffmpeg` | Signal Lost Video node | Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org). Must be on PATH. |
| NVIDIA GPU (CUDA) | All model inference | Any modern NVIDIA GPU with >=8 GB VRAM for Gemma 4 E4B. >=12 GB for Gemma 4 12B. |
| ~40 GB disk space | Model downloads | First run downloads Gemma 4 + Bark models to `ComfyUI/models/huggingface/` |

---

## Architecture & Design Patterns

This project follows the [ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide). Key patterns used:

| Pattern | Survival Guide Section | Why |
|---|---|---|
| Isolated per-node loading | Section 8 | One broken node doesn't take down the pack. If Parler-TTS isn't installed, only that node fails — the other 10 load fine. |
| Lazy heavy imports | Section 3 | `torch`, `transformers`, `bark` are only imported when a node actually executes, not at ComfyUI startup. |
| Device-tracked model cache with explicit unload | Sections 3, 34, 40 | Gemma 4 is explicitly unloaded from VRAM before Bark loads. No `device_map="auto"`. |
| No `trust_remote_code` | Section 49 | Gemma 4 is natively supported by transformers — no need to run arbitrary code from HuggingFace. |
| `prestartup_script.py` | Section 44 | Sets `HF_HOME` before any module imports, so models cache in the right place. |
| AUDIO type contract | Section 26 | Every audio output is `{"waveform": Tensor[1, channels, samples], "sample_rate": int}`. |
| `IS_CHANGED` on ScriptWriter | Section 12 | Forces re-execution every queue so you always get fresh news headlines. |
| Offline resilience | Section 37 | If RSS feeds are unreachable, the ScriptWriter falls back to built-in science headlines. |

---

## File Structure

```
ComfyUI-OldTimeRadio/
├── __init__.py                    # Isolated per-node loader (11 nodes)
├── prestartup_script.py           # HF_HOME + env setup (runs before imports)
├── requirements.txt               # Python dependencies (no torch!)
├── pyproject.toml                 # Package metadata
├── README.md                      # You are here
├── otr_monitor.py                 # Live render dashboard (standalone script)
├── otr_dashboard.json             # Machine-readable render status
├── otr_runtime.log                # Runtime heartbeat log
├── nodes/
│   ├── __init__.py
│   ├── gemma4_orchestrator.py     # ScriptWriter + Director (Gemma 4 LLM)
│   ├── bark_tts.py                # Single-line Bark TTS
│   ├── batch_bark_generator.py    # Batch Bark (full script in one pass)
│   ├── parler_tts.py              # Parler-TTS with text-described voices
│   ├── vintage_radio_filter.py    # 6-stage audio degradation chain
│   ├── sfx_generator.py           # Procedural sound effects
│   ├── scene_sequencer.py         # Sequencer + Assembler
│   ├── audio_enhance.py           # 48 kHz spatial mastering
│   ├── audio_batcher.py           # Audio batch utility
│   └── video_engine.py            # Signal Lost CRT video renderer
├── workflows/
│   ├── old_time_radio_scifi_lite.json   # Lite workflow (E4B model)
│   └── old_time_radio_scifi_full.json   # Full workflow (12B model)
├── samples/                       # Example episode outputs
└── tests/                         # Unit tests
```

---

## Live Render Monitor

The `otr_monitor.py` script is a standalone dashboard you run in a separate terminal during renders. It gives you real-time visibility into what's happening inside ComfyUI without touching the UI.

### Usage

```bash
# From the ComfyUI-OldTimeRadio directory:
python otr_monitor.py

# Or with explicit port:
python otr_monitor.py --port 8000
```

### What It Watches

Three daemon threads run in parallel:

| Thread | Data Source | What It Reports |
|---|---|---|
| Log Tailer | `comfyui_8000.log` | State transitions: prompt received → executing → complete/crashed |
| WS Listener | ComfyUI WebSocket | Which node is currently executing, progress percentage |
| Heartbeat | `otr_runtime.log` | Live script generation progress, BatchBark line-by-line progress |

All three update a shared status dict, flushed to `otr_dashboard.json` after every change. External tools (or your own scripts) can poll this single JSON file.

### Terminal Output

```
[14:32:05] EXECUTING      | Gemma4 ScriptWriter      |  45% | Bark: —            [12m30s]
[14:32:10] EXECUTING      | Gemma4 ScriptWriter      |  52% | Bark: —            [12m35s]
[14:42:18] EXECUTING      | Batch Bark Generator     |  33% | Bark: 4/12 lines   [22m43s]
```

### Optional Dependency

The WebSocket listener requires `websocket-client`. Without it, the monitor still works but only uses log tailing and heartbeat:

```bash
pip install websocket-client
```

---

## Setting Up Continuous Output with OBS

Run SIGNAL LOST as a live generative broadcast — each output episode auto-loads into OBS as it finishes.

### Prerequisites

- [OBS Studio](https://obsproject.com/download)
- [Python 3.11.x](https://www.python.org/downloads/release/python-3119/)
- [Media Playlist Source (OBS Plugin)](https://obsproject.com/forum/resources/media-playlist-source.1765/)
- [Directory Sorter for OBS](https://github.com/CodeYan01/directory_sorter_for_obs)

### Setup

1. Install OBS and the Media Playlist Source plugin
2. In OBS: **Tools → Scripts** → Python Settings → point to your Python 3.11 path
3. Load the `directory_sorter_for_obs` script and point it to `ComfyUI/output/old_time_radio/`
4. Add a **Media Playlist Source** scene item pointed to the same folder
5. OBS will automatically pick up each new episode MP4 as soon as the Signal Lost Video node finishes rendering.

> **Pro tip:** If your main GPU is maxed out on inference, set OBS to encode via your integrated GPU (**QSV AV1** or **HEVC**). This keeps the stream smooth while the NVIDIA GPU handles Gemma 4 and Bark inference.

---

## Troubleshooting

<details>
<summary><strong>Console says "[OldTimeRadio] ⚠️ Skipped 'OTR_ParlerTTS': No module named 'parler_tts'"</strong></summary>

Parler-TTS is optional and not in the default `requirements.txt`. Install it separately:

```bash
pip install parler-tts>=0.2
```

The other 10 nodes work fine without it. Isolated loading means this is not a fatal error.
</details>

<details>
<summary><strong>CUDA out of memory during BatchBark</strong></summary>

Gemma 4 didn't fully unload before Bark loaded. Restart ComfyUI and re-queue. If it persists, switch to the Lite workflow (`gemma-4-E4B-it`) which uses ~3 GB VRAM instead of ~8 GB.
</details>

<details>
<summary><strong>Signal Lost Video node fails with "ffmpeg not found"</strong></summary>

The video node shells out to `ffmpeg` for MP4 encoding. Install it and make sure it's on your system PATH:

- Windows: `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org)
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

Verify: `ffmpeg -version` should print version info in your terminal.
</details>

<details>
<summary><strong>Script Writer produces garbage or repetitive text</strong></summary>

Lower the `temperature` widget (try 0.6). If using the 12B model and seeing incoherent output, check that you have enough RAM — the model needs ~24 GB disk and ~8 GB VRAM. The E4B model is more forgiving.
</details>

<details>
<summary><strong>Episode audio has long dead-air pauses</strong></summary>

Beat pauses default to 200 ms. If you're hearing longer gaps, check the Director node's `pacing` widget — set it to `moderate` or `tight`. The Scene Sequencer also caps pauses at 200 ms regardless of what the Director requests.
</details>

<details>
<summary><strong>RSS feeds fail / "No news articles found"</strong></summary>

The Script Writer falls back to built-in science headlines when RSS is unreachable. If you're behind a corporate firewall or proxy, set `HTTP_PROXY` / `HTTPS_PROXY` environment variables before launching ComfyUI. The episode will still generate — it just won't reference today's actual headlines.
</details>

---

## Render Timeline

Typical render times on an RTX 3080 / RTX 4070 class GPU:

| Stage | Time | What's Happening |
|---|---|---|
| Script Writer | ~10 min | Gemma 4 generating multi-act script from news |
| Director | ~10 min | Gemma 4 planning production (voices, SFX, pacing) |
| Batch Bark | ~8 min | Bark TTS voicing all dialogue lines sequentially |
| Scene Sequencer + Enhance + Assembler | ~1 min | Audio assembly, spatial mastering, crossfades |
| Signal Lost Video | ~4 min | CPU-based procedural CRT video render + ffmpeg encode |
| **Total** | **~33 min** | For a ~4 minute episode |

---

## Genre Presets

The Script Writer's `genre` widget controls the tone, tropes, and narrative structure:

| Preset | Flavor |
|---|---|
| `hard_sci_fi` | Grounded speculation — real physics, plausible technology, ethical dilemmas |
| `cosmic_horror` | Lovecraftian dread — incomprehensible signals, reality distortion, isolation |
| `noir_detective` | Hardboiled narration — cynical detective, femme fatale, double-crosses |
| `space_opera` | Grand scale — interstellar empires, heroic captains, space battles |

---

## License

MIT

---

## Credits

- **Bark** by [Suno AI](https://github.com/suno-ai/bark)
- **Parler-TTS** by [Hugging Face](https://github.com/huggingface/parler-tts)
- **Gemma 4** by [Google DeepMind](https://ai.google.dev/gemma)
- Built with patterns from the [ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide)
- Created by [Jeffrey Brick](https://github.com/jbrick2070)
