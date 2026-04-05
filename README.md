# ComfyUI-OldTimeRadio (SIGNAL LOST)

**Real science news → AI radio drama script → Bark TTS voice acting → procedural SFX → 48kHz stereo master → audio-reactive CRT video.**

Fully automated. Zero API keys. Drop into `custom_nodes/` and queue.

---

## Download
[![Download ComfyUI-OldTimeRadio v1.0](https://img.shields.io/badge/Download-OldTimeRadio_v1.0-blue?style=for-the-badge)](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/releases)

**[Click here to download the full package (v1.0)](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/releases)** — includes workflow JSONs + this guide.

---

## What It Does
"SIGNAL LOST" (The OldTimeRadio engine) fetches today's real science headlines via RSS, feeds them to a local Gemma 4 LLM to write a multi-act sci-fi radio drama, acts as a Director to cast specific Bark TTS voices for each character, voices every line with expressive emotions (sighs, laughs, whispers), adds procedural theremins and radio tuning, masters the final mix with spatial audio, and renders a procedural CRT-aesthetic MP4. 

Every run is a brand new, complete episode generated entirely from scratch.

---

## New to ComfyUI? Start Here
ComfyUI is a free, node-based interface for running AI image, video, and audio models locally on your GPU.

> **Already have ComfyUI installed?** Skip to Step 2.

### Step 1 — Install ComfyUI
Use the official desktop installer — it handles Python, Git, and dependencies automatically:

**https://www.comfy.org/download**

Advanced users can also install manually from [GitHub](https://github.com/comfyanonymous/ComfyUI).

### Step 2 — Install Required Models
> **☕ Grab a coffee — the models are large and will download automatically on first run.**

| Model | Download | Size | Save To |
|-------|----------|------|---------|
| **Gemma 4 E4B** (default) | Auto-downloads on first run | ~5 GB | `ComfyUI/models/huggingface/` |
| **Gemma 4 12B** (optional) | Auto-downloads on first run | ~24 GB | `ComfyUI/models/huggingface/` |
| **Bark TTS** (required) | Auto-downloads on first run | ~5 GB | `ComfyUI/models/huggingface/` |
| **Parler-TTS Large** (optional) | Auto-downloads on first run | ~4 GB | `ComfyUI/models/huggingface/` |

> *VRAM management is handled automatically. The pipeline unloads Gemma from VRAM before loading Bark, so you will never run out of memory doing both.*

### Step 3 — Install ComfyUI-OldTimeRadio
**Option A — ComfyUI Manager (recommended):**
1. Open ComfyUI Manager → Install Custom Nodes
2. Search "OldTimeRadio" → Install → Restart

**Option B — Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
```
Then install Python dependencies (use your ComfyUI python/venv):
```bash
pip install transformers soundfile numpy feedparser tokenizers sentencepiece
```

### Step 4 — Load the Workflow
The pack ships with two workflow presets in the `workflows/` folder:

| Workflow | Target Mode | Episode Length | File |
|----------|-------------|----------------|------|
| **Lite** | Quick Tests | ~5 mins | `workflows/old_time_radio_scifi_lite.json` |
| **Full** | Productions | ~15+ mins | `workflows/old_time_radio_scifi_full.json` |

1. Open ComfyUI at `http://127.0.0.1:8000`
2. Click **Load** and select the workflow you want
3. Hit **Queue** — The system grabs the news and starts building the episode.

**Which GPU do I need?**

| Setup | GPU | VRAM |
|-------|-----|------|
| Recommended | RTX 5080 / 4090 | 16 GB+ |
| Minimum (Lite Mode) | RTX 4070 / 3060 | 8 GB |

---

```
ScriptWriter
    │
    ▼
 Director ───────────────┐
    │  script_json       │ production_plan
    ▼                    ▼
 BatchBark             SFX ────────┐
    │  audio_clips       │ effects │
    ▼                    ▼         │
SceneSequencer ◄───────────────────┘
    │  scene_audio
    ▼
 AudioEnhance
    │
    ▼
 EpisodeAssembler ───► .WAV Audio
    │
    ▼
SignalLostVideo ─────► .MP4 Result
```

| Node | What It Does |
|------|-------------|
| **ScriptWriter** | Grabs real RSS headlines and uses Gemma 4 to write a multi-act script. |
| **Director** | Scans the script to cast voices and set the pace. |
| **BatchBark** | Generates TTS for every line sequentially using strict explicit character voices. |
| **SFX / Filters** | Adds procedural theremins, static, and vintage tube-saturation degradation. |
| **SceneSequencer** | Handles the deterministic `(beat)` pauses and stitches lines and SFX together. |
| **AudioEnhance** | Masters the mix to 48kHz stereo with Haas-effect widening. |
| **EpisodeAssembler** | Sandwiches scenes with intro/outro theme music. |
| **SignalLostVideo** | Pure CPU rendering (no VRAM required) of audio-reactive CRT data-viz graphics. |

---

## Live Render Monitor
Long audio processing runs without any visual UI can feel like a black box. **SIGNAL LOST** includes a built-in "Granular Heartbeat Observability" system. 

Run the standalone dashboard in a terminal during renders to watch the AI write the script line-by-line in real time:

```bash
cd custom_nodes/ComfyUI-OldTimeRadio
python otr_monitor.py
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

> **Pro tip:** If your main GPU is maxed out on inference, set OBS to encode via your integrated GPU (**QSV AV1** or **HEVC**). This keeps the stream smooth while the NVIDIA RTX handles Gemma 4 and Bark calculations.

---

## Troubleshooting

<details>
<summary><strong>CUDA out of memory during BatchBark</strong></summary>

**Cause:** Gemma 4 occasionally didn't fully un-pin its VRAM.

**Fix:** Restart ComfyUI and re-queue. The engine utilizes a "Strict Device Sentry," but occasionally PyTorch memory fragmentation can leak. If it persists, ensure you're using the Lite workflow (`gemma-4-E4B-it`), which takes only ~3GB.
</details>

<details>
<summary><strong>Signal Lost Video node fails with "ffmpeg not found"</strong></summary>

**Cause:** System doesn't have ffmpeg installed or it isn't on the system PATH.

**Fix:** Windows: run `winget install ffmpeg`. 
</details>

<details>
<summary><strong>"No news articles found" / RSS feed errors</strong></summary>

**Cause:** Firewall issues or the RSS fetch timed out.

**What happens:** Total fallback. The script writer defaults to built-in fallback sci-fi premises, so the generation will succeed flawlessly, it just won't be referencing today's explicit news cycle.
</details>

---

## Technical Appendix (For Developers)

### Requirements Deep Dive
- `transformers >=4.40, <6.0`
- `soundfile >=0.12`
- `feedparser >=6.0`

> **Note:** We do NOT pin torch. Pinning torch crashes ComfyUI, as it handles its own CUDA pytorch installation.

### Architecture & Design Patterns
Built adhering to the [ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide).

1. **Isolated Module Loading:** Failing dependencies in one node (like the optional Parler-TTS) will not crash the rest of the pack.
2. **Strict Device Sentry:** Enforces automatic PyTorch Garbage Collection `gc.collect()` and VRAM flushes between HuggingFace model lifetimes.
3. **Canonical 1.0 Tokens:** Highly deterministic script parsing relying on `[VOICE: NAME, traits]`, `[SFX:]`, `[ENV:]`, and `(beat)` tags.
4. **Transparent Legacy Aliasing:** Prevents legacy pipeline layouts from breaking when updating to the `OTR_` node registry prefixes.

### License & Credits
MIT License

- **Bark** by [Suno AI](https://github.com/suno-ai/bark)
- **Parler-TTS** by [Hugging Face](https://github.com/huggingface/parler-tts)
- **Gemma 4** by [Google DeepMind](https://ai.google.dev/gemma)
- Built with patterns from the [ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide)
- Created by [Jeffrey Brick](https://github.com/jbrick2070)
