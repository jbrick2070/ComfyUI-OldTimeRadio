# ComfyUI-OldTimeRadio (SIGNAL LOST)

**Real science news → AI radio drama script → Bark TTS voice acting → procedural SFX → 48kHz stereo master → audio-reactive CRT video.**

Fully automated. Zero API keys. Drop into `custom_nodes/` and queue.

---

## Download
[![Download ComfyUI-OldTimeRadio v1.3](https://img.shields.io/badge/Download-OldTimeRadio_v1.3-blue?style=for-the-badge)](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/releases)

**[Click here to download the full package (v1.3)](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/releases)** — includes all three workflow JSONs + this guide.

---

## What It Does

"SIGNAL LOST" fetches today's real science headlines via RSS, feeds them to a local Gemma 4 LLM to write a multi-act sci-fi radio drama with story-first writing. Each episode randomly draws from 12 proven story arc templates — Shakespeare tragedies and comedies, Larry David comedic spirals, Marvel-style escalation, Twilight Zone twists, and more.

Character names, gender, age, demeanor, accent, and voice model are all procedurally generated from the episode seed — every cast is unique. The pipeline voices every line with expressive emotions (sighs, laughs, whispers), adds procedural theremins and radio tuning, masters the final mix with spatial audio, and renders a procedural CRT-aesthetic MP4.

Every run is a brand new, complete episode generated entirely from scratch.

---

## Current Realities & Beta Quirks

While SIGNAL LOST aims for a fully automated pipeline, **we are bound by the realities of local hardware.** Generating a complete 5-act to 10-act radio drama entirely from scratch is a massive computational lift. 

Please be aware: **Full epic episodes can take over an hour to render on local hardware.** Because of the sequential generation of tokens, voices, Foley effects, and CRT filters, the pipeline runs deep.

Our advice? **Queue the prompt, walk away, and let it cook.** The results are worth the wait.

---

## What's New in v1.2

### Narrative Patterns 1–6
Six new story-craft patterns embedded in the Gemma prompt chain:
**AISM Filter**, **Scaffolding Preamble**, **Verbalized Sampling Epilogue**, **Yes-But / No-And conflict rule**, **Vocal Blueprints**, and **Locked Decisions**. The result: tighter arcs, sharper character differentiation, and fewer "AI voice" tells in the dialogue.

### 8,316 Procedural Cast Combos
First/last name pools expanded to **154 × 54** — Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk, pulp adventure, classic Simpsons/Office generics, and 24 pre-1931 public domain literary characters. Zero trademark-specific franchise names. Every episode draws a unique cast.

### Lemmy Statistical Audit
New `tests/lemmy_rng_check.py` harness runs 10,000 trials against the 11% `SystemRandom` roll with ±1.5% tolerance. Confirms the easter egg stays statistically honest across builds.

### v1.2 Bug Fixes (v1.2.0.5)
- **Revision token budget** scales from draft length (`draft_chars / 3.5 × 1.25`) instead of `target_words`, ending Scene 4 decapitation that caused "weak ending" reviews
- **Minced oaths pool** — ContentFilter rotates through period-authentic 1940s radio euphemisms (*Stars above*, *Jiminy*, *Great Scott*, *Thunderation*) instead of `[BLEEP]`
- **Female preset pool expansion** — `en_speaker_7` promoted to female, giving 3 distinct female voices (4 / 7 / 9) so VEX and ZARA no longer collide on the same preset
- **NameLeakGuard post-pass** — `difflib` fuzzy-matches stray stock names in dialogue body against the real `[VOICE:]` roster, catching "Rex" → "Vex" type errors with zero hardcoded name lists

---

## What's New in v1.1

### Friendlier UI — 5th-Grader Approved
All three workflows now use numbered nodes with emojis and color-coded groups so anyone can follow the pipeline at a glance:

- **Blue zone** → `1. WRITE THE STORY 📝`
- **Green zone** → `2. MAKE THE VOICES & SOUNDS 🎙️`
- **Orange zone** → `3. PUT IT ALL TOGETHER 🎬`

Node names: `1. Gemma Writes the Story` → `2. Gemma Directs the Show` → `3. Voice Maker Machine` → `4. Scene Builder` → `5. Make It Sound Awesome` → `6. Glue Everything Together` → `7. Make the Final Video`

### Three New Control Dials on Node 1
| Widget | Options | What It Does |
|--------|---------|-------------|
| `runtime_preset` | 🧪 test / ⚡ quick / 📻 standard / 🎬 long / 🎭 epic / 🔧 custom | Pick your episode length without typing numbers |
| `target_length` | short (3 acts) / medium (5 acts) / long (7-8 acts) / epic (10+) | Controls act count and mandatory dialogue volume |
| `style_variant` | tense claustrophobic / space opera epic / psychological slow-burn / hard-sci-fi procedural / noir mystery / chaotic black-mirror | Injects a tonal directive into the script prompt |
| `creativity` | safe & tight / balanced / wild & rough / maximum chaos | Maps to temperature/top_p (0.6 → 0.85 → 1.1 → 1.35) |

### The Lemmy Fix
LEMMY — the 11% easter-egg engineer named after Lemmy Kilmister — had two bugs in v1.0:

1. **Voice collision**: Drake could steal his `v2/en_speaker_8` preset before Lemmy's locked branch ran. Fixed with two-pass cast iteration (LEMMY and ANNOUNCER processed first).
2. **Deterministic freeze**: The per-episode RNG seed was freezing the 11% roll, so the same widget config always hit or always missed. Fixed by routing the Lemmy coin flip to `SystemRandom` (OS entropy, immune to `random.seed()`).

New `summon_lemmy` toggle on Node 1 guarantees Lemmy for testing. Defaults to OFF in production — he stays a genuine 11% surprise.

### Pacing Overhaul
- Banned consecutive `[PAUSE/BEAT]` tags at the system-prompt level
- Hardened length directives to MANDATORY minimum line counts (not soft targets)
- Default `target_minutes` lowered 25 → 8 for punchy, dense dialogue out of the box
- `active_top_p` (creativity dial) now correctly reaches the chunked generation path for long episodes

### Lean Node Count
Removed unused nodes from the repo: `parler_tts.py`, `vintage_radio_filter.py`, `audio_batcher.py`. Node count: 12 → **9**. Boot log confirms: `[OldTimeRadio] ✓ All 9 nodes loaded successfully`.

---

## New to ComfyUI? Start Here

ComfyUI is a free, node-based interface for running AI models locally on your GPU.

> **Already have ComfyUI installed?** Skip to Step 2.

### Step 1 — Install ComfyUI
Use the official desktop installer — it handles Python, Git, and dependencies automatically:

**https://www.comfy.org/download**

Advanced users can install manually from [GitHub](https://github.com/comfyanonymous/ComfyUI).

### Step 2 — Install Required Models

> **☕ Grab a coffee — models download automatically on first run.**

| Model | Size | Notes |
|-------|------|-------|
| **Gemma 4 E4B** | ~5 GB | Default LLM. Auto-downloads to `models/huggingface/` |
| **Gemma 4 12B** | ~24 GB | Optional higher-quality LLM |
| **Bark TTS** | ~5 GB | Voice engine. Auto-downloads on first run |

> VRAM management is automatic. The pipeline unloads Gemma before loading Bark so you never run out of memory.

### Step 3 — Install ComfyUI-OldTimeRadio

**Option A — ComfyUI Manager (recommended):**
1. Open ComfyUI Manager → Install Custom Nodes
2. Search "OldTimeRadio" → Install → Restart

**Option B — Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
pip install transformers soundfile numpy feedparser tokenizers sentencepiece
```

### Step 4 — Load a Workflow

Three workflows ship in the `workflows/` folder:

| Workflow | Runtime Preset | Best For | File |
|----------|---------------|----------|------|
| **Test** | 🧪 1 min | Smoke testing after code changes | `old_time_radio_test.json` |
| **Lite** | 📻 5 min | Quick episodes, `open_close` OFF | `old_time_radio_scifi_lite.json` |
| **Full** | 📻 8 min | Production episodes, all features ON | `old_time_radio_scifi_full.json` |

1. Open ComfyUI at `http://127.0.0.1:8000`
2. Click **Load** → select a workflow
3. Hit **Queue** — the system grabs today's news and starts writing

**GPU requirements:**

| Setup | GPU | VRAM |
|-------|-----|------|
| Recommended | RTX 5080 / 4090 | 16 GB+ |
| Minimum | RTX 4070 / 3060 | 8 GB |

---

## Pipeline Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  1. WRITE THE STORY 📝                                              │
 │                                                                     │
 │  ┌────────────────────────┐    ┌──────────────────────────────────┐ │
 │  │ 1. Gemma Writes        │───►│ 2. Gemma Directs the Show        │ │
 │  │    the Story           │    │  Procedural cast: name/gender/   │ │
 │  │  RSS → Gemma 4 → Script│    │  age/demeanor/accent/voice model │ │
 │  │  Open-Close expansion  │    │  LEMMY stays LEMMY (en_speaker_8)│ │
 │  │  Self-critique loop    │    └──────────────────────────────────┘ │
 │  └────────────────────────┘                                         │
 └─────────────────────────────────────────────────────────────────────┘
                │ script_json + production_plan
                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  2. MAKE THE VOICES & SOUNDS 🎙️                                     │
 │                                                                     │
 │  ┌──────────────────┐   ┌────────────────┐   ┌───────────────────┐ │
 │  │ 3. Voice Maker   │   │ Opening Radio  │   │ Closing Space     │ │
 │  │    Machine       │   │    Static      │   │    Music          │ │
 │  │  Bark TTS batch  │   │  (radio_tuning)│   │  (theremin)       │ │
 │  └──────────────────┘   └────────────────┘   └───────────────────┘ │
 │            │ audio_clips        │ sfx_audio          │ sfx_audio    │
 │            ▼                   ▼                    ▼              │
 │  ┌──────────────────────────────────────────────────────────────┐  │
 │  │ 4. Scene Builder — stitches TTS lines + SFX into scene audio │  │
 │  └──────────────────────────────────────────────────────────────┘  │
 │            │ scene_audio                                            │
 │            ▼                                                        │
 │  ┌──────────────────────────────────────────────────────────────┐  │
 │  │ 5. Make It Sound Awesome — 48kHz Haas widening + mastering   │  │
 │  └──────────────────────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────────────┘
                │ enhanced_audio
                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  3. PUT IT ALL TOGETHER 🎬                                          │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────────────────┐  │
 │  │ 6. Glue Everything Together — intro/outro + episode assembly │  │
 │  └──────────────────────────────────────────────────────────────┘  │
 │            │ episode_audio                                          │
 │            ▼                                                        │
 │  ┌──────────────────────────────────────────────────────────────┐  │
 │  │ 7. Make the Final Video — CRT frame + h264_nvenc → .mp4      │  │
 │  │    + _treatment.txt (cast · voices · full script · stats)    │  │
 │  └──────────────────────────────────────────────────────────────┘  │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Node Reference

| Node | What It Does |
|------|-------------|
| **1. Gemma Writes the Story** | Fetches real RSS science headlines, then uses Gemma 4 to write a multi-act script. Open-Close expansion generates 3 competing outlines and picks the best. Self-critique loop revises the draft. 12 dramatic story arc templates (Shakespeare, Larry David, Marvel, Twilight Zone, and more). |
| **2. Gemma Directs the Show** | Scans the script and generates a production plan. Character names, traits, accents, and voice models are procedurally overridden. LEMMY always gets `v2/en_speaker_8`. ANNOUNCER gets a gender-balanced random preset. International presets produce accented English with safety rails. |
| **3. Voice Maker Machine** | Generates TTS for every line sequentially using Bark with the Director's voice assignments. ASCII sanitizer strips non-ASCII before Bark. Temperature cap (0.55 for international, 0.5 for first lines). GPU-accelerated. |
| **4. Scene Builder** | Stitches TTS lines, SFX cues, and `(beat)` pauses into scene audio in script order. |
| **5. Make It Sound Awesome** | Masters the mix to 48kHz stereo with Haas-effect spatial widening, bass warmth, and loudness normalization. |
| **6. Glue Everything Together** | Sandwiches scenes with intro/outro theme music. Configurable crossfade and duration. |
| **7. Make the Final Video** | Procedural CRT frame rendering + NVIDIA hardware video encoding (`h264_nvenc`, CPU fallback). Saves `_treatment.txt` alongside the MP4 — full cast, voice assignments, complete script, and production stats. |
| **Opening Radio Static** | Procedural `radio_tuning` SFX (scales with target runtime). |
| **Closing Space Music** | Procedural `theremin` SFX (scales with target runtime). |

---

## The LEMMY Easter Egg

There's an 11% chance per run that a character named **LEMMY** — a grizzled, wrench-wielding engineer — crashes the episode. He always gets `v2/en_speaker_8` (gravelly, English, 50s), always has a signature wrench-clank SFX on his first line, and always speaks in blunt, colorful mechanic metaphors.

Watch for this in the ComfyUI console:
```
[Gemma4ScriptWriter] 🎲 Lemmy rolled in on his own (lucky 11%)  [force=False, rng_hit=True]
```

For testing, flip the `summon_lemmy` toggle on Node 1 to guarantee his appearance.

### Lemmy's Lineage — A Tribute Across Three Generations

Our Lemmy is the third link in a chain that stretches back to the dawn of sci-fi radio drama:

**1. Lemuel "Lemmy" Barnet** — the cockney engineer in Charles Chilton's BBC radio serial *Journey into Space* (1953–1958), the original British sci-fi audio epic. The role was voiced across the original *Journey to the Moon*, *The Red Planet*, and *The World in Peril* arcs by Andrew Faulds, Guy Kingsley Poynter, Bruce Beeby, David Kossoff, Don Sharp, Alfie Bass, David Williams, John Pullen, Ed Bishop, Nigel Graham, and Anthony Hall — and later in the modern revivals (*Frozen in Time*, *The Host*) by David Jacobs, Alan Marriott, Michael Beckley, Chris Moran, Toby Stephens, Jot Davies, and Chris Pavlo.

**2. Ian "Lemmy" Kilmister** — Motörhead founder and rock icon. Chose his stage name directly after Lemmy Barnet from *Journey into Space*, which he listened to as a kid in the 1950s. ([source](https://www.reddit.com/r/todayilearned/comments/mdeffs/til_lemmy_chose_his_name_after_a_character_from/))

**3. Our Lemmy** — the 11% ghost in the SIGNAL LOST pipeline. Grizzled, wrench-wielding, gravel-voiced, crashes episodes unannounced. Stands on the shoulders of every actor who ever climbed into Jet Morgan's rocket, and the rock legend who took the name to the stage.

It's radio drama → rock and roll → AI radio drama. History works in strange loops. SIGNAL LOST is fundamentally a love letter to the *Journey into Space* era, and the 11% Lemmy roll is how we keep the ghost on the bridge.

---

## Live Render Monitor

Long renders without visual feedback can feel like a black box. SIGNAL LOST includes a built-in heartbeat observability system.

Run in a terminal during renders to watch the AI write the script in real time:

```bash
cd custom_nodes/ComfyUI-OldTimeRadio
python otr_monitor.py
```

---

## Setting Up Continuous Output with OBS

Run SIGNAL LOST as a live generative broadcast — each output episode auto-loads into OBS as it finishes.

**Prerequisites:**
- [OBS Studio](https://obsproject.com/download)
- [Media Playlist Source (OBS Plugin)](https://obsproject.com/forum/resources/media-playlist-source.1765/)
- [Directory Sorter for OBS](https://github.com/CodeYan01/directory_sorter_for_obs)

**Setup:**
1. Install OBS and the Media Playlist Source plugin
2. In OBS: **Tools → Scripts → Python Settings** → point to your Python path
3. Load the `directory_sorter_for_obs` script → point to `ComfyUI/output/old_time_radio/`
4. Add a **Media Playlist Source** scene item pointed to the same folder
5. OBS picks up each new MP4 automatically as Signal Lost Video finishes

> **Pro tip:** If your GPU is maxed on inference, set OBS to encode via integrated GPU (QSV AV1 or HEVC). Keeps the stream smooth while RTX handles Gemma 4 and Bark.

---

## Troubleshooting

<details>
<summary><strong>CUDA out of memory during Voice Maker Machine</strong></summary>

**Cause:** Gemma 4 occasionally didn't fully un-pin its VRAM.

**Fix:** Restart ComfyUI and re-queue. The pipeline uses automatic VRAM management between model lifetimes. If it persists, switch to the Lite workflow (`gemma-4-E4B-it`, ~3 GB VRAM).
</details>

<details>
<summary><strong>Signal Lost Video node fails with "ffmpeg not found"</strong></summary>

**Cause:** ffmpeg is not installed or not on your system PATH.

**Fix:** Windows: `winget install ffmpeg`
</details>

<details>
<summary><strong>"No news articles found" / RSS feed errors</strong></summary>

**Cause:** Firewall issues or the RSS fetch timed out.

**What happens:** The pipeline falls back to built-in sci-fi premises. Generation succeeds — it just won't reference today's news cycle.
</details>

<details>
<summary><strong>Lemmy appears on every single run</strong></summary>

**Cause:** The `summon_lemmy` toggle was set to ON in your browser session and got cached.

**Fix:** Hard-refresh the ComfyUI browser tab (Ctrl+Shift+R), reload the workflow JSON via File → Open. Confirm `summon_lemmy` is OFF on Node 1. The 11% RNG uses OS entropy and is not affected by the per-episode seed.
</details>

<details>
<summary><strong>Script is too short / under-running the target runtime</strong></summary>

**Cause:** Gemma occasionally undershoots even with MANDATORY line count directives.

**Fix:** Try `creativity = wild & rough` or `maximum chaos` to push more generation. The self-critique loop will catch the worst undershots. A post-generate line-count enforcement loop is planned for v1.2.
</details>

---

## Technical Appendix

### Requirements
```
transformers>=4.40,<6.0
soundfile>=0.12
numpy>=1.24
feedparser>=6.0
tokenizers>=0.15
sentencepiece>=0.1.99
```

> **Do NOT pin torch.** ComfyUI manages its own CUDA PyTorch installation.

### Architecture & Design Patterns

Built adhering to the [ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide).

**Isolated Module Loading** — A failing dependency in one node will not crash the rest of the pack. Each of the 9 nodes loads independently with per-node exception handling.

**RNG Architecture** — Two separate RNG systems prevent determinism leaking into places it shouldn't:
- `random` (seeded per-episode from fingerprint) — used for reproducible story arc selection, Open-Close arc choices, character voice pool draws
- `SystemRandom` (`_LEMMY_RNG`) — OS entropy, used only for the Lemmy 11% coin flip so it stays genuinely random across repeated runs of the same config

**Canonical 1.0 Token Format** — Highly deterministic script parsing relying on `[VOICE: NAME, traits]`, `[SFX:]`, `[ENV:]`, and `(beat)` tags. Tags always start at the beginning of a line. Consecutive `(beat)` / `[PAUSE/BEAT]` tags are banned at the prompt level.

**VRAM Sentry** — Automatic `gc.collect()` and CUDA cache flush between Gemma 4 and Bark model lifetimes. Chunked generation for long episodes (>5 min) prevents OOM on 8–12 GB cards.

**Voice Collision Prevention** — LEMMY and ANNOUNCER are assigned voice presets before regular characters draw from the pool. Regular characters use a 10-attempt re-roll loop to avoid duplicates. Pool exhaustion is logged with a warning rather than crashing.

### Output Files

Every completed episode produces two files in `ComfyUI/output/`:
- `signal_lost_<title>_<timestamp>.mp4` — the final video
- `signal_lost_<title>_<timestamp>_treatment.txt` — full cast list, voice assignments, complete script in scene order, and production stats (duration, resolution, file size)

---

## What's New in v1.3

### Arc Enhancer — Full Phase A/B/C Pipeline
The Arc Enhancer is now fully instrumented and on by default in all three workflows.

**Phase A — Structural Coherence Scoring:** Before any rewrite, the system scores the draft arc across 5 checks: truncation guard, strong scene (≥4 distinct voices), payoff (keyword overlap between opening and closing), echo (shared thematic words), and epilogue (ANNOUNCER present in final 500 characters). Score and all 5 check values are logged per-episode so you can see exactly what the arc looks like before Phase B touches it.

**Phase B — Plot Spine Injection:** The middle acts are summarized into a ~50-word "plot spine" and injected directly into the Phase B bookend rewrite prompt. Phase B can no longer hallucinate an ending that contradicts what actually happened in the middle of the episode.

**Phase C — Echo Phrase Logging:** After the bookend rewrite, the system extracts the shared noun that bridges the new opening and closing lines and logs it — the concrete thematic echo that ties the episode together.

### OpenClose Stability Fix
Reduced outline token budget from 600 to 450 tokens and raised the generation wall from 300s to 480s. Eliminates the timeout failure that was hitting when SDPA inference ran at ~2 tok/sec on long outline passes.

### Test Workflow Speed Pass
Test workflow now runs `short (3 acts)` instead of `medium (5 acts)` — cuts smoke-test time by roughly half while retaining 100% feature coverage including Arc Enhancer, Plot Spine, and echo logging.

### Attention Backend Clarity
The Flash Attention 2 probe now logs a precise platform message instead of a generic "not installed" warning:
```
[Gemma4] Flash Attention 2: NOT AVAILABLE — no prebuilt wheel exists for torch 2.10 + CUDA 13 + Blackwell sm_120 on Windows. SageAttention + SDPA active. Performance unaffected. Do not attempt install.
```

### Stability (from v1.3-beta)
`prestartup_script.py` injects a no-op mock for `transformers.safetensors_conversion` before any node imports, permanently eliminating the `JSONDecodeError` crash in offline/air-gapped environments. Bark's VRAM health probe is deferred to the `Gemma4Director` stage so Gemma has full VRAM during Arc Enhancer operations.

---

## 🛠️ Developer Note: Architecture Gotchas
*For the next AI assistant or contributor:*

1. **Offline First:** This project is designed for air-gapped performance. Never add code that relies on real-time `transformers` Hub pings or `requests.get` to remote servers without the safety mocks in `prestartup_script.py`.
2. **VRAM Sequencing:** Do not load Bark models while Gemma is active. The `Gemma4Director` is the bridge where Gemma unloads and Bark prepares. Keep this boundary clean.
3. **Regex Parity:** The dialogue filter `_clean_text_for_bark` is duplicated in `batch_bark_generator.py` and `scene_sequencer.py`. Any change to one must be applied to both.
4. **Test Sovereignty:** Run `python -m pytest tests/` before committing. The arc coherence tests in `tests/test_arc_check.py` cover Phase A scoring and Plot Spine extraction — do not skip them.
5. **Flash Attention 2:** No prebuilt wheel exists for torch 2.10 + CUDA 13 + Blackwell sm_120 on Windows. SageAttention is already active. Do not attempt FA2 installation on this platform.

---

## v1.4 Roadmap
- **Kokoro Announcer (shipped, in progress)** — Dedicated non-Bark narrator bus. ANNOUNCER dialogue lines are routed to Kokoro v1.0 via the new `🎙️ Kokoro Announcer` node, eliminating Bark's "ums" and "ahs" from the opening and closing bookends. Voice is picked per episode from a curated 4-voice British grab bag (`bm_george`, `bm_fable`, `bf_emma`, `bf_lily`), seeded from the episode seed so the same episode always draws the same narrator. `BatchBarkGenerator` now skips ANNOUNCER lines, and `SceneSequencer` consumes the Kokoro clips on a separate `announcer_audio_clips` bus. Voice `.pt` files are lazy-downloaded from `1038lab/KokoroTTS` on first use.
- **OpenClose parallel evaluator** — Re-enable the 3-outline race when faster attention backends land.
- **Chunked context continuity** — Richer act-by-act character state summaries for long multi-act runs.
- **Automatic Scene Transitions** — Procedural crossfades based on [ENV:] tag changes.

---

## License & Credits

MIT License

- **Bark** by [Suno AI](https://github.com/suno-ai/bark)
- **Gemma 4** by [Google DeepMind](https://ai.google.dev/gemma)
- Built with patterns from the [ComfyUI Custom Node Survival Guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide)
- Created by [Jeffrey Brick](https://github.com/jbrick2070)
