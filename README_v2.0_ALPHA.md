# ComfyUI-OldTimeRadio v2.0 (SIGNAL LOST — Visual Engine Alpha)

## 🔴 HEADLINE — Next-Gen Animation Pipeline for AI Radio Drama

**LTX-Video I2V + Scene Animation + VRAM Optimization + 93-Entry Regression Suite** → Full multi-model pipeline with real-time keyframe generation, procedural fallback clips, and automated quality assurance. **ALPHA on v2.0-visual-engine branch.**

Real science news → Model-Independent LLM Scriptwriting → Scene Segmentation → Director Reconciliation → Prompt Builder → **LTX-Video Scene Animation** → Kokoro Narration → Bark TTS Voice Acting → MusicGen Themes → Procedural SFX → 48kHz Master → CRT Video.

Fully automated. Zero API keys. Drop into `custom_nodes/` and queue.

---

## ⚠️ Alpha Status

**v2.0 is in active development.** The visual engine works on production hardware (RTX 5080, 16 GB VRAM, Windows). All nodes pass regression testing (21 passed, 2 xfailed). See [comfyui-custom-node-survival-guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide) for the test suite.

Expect to find and report bugs. PRs welcome.

---

## What's New in v2.0

**v1.5 (main branch)** — Audio-first radio drama pipeline. Full episodes in ~12 minutes.

**v2.0-visual-engine (this branch)** — Adds visual storytelling:
- **LTX-Video I2V** — converts keyframe images to smooth animated video (768x512, fp8)
- **Scene Animator** — orchestrates dialogue, director notes, and animation timing
- **Director Reconciler** — syncs scriptwriter output with animator requirements (Jaccard overlap)
- **Memory Boundary** — enforces VRAM discipline (14.5 GB ceiling, 14.3 GB watermark)
- **VRAM Guard** — pre-flight checks before animation loads
- **Regression Suite** — 23 pytest tests catch bugs before they ship (2-second static analysis, no runtime)
- **Three-File Contract** — Bible + tests + README stay in sync automatically

---

## Quick Start (The "Zero-Click" Path)

1. **Get ComfyUI**: Use the [Official Desktop Installer](https://www.comfy.org/download).
2. **Install v2.0**: Use **Install via Git URL** in ComfyUI Manager:
   ```
   https://github.com/jbrick2070/ComfyUI-OldTimeRadio
   ```
   Then switch to the `v2.0-visual-engine` branch.
3. **Run**: Drag `workflows/old_time_radio_v2_full.json` into your browser and hit **Queue Prompt**.
4. **Walk Away**: Everything else (Models, News, Scripts, Voices, Animation, Mastering, Video) is automatic.

---

## Hardware Requirements

| Setup | GPU | VRAM | Runtime (Full) |
|-------|-----|------|---|
| **Recommended** | RTX 5080 | 16 GB | ~60 min |
| **Minimum** | RTX 4090 | 24 GB | ~90 min |
| **Budget** | RTX 4080 | 16 GB | ~120 min |

VRAM is managed automatically. The pipeline unloads the LLM before loading animation models.

---

## Install v2.0-visual-engine

**Option A — ComfyUI Manager (recommended):**
1. Open ComfyUI Manager → **Install via Git URL**
2. Paste: `https://github.com/jbrick2070/ComfyUI-OldTimeRadio`
3. Click **Install** → Restart ComfyUI
4. In ComfyUI, switch to branch `v2.0-visual-engine`

**Option B — Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jbrick2070/ComfyUI-OldTimeRadio
cd ComfyUI-OldTimeRadio
git checkout v2.0-visual-engine
pip install transformers soundfile numpy feedparser tokenizers sentencepiece torch torchvision
```

---

## Run the Regression Suite

Before running a full episode, verify the pipeline passes all checks:

```bash
pip install pytest
cd <survival-guide-repo>
python -m pytest tests/bug_bible_regression.py -v --pack-dir <path-to-OldTimeRadio>
```

Expected output: **21 passed, 2 xfailed in 1.65s**

---

## Known Issues & Workarounds

- **LTX-Video memory spikes on long dialogue** — the pipeline auto-chunks >8-line scenes by 4. If you hit OOM, reduce scene length in the Director node.
- **Fallback clip integration** — if animation drift exceeds 500ms, the system uses an FFmpeg loop clip. Max 2 fallbacks per episode.
- **Cold-start CUDA stall** — first generate() call takes 8-10 seconds for kernel JIT. Subsequent calls are 13+ tok/s.

---

## Architecture

See [V2_BUILD_ORDER.md](./V2_BUILD_ORDER.md) for the complete node pipeline and [comfyui-custom-node-survival-guide](https://github.com/jbrick2070/comfyui-custom-node-survival-guide) for bug documentation.

**Core nodes:**
- `MemoryBoundary` (T2) — VRAM discipline checkpoint
- `VRAMGuard` (T3) — pre-flight watermark check
- `SceneSegmenter` (T6) — dialogue chunking
- `DirectorReconciler` (T7) — alignment verification
- `PromptBuilder` (T8) — fully-materialized scene matrix
- `SceneAnimator` (T10) — LTX-Video orchestrator with 6 entry guards

---

## Feedback & Contributing

This is alpha software. If you hit bugs:
1. Run the regression suite (see above)
2. Check [Bug Bible](https://github.com/jbrick2070/comfyui-custom-node-survival-guide/blob/main/BUG_BIBLE.yaml) for known issues
3. File an issue on GitHub with the regression output

PRs welcome. Follow the [Three-File Contract](https://github.com/jbrick2070/comfyui-custom-node-survival-guide#maintenance-rule-the-three-file-contract): every code change touches the Bible + tests + README.

---

**By Jeffrey A. Brick** · April 2026 · ALPHA
