# Running Google's Gemma 4 12B locally in ComfyUI (the easy Ollama way) — actually tested

Gemma 4 12B is a great little local LLM for building/expanding prompts, but its brand-new **unified architecture** (unified K/V + proportional RoPE) means most ComfyUI text loaders either crash or spit out garbage / rows of `.` characters. Here's the path that actually works — including the one gotcha that wastes everyone's first hour.

## TL;DR
Run Gemma through **Ollama** as a background server and call it from a ComfyUI Ollama node. The 4-bit build is ~6.75 GB. **You need Ollama ≥ 0.30.4** or it will not load.

## ⚠️ The #1 gotcha — read this first
gemma4 support was merged into llama.cpp / Ollama only very recently. If your Ollama is older (winget and most package managers still ship 0.24.x), you'll get this and think the model is broken:

```
Error: unable to load model ... unknown model architecture: 'gemma4'
```

It's not the model — it's the runtime. Install the **latest Ollama** directly from https://ollama.com/download (you need **0.30.4+**). Don't rely on winget/choco; they lag a few days behind. Verify with `ollama --version` → should say `0.30.x`.

## Step 1 — Ollama 0.30.4+
Download/update from ollama.com, then confirm:
```
ollama --version
```

## Step 2 — Pull the 4-bit model (~6.75 GB)
```
ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M
```
That's the instruction-tuned 12B at Q4_K_M (best quality/size balance). Fits a 16 GB card with room left for an image model; runs on 8 GB too with the keep_alive trick below.

(Quick sanity check it generates real text, not dots:)
```
ollama run hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M "Write one vivid sentence about a lighthouse at dawn."
```

## Step 3 — Install a ComfyUI Ollama node
ComfyUI Manager → **Install Custom Nodes** → search and install **one** of:
- **ComfyUI-Ollama** (by stavsap)
- **ComfyUI-IF_AI_tools**

Then **fully restart ComfyUI**.

## Step 4 — Wire the workflow
```
[Text / Primitive: your idea]  →  [Ollama Generate: prompt input]  →  text output  →  [CLIP Text Encode]
```
In the Ollama node, set the model field **exactly** to:
```
hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M
```

## 🔑 The Golden VRAM Rule
When you hit Queue, your GPU holds **both** Gemma and your image model (SDXL/Flux).
- **16 GB+:** you're fine — run both. Set `keep_alive: -1` to keep Gemma resident for faster repeats.
- **8–12 GB:** set **`keep_alive: 0`** in the Ollama node. This unloads Gemma the instant it finishes writing the prompt, freeing ~7 GB so your image model can render. Skip it and you'll OOM-crash.

## One more gotcha — "thinking" tags
Gemma 4 has a thinking/reasoning mode and can emit channel tags like `<|channel>thought ... <channel|>` around its answer. If those show up in your prompt text, use the node's **chat/instruct** path (it applies the proper template), or strip everything up to and including the closing `<channel|>`. Raw "generate" mode is where you'll see them.

## Want it faster? (16 GB only)
Gemma 4 ships an **MTP speculative-decoding drafter** (`...-it-assistant`). Load it alongside the 12B for up to ~3× faster generation. It needs both models resident at once, so 16 GB+ only.

---
**Recap of the pitfalls that get everyone:** (1) Ollama too old → `unknown model architecture: 'gemma4'` — update to 0.30.4+. (2) Forgetting `keep_alive: 0` on a small card → OOM. (3) Stray `<|channel>thought` tags → use chat mode or strip them. Nail those three and you've got a 12B writing your prompts locally in a couple of minutes.
