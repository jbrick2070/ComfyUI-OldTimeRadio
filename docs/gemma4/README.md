# Run Gemma-4 (12B) locally — and the ONE setting that makes it actually work

[![test JSON](https://img.shields.io/badge/test-JSON-2ea44f)](./gemma4_ollama_request.json) [![raw JSON (copy-paste)](https://img.shields.io/badge/raw-copy--paste-1f6feb)](https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/docs/gemma4/gemma4_ollama_request.json) [![tester](https://img.shields.io/badge/run-gemma4__test.py-orange)](./gemma4_test.py)

Gemma-4 runs great locally, but it's a **thinking model**: out of the box it spends its
whole output budget on a hidden reasoning pass and frequently returns **empty text**
(`finish_reason: "length"`). That silently breaks anything that expects an answer —
chats look dead, pipelines abort. The fix is literally one line. Here's the whole
recipe, plus every way to serve it (Ollama, llama-server, LM Studio, ComfyUI-native).

## 0) Which Gemma 4? (12B is the 16 GB sweet spot)

The family is **E2B, E4B, 12B Unified, 26B-A4B (MoE), 31B**. Memory needed at 4-bit
(GGUF, total RAM+VRAM, per Unsloth's tables) vs. what you get:

| Variant | 4-bit memory | MMLU-Pro | On a 16 GB card |
| --- | ---: | ---: | --- |
| E2B | ~4 GB | 60.0 | runs anywhere, weakest writer |
| E4B | ~6 GB | 69.4 | fine, still a clear quality step down |
| **12B Unified** | **7–8 GB** | **77.2** | **the pick — fits with VRAM to spare** |
| 26B-A4B (MoE) | 16–18 GB | 82.6 | does NOT fit in VRAM; CPU-expert offload only |
| 31B | 17–20 GB | 85.2 | doesn't fit |

12B is the knee of the curve: the biggest Gemma 4 that fits a 16 GB GPU **with room
left over** for whatever else your pipeline runs (video/image models, TTS). The 26B
MoE is the only quality upgrade reachable from a 16 GB card — but only by spilling
experts to system RAM (llama.cpp `--n-cpu-moe`, needs ~32 GB RAM, noticeably slower),
and it can't share the GPU with a render pipeline. Prefer Google's **QAT** 4-bit
quants where available — same size, less quality loss.

## 1) Install Ollama
Get it from https://ollama.com and make sure it's running (`ollama serve`, or the tray
app). It exposes an OpenAI-compatible API at `http://localhost:11434/v1` — no API key.

**Use a recent Ollama.** `reasoning_effort` is only honoured on newer builds — tested on
**0.30.5** (check with `ollama --version`). If gemma-4 still comes back empty *with a full
hidden "reasoning" field*, your build is ignoring the flag → update Ollama. That's the #1
cause of "it just returns nothing."

## 2) Pull Gemma-4
The official library tag exists now — simplest:
```
ollama pull gemma4:12b
```
That's the ~7.6 GB Q4 build (256K context). The Unsloth GGUF also works and gives you
quant choice (`Q4_K_M`, QAT, etc.):
```
ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M
```

## 3) THE GOTCHA — turn thinking OFF
If you skip this, Gemma-4 burns the token budget on its thought channel and hands back
empty content. Disable it based on how you call the model:

- OpenAI-compatible `/v1/chat/completions`:  add  `"reasoning_effort": "none"`
- Ollama native `/api/chat`:                 add  `"think": false`
- CLI:                                       `ollama run gemma4:12b --think=false "..."`

Heads-up: the `/v1/responses` endpoint currently **ignores** `reasoning_effort` — use
`/v1/chat/completions`.

## 4) Test it (copy-paste)
Request body is in `gemma4_ollama_request.json` (the `reasoning_effort: "none"` line is
the whole trick):
```json
{
  "model": "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M",
  "messages": [{ "role": "user", "content": "Write one vivid sentence about an old radio." }],
  "temperature": 0.6,
  "max_tokens": 256,
  "reasoning_effort": "none"
}
```
Fire it:
```
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer ollama" -d @gemma4_ollama_request.json
```
You get a real sentence back. Delete the `reasoning_effort` line and re-run — you'll
often get **empty** content. That's the trap that wastes everyone an afternoon.

## 5) Want JSON / structured output? Don't send raw GBNF
Ollama's `/v1` does **not** accept a raw top-level `grammar` (GBNF) field. Pass a JSON
schema in `response_format` instead (Ollama converts it to a grammar internally).
Sending raw `grammar` is a no-op at best, an error at worst.

## 6) Don't want Ollama? `llama-server` is a drop-in
Ollama is just a wrapper around llama.cpp — you can run the engine directly. Grab a
release binary of [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) and:
```
llama-server -hf unsloth/gemma-4-12b-it-GGUF:Q4_K_M --port 8080 --reasoning off ^
  --temp 1.0 --top-p 0.95 --top-k 64 --alias "hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M"
```
Same OpenAI `/v1` API, one exe, no daemon, no model store magic. Differences that bite:

- **Thinking is disabled server-side, not per-request.** llama-server does not honour
  `reasoning_effort` in the request body (it's ignored — harmless to keep sending it).
  Use `--reasoning off` on current builds; on older ones use
  `--chat-template-kwargs '{"enable_thinking":false}'`
  (PowerShell: `--chat-template-kwargs "{\"enable_thinking\":false}"`). Note: for
  gemma-4 specifically, `--reasoning-budget 0` alone reportedly does NOT stop it —
  use `--reasoning off`.
- `--alias` sets the model id the server answers to — set it to exactly the model
  string your client sends.
- Unlike Ollama, a real llama.cpp server DOES accept raw GBNF `grammar` — but the JSON
  `response_format` path works on both, so prefer it.

**LM Studio** works too: load the same GGUF, start its OpenAI server
(`http://localhost:1234/v1`), thinking has a UI toggle. Your client's `model` id must
match LM Studio's API identifier for the model.

## 7) ComfyUI-native Gemma 4 (no server at all) — know its limits
ComfyUI now ships Gemma 4 natively: the **TextGenerate** node loads
[Comfy-Org/gemma-4](https://huggingface.co/Comfy-Org/gemma-4) weights from
`models/text_encoders/` ([tutorial](https://docs.comfy.org/tutorials/llm/gemma4/gemma4)).
Two catches before you ditch your LLM server for it:

- Only **E2B / E4B / 31B** are packaged. There is **no 12B** (its encoder-free
  "unified" arch isn't in the text-encoder repack) and no 26B MoE. On a 16 GB card
  that means E4B — a real writer-quality step down from 12B.
- It runs **inside the ComfyUI process**, so the LLM competes for the exact VRAM your
  image/video models need. Fine for standalone LLM workflows; wrong for a pipeline
  that renders. (That's why the OTR writer stays out-of-process over `/v1` — the LLM
  uses zero ComfyUI VRAM.)

---

## OTR / ComfyUI users — switch the writer to Gemma-4
If you drive an LLM writer node over the local lane (OTR-style):
1. A local `/v1` server running: `ollama serve` + the pull above, **or** the
   `llama-server` line from section 6 (then set env
   `OLLAMA_BASE_URL=http://localhost:8080/v1` — the lane speaks to any local
   OpenAI-compatible server and never falls back to cloud).
2. On the writer node (`OTR_LedgerScriptWriter`), set **both** model slots —
   `creative_writing_model` AND `technical_model` — to `google/gemma-4-12b-it`.
3. Set env `OLLAMA_REASONING_EFFORT=none`, then **restart ComfyUI** (or use a build
   whose Ollama lane defaults it). On llama-server also pass `--reasoning off`
   (request-body `reasoning_effort` is Ollama-only). Without thinking disabled, the
   style/structure passes return empty and the episode aborts at the style picker.
4. The lane is fail-closed local-only (no key, no cloud fallback).

Same recipe we used to run a full episode end-to-end (script -> voice -> video) on a
16 GB RTX 5080.

## Quick gotcha checklist
- thinking disabled  (`reasoning_effort:"none"` / `think:false` / `--reasoning off`)
- using `/v1/chat/completions`, NOT `/v1/responses`
- enough `max_tokens` (a thinking model wants headroom even with thinking off)
- structured output via `response_format` JSON schema (works on Ollama AND llama-server)
- Google's recommended sampling: `temperature 1.0`, `top_p 0.95`, `top_k 64`
- 26B-A4B on a 16 GB card = CPU-offload territory, not a VRAM fit — 12B is the fit
