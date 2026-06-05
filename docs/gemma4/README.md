# Run Gemma-4 (12B) locally — and the ONE setting that makes it actually work

[![test JSON](https://img.shields.io/badge/test-JSON-2ea44f)](./gemma4_ollama_request.json) [![raw JSON (copy-paste)](https://img.shields.io/badge/raw-copy--paste-1f6feb)](https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/docs/gemma4/gemma4_ollama_request.json) [![tester](https://img.shields.io/badge/run-gemma4__test.py-orange)](./gemma4_test.py)

Gemma-4 runs great locally through **Ollama**, but it's a **thinking model**: out of
the box it spends its whole output budget on a hidden reasoning pass and frequently
returns **empty text** (`finish_reason: "length"`). That silently breaks anything that
expects an answer — chats look dead, pipelines abort. The fix is literally one line.
Here's the whole recipe.

## 1) Install Ollama
Get it from https://ollama.com and make sure it's running (`ollama serve`, or the tray
app). It exposes an OpenAI-compatible API at `http://localhost:11434/v1` — no API key.

## 2) Pull Gemma-4
```
ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M
```
That's the ~7-8 GB Q4 build (fits a 16 GB card with room for other models). The
official library tag `ollama pull gemma4` also works — just pick the 12B variant.

## 3) THE GOTCHA — turn thinking OFF
If you skip this, Gemma-4 burns the token budget on `<think>` and hands back empty
content. Disable it based on how you call the model:

- OpenAI-compatible `/v1/chat/completions`:  add  `"reasoning_effort": "none"`
- Ollama native `/api/chat`:                 add  `"think": false`
- CLI:                                       `ollama run gemma4 --think=false "..."`

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

---

## OTR / ComfyUI users — switch the writer to Gemma-4
If you drive an LLM writer node over the Ollama lane (OTR-style):
1. `ollama serve` running + `ollama pull` the tag above.
2. On the writer node (`OTR_LedgerScriptWriter`), set **both** model slots —
   `creative_writing_model` AND `technical_model` — to `google/gemma-4-12b-it`.
3. Set env `OLLAMA_REASONING_EFFORT=none`, then **restart ComfyUI** (or use a build
   whose Ollama lane defaults it). Without it, the style/structure passes return empty
   and the episode aborts at the style picker.
4. The lane is fail-closed local-only (`http://localhost:11434/v1`, no key, no cloud
   fallback).

Same recipe we used to run a full episode end-to-end (script -> voice -> video) on a
16 GB RTX 5080.

## Quick gotcha checklist
- thinking disabled  (`reasoning_effort:"none"` or `think:false`)
- using `/v1/chat/completions`, NOT `/v1/responses`
- enough `max_tokens` (a thinking model wants headroom even with thinking off)
- structured output via `response_format` JSON schema, never raw `grammar`
