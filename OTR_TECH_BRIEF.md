# SIGNAL LOST — ComfyUI-OldTimeRadio Technical Brief
**Version:** 1.0 | **Author:** Jeffrey Brick | **Date:** 2026-04-05

---

## System Overview

**ComfyUI-OldTimeRadio** generates fully produced sci-fi audio dramas from real science headlines. One ComfyUI workflow — no manual steps. A real RSS article becomes a voiced, spatially mastered, SFX-bookended radio episode.

**Confirmed working end-to-end.** Lite render (1-min) completed successfully at `01:19:54` producing `episode_001_lite_the_last_frequency_20260405_011954.wav` (47 seconds, 48kHz stereo). A separate AI (Google Gemini Flash) cold-analyzed the audio and reconstructed the episode's plot, characters, and stakes without any context — confirming the pipeline is operating at a creative level, not just technical.

---

## Pipeline

```
RSS News Feeds (13 sources, parallel fetch)
    ↓
Gemma4ScriptWriter  — LLM writes full script in Canonical 1.0 format
    ↓
Gemma4Director      — LLM assigns voice presets + SFX plan as JSON
    ↓
BatchBarkGenerator  — Pre-renders all dialogue TTS grouped by voice preset
    ↓
SceneSequencer      — Assembles script items: dialogue, SFX, beats, ENV
    ↓
AudioEnhance        — 48kHz stereo, Haas, bass warmth, LPF, normalize
    ↓
SFXGenerator ×2     — Procedural theremin + radio_tuning themes
    ↓
EpisodeAssembler    — Crossfade intro/outro, save timestamped WAV
    ↓
PreviewAudio        — ComfyUI preview
```

**Node files:** `gemma4_orchestrator.py`, `batch_bark_generator.py`, `bark_tts.py`, `scene_sequencer.py`, `sfx_generator.py`, `audio_enhance.py`, `audio_batcher.py`, `vintage_radio_filter.py`, `parler_tts.py` + `__init__.py`

**Monitoring:** `otr_monitor.py` + `otr_dashboard.json` + `otr_runtime.log`

---

## Models

| Model | Role | Load |
|---|---|---|
| `google/gemma-4-E4B-it` | Script writer + Director | Lazy, bfloat16, SDPA attn, unloaded before Bark |
| `suno/bark` | TTS for all dialogue | float16, device_map=cuda:0, local_files_only |

---

## Canonical Audio Engine v1.0 — Script Format

Every generated script uses exactly four token types. No other formats are accepted by the parser.

```
=== SCENE X ===
[ENV: description]
[SFX: description]
[VOICE: CHARACTERNAME, gender, age, tone, energy] Dialogue text here.
(beat)
```

**Critical rules enforced by the prompt:**
- `[VOICE:]` first field is ALWAYS the character name in ALL CAPS — no exceptions
- Character names must be consistent across all scenes (same spelling every time)
- Dialogue uses Bark non-verbal tokens inline, not parenthetical stage directions
- `(beat)` is the only use of parentheses

---

## Key Bugs Fixed This Session

### 1. `NameError: AutoProcessor is not defined`
**File:** `gemma4_orchestrator.py` → `_load_gemma4()`
**Cause:** Gemini Flash refactored the loader but dropped the `from transformers import AutoProcessor, AutoModelForCausalLM` lazy import inside the function. It was previously a module-level import that got removed.
**Fix:** Added the import as the first statement inside the `if _GEMMA4_CACHE["model"] is None:` block.

---

### 2. Character Name Mangling — "MALE" / "FEMALE" as Characters
**Files:** `gemma4_orchestrator.py` — `SCRIPT_SYSTEM_PROMPT`, `DIRECTOR_PROMPT`, `_parse_script()`
**Root Cause:** `SCRIPT_SYSTEM_PROMPT` Section 1 showed the example format as:
```
[VOICE: gender, age, tone, energy] Short, natural dialogue line.
```
No NAME field. Section 2 correctly showed `[VOICE: NAME, gender, age, tone, energy]`. Gemma saw the inconsistency and dropped the name ~50% of the time, causing `_parse_script()` to extract `"MALE"` or `"FEMALE"` as the character name. All affected lines got hash-assigned to the same voice preset, destroying character voice consistency.

**Fix — three layers:**

**Layer 1 — Prompt fix:** Section 1 example now shows:
```
[VOICE: CHARACTERNAME, gender, age, tone, energy] Short, natural dialogue line.
```
Plus an explicit WRONG/RIGHT callout added directly below the format definition.

**Layer 2 — Director prompt fix:** `DIRECTOR_PROMPT` changed `"VOICE_TAG_STRING"` key to `"CHARACTER_NAME"` with a WRONG/RIGHT example:
```
WRONG key: "HAYES, male, 40s, calm"
RIGHT key: "HAYES"
```
This aligns the Director's voice_assignments JSON with what `_voice_preset_for_character()` looks up.

**Layer 3 — Parser hardening:** `_parse_script()` now detects the failure mode. If the first field of a `[VOICE:]` tag is a gender/age word (`male`, `female`, `young`, `old`, etc.), it assigns a positional fallback name (`CHAR_A`, `CHAR_B`, ...), logs a warning, and continues generating audio instead of silently producing a broken episode.

**Confirmed fixed:** Verified render shows `ANNOUNCER`, `DR_CHEN`, `HAYES` — no gender words as character names.

---

### 3. Bark Dialogue Token Contamination
**Files:** `batch_bark_generator.py` → `_clean_text_for_bark()`, `scene_sequencer.py` → `_clean_text_for_bark()`
**Cause:** Old implementation had an incomplete token map and passed unsupported tokens to Bark verbatim. `[whispers]`, `[nervous laugh]`, `[shouts]` are NOT in Bark's token set — Bark speaks them as literal words ("whispers", "nervous", "laugh"). Also, structural tags (`[VOICE:]`, `[SFX:]`, `[ENV:]`) were not being stripped before sending text to Bark.

**Bark's actual supported token set:**
```
[laughter]  [laughs]  [sighs]  [music]  [gasps]
[clears throat]  [coughs]  [pants]  [sobs]
[grunts]  [groans]  [whistles]  [sneezes]
♪ text ♪   (sung/hummed)
```

**Fix — `_clean_text_for_bark()` completely rewritten in both files:**
1. Strip all structural tags: `[VOICE:...]`, `[SFX:...]`, `[ENV:...]`, `[MUSIC:...]`, `=== ... ===`
2. Convert parenthetical stage directions to nearest valid Bark token (ordered by specificity)
3. Convert `*asterisk actions*` to Bark tokens
4. Preserve `♪ text ♪` for singing/humming
5. Whitelist-filter remaining bracket tags — anything not in the 13-token set gets dropped
6. Collapse whitespace

Also removed: `shout → [laughs]` hack from old scene_sequencer (was causing wrong emotional delivery).

---

### 4. Bark `local_files_only` Missing on Model Load
**File:** `bark_tts.py` → `_load_bark()`
**Cause:** `local_files_only=True` was set on `AutoProcessor.from_pretrained()` but NOT on `BarkModel.from_pretrained()`. On every load, transformers made an HTTP request to the HuggingFace API attempting a safetensors conversion. That request returned an empty body, causing:
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
This flooded the log with a full traceback on every Bark load.

**Fix:** `BarkModel.from_pretrained()` now uses the same try/except pattern as the processor:
```python
try:
    model = BarkModel.from_pretrained(model_id, ..., local_files_only=True)
except OSError:
    model = BarkModel.from_pretrained(model_id, ...)  # first-time download fallback
```

---

### 5. Bark `max_length=20` / `pad_token_id` Warning Flood
**File:** `bark_tts.py` → `_load_bark()`
**Cause:** Bark's `BarkModel` and its three sub-models (`semantic`, `coarse_acoustics`, `fine_acoustics`) ship with `max_length=20` in their `GenerationConfig`. When `model.generate()` is called with `max_new_tokens`, transformers detects both values and logs a deprecation warning — once per internal generate call per sub-model, producing ~20 warning lines per single dialogue line. Previous fix only patched the three sub-models; the parent `BarkModel.generation_config` was not patched, so warnings continued.

Additionally, `pad_token_id` was not set on sub-model configs, producing a separate "pad_token_id not set" warning on every call.

**Fix:** Patching loop now covers parent model + all three sub-models:
```python
_configs_to_patch = [model]
for sub_name in ("semantic", "coarse_acoustics", "fine_acoustics"):
    sub = getattr(model, sub_name, None)
    if sub is not None:
        _configs_to_patch.append(sub)

for obj in _configs_to_patch:
    if hasattr(obj, "generation_config"):
        obj.generation_config.max_length = None
        if obj.generation_config.pad_token_id is None:
            eos = obj.generation_config.eos_token_id
            obj.generation_config.pad_token_id = (
                eos[0] if isinstance(eos, list) else eos
            )
```

---

### 6. Stale Variable `character` in BatchBark Exception Handler
**File:** `batch_bark_generator.py`
**Cause:** Exception handler referenced `character` (old variable name) instead of `character_name` (current). Would have caused a secondary `NameError` if any Bark generation failed, masking the real error.
**Fix:** `character` → `character_name` in both the `log.warning()` and `batch_log.append()` calls.

---

## Bark CUDA Device Fix (Monkey-Patch — Carried Forward)

The core Bark silent-audio bug was fixed in a prior session and is still in place. Root cause: Bark's internal sub-model loops call `torch.tensor()` and `torch.arange()` without a `device` argument at the C level. These land on CPU and clash with CUDA weights during `index_select` (embedding lookup). Context managers, `set_default_device`, `device_map`, and `_move_to_device` all fail because the problem is inside C-level operations.

**Fix applied to:** `batch_bark_generator.py`, `bark_tts.py`, `scene_sequencer.py`

```python
_orig_tensor = torch.tensor
_orig_arange = torch.arange
def _tensor_cuda(*args, **kwargs):
    if "device" not in kwargs:
        kwargs["device"] = "cuda"
    return _orig_tensor(*args, **kwargs)
def _arange_cuda(*args, **kwargs):
    if "device" not in kwargs:
        kwargs["device"] = "cuda"
    return _orig_arange(*args, **kwargs)
torch.tensor = _tensor_cuda
torch.arange = _arange_cuda
try:
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=True, temperature=temperature)
finally:
    torch.tensor = _orig_tensor
    torch.arange = _orig_arange
```

`_move_to_device()` is also recursive, handling the nested `history_prompt` dict that Bark's processor returns for voice presets.

---

## Gemma 4 — Key Details

- **Model:** `google/gemma-4-E4B-it` (4B instruction-tuned)
- **Load:** `bfloat16`, `attn_implementation="sdpa"` (Scaled Dot Product Attention — significant speed gain on RTX 5000-series)
- **`local_files_only=True`** with OSError fallback — no HTTP ETag checks after first download
- **`eos_token_id` is a list** `[1, 107]` — `pad_token_id` extracted as `eos_id[0]` to avoid `GenerationConfig.validate()` crash
- **Unloaded before Bark:** `_unload_gemma4()` calls `gc.collect()` then `torch.cuda.empty_cache()` — mandatory for long episodes to avoid OOM
- **Token limit:** 1-min episodes use single-pass `max_new_tokens = target_words * 2.0`; episodes >5 min use act-by-act chunked generation

---

## GemmaHeartbeatStreamer

Custom `BaseStreamer` subclass hooked into `model.generate()`. Provides real-time script visibility without waiting for generation to finish.

**Tracks:**
- Scene count (from `=== SCENE ===` tags)
- Dialogue line count (from `[VOICE:]` tags)
- SFX cue count (from `[SFX:]` tags)
- Unique character names seen so far
- Token generation speed (tokens/sec, reported every 100 tokens)

**Output goes to `otr_runtime.log`** — timestamped, tailed live by `otr_monitor.py` Thread 3.

**Sample output in log:**
```
[01:15:32] ScriptWriter: === SCENE 1 ===
[01:15:42] ScriptWriter: [1] ANNOUNCER: Tonight, we track a breakthrough in va
[01:16:05] ScriptWriter: [2] DR_CHEN: The 10 mmHg reduction is statistically s
[01:16:23] ScriptWriter: [3] HAYES: Aggressive is an understatement, Doctor.
[01:16:43] ScriptWriter DONE: 86.9s (3.0 tok/s) | 1 scene | 8 dialogue lines | Characters: ANNOUNCER, DR_CHEN, HAYES
```

---

## News Pipeline

**13 RSS feeds, parallel fetch via `ThreadPoolExecutor`.** All open-access: ScienceDaily, EurekaAlert ×5, NASA, NIH, NSF, BBC Science, Ars Technica, The Conversation, Cosmos.

**Per-feed socket timeout:** 7 seconds (prevents any single slow feed from stalling the pool).

**Article resolution — 3-tier:**
1. RSS `content` field (ScienceDaily, Ars Technica — full text inline, up to 8000 chars)
2. URL fetch via `requests` + `BeautifulSoup` (strips nav/footer/ads, extracts `<article>` or `<main>`)
3. RSS `summary` fallback (if URL is paywalled/blocked)

**Selection:** shuffle feed list → 6 articles per feed → shuffle pool → `random.choice(pool)` → 1 story per render. Genuine variety across renders.

**Confirmed working:** Fetched 30 headlines from 5 feeds in 1.61 seconds. Selected ScienceDaily blood pressure story, fell back to RSS summary when article URL was blocked.

---

## OTR Monitor (`otr_monitor.py`)

Standalone watchdog. Run in a separate terminal during renders.

**Three daemon threads:**

| Thread | Source | What it tracks |
|---|---|---|
| LogTailer | `comfyui_8000.log` | State (STARTING/EXECUTING/COMPLETE/CRASHED), news headline, Bark progress |
| WSListener | `ws://127.0.0.1:8000/ws` | Active node ID → friendly name, progress % |
| Heartbeat | `otr_runtime.log` | Gemma live script output, act generation progress |

**Key fixes over original:**
- Port auto-detected from log filename (`comfyui_8000.log` → 8000) with `--port` CLI override
- WebSocket reconnects with exponential backoff (1s → 30s max) instead of dying silently
- Error detection uses specific exception class names (no false-positives on "ErrorHandler", "error_count")
- OTR node ID → friendly name mapping (`"1"` → `"Gemma4 ScriptWriter"`, `"11"` → `"Batch Bark Generator"`, etc.)
- Elapsed time tracked from `got prompt` event
- Atomic dashboard write via `os.replace()` (prevents half-written JSON on crash)
- Graceful `websocket-client` import guard (warns if not installed, degrades instead of crashing)

**Terminal output (every 5 seconds):**
```
  [01:17:30] EXECUTING       | Gemma4 Director         |  62% | Bark: —  [2m31s]
  [01:18:35] EXECUTING       | Batch Bark Generator    |   0% | Bark: 1/8 lines  [3m36s]
```

**Dashboard JSON fields:** `state`, `current_node`, `current_node_name`, `progress`, `elapsed_sec`, `last_heartbeat`, `bark_progress`, `news_headline`, `last_error`, `error_count`, `active_run`, `last_update`

---

## Render Configs

| File | `target_minutes` | `include_act_breaks` | `news_headlines` | Purpose |
|---|---|---|---|---|
| `otr_lite_prompt.json` | 1 | false | 1 | Smoke test / fast iteration |
| `otr_prompt_final.json` | 25 | true | 1 | Full production episode |

Both wire through `OTR_BatchBarkGenerator` (node 11). API queue command:

```powershell
$raw = [System.IO.File]::ReadAllText("C:\Users\jeffr\Documents\ComfyUI\otr_lite_prompt.json")
$raw = $raw.TrimStart([char]0xFEFF)
$body = '{"prompt":' + $raw + '}'
Invoke-RestMethod -Uri "http://127.0.0.1:8000/prompt" -Method POST -Body $body -ContentType "application/json"
```

---

## Regression Test Results

All 11 files pass Python syntax check (`python3 -m py_compile`). Logic regression: parser correctly rejects gender-word character names, assigns `CHAR_A`/`CHAR_B` fallbacks, passes proper names through unmodified. Bark token cleaner smoke-tested against 9 edge cases — all output verified.

**Live render confirmed:** `prompt_id: e19cd10b` completed in 294 seconds. Output: 47 seconds of stereo audio, 3 distinct voices, SFX bookends, spatial mastering applied.

---

---

## Citation Hallucination Guard

**Problem:** Gemma confidently invents plausible-looking ArXiv IDs (`arXiv:2401.XXXXX`) and DOIs even when given real source material. For a show whose credibility rests on using real science, fabricated citations are a direct integrity failure.

**Fix — three-layer approach:**

**Layer 1 — Prompt change:** `SCRIPT_SYSTEM_PROMPT` Section 4 now has an explicit CITATION RULE block:
> "Cite ONLY the real article provided above — its exact headline, source name, and date. DO NOT invent ArXiv IDs, paper titles, DOIs, or journal names that were not in the article."

**Layer 2 — Epilogue instruction updated:** The closing line in `user_prompt` previously said "cite real sources: ArXiv, Nature, Reuters, etc." — this was an open invitation to hallucinate. Now says "cite ONLY the real article provided above. Headline, source, date. No invented IDs."

**Layer 3 — Post-processing detector:** After generation, `write_script()` runs regex against the output for ArXiv ID and DOI patterns (`arXiv:\d{4}.\d{4,5}`, `doi.org/10.XXXX/...`). Any found IDs are cross-checked against the real article text that was fed to Gemma. IDs not present in the source material log a `[CitationGuard] WARNING` flagging them for review before publishing. IDs are left in the text (stripping creates jarring gaps) but the warning is unambiguous.

---

## Lemmy Easter Egg — SFX Signature

**11% probability** per render. When triggered, LEMMY (grizzled engineer/mechanic, named after Lemmy Kilmister) is added as a required character with at least 3 dialogue lines.

**SFX signature added:** When Lemmy rolls, the directive now requires a specific sound cue before his first line:
```
[SFX: heavy wrench strike on metal pipe, single resonant clank]
```
This runs once — on his entrance only, never repeated. It's his calling card: you hear the clank before you hear him speak.

The SFX is injected via the Lemmy directive in `write_script()` and carried through `_generate_chunked()` for long episodes. The `GemmaHeartbeatStreamer` will log it as `ScriptWriter: SFX #N: heavy wrench strike on metal pipe...` in `otr_runtime.log` when it appears.

---

## Pre-1.0 Seal QA Checklist (Gemini Flash Audit)

| Test | Target | Success Criteria |
|---|---|---|
| Duration stress test | 25-minute render | Completes without OOM; act-by-act chunking maintains plot coherence |
| RSS edge case | Disconnected network | `RuntimeError` caught and surfaced cleanly |
| Bark token test | `♪ text ♪` singing | Melodic delivery confirmed in output audio |

---

## Permanent Rules

- Version is always **1.0** — no changelogs, no version bumps
- `news_headlines` input is always **1** in both JSON files
- No `trust_remote_code=True` on any model load
- No `device_map="auto"` (conflicts with ComfyUI's `torch.set_default_device`)
- Gemma must be unloaded before Bark loads (`_unload_gemma4()`)
- All Bark code paths get the monkey-patch: `batch_bark_generator.py`, `bark_tts.py`, `scene_sequencer.py`
- Epilogue citations must only reference the real article provided — never invented IDs
