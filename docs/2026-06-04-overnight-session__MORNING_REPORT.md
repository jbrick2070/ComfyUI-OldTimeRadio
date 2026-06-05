# OTR Session -- Morning Report (2026-06-04 / overnight)

Branch `v2.0-alpha`. Everything below is committed + pushed (HEAD == origin).
Nothing is at risk.

## TL;DR
- **Your render will COMPLETE now** (it failed at character voices before). A
  graceful bark fallback renders any line a cloning engine can't (no reference
  clip), so the episode never hard-fails. **You must RESTART ComfyUI** for the
  fix to load (Python caches the old modules).
- **indextts2 now produces real voices.** I picked a genuinely public-domain
  source (kyutai/tts-voices `voice-zero/` = curated LibriVox, **CC0**),
  downloaded 4 clips -- 3 male (bill_boerst, peter_yearsley, stuart_bell) + 1
  female (caro_davy), sha256-verified -- placed them at the model roots, and
  pointed the voice bank at them. indextts2 now clones a real per-character
  voice (mapped by gender). RESTART ComfyUI to use it. One gap: only 1
  confidently-gendered CC0 female reader was in that set, so female characters
  reuse it -- add more via `scripts/_otr_dl_indextts2_refs.py` (edit VOICES, re-run).
- **gemma-4-12b now has its own LOCAL lane** (Ollama), separate from OpenRouter
  (cloud) and Comfy Credits (cloud). Pick `google/gemma-4-12b-it` as the writer
  and it routes through llama.cpp/Ollama at 127.0.0.1:11434 -- no more
  `gemma4_unified` transformers crash.
- Full `tests/` GREEN throughout (3755 collected, 0 failed). Bug Bible green.

## Commits pushed (newest first)
- `d199515` Resolve indextts2 reference clips per-line (by id or gender) before bark fallback
- `dd2b95f` Add LOCAL llama.cpp/Ollama writer lane + ref-less voice fallback (+ the prior gemma reasoning_effort fix, now committed -- it was the only uncommitted thing you were worried about)
- `0c50793` Promote indextts2 to the shipped char_voice default

## 1. The render crash -- FIXED (restart required)
`OTR_BatchCharacterVoices` got `ref_clip=None` -> "Invalid file: None". Root
cause: indextts2 is a voice-CLONING engine; it needs a per-character reference
WAV, but (a) no reference clips are installed and (b) preserve_ledger never
assigns clip refs. Two-layer fix in `nodes/_otr_voice_node_common.py`:
1. Resolve a reference WAV per line -- from the cast's `voice_ref_id`, else
   assigned by gender from the voice bank -- and use it only if the file exists.
2. If none resolves, render that line with **bark** (preset voices) using the
   replayed `voice_preset`, so the episode always completes (audio is king).

**ACTION: restart ComfyUI** (you have the desktop app on :8000). The fix is in
the cached `.py`; a restart reloads it.

## 2. indextts2 real voices -- the ONE remaining step
The engine works (the worker ran; it only complained about the missing ref).
What's missing is the reference audio. The voice bank
(`config/voice_reference_bank.json`) already lists 4 indextts2 refs:
`ix_male_neutral`, `ix_male_warm`, `ix_female_neutral`, `ix_female_bright`,
with ref_paths under `models/TTS/refs/indextts2/`. Those WAV files are NOT on
disk -- the public-domain download never ran.

You wanted >=3M / 3F / ~3 neutral. To get there:
1. Put reference WAVs (clean ~8-15s mono speech, one clear speaker each) at
   `<ComfyUI models dir>/TTS/refs/indextts2/*.wav`. NOTE: your model dir is
   ENV-pointed after the Comfy Desktop 1.0.4 migration (the default
   `...\Documents\ComfyUI\models\TTS\KokoroTTS\voices` is empty), so confirm
   where `folder_paths.models_dir` actually resolves before placing files.
2. Expand `config/voice_reference_bank.json` to 3M/3F/3neutral (mirror the
   existing entries; set real `ref_sha256`).
3. That's it -- the per-line resolver (committed) maps each character to a
   gender-matched ref automatically. No code change.

I did NOT auto-generate or substitute synthetic voices because (a) you
specifically wanted public-domain human voices and (b) I couldn't verify the
real model paths or download binaries from my side. **Easiest next session:**
tell me the live `folder_paths.models_dir` value and your source (LibriVox /
Wikimedia public-domain, or a folder of clips), and I'll write + run the
download/trim + bank-expansion in one pass and verify a live indextts2 render.

## 3. gemma-4-12b LOCAL lane (the writer fix)
`google/gemma-4-12b-it` via the local HF transformers path crashes
(`gemma4_unified` unsupported by transformers 5.5.0; a fix would need
transformers-from-source, which bricks the Blackwell venv). New dedicated lane
`nodes/_otr_ollama_backend.py`: a self-contained LOCAL OpenAI-/v1 client
hardwired to `127.0.0.1:11434`, fail-closed (never cloud), no API key, no
credits. The `google/gemma-4-12b-it` catalog row now routes here against the
GGUF you already pulled (`hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M`). It carries
`reasoning_effort` (kills the `<think>` budget blowup) and `grammar` (the GBNF
exact-5 constraint for the style inventor). Select it as the writer model and
it runs locally through Ollama.

Note: this is separate from the inventor's 63-vs-5 overgeneration -- the grammar
plumbing exists; wiring the inventor to pass its GBNF to the ollama lane is a
small follow-up if you want gemma as the live writer.

## State / guardrails
- UTF-8 no BOM, ASCII-only, no "dummy". Regression run after every change.
- One push attempt each, all succeeded; HEAD == origin/v2.0-alpha at d199515.
- Untracked + intentionally NOT committed: `session_handoff.md` (your prior
  doc), `custom_nodes.lnk` (a stray shortcut).
