# Session Handoff -- ComfyUI-OldTimeRadio (OTR) -- CURRENT (written 2026-06-05, overnight session 06-04 -> 06-05)

## Core goal
Three things shipped on `v2.0-alpha` this session; the LAST remaining work is
**live GPU verification of two of them** -- do NOT lose sight of LIVE-TESTING
indextts2 AND gemma-4 (the user's explicit priority):
1. **indextts2 is now the DEFAULT character voice** + has real public-domain
   reference clips downloaded and wired. NOT yet proven by a live render.
2. **gemma-4-12b has a dedicated LOCAL Ollama writer lane** (no more
   `gemma4_unified` transformers crash). NOT yet proven by a live render as the
   writer; the style-inventor 63-vs-5 overgeneration still needs its GBNF
   constraint wired through this lane.
3. A bark fallback so a cloning engine with no usable ref never breaks a render
   (safety net; already done).

## Tech stack & constraints (session-specific; CLAUDE.md auto-loads the rest)
- **ComfyUI must be RESTARTED to load any .py edit** (Python module cache). User
  restarted at end of session, so the latest code is now live for the next run.
- **Cloud lanes are ALLOWED now** -- the "100% local" rule was LIFTED this
  session (OpenRouter + Comfy Credits OK). The checked-in CLAUDE.md "100% local"
  text is STALE pending the user's edit.
- **Models live at `C:\ComfyUI-Models`** (Comfy Desktop 1.0.4 migration;
  base-directory = `C:\Users\jeffr\Documents\ComfyUI`). `C:\ComfyUI-Models` is
  OneDrive-virtualized: **cmd `dir` / `if exist` CANNOT see files there, but
  Python `os.path.exists` CAN** -- always verify file presence with the venv
  python, never `dir`.
- Venv: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. GUI on :8000.
  Use Desktop Commander (cmd) for git/pytest. indextts2 = Path B oop_venv
  subprocess worker (env-pointed; it ran this session, so installed). gemma-4 =
  local Ollama at 127.0.0.1:11434 (GGUF `hf.co/unsloth/gemma-4-12b-it-GGUF:Q4_K_M`
  already pulled).

## What's done & decided (HEAD = 2588974 on v2.0-alpha, == origin)
- `0c50793` indextts2 -> shipped char_voice default (mirrors SA3 music promotion):
  eng_indextts2 default_roles=("char_voice",)+requires_flag=None; bark demoted to
  selectable fallback; `_LEGACY_FIRST_ENGINES["char_voice"]` =
  ("indextts2","chatterbox","bark"); node 81 of otr_scifi_16gb_full.json =
  indextts2; dep-pilot coverage decoupled (oop_venv, not flag).
- `dd2b95f` LOCAL gemma Ollama lane + bark fallback (+ the prior gemma
  reasoning_effort fix, NOW COMMITTED -- was the user's "don't lose it" worry):
  - `nodes/_otr_ollama_backend.py` (NEW): self-contained LOCAL OpenAI-/v1 client,
    hardwired 127.0.0.1:11434 (env `OLLAMA_BASE_URL`), fail-closed (never cloud),
    no key, no cost guard. Carries reasoning_effort (`OLLAMA_REASONING_EFFORT`) +
    grammar (GBNF). loader_backend "ollama_local_http", provider "ollama".
  - `google/gemma-4-12b-it` catalog row routes here (ollama_model_tag = the GGUF).
    Dispatch table + loader make_generate_fn/make_polish_generate_fn + writer +
    constrained-generate seams all branch provider=="ollama".
  - bark fallback in `_otr_voice_node_common._render_per_line`: a clone char
    engine in `_OTR_CLONE_ENGINES = ("indextts2","chatterbox")` with no usable
    ref renders that line on bark using the replayed voice_preset.
- `d199515` per-line ref resolver: `_resolve_clone_ref_path` resolves a ref by
  voice_ref_id else by gender via the bank caster; `_resolve_ref_to_disk` tries
  multiple model-dir layouts incl. C:\ComfyUI-Models. No ref -> bark.
- `2588974` REAL CC0 reference voices: `scripts/_otr_dl_indextts2_refs.py`
  (gitignored scratch) downloaded 4 CC0 LibriVox clips from
  kyutai/tts-voices `voice-zero/` -> C:\ComfyUI-Models\TTS\refs\indextts2\*.wav +
  the base models dir (Python-verified present). Bank entries (engine indextts2):
  vz_bill_boerst / vz_peter_yearsley / vz_stuart_bell (male), vz_caro_davy
  (female), commercial_clean=true.
- Rejected/decided: did NOT revert node 81 to bark (kept indextts2 default + bark
  safety net); gemma lane kept SEPARATE from openrouter(cloud)/comfy(cloud);
  voice-donations rejected as a source (un-genderable hex IDs) in favor of
  voice-zero. Full tests/ green throughout (3755 collected, 0 failed).

## State of the art
- HEAD 2588974 == origin/v2.0-alpha. Uncommitted: `session_handoff.md` (this),
  `scripts/_otr_dl_indextts2_refs.py` (gitignored), `custom_nodes.lnk` (junk).
- indextts2 refs ON DISK at C:\ComfyUI-Models\TTS\refs\indextts2\ : vz_bill_boerst,
  vz_peter_yearsley, vz_stuart_bell, vz_caro_davy (.wav). ONLY 1 female -> 3F unmet.
- NOTHING has been live-rendered yet this session (ComfyUI was being restarted).

## Immediate next steps
1. **LIVE-VERIFY indextts2 (priority).** Queue an episode (writer = mistral-nemo
   is fine). Tail logs (`scripts/otr_tail_logs.py` or the console). CONFIRM:
   - log `char_voice: rendering N lines on 'indextts2'` (NOT bark),
   - NO warning `engine 'indextts2' has no reference clip ... falling back to bark`,
   - episode completes with audio. If the indextts2 worker rejects the WAV
     (format/sample-rate), resample the 4 refs to 24kHz mono with ffmpeg + re-run.
2. **LIVE-VERIFY gemma-4-12b as writer.** Set node-1 creative_writing_model +
   technical_model = `google/gemma-4-12b-it`; ensure `ollama serve` is up. Queue
   an episode. CONFIRM it routes through Ollama (no `gemma4_unified` crash; log
   `[Ollama] load google/gemma-4-12b-it -> ... base_url=http://localhost:11434/v1`).
   Then the 63-vs-5 inventor: `make_ollama_generate_fn` already sets
   `_otr_supports_grammar=True`; verify `nodes/_otr_style_picker._run_inventor`
   passes its exactly-N GBNF grammar to grammar-capable backends. Set
   `OLLAMA_REASONING_EFFORT=none` if gemma burns its budget on a `<think>` preamble.
3. **Add more female indextts2 voices (3F target).** Edit VOICES in
   scripts/_otr_dl_indextts2_refs.py (more female CC0 readers -- OwenTyme/voice-zero
   LibriVox set or named female LibriVox readers), re-run, add bank entries.
4. Run full tests/ after any code change (baseline 3755/0); commit + push.

## Open questions
- Does the indextts2 Path B worker accept the kyutai voice-zero WAVs as-is, or do
  they need resampling to 24kHz mono? (live render answers).
- Does the resolver hit the right on-disk ref in a LIVE render (live
  folder_paths.models_dir vs the C:\ComfyUI-Models candidate)? (live render answers).
- Is the gemma 63-vs-5 inventor break fully fixed by the GBNF pass, or does the
  inventor-side wiring still need work? (grammar plumbing exists in
  _otr_openrouter_backend + the ollama lane; the _otr_style_picker wiring is the
  open piece).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
