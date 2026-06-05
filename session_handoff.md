# Session Handoff -- ComfyUI-OldTimeRadio (OTR) -- 2026-06-05

## Core goal
Made **gemma-4-12b viable as the OTR writer** and **recovered indextts2 as the default
character voice**, then proved the whole pipeline end-to-end on a 30-word episode
(script -> voice -> HuMo -> composite -> upscale -> procgen). The point of this pass was
to confirm the writer/voice base is solid before Jeffrey's video-engine refactor.

## Tech stack & constraints (session-specific; CLAUDE.md/ROADMAP/BUG_LOG auto-load the rest)
- ComfyUI Desktop 0.24.1 (Electron), Windows, RTX 5080 Laptop 16GB (Blackwell sm_120),
  torch 2.10+cu130. Venv `C:\Users\jeffr\Documents\ComfyUI\.venv`; server on :8000.
- **`.py` edits need a ComfyUI RESTART to load** (module cache).
- Git via Desktop Commander (sandbox git is read-OK; use DC for writes). The DC/PowerShell
  spawn often fails to capture external-exe stdout -> run git/pytest via the venv python's
  `subprocess` or Start-Process `-RedirectStandardOutput` to a file, then read the file.
- **`scripts/_*.py` is gitignored**; essential scripts are kept by `!` negation (the
  indextts2 worker was lost to this -- see below).
- Ollama 0.30.5 local at 127.0.0.1:11434 for the gemma lane.
- **HEAD = d076898 on v2.0-alpha (== origin).** Repo default branch is `main`; all of this
  session's work is on `v2.0-alpha`.

## What's done & decided (all pushed to origin/v2.0-alpha)
- **`af75ab1`** -- Ollama lane no longer advertises raw-GBNF grammar support (Ollama /v1
  doesn't accept a raw `grammar`); `OllamaBackend.generate` fails loud if one is passed.
  Roundtable Option A (unanimous GPT+Gemini+Grok+DeepSeek; ~$0.20).
- **`e4cb3ac`** -- Ollama lane now DEFAULTS `reasoning_effort=none`. THE gemma fix:
  gemma-4-12b is a thinking model; unset, it spent its whole budget on `<think>`, returned
  empty (finish_reason=length), and aborted the style inventor. `OLLAMA_REASONING_EFFORT=none`
  also set as a User env var.
- **`bb140a4` + `858a9b2`** -- committed `scripts/_otr_indextts2_worker.py` and UN-IGNORED it
  from `scripts/_*.py` (root cause: the worker was scratch-only, went missing, silently broke
  the default char voice -> node 81 failed closed). Doc: `docs/indextts2_pathb_setup.md`.
- **`721ecf6`/`d892a54`/`d076898`** -- `docs/gemma4/` shareable Reddit guide + test JSON +
  one-file tester + badges + Ollama-version note.
- **gemma-4-12b PROVEN as writer** (cleared the style inventor that previously rejected it --
  reverses the writer-bakeoff "rejected" verdict). **indextts2 PROVEN as voice** (rendered
  in-pipeline). Full `tests/` 3744/13/0 + Bug Bible green after every change.
- Rejected: sending raw GBNF to Ollama; leaving the worker as scratch.

## State of the art
- indextts2 Path-B install is ON DISK and working: venv
  `C:\Users\jeffr\Documents\ComfyUI\index-tts\.venv`, weights `...\index-tts\checkpoints`
  (gpt.pth/s2mel.pth/qwen emo + facebook/w2v-bert-2.0 under `checkpoints\hf_cache`). Env
  vars (User): `OTR_INDEXTTS2_VENV` / `_DIR` / `_WORKER`. Worker protocol = stdin/stdout JSON
  (ready -> per-line {text,ref_clip,emo_vector[8],emo_alpha,seed,out_path} -> {ok,...,22050});
  worker dups fd1->fd2 so model prints can't corrupt the channel.
- Full 30-word smoke (gemma + indextts2) rendered SUCCESS:
  `output\otr\episodes\signal_lost_melting_glass_pressure_20260605_093330\` + final
  `output\otr\obs\..._procgen_blended.mp4`. `/history` status=success. No VRAM thrash through
  HuMo/upscale/procgen.
- KNOWN issue (NOT ours): Comfy Desktop's Electron window goes BLACK mid-render (GPU-process
  crash under VRAM pressure) -- cosmetic; backend completes via API. Comfy-Org/desktop
  #1643/#1046. For heavy renders use the browser tab at :8000 or disable HW accel; the
  /prompt API needs no UI.
- Harness: `scripts/queue_smoke.py` (30w/2char/1act) on `scripts/otr_api.py`; writer = node 1
  `OTR_LedgerScriptWriter`, slots `creative_writing_model` + `technical_model`. Monitor via
  `otr_runtime.log` + `/otr/latest_ledger` + `/history/<pid>`. (See memory: otr-full-smoke-harness.)

## Immediate next steps
1. **Pre-refactor confirmation (cheap, headless, no video cost):** run the writer x voice
   matrix pruned to `OTR_EpisodeAssembler` -- {mistral-nemo, gemma-4-12b} x {bark, indextts2}.
   Confirms all 4 combos produce a script + voiced audio. Then ONE full baseline render
   (mistral-nemo + bark = the byte-identical default) to confirm the full pre-refactor pipeline
   is green. (Avoid full video for every combo -- HuMo is the long pole; the refactor replaces
   the video stage anyway.)
2. **Start the video-engine refactor** -- use the **otr-video-handoff** skill (NOT this general
   one); it pins the video mission + anti-drift rules. Plan + artifacts live OUTSIDE the repo at
   `C:\Users\jeffr\Documents\otr-video-roundtable\` (waves W0-W6).
3. Housekeeping (non-blocking): 5 deleted `workflows/GO_FORWARD_PLAN_v7-v11_*.md` are unstaged
   -- decide commit-deletion vs restore. Optionally mirror `docs/gemma4/` onto `main` for a
   clean share link.

## Open questions
- Nothing from this session blocks anything. BUG_LOG's open items are the historical
  BUG-LOCAL-231 FLUX VRAM-thrash family (out-of-band loaders) -- superseded by the NORMALVRAM
  fix and dissolved by the coming video refactor; the full render this session showed no thrash.
- gemma-4 is proven through the inventor; the rest of the writer (casting/compose/doctor) ran
  clean once but hasn't been soaked across many episodes -- worth a soak before making gemma a
  default.

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
