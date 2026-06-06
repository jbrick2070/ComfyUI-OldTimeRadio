# Session Handoff -- ComfyUI-OldTimeRadio (OTR) -- 2026-06-05 (TTS engines + delivery wiring + live chatterbox)

## Core goal
The project is becoming a collection of the greatest TTS engines (a TTS
experiment). This session SHIPPED two new commercial-clean clone voices on the
proven IndexTTS2 Path-B sidecar pattern -- **chatterbox** (MIT) and **Dia**
(Apache-2.0) -- plus the foundation refactor (adapter metadata instead of a
hard-coded engine tuple) and **wired the per-line delivery (emotion) vector**
through dispatch so indextts2/chatterbox are expressive. Chatterbox was
**live-validated rendering on the RTX 5080**; the one bug found was fixed. All
committed + pushed (HEAD **97c141e** on v2.0-alpha, even with origin).

## Tech stack & constraints (session-specific; CLAUDE.md + memory auto-load the rest)
- ComfyUI Desktop (Electron, "Comfy Desktop.exe") on :8000, Windows, RTX 5080
  16GB (Blackwell sm_120). Main venv `C:\Users\jeffr\Documents\ComfyUI\.venv`
  (torch 2.10/cu130). `.py` edits need a ComfyUI RESTART (module cache); voice
  bank JSON hot-reloads.
- GOTCHAS discovered this session (durable):
  - `setx` is NOT on the Desktop-Commander shell PATH -> it fails silently.
    Set USER env via `[Environment]::SetEnvironmentVariable(name,val,'User')`.
  - PowerShell `& $exe ... | Out-String` errors ("document in the middle of a
    pipeline"); use `Start-Process -Wait -RedirectStandardOutput` (DC also does
    not capture external-exe stdout -> always redirect to a file and read it).
  - Comfy Desktop's python server canNOT be (re)started from the DC
    non-interactive shell (the Electron app spawns 7 procs but never inits the
    server / writes no logs). Manager has NO reboot endpoint (all /reboot 404).
    Only a real app relaunch (Electron) reloads USER env + code. computer-use
    needs an on-screen approval (unavailable when Jeffrey is away).
  - The live server runs as the uv python `...\uv\python\cpython-3.12.11...
    \python.exe -s ComfyUI\main.py --port 8000 --enable-manager`, parented by
    "Comfy Desktop.exe" (PID chain: server <- .venv python <- Comfy Desktop.exe
    <- explorer). main.py is bundled in the app (not in Documents\ComfyUI).
- git: changes land via file tools on disk; commit/push via DC `git -C <repo>`
  with Start-Process redirect. `scripts/_otr_*` is gitignored -> new worker /
  install / smoke scripts must be `git add -f`'d (the recurring trap).

## What's done & decided (all committed on v2.0-alpha, suite 3786/0 + Bug Bible green)
- **Adapter-metadata refactor (casting MUST-FIX #6):** deleted the
  `_OTR_CLONE_ENGINES` tuple in `nodes/_otr_voice_node_common.py`; dispatch now
  branches on adapter metadata `requires_voice_ref` / `voice_ref_kind="wav_path"`
  / `missing_ref_fallback="bark"` (defaults on `base.AudioEngineAdapter`; set on
  indextts2/chatterbox/dia). `_resolve_clone_ref_path` is role-aware; a stale
  (missing-on-disk) ref is nulled -> resolution + fallback (PD1); bark fallback
  guard admits char_voice AND announcer_voice.
- **chatterbox** (`eng_chatterbox.py` rewritten in-process->Popen sidecar +
  `scripts/_otr_chatterbox_worker.py`): MIT, flag OTR_ENABLE_CHATTERBOX, sr 24000,
  char+announcer. Worker saves via **soundfile** (NOT torchaudio.save -- torch
  2.x routes save through torchcodec). LIVE-PROVEN on GPU.
- **Dia** (`eng_dia.py` + `scripts/_otr_dia_worker.py`): Apache-2.0
  (commercial-clean -> fixes indextts2's bilibili liability), flag OTR_ENABLE_DIA,
  sr 44100, char_voice ONLY, `[S1]` tagged, audio_prompt-only clone (optional
  `config/dia_ref_transcripts.json` keyed by WAV basename). NOT yet installed live.
- **Shared `nodes/_otr_audio_engines/_otr_sidecar.py`:** bounded `read_protocol_line`
  (reader thread; Windows pipes), idempotent `close_worker` (closes stdin/stdout
  +stderr, kill+wait, tolerates double-close), `remove_quietly`. Env
  OTR_SIDECAR_STARTUP_TIMEOUT(1800)/_REQUEST_TIMEOUT(600).
- **Delivery wiring:** `_render_per_line` derives a per-line 8-dim vector via
  `_otr_delivery_vector.deterministic_delivery_vector(text, scene_tension)` (pure,
  C7) and passes it to prep + generate_voice. bark/kokoro/dia IGNORE it
  (byte-identical); indextts2 `emo_list` + chatterbox `_project` consume it (both
  hardened: non-numeric/out-of-range -> safe default + clamp). `OTR_DELIVERY_VECTOR=0`
  = true no-import old path.
- **Bank:** `scripts/_otr_mirror_clone_refs.py` mirrored the 36 CC0 indextts2 refs
  to cb_*/dia_* (110 voices); dropped the 5 dangling placeholder chatterbox rows;
  +1 chatterbox announcer ref. Profiles: added `char_dia_v1`, chatterbox profiles
  runtime->oop_venv. dia added to dropdown + dep-pilot OPT_IN_ENGINES.
- **5 roundtables** (1 design + 3 polish + 1 wiring QA, ~$1.5 OpenRouter, panel
  GPT-5.5+Gemini-3.1+Grok-4.3+DeepSeek-v4, Opus judge) under
  `docs/2026-06-05-tts-engine-sidecars/` + `docs/2026-06-05-delivery-wiring/`.
- Rejected: torchaudio.save in the sidecar (torchcodec); chatterbox `--variant`;
  mktemp->mkstemp (defeats the worker's no-file check); per-adapter request Lock
  (ComfyUI is serial); restarting the proven indextts2 path.

## State of the art (live, on the box)
- chatterbox venv `C:\Users\jeffr\Documents\ComfyUI\chatterbox\.venv` INSTALLED:
  chatterbox-tts 0.1.7 + **torch 2.11.0+cu128 / torchaudio 2.11.0+cu128** (the
  Blackwell sm_120 override; PyPI torch was 2.6 CPU-only). GPU verified cap (12,0).
- LIVE render proven: worker -> `RENDER_OK 3.80s @24kHz`; full adapter path ->
  `ADAPTER_OK shape=(1,96000) @24kHz`. Bug fixed = soundfile save (commit e5d22a4).
- USER env persisted (HKCU): `OTR_ENABLE_CHATTERBOX=1`,
  `OTR_CHATTERBOX_VENV=...chatterbox\.venv\Scripts\python.exe`.
- Comfy Desktop is CLOSED (0 procs, :8000 free) -- clean. Jeffrey reopens it to
  load the flag + new code.
- Ready driver: `scripts/_otr_chatterbox_smoke.py` (30-word workflow, node 81
  engine=chatterbox, node 80 voice_bank=default). Full detail:
  `docs/2026-06-05-tts-engine-sidecars/LIVE_TEST_RESULTS.md` + OPERATOR_HANDOFF.md.

## Immediate next steps -- START HEADLESS TESTING (Jeffrey's call: do NOT kill/launch his ComfyUI uninvited)
The engine is already proven (worker + full adapter path render a 3.8s 24kHz clip
on the GPU; sample at docs/2026-06-05-tts-engine-sidecars/chatterbox_render_sample.wav,
peak 0.47/rms 0.06). The ONE remaining validation is the in-ComfyUI-GRAPH render,
driven HEADLESS via the API. Plan:
1. **Get ComfyUI :8000 up WITH the flag + new code.** It must be a process whose
   env has OTR_ENABLE_CHATTERBOX=1 (already in HKCU) AND that loaded the new .py.
   - Easiest: Jeffrey opens Comfy Desktop himself (fresh launch reads HKCU env +
     re-imports OTR). Confirm `Invoke-RestMethod http://127.0.0.1:8000/system_stats`.
   - Headless server launch (if wanted): the Desktop server is
     `<uv python ...\cpython-3.12.11...\python.exe> -s ComfyUI\main.py --port 8000
     --enable-manager --base-directory C:\Users\jeffr\Documents\ComfyUI
     --user-directory ...\user --database-url sqlite:///...\user\comfyui.db ...`.
     BLOCKER: that `main.py` is bundled in the Electron asar -- it was NOT found in
     app.asar.unpacked or Documents\ComfyUI. To launch headless, FIRST capture the
     RUNNING server's working dir + exe via `Get-CimInstance Win32_Process` (do this
     while Comfy Desktop is up), then relaunch that exact main.py with
     OTR_ENABLE_CHATTERBOX=1 in env on a FREE port. Do NOT Start-Process the
     Electron "Comfy Desktop.exe" from the DC shell -- it spawns ~7 procs but never
     inits the server (no :8000, no log writes). See
     [[reference_comfy_desktop_restart_gotchas]] / docs LIVE_TEST_RESULTS.md.
2. **Run the smoke (fully headless API):**
   `& "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
   "...\ComfyUI-OldTimeRadio\scripts\_otr_chatterbox_smoke.py"` -> queues the
   30-word workflow with node 81 engine=chatterbox, node 80 voice_bank=default.
   Watch otr_runtime.log + /history/<pid>; expect chatterbox char voices,
   Bark loaded=0, output under output\otr\episodes\.
3. **Install Dia live** then repeat with node 81 engine=dia. `scripts\_otr_dia_install.ps1`
   note: `py -3.11` is absent -> build the venv from the main 3.12 python (`python
   -m venv`), torch 2.8+ cu128, `pip install git+https://github.com/nari-labs/dia.git`
   + soundfile; set OTR_ENABLE_DIA=1 via `[Environment]::SetEnvironmentVariable`.
4. Decide chatterbox/dia default-promotion (still opt-in; indextts2 is the default).

## Open questions
- Dia clone quality with audio_prompt-only (no transcript) -- acceptable, or add
  `config/dia_ref_transcripts.json` (faster-whisper)? Verify on GPU.
- chatterbox external `torch.Generator` for bit_exact (G1) -- run
  `scripts/otr_audio_dep_pilot.py --engines chatterbox,dia` on the box; keep
  `supports_external_generator=False` until confirmed.
- Dia 1.6B-0626 vs Dia2 (2025-11-19) -- targeted 0626; evaluate Dia2 later.
- The other casting MUST-FIX items (#1 voice_ref_path stamping, #2
  commercial_clean-effective, #3 gender guarantee, #4 kokoro announcer pool, #5
  resample-every-clip) remain staged (see the casting roundtable plan).

---
## Resume instructions
Open a fresh window, attach this file, and say:
"Read this handoff file and prepare to execute the immediate next steps.
Acknowledge when you're ready to start."
