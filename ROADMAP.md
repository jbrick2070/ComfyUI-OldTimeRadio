# OTR Roadmap

**Branch:** `v2.0-alpha` | **Owner:** Jeffrey A. Brick | **Stack head:** `f1467a2` | **Last refactored:** 2026-05-03 EVENING

This file is the **canonical going-forward plan**. Forward-only. Historical session logs and "what shipped" archives are in `docs/ROADMAP_HISTORY.md`.

---

## Status snapshot — 2026-05-03 EVENING (post BUG-027 + BUG-028 soak fixes)

**Code work for the v2.0-alpha cycle is now 19 entries deep.** All 19 BUG-LOCAL entries below are `[FIXED]` in code and pushed to `origin/v2.0-alpha`. The 2026-05-03 EVENING soak surfaced two new failure modes (BUG-027 dialogue wipe + BUG-028 FLUX legacy save paths); both were fixed in the same autonomous session per direct user directive ("yes ofrget rop8u7hnd robins just fix fix fix"). Round-robin consult was SKIPPED for both fixes per the same directive — extra verification in lieu (AST + format-safety + targeted regression + Bug Bible regression all green pre-commit). The remaining work is **a single real-run acceptance soak** to confirm the live behavior on Jeffrey's RTX 5080.

**Committed and pushed (in chronological order):**

| Bug | Phase | Commit | What it fixed |
|---|---|---|---|
| 003 | Sprint 1 | (pre-QA-pass mega-commit) | `scripts/run_comfyui.cmd` reads HF_HOME from HKCU\Environment |
| 004 | Sprint 1 | (same) | LLM script-writer OOM — `_flush_vram_keep_llm()` + `MAX_PARSE_RETRIES=2` |
| 005 | Sprint 1 | (same) | 30-word preset CHARACTER:/SCENE: enforcement + ULTRA_SMOKE strict-VOICE parse |
| 006 | Sprint 1 | (same) | `tests/conftest.py` CUDA mask; later promoted from `[PARTIAL]` to `[FIXED]` after re-verification |
| 014 | A | `d2c2df8` | Spacesaver wrong-episode wipe via global mtime ledger scan |
| 015 | B | `29295c9` | production_ledger treatment rename gap + os.replace silent split state |
| 016 | C | `3e1d995` | Filename pattern audit — slug-reconstruction regression guard |
| 017 | D | `e43695d` | MusicGen + AudioGen cache miss every run — `_cache_key` returned fresh ts |
| 018 | E | `7c84ee8` | Ledger schema bump l3-2026-05-02 + meta.paths block |
| 019 | (cleanup) | `ca85a01` | Sprint 1 full-suite acceptance — pre-existing test rot fixed |
| 020 | G | `1fabd5c` | video_engine.py procgen mp4 written to legacy `output/otr/audio/` (SOAK BLOCKER from 2026-05-02 23:00 run) |
| 021 | G | `1fabd5c` | Audio-side nodes used global mtime walker (latent BUG-LOCAL-014 wrong-episode shape in 7 sites) |
| 022 | G | `1fabd5c` | BatchHumoRender stem-swap broken when `safe_title[:40]` truncates the title |
| 023 | H | `5075b9e` | ANNOUNCER portrait wasted FLUX context + skewed scene composition |
| 024 | H | `5075b9e` | Radio bookend FLUX prompt fell back to generic when style missing OR ledger stale |
| 025 | H | `5075b9e` | LTX role prompts ignore story style + scene context (every episode looked the same) |
| 026 | G/H hotfix | `03dfbfa` | DIRECTOR_PROMPT.format crash from Phase H unescaped curly braces (caused soak crash 23:46) |
| **027** | **soak fix** | **`f1467a2`** | **Critique/revision pass strips all CHARACTER dialogue (parser regex didn't accept `[N] CHARNAME:` format + acceptance gate had no total-collapse check + revision LLM under temp=0.95 would happily produce SCENE/ENV/SFX-only output). 3-part fix: regex + total-collapse hard gate + ABSOLUTE REQUIREMENT prompt clause.** |
| **028** | **soak fix** | **`f1467a2`** | **FLUX env stills + radio bookend save to legacy flat dirs (`_legacy_stills/` + flat `otr/stills/` shared global counter) instead of per-episode workspace — VideoComposite + BatchHumo + BatchLTX all looked in the wrong places after Phase B reorg. 4-site write+read alignment fix.** |
| **078** | **portraits** | **(BUG_LOG)** | **Per-cast portrait pass (`OTR_BatchFluxPortraitRender`) — renders one clean head-and-shoulders FLUX portrait per cast member to `<ep>/portraits/<char_id>_portrait.png`, stamps `cast[i].portrait_path` into the ledger so HuMo's tier-1 lookup hits instead of falling through to env-still tier-4 stopgap.** |
| **081** | **workflow-wiring** | **`413ef3a`** | **Portrait node never executed in workflow — Node 59 `ledger_json` socket was wired to Node 12 `video_path` (a `.mp4` filesystem path) so `_load_ledger` raised `RuntimeError`; AND the Node 12 dependency forced portraits to run AT THE END of the workflow, after HuMo had already needed them. Fix (workflow JSON only): drop link 100, set `ledger_json` widget to empty for in-flight auto-pickup, re-route link 45 from `(23 → 24)` to `(59 → 24)` so chain is FLUX env stills → Portraits → UnloadAll → HuMo. Portraits confirmed live in run `signal_lost_skindeep_microneedle_..._222516` — `c01/c02/c03_portrait.png` all rendered.** |
| **082** | **filename-derivation** | **`b34d272`** | **VideoComposite missing the BUG-118 underscore-mismatch fallback. SignalLostVideo writes procgen mp4 with `__` (double underscore) before the timestamp; ledger writer uses `_` (single). VideoComposite's naive `mp4 → _ledger.json` derivation got the wrong path and crashed `derived ledger from .mp4 not found`. BatchLTXRender already had the fallback; ported it to VideoComposite (when `__` in stem, also try single-underscore variant before raising).** |
| **083** | **kwarg-signature** | **`e601ee8`** | **`probe_duration_s(...)` called with `ffmpeg=ffprobe` kwarg but the function signature names it `ffprobe`. Caught by smoke harness on first run after BUG-082 landed — TypeError silenced by strict_c7 exception handler. Fix: rename kwarg at both call sites in `video_composite.py` (lines 1033 + 1135).** |
| **084** | **composite-sync** | **`7f2d03f`** | **VideoComposite per-clip-mux concatenated 6 line clips back-to-back at t=0 with no gap-fill — audio timeline has 9.5s pre-roll music + 0.6s inter-line silences + post-roll, video timeline had none. Cumulative 9.5s+ drift made wrong-mouth-on-wrong-voice; trailing audio truncated by `-shortest`. 4-site fix: (1) LTX clip stamps real `start_s` + ffprobed `dur_s` into ledger.clips, (2) per-clip BUG-031 duration matching (already wired), (3) NEW gap-fill pass walks sorted timeline + inserts static-radio segments for gaps >0.1s + trailing tail-fill, (4) NEW duration-contract assertion before mux with tail-pad fallback if audio overruns.** |
| **085** | **hf-cache** | **`56cf493`** | **Mistral-Nemo OOM at SDPA prefill with 24 GiB allocated on 16 GiB GPU. Cause: ComfyUI Desktop's Electron parent process didn't inherit `HF_HOME` from `HKCU\Environment`, so OTR's `_load_llm` fell through to `~/.cache/huggingface` default. With wrong cache_dir + `local_files_only=True` + sharded-safetensors layout on Windows, transformers misresolved the model location, fell back to fp16 silently despite `BitsAndBytesConfig(load_in_4bit=True)` being passed. Fix: NEW `nodes/_otr_hf_env.py` (winreg HF_HOME resolver + canonical snapshot directory resolver) wired into `_load_llm` so the loader passes the absolute snapshot path (bypasses transformers' Hub-resolution). Standalone check confirms NF4 working at 7.79 GiB allocated, 280/281 modules quantized.** |

**Cumulative regression test count (post-027/028):** 155 passed in 3.27s (targeted set: production_ledger + radio_still_resolver + filename_pattern_audit + cache_key_mutations + meta_paths + ledger_rename + critique_dialogue_preservation + save_to_episode_workspace + prompt_format_safety) PLUS Bug Bible regression 24 passed / 1 skipped / 1 xfailed in 1.24s. Full `tests/` directory NOT re-run (BUG-LOCAL-006 dropdown_guardrails hang resurfaced under live ComfyUI; pre-existing, not caused by these fixes; documented as known regression in cohabit mode).

**Promotion to Bug Bible:** All 19 entries are Bible candidates. Promotion happens after the next real-run soak confirms behavior end-to-end.

### What still needs Jeffrey's hands

1. **Restart ComfyUI Desktop** so the new code is loaded (custom node `.py` files are cached in `sys.modules`; mid-process changes don't hot-reload). Especially important after BUG-028 because a NEW node class (`OTR_SaveToEpisodeWorkspace`) was registered in `__init__.py` and the workflow JSON now references it.
2. **Re-queue any episode** — the BUG-027 + BUG-028 fixes are general-purpose, no special title needed.
3. **Tail the run** and confirm the new acceptance signatures:
   - `CRITIQUE: Character line counts - draft={'CHAR1': N, ...} revised={...}` with NON-EMPTY draft dict (BUG-027 parser fix)
   - If revision wipes dialogue: `CRITIQUE: CRITIQUE_REJECTED - total character lines collapsed from N to M` (BUG-027 hard gate fires)
   - `[BatchBark] Found >=1 dialogue lines in Canonical 1.0 format` (downstream confirms dialogue survived)
   - `output/otr/episodes/<ep>/stills/full_env_NNNNN_.png` files exist with counter starting at 1 (BUG-028 writer fix)
   - `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` exists (BUG-028 writer fix)
   - `[BatchHumoRender] cast-still binding: N/M cast members matched to fresh stills` reports N>0 (BUG-028 reader fix)
4. **On a green soak,** promote all 19 BUG-LOCAL entries to the Bug Bible together.

### Known remaining suspects (NOT blocking the soak — Phase H+ candidates)

- `nodes/scene_sequencer.py:147` `DEFAULT_OUT = output/otr/audio` legacy default. Only matters if it's ever the actual write target.
- `nodes/batch_humo_render.py:1773` uses `otr_legacy_audio_dir()` in the auto-pick fallback. Only fires when `ledger_json` input is empty.
- `nodes/batch_ltx_render.py:300/846` use `otr_stills_dir()` / `otr_audio_dir()` with NO episode_id (returns legacy dirs).
- `nodes/video_composite.py:282` legacy audio dir scan.
- `nodes/story_orchestrator.py:6276` hardcoded `output/otr/audio/` path.
- `nodes/post_audio_video_pipeline.py:126` empty-input fallback uses mtime walker (intentional for headless mode).

These are documented in the Phase G consult (`docs/2026-05-03-phase-g-path-reorg-blast-radius__01_chatgpt.md` Section 3) and queued for a future pass.

---

## Original P0/P1/P2 sections below are NOW HISTORICAL — Sprint 1 is DONE

**Canonical narrative hierarchy** — every ledger, workflow, and doc in this repo follows this:

```
Scene  >  Shot  >  Beat  >  Clip
```

- **Scene** — high-level narrative location (`AstroTech Research Facility`, `Control Room`, ...). One per `scene_id`.
- **Shot** — continuous visual unit. Same framing, same lighting. May contain multiple speakers.
- **Beat** — single-speaker continuous turn within a shot. The unit at which the 7 s clip-fill rule applies — beats never cross speakers, so HuMo audio windows align to one voice.
- **Clip** — one HuMo render call. Length must be `4n + 1` frames (Wan VAE temporal compression of 4) and ≤ 177 (verified ceiling on 16 GB).

Every consumer of `ledger.json` must understand all four levels.

---

## Platform Pins

Lock these. Any work item that contradicts this list is wrong.

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0.

**Canonical stack (do not downgrade):**
- CUDA 13.x / cu130
- PyTorch cu130 matching the ComfyUI environment
- SDPA as guaranteed fallback
- SageAttention only when the cu130 wheel/source build matches Python + Torch exactly
- FlashAttention not required for shipped OTR

CUDA 13 is non-negotiable because (1) Blackwell sm_120 support is the point, (2) NVFP4 / FP4 model support in ComfyUI requires `comfy-kitchen` which requires CUDA 13+, (3) Task #2 SeedVR2 v2.5 NVFP4 path needs cu130 downstream. The cu128 SageAttention path exists in the wild and is the easier wheel target, but it belongs in a SEPARATE experimental ComfyUI folder if needed for sandbox work — never in the production OTR pipeline.

**Attention backend policy:**
- Default: PyTorch SDPA (boring, safe, in-tree).
- Preferred acceleration: SageAttention via KJNodes "Patch Sage Attention" node, tested per-workflow only.
- Do NOT use global `--use-sage-attention` unless a specific model/workflow has passed smoke testing — Triton route can produce black outputs with some models.
- FlashAttention 2/3: out of scope on Windows Blackwell. Do not chase community wheels for the shipped pipeline.
- FlashAttention 4: real and worth tracking (`pip install flash-attn-4`, exposes `flash_attn.cute` namespace), but NOT a ComfyUI production dependency yet. Older FA2-style custom nodes hard-coding the top-level import won't see it. FA4 is the future-looking transformer/training answer; SageAttention is the practical diffusion/ComfyUI answer today.
- Any third-party attention wheel must pass before shipping: import test → one FLUX smoke → one Wan/HuMo smoke → no black frames → no VRAM regression → no audio-path impact. Then it's blessed.
- Note on SageAttention wheel sourcing: `mobcat40/sageattention-blackwell` is the leading prebuilt wheel repo for sm_120, but its primary build line is PyTorch 2.11 nightly + CUDA 12.8. A cu130 build exists in that repo, but verify with smoke workflow on our pinned torch 2.10.0 / CUDA 13.0 stack before blessing.
- 100% local, offline-first, open source, no API keys for the shipped pipeline. Cloud LLMs (OpenAI / Gemini / NVIDIA NIM) are for **internal QA round-robins only**, never shipped output.
- VRAM ceiling: **14.5 GB audio** / **15.5 GB video** (lifted 2026-04-17 for the video stack only — audio stays at 14.5 GB).
- Audio is king (rule **C7**). Full narrative output must never break, shorten, or degrade. If video breaks audio, revert immediately. Audio output must remain byte-identical to v1.5 baseline at every gate.

---

## P0 — Sprint 1: make smoke green (code work, blocks everything else)

Tonight's smoke (2026-05-02, prompt_id `e6b87239-16d4-4318-bfde-134468d32904`) failed end-to-end. Six new entries in `docs/BUG_LOG.md`. The four fixes below unblock the entire BUG-128/129 acceptance verification — that work is already shipped in code, but cannot be observed because the pipeline cannot reach the audio path on the 30-word smoke.

### BUG-LOCAL-005 — 30-word ultra-smoke ScriptWriter output unparseable

**Fix:** port the BUG-007 `CHARACTER:` / `SCENE:` enforcement clause from the "short (3 acts)" prompt into the "30 words (smoke, 1 act)" preset prompt in `nodes/story_orchestrator.py`. Add a unit test in `tests/test_dropdown_guardrails.py` (or a new `tests/test_30word_preset.py`) that asserts the compiled prompt contains the literal substrings `CHARACTER:` and `SCENE:` whenever `target_length.lower().startswith("30 words")`.

**Verify:** re-queue the 30-word smoke, expect ≥3 dialogue lines parsed, ≥1 scene, 2 named characters in `ledger.cast`.

### BUG-LOCAL-004 — OOM in script writer after parse-retry loop (peak 29.5 GB on 16 GB device)

**Fix:** in `nodes/story_orchestrator.py::write_script`, add (a) explicit `_LLM_CACHE` cleanup between the OpenClose synthesizer and the main script-writer call, (b) a hard parse-retry cap (`MAX_PARSE_RETRIES = 2`) so a runaway 0-line parse fails with a clear `MAX_PARSE_RETRIES_EXCEEDED` instead of OOMing on the fourth forward pass. Audit `_generate_with_llm`'s finally block: `torch.cuda.empty_cache()` is in place but the model's internal `past_key_values` may need an explicit `del` before it fires. Log `prompt_token_count` alongside `vram_snapshot("llm_generate_entry")` so future OOMs can be bisected.

**Verify:** re-queue 30-word smoke, expect peak_gb < 14.5 across the LLM ladder; if parse keeps failing, expect `MAX_PARSE_RETRIES_EXCEEDED` not `torch.OutOfMemoryError`.

### BUG-LOCAL-006 — `pytest tests/` hangs at session-start when ComfyUI is on the GPU

**Fix:** add `tests/conftest.py` with an autouse fixture that sets `CUDA_VISIBLE_DEVICES=""` for unit tests so collection never tries to bind to GPU. Optionally also lazy-import the heavy OTR modules from `__init__.py` so collection imports don't pull torch on path-only tests.

**Verify:** `python -m pytest tests/ -q` runs to completion in <60 s with ComfyUI Desktop up on `:8000`.

### BUG-LOCAL-003 — ComfyUI Desktop launch `HF_HOME` inheritance

**Fix:** add `scripts/run_comfyui.cmd` that reads `HF_HOME` + `HUGGINGFACE_HUB_CACHE` from `HKCU\Environment` via PowerShell + `[Environment]::GetEnvironmentVariable(...,'User')` and exports them into the launch shell before `start "" "...\ComfyUI.exe"`. Document in `README.md` under "Running ComfyUI Desktop" section. Source patch into Electron is out of scope (third-party).

**Verify:** kill ComfyUI, run `scripts/run_comfyui.cmd`, queue any episode that touches an HF model — expect `LLM tokenizer loaded from cache (no HTTP checks)` log line, no `local_files_only=True failed` errors.

### Sprint 1 acceptance

All four bugs marked `[FIXED]` in `docs/BUG_LOG.md`. `python -m pytest tests/` runs to completion. 30-word smoke produces a parseable script, reaches `master_mix_per_clip_mux`, ledger.json on disk.

---

## P0 — Live-test verification (already coded, awaits clean smoke + your manual cycle)

The work below is **observation against shipped code**, not new development. Items can be checked off only after Sprint 1 lands and a clean smoke completes.

### BUG-128/129 acceptance list (locked 2026-05-01)

1. No HuMo render job ever receives the radio still (assertion in dispatch — already in `nodes/batch_humo_render.py`).
2. ANNOUNCER clips l001 and l021 in a regression episode resolve to the same announcer portrait family — no generic-blonde drift.
3. `music_*` / standalone-`sfx` segments render through the static-video path (`ledger.clips[].source_kind == "static_ffmpeg"` vs `"humo"`).
4. Final mp4's extracted audio packet-hash matches procgen's audio stream byte-for-byte.
5. Peak VRAM stays below 14.5 GB.
6. Final video duration ≈ master mix duration (no `-shortest` truncation).
7. `tests/test_dropdown_guardrails.py`, `tests/test_core.py`, and the Bug Bible regression all pass.

### Live-test verification of the radio-coverage + bit-perfect-audio architecture

Confirmation items, not new design work:

- `ledger.lines[]` carries a `speaker_role` on every entry. No nulls, no missing rows. Roles: `character` / `announcer` / `music_open` / `music_close` / `music_inter` / `sfx`.
- `ledger.meta.audio_path_selected = "master_mix_per_clip_mux"` and `audio_path_reason = "ok (zero audio re-encodes downstream of SignalLostVideo)"`.
- BUG-129 routing (locked 2026-05-02 — see Architecture Truth section below):
  - `character` lines ONLY: `BatchHumoRender` dispatches HuMo with the cast portrait. Log line: `ref=full_env_NNNNN_.png source=ledger-cast-fresh` (or composite/portrait fallback).
  - `announcer` / `music_*` / standalone `sfx` lines: `BatchHumoRender` log line shows `SKIP HuMo (role=<role>, covered by VideoComposite static-radio fill)`. `is_never_humo_role()` short-circuits before any portrait lookup. No HuMo render fires for these.
  - NO log line should ever show `source=radio-still (...)` -- if one does, BUG-129 has regressed (`_RADIO_ROLES` was re-populated in `_otr_speaker_role.py`).
- `ledger.meta.radio_bookend_prompt_source` populated with the dynamic-build branch tag (e.g. `"dynamic (style='space opera epic')"`).
- BUG-129a static-fill fires for any line with no clip on disk. VideoComposite report includes `[<n_humo> humo + <n_static> static]` summary; expect static count > 0 if any music_*/sfx lines exist.
- BUG-128 tail-pad: VideoComposite report shows `tail-pad: +0.500s on <line_id>` after the pillarbox loop completes. The line_id matches the actual surviving last clip, not necessarily the last in the original timeline.
- Music tracks > 7s show up as multiple chunked entries (`music_open_001`, `music_open_002`, ...) — chunking math fired.
- ffprobe on the final mp4: video + audio streams both present; final mp4 audio `codec_name == aac` (passthrough from procgen); duration ≈ master mix duration.
- No `[VideoComposite] master_mix_per_clip_mux FAILED` in the log. With `strict_c7=True` (default), any failure would have raised.

### P1 audio pipeline — live-test verification (7 items, code-shipped on `v2.0-alpha`)

| Item | Confirmed in code | Awaits real-run observation |
|---|---|---|
| `min_line_count_per_character` self-critique guard | `nodes/story_orchestrator.py:6624` (default=2) | CRITIQUE_REJECTED log line on a real run where revision drops a character below 2 lines |
| Director JSON schema + validator | `_DIRECTOR_SCHEMA` at `:9239`, `_validate_director_plan` at `:10332` | DIRECTOR_SCHEMA_REPAIR log line on a malformed director plan |
| Length-sorted Bark batching | `nodes/batch_bark_generator.py:478` `@vram_sentinel` decorator | Throughput improvement vs unsorted baseline (10-15% expected) |
| VRAM-Sentinel decorator | `nodes/_vram_log.py::vram_sentinel`, used in 4 nodes | VRAM_SENTINEL_ENTRY/EXIT lines bracketing every decorated phase |
| High-creativity soak profile | "maximum chaos" in CREATIVITIES dropdown, temp 0.95 | One soak run on this tier; expect format-resilient output (no SFX loops, no [ACT N] injection) |
| Per-LLM-call VRAM snapshots | `vram_snapshot("llm_generate_entry/exit")` at every `_generate_with_llm` boundary | Snapshot lines visible in runtime log; peak summable across phases |
| ScriptCritic + Reviser advisory gate | `nodes/script_critic.py`, `block_on_reject` defaults False | 3-5 successful runs; flip to True after critic rate stabilizes |

### Open follow-ups (P2/P3-flavored, not blocking smoke green)

- **Audio codec ffprobe pre-flight** (P2) — confirm procgen audio stream is AAC before `-c:a copy` mux. One-line subprocess.run + assertion. Trivial; deferred until first run confirms procgen output codec.
- **Post-mux audio stream identity validation** (P3) — extract per-stream packet hash on procgen vs final mp4, fail tier on mismatch. Concrete proof of bit-identity. Ship as a separate validation node since the ffmpeg incantation needs care on Windows.
- **Low-motion observability for radio HuMo clips** (P3) — frame-difference metric on non-dialogue clips so "static" failures (Whisper OOD producing flat frames) surface as warnings instead of going unnoticed. No behavior change.
- **HuMo continuity layer for >7s narrative beats** (v2.0-beta) — hybrid blending across HuMo windows so 30s narrative beats don't show 7s jump-cuts. Decoupled from the audio path; gates "production unattended."
- **Per-scene environment FLUX still + LTX/zoompan animated background** (v2.0-beta) — bottom layer under the HuMo center pillarbox in dialogue windows.
- **Procgen-CRT lighten layer on top** (v2.0-beta) — audio-reactive scanlines + flicker as the SIGNAL LOST signature.
- **Drifted-filename smoke for BUG-LOCAL-118** — force an underscore-drifted .mp4 stem to verify the fallback chain fires before relying on it in a long soak.
- **Reconcile `16294df` ROADMAP-vs-git-log mismatch** — git log says "BUG-LOCAL-112 news-history reset"; prior narrative had it as "Wire ScriptCritic." Likely a rebase artifact. Decide canonical message before the next QA pass walks the history.

#### Hardware floor (locked 2026-04-25, do not relitigate)

- HuMo 14B fp8 e4m3fn scaled (Kijai) — `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`. Stock `UNETLoader`. Tuned by Kijai for 16 GB cards.
- Fallback ladder (kept on disk, do NOT delete): `humo_17B_fp8_e4m3fn.safetensors` (highest quality, slower ~6 min/clip), `Wan2_1-HuMo-17B_Q5_K_M.gguf` (speed-tuned).
- Stable shape: `length=97` (3.88 s @ 25 fps), 480x832, batch=1. Or `length=177` at 640x640 (7 s, OOD but verified working).
- Frame count must be `4n + 1`. Helper `humo_length_for_dur(dur_s)` snaps. Cap mirrored to `7.0s` in EpisodeAssembler music chunking.
- Per-step: 42 s. Per-clip: ~4:30 native, ~6:15 in TEST_humo. Cold load: ~50 s.

---

## Sprint 2 — harness + test-rot cleanup

Pre-existing test infrastructure rot blocking the regression contract from being measurable.

### BUG-LOCAL-001 — 8 stale test collectors importing `otr_v2.visual`

**Fix:** delete the 8 orphan test files (`tests/test_anchor_gen.py`, `test_camera_path_determinism.py`, `test_character_regression.py`, `test_cold_open_canary.py`, `test_episode_dry_run.py`, `test_lhm_monitor.py`, `test_three_minute_continuous.py`, `test_visual_phase_a.py`) OR rewrite them against the active video-stack code path. `otr_v2/visual/` was deleted in commit `7706660`; the test files were never updated. Triage during the cleanup: any test still asserting current behavior gets ported, the rest get deleted.

**Verify:** `python -m pytest tests/ --collect-only -q` reports zero collection errors.

### BUG-LOCAL-002 — `scripts/soak_operator.py` + `scripts/supersoaker.py` widget indices stale

**Fix:** delete both scripts. Replace with `scripts/otr_api.py` containing: (a) `patch_widget(workflow, node_id, widget_name, value)` that reads `/object_info` for the node's input order and writes by name (no fragile `WV_*` positional indices), (b) `workflow_to_api_prompt(workflow, schemas)` ported from soak_operator's working converter, (c) `submit_prompt(api_prompt) -> prompt_id` and `poll_history(prompt_id, timeout_s) -> status` helpers. Rewire `scripts/queue_smoke.py` onto `otr_api.py`.

**Verify:** running `scripts/queue_smoke.py` against `otr_scifi_16gb_full.json` produces a `/history` entry with `current_inputs` matching the patched values exactly (`target_words=30`, `num_characters=2`, `target_length="30 words (smoke, 1 act)"`).

### Triage 14 `tests/test_backend_dispatch.py` failures (logged, root cause not yet bisected)

Investigate during Sprint 2 — may be tied to the `otr_v2.visual` rot or to backend-dispatch refactors. Captured at baseline 2026-05-02: pytest -q output showed `FFFFFFFFFFFFFF` (14 failures) for this file. After Sprint 1's conftest CUDA-mask fixture is in place, re-run with `--tb=short` to capture exception types; fix or mark `xfail` with reason.

---

## Sprint 3 — MEGA-SPRINT: status (2026-05-02)

**Wiring SHIPPED on `v2.0-alpha`. Live acceptance BLOCKED on BUG-LOCAL-010 (pre-existing LLM-phase OOM regression).**

The Sprint 3 mega-sprint code is in place: LTX wiring (LowVRAMCheckpointLoader + OTR_BatchLTXRender), RTX VSR upscale (OTR_RTXUpscale), VideoComposite rewired downstream of LTX, anti-clobber + pipe-deadlock + cache-buster fixes from the round-robin consult. AST-clean, regression-clean (225 tests pass), workflow JSON valid, all three new nodes register, ComfyUI accepts the patched workflow at /prompt. The smoke OOM'd at OTR_LLMScriptWriter (BUG-LOCAL-010 in `docs/BUG_LOG.md`) -- the wiring code never executed because the LLM phase couldn't progress.

Once BUG-LOCAL-010 is fixed in a separate bisect window, re-queue the same workflow JSON and the S3.x acceptance bullets become directly observable. The full shipped scope and consult transcripts live in `docs/ROADMAP_HISTORY.md` under the 2026-05-02 mega-sprint entry; the Architecture Truth (locked 2026-05-02) is preserved there too.

**Locked-but-not-yet-verified S3.x acceptance bullets** (move to Done after a clean post-LLM-fix smoke):

- `ledger.clips[].source_kind == "ltx"` on announcer / music / sfx rows.
- VideoComposite report logs `[N humo + N ltx + N static]`.
- Pre-upscale ffprobe: width=832 height=480.
- Post-upscale ffprobe: width=1920 height=1080.
- Bypass path produces 832x480 unchanged.
- Audio byte-identical between pre- and post-upscale (stream MD5 match).
- Peak VRAM stays below 14.5 GB audio / 15.5 GB video.

### Architecture Truth (locked 2026-05-02 — do not relitigate)

The decisions below are settled. Any future session that tries to "improve" them must show a real-run failure first, not theory.

**Resolution policy — native 832x480 end-to-end:**
- `SignalLostVideo` procgen: 832x480 (canonical OTR landscape).
- `OTR_BatchLTXRender`: 832x480 (matches procgen + canvas; no upscale at composite time).
- `VideoComposite` canvas: 832x480 default (was 1920x1080 — corrected to native).
- `BatchHumoRender`: stays portrait pillarbox (480x832 internal, 832x480 letterboxed on canvas).
- `BatchFluxRender` cast portraits: 1024x1024 (FLUX-native square; HuMo `ref_image` is face-centered conditioning, not first-frame I2V).
- `BatchFluxRender` radio bookend: renders at **1248x720** then Lanczos-downscales to 832x480 in-node. Pixel budget locked — do NOT switch to 1344x768 or 1280x720.

**Role routing — `_NEVER_HUMO_ROLES` is the single source of truth:**
- Defined in `nodes/_otr_speaker_role.py` as a frozenset including `announcer`, `music_open`, `music_close`, `music_inter`, `sfx`. `_RADIO_ROLES` is empty (defense-in-depth).
- `BatchHumoRender` short-circuits via `is_never_humo_role()` BEFORE any portrait lookup. HuMo's `ref_image` is face-locked conditioning — it cannot animate the radio still as a non-face reference (verified in `comfy_extras/nodes_wan.py:1070-1108`).
- Coverage for non-character lines: `OTR_BatchLTXRender` (motion radio loops) takes precedence; `VideoComposite` static-radio fallback (BUG-129a) covers any line LTX skipped.

**LTX seamless-loop architecture — radio still as both start AND end keyframe:**
- `OTR_BatchLTXRender` uses `LTXVAddGuide` twice in the conditioning chain: `frame_idx=0` with strength 0.75 (start), `frame_idx=-1` with strength 0.6 (end). Both reference the same radio still PNG so the clip loops cleanly back to the bookend frame — no visible cut at loop boundary.
- Frame-count rule: `8n + 1` (LTX VAE temporal compression of 8). `LTX_MAX_FRAMES = 177` to match HuMo's verified ceiling on 16 GB; do NOT raise to 257 without a fresh VRAM smoke.
- Tiling: `LTX_TILE_SIZE=512`, `OVERLAP=64`, `TEMPORAL_SIZE=4096`, `TEMPORAL_OVERLAP=8` (Goofer-proven Blackwell params; see Jeffrey's `ComfyUI-Goofer` project).
- Strict teardown after the per-line loop: `unload_all_models()` + `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` in `finally`. LTX must fully release VRAM before the next pipeline stage.

**Loader policy — UNETLoader chain, NO C2 carve-out:**
- LTX 2B fp16 wires through `UNETLoader` + `CLIPLoader` (T5) + `VAELoader`. NOT `CheckpointLoaderSimple`.
- Reason: C2 stays intact (no carve-out drift); split-load lets ComfyUI offload T5 / VAE independently; bundled-load on a hot HuMo cache is the OOM shape C2 was written to prevent.

**DAG sequencing — `humo_clips_dir` optional dependency edge:**
- `OTR_BatchLTXRender` accepts an optional `humo_clips_dir` STRING input. When present, LTX waits for HuMo to finish writing its clips before starting — this is a pure dependency edge, not data flow. Sequential model load: HuMo loads → renders character clips → unloads → LTX loads → renders radio loops → unloads.
- LTX clips stamp `ledger.clips[].source_kind == "ltx"` (NOT `"humo"`). One-line clip-emit fix in `batch_ltx_render.py`; ship in the same commit as the wiring.

**Round-robin ladders (locked 2026-05-02):**
- OpenAI: `gpt-5.5` via `/v1/responses`. Gemini: `gemini-3.1-pro-preview-customtools`. NVIDIA: `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
- See `scripts/_consult_round_robin.py` + `scripts/_consult_nvidia.py`. Typed error logging (404/400/403/429 fall through; 401/transport re-raise).
- Internal QA only — never shipped output.

### S3.1 — Wire `OTR_BatchLTXRender` into `workflows/otr_scifi_16gb_full.json`

Node already built (`nodes/batch_ltx_render.py`, registered `__init__.py:155`). This is JSON wiring, not Python.

**Scope:**
1. Add `UNETLoader` + `CLIPLoader` (T5) + `VAELoader` triplet for LTX 2B fp16. Distinct `_meta.title` per loader.
2. `EpisodeAssembler.ledger_json` → `OTR_BatchLTXRender.ledger_json`.
3. `BatchHumoRender.clips_dir` → `OTR_BatchLTXRender.humo_clips_dir` (optional STRING dependency edge; sequencing only).
4. `OTR_BatchLTXRender.clips_dir` → `VideoComposite` as sibling source to HuMo's `clips_dir`. VideoComposite already merges by `line_id`.
5. Add `humo_clips_dir` optional STRING to `INPUT_TYPES` if missing.
6. Confirm clip-emit stamps `source_kind="ltx"`.

**Acceptance:**
- `ledger.clips[].source_kind == "ltx"` on announcer / music / sfx rows.
- Final mp4 shows LTX motion on those windows, looping seamlessly back to bookend.
- VideoComposite report logs `[N humo + N ltx + N static]`.
- Peak VRAM < 14.5 GB.
- Audio byte-identical to no-LTX baseline.

### S3.2 — FLUX radio bookend visual confirmation

Already coded. Observation only on next smoke.

**Acceptance:**
- Saved radio bookend PNG is exactly 832x480.
- Image is sharp (Lanczos downscale, not box / nearest).
- Same PNG hash feeds VideoComposite static fallback AND LTX start/end keyframes.

### S3.3 — 832x480 native end-to-end audit

**Acceptance:**
- `ffprobe` on the final composited mp4 (pre-upscale): `width=832 height=480` exactly.
- All segments (procgen / LTX / HuMo-pillarboxed / static-radio) composite onto 832x480 with no scale ops.

### S3.4 — RTX VSR ULTRA upscale to 1080p

Wire NVIDIA's RTX Video Super Resolution ULTRA ComfyUI node as the final stage after VideoComposite. ~0 GB VRAM (HW-accelerated via RTX driver), near-real-time. Output is the saved deliverable.

**Scope:**
1. Add RTX VSR ULTRA node to `workflows/otr_scifi_16gb_full.json` after VideoComposite's mp4 output.
2. Target resolution: 1920x1080 (16:9 from 832x480 source — the upscaler's standard 1080p mode).
3. Workflow toggle (Ctrl+B bypassable) so the user can disable per-run for raw 832x480 output.
4. Saved deliverable: `output/episodes_for_obs/<ep>/<ep>_1080p.mp4` when upscale on; `<ep>.mp4` when bypassed.

**Acceptance:**
- `ffprobe` on the upscaled mp4: `width=1920 height=1080`.
- Audio stream byte-identical to pre-upscale mp4 (RTX VSR is video-only; passthrough audio).
- Wall-clock for upscale stage: target near-real-time (≤ episode duration on a 5 min episode).
- Bypass path produces the original 832x480 mp4 unchanged.

**Deferred (NOT this sprint):** SeedVR2 v2.5 NVFP4 quality upscale lane — adds as second toggle once the RTX VSR fast path is validated. Wall-clock for SeedVR2 is ~2-3 h per 5 min episode, so it needs its own session and a dedicated VRAM smoke.

### B1 — Workflow JSON path scrub — VERIFIED SHIPPED 2026-05-02

Re-audit on 2026-05-02 found zero hardcoded user paths in `workflows/otr_scifi_16gb_full.json`, `workflows/otr_humo_smoke.json`, `workflows/otr_flux_smoke.json`, or `workflows/otr_humo_radio_experiment.json`. The "Resonance Chamber" `LoadAudio` widget on the smoke workflow already has an empty default. The portability concern is closed; everything goes through `OTR_OUTPUT_DIR` / `folder_paths.get_output_directory()` as designed.

The only remaining B1 work is documentation: `README.md` should explicitly state the env override pattern (`OTR_OUTPUT_DIR=/path/to/out`) for cloud / non-Windows installs.

---

## P2 — Continuity layer

Blocked on video-stack maturity. Design begins once stack empirics exist from the live-test cycle.

| Item | Summary |
|---|---|
| Scene-Geometry-Vault | Series-scale persistent geometry vault so Act 3's bridge matches Act 1's bridge across episodes. Seeded by FLUX anchor outputs |
| Style-Anchor cache | Reuse engine over the vault. Same geometry, N relight passes. `style_anchor_hash` in Director schema keys the split |
| Head-Start async pre-bake (Phase B.5) | Kick off VisualBridge on `outline_json` while ScriptWriter + Director run. Wall-clock win. Blocked on vault stability |
| ASCII sanitizer in prompt_compiler | Strip non-ASCII before Tencent text encoders. Preserve case. Collapse whitespace |
| Diff 3 — spine ledger-stamping + schema bump l3 → l4 | New ledger fields (`outline`, `beats[]`, `spine_meta`) + bundled metadata (`episode_title`, `meta.gen_params`, `meta.news_seed`, `meta.bug_109_retries`, `meta.word_ratio_pct`, `meta.title_source`, `meta.episode_breakdown_s`). See `docs/2026-04-29-spine-ledger-stamping-ticket.md`. **Unblocked by:** 2-3 real-episode runs of `voice_warnings[]` + Mistral-Nemo + Gemma 4 E4B both PASSing the LLM edge-case matrix + v2.0-alpha video stack feature-complete |

---

## P3 — Experiments & polish

| Item | Summary |
|---|---|
| `torch.compile` on Bark sub-models | `mode="reduce-overhead"` on semantic, coarse, fine acoustic. Needs isolated A/B timing; variable-length loops may fight the compiler |
| Skip/shorten Bark fine acoustic pass | Fine pass detail that AudioEnhance destroys via tape emu / LPF / Haas. Needs listening test, not spectrogram |
| `episode_title` socket on `OTR_SignalLostVideo` | Replace implicit `script_json` title-token read with explicit socket. v2.1 cleanup |
| News-history fuzzy dedup for syndication edge case | URL dedup catches direct repeats; same content with different URLs needs a fuzzy headline match |
| Empty-section pruning in filtered rubric | 1-character runs keep `### Ensemble-voice collapse` heading after all 3 rules filter out. Wastes tokens, doesn't break anything |
| VideoComposite cleanup deletion logic | Widget shipped (`cleanup_clips_after_assembly`), no-op for now. Wire actual deletion when stable enough to trust |
| Auto-update `OTR-CANON.md` from passing critic verdicts | `_canon_update()` helper exists in `script_critic.py` but is intentionally not called yet. Wire in once 3-5 runs of critic data accumulate |
| Tune `_MODEL_CONTEXT_CAPS` from real `OTR_VRAMContextTest` data | Currently conservative defaults |
| Update stale dropdown-guardrail tests in same commit as widget changes | Lesson from 2026-04-30: when widget mins/defaults change, update `tests/test_dropdown_guardrails.py` in the same commit so the test suite never drifts behind production |

---

## v2.0 release blockers

### B0 — Portrait pass polish (post BUG-LOCAL-081 verification)

**Status:** queued 2026-05-03 LATE EVENING. Discovered live in run `signal_lost_skindeep_microneedle_..._222516` after BUG-081's wiring fix landed and portraits actually rendered for the first time. Two cosmetic-but-real issues:

**B0.1 — Portraits duplicated into `stills/` as `full_env_NNNNN_.png`.** When I re-routed link 45 from `(Node 23 → Node 24 UnloadAll)` to `(Node 59 → Node 24 UnloadAll)`, the downstream `OTR_SaveToEpisodeWorkspace` (Node 25) inherited the new IMAGE source. It now writes the portrait_batch tensors out as `stills/full_env_00001-3_.png` thinking they're env stills. Real portraits are still correctly at `portraits/c0X_portrait.png`, so HuMo's tier-1 lookup is unaffected, but it's ~6 MB of duplicate data per episode with misleading filenames. **Fix options:** (a) detect the source node in SaveToEpisodeWorkspace and route portrait_batch tensors to `portraits/` instead of `stills/`, OR (b) leave SaveToEpisodeWorkspace wired only to genuine env-still sources and let the portrait node manage its own saves (it already does — `<ep>/portraits/<char_id>_portrait.png`). Option (b) is cleaner: just unwire link 46 from UnloadAll → Node 25 when env stills are skipped.

**B0.2 — `skip_announcer=True` widget never fires.** Cast field `cast[i].speaker_role` is empty in the ledger (`role=` for all entries — confirmed via PowerShell on the 222516 run). The portrait node's announcer-skip logic has nothing to match against, so it renders a portrait for ANNOUNCER (c01) too. Cost: ~10s extra FLUX time + one unused 1024x1024 PNG per episode. **Fix:** either (a) populate `speaker_role` field on cast at LLMDirector time (canonical fix; benefits any future role-aware logic), OR (b) fall back to `name.upper() == "ANNOUNCER"` substring match in the portrait node when `speaker_role` is empty (cheap defensive fix). Probably both — populate the field upstream AND keep the substring fallback as defense-in-depth.

**Why release blocker:** v2.0 ships when the per-episode workspace is clean. Phantom env stills + unused announcer portrait are both visible to anyone who opens the workspace folder, and both make the JSON layout harder to reason about during debugging. Cheap to fix once HuMo soak completes.

### B1 — Generic / relative paths (no Windows-hardcoded absolutes)

**Status:** Step 0 paths refactor shipped 2026-04-28 (`70f4a5c`) — `nodes/_otr_paths.py` helper module with resolution order: `OTR_OUTPUT_DIR` env → `folder_paths.get_output_directory()` → walk-up to ComfyUI root → cwd fallback. ~12-15 hardcoded `r"C:\Users\jeffr\..."` strings replaced.

**Remaining:** see Sprint 3 above.

**Why it's a release blocker:** every Windows-absolute path is a portability blocker for any non-Jeffrey user (Linux/Mac/RunPod/cloud) and a portability blocker for the 8GB-tier work. v2.0 cannot ship while paths are user-and-OS-specific.

### B2 — 8GB-VRAM-class user experience

**Stance:** v2.0 doesn't release until 8GB-class users get an enhanced visual output too.

**Architecture (Locked 2026-04-30):** Single master JSON with bypassable video-stack groups. Shared audio chain → procgen, then multiple side-by-side render groups — each group bypassable via Ctrl+B. Final VideoComposite takes whichever group is active.

**Stance:** 8 GB tier does NOT get "full animated backgrounds" or generative character video. They get an **enhanced visual mode** optimized for their VRAM limits: still + parallax + interpolation for motion, with optional Wan 2.2 5B B-roll for users who want to gamble on render time.

**Do NOT offer:** HuMo, LTX-2, LTX-2.3, or 14B Wan to 8 GB users. The support burden and OOM risk are too high.

**Locked picks (2026-04-30, after evaluating LTX 2.3, LTX-2 19B, ERNIE Image, NVIDIA CES 2026 NVFP4, and round-robin consult on background models):**

| Component | 16 GB tier | 8 GB tier | Why |
|---|---|---|---|
| **Stills** | **NVFP4 FLUX.2** (RTX 50 Series, ~5 GB; falls back to FLUX-fp8 ~12 GB if NVFP4 unavailable) | **FLUX.1-dev Q4_K_S** (city96 GGUF, ~5-6 GB) | FLUX is the visual anchor for both tiers. NVFP4 is the new official quantization NVIDIA announced at CES 2026 — 3x faster, 60% less VRAM than fp8 on RTX 50 Series. Q4_K_S is the safe 8GB GGUF option. |
| **Motion** | **HuMo 14B fp8** + master_mix_per_clip_mux + LTXV background layer | **Still + Parallax + Interpolation** (deterministic Ken-Burns + frame interp on FLUX stills) | HuMo for 16 GB character lip-sync. 8 GB gets safest, fastest, most deterministic motion — high quality, zero VRAM spikes, no diffusion-per-beat. |
| **Optional B-roll** | n/a (HuMo covers all character beats; LTXV covers backgrounds) | **Wan 2.2 5B TI2V** (native ComfyUI template, optional toggle) | Strictly optional B-roll lane for 8 GB users who want generative motion on non-dialogue beats. Slow, not guaranteed; document expectation upfront. |
| **Upscale — Speed option** | **RTX Video Super Resolution ULTRA** (~0 GB, HW-accelerated, target 4K, real-time) | **RTX VSR ULTRA** (same node, same zero VRAM cost) | Default. NVIDIA CES 2026 ComfyUI node. Whole-episode upscale, near-real-time, ships with RTX driver. Use this when speed matters more than maximum diffusion-based detail. |
| **Upscale — Quality option** | **SeedVR2 v2.5 NVFP4** (7B, ~6 GB on RTX 50 NVFP4, ~78 s per 65-frame 720p→1080p clip — full episode ~2-3 h on a 5-min run) | not viable on 8 GB | Whole-episode upscale via the diffusion upscaler. Quality king for AI-generated content. SeedVR2 v2.5 NVFP4 support landed via [PR #486](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/pull/486). On RTX 50 NVFP4: 3x faster + 60% less VRAM vs fp16 baseline. |

Both upscale options run on the WHOLE episode (every clip), exposed as a workflow toggle so the user picks per-run. Default = RTX VSR (fast). Quality run = SeedVR2 v2.5 (slow but state-of-the-art on AI content). Either can be toggled off entirely for raw 480x832 / 720p output.
| **TTS / Audio** | Bark + Kokoro + MusicGen + AudioGen → master mix (canonical) | Same | OTR's TTS pipeline is the project. NEVER replaced by model-internal A/V generation (LTX-2's prompt-driven audio is a paradigm mismatch). |

**Picks REJECTED after evaluation:**
- **LTX 2.3 22B distilled** — smallest GGUF (Q5_K_M ~14 GB) doesn't fit 8GB; "distilled" = step-distilled NOT param-distilled; 22B is the only param size Lightricks publishes.
- **LTX-2 19B distilled (Kijai)** — Q4_K_M ~12 GB still over 8GB.
- **LTX-2's built-in audio for character dialogue** — model GENERATES speech from text prompt, doesn't accept input audio. Replacing OTR's TTS would lose Bark/Kokoro voice control + script→voice mapping. Unless an audio-input ControlNet/LoRA ships, LTX-2 is visuals-only for OTR.
- **Wan 2.2 14B GGUF Q3/Q4** — RAM-thrashes on Windows under aggressive offload; support-ticket bait.
- **FLUX.1-dev Q5_K_S** — over 8GB budget once T5 + VAE + OS overhead added.
- **Z-Image Turbo / PixArt-Sigma** — weaker prompt-adherence than FLUX for radio-drama series consistency.
- **ERNIE Image 8B** — parked pending model card review (Jeffrey to provide spec link).

**Acceptance for 8GB path:**
- Full audio pipeline (LLM + Bark + AudioGen + MusicGen + SceneSequencer + EpisodeAssembler) — same as 16 GB.
- SignalLostVideo procgen base — same.
- Stills via FLUX.1-dev Q4_K_S; video via Wan 2.2 5B (atmospheric B-roll / scene loops).
- Final mp4 lands in `output/episodes_for_obs/<ep>/<ep>.mp4` same as 16 GB.
- Wall-clock expectation: FLUX still ~45-90 s/still, Wan 5B clip ~4-8 min/clip (significantly slower than 16GB tier; document upfront).

**Distribution requirements before tagging v2.0:**
- Pin exact ComfyUI version + GGUF model versions in README; include checksums for the GGUF files.
- README must set time expectations explicitly so 8GB users don't think the run hung.
- Both tier workflows live in the same JSON; the README screenshot shows the "8GB mode" group toggles to enable.

**Related:** flip default `optimization_profile` to `Pro (Ultra Quality)` once 16 GB FULL has shipped clean — Jeffrey: *"I almost feel we should default to Pro Ultra"*.

---

## v2.0-beta candidates

### Animated backgrounds (3-layer composite, 16 GB only)

Promotes the current 2-layer composite (procgen-base + HuMo-overlay, BUG-092) into a 3-layer composite. **8 GB tier does NOT get a background layer** (procgen sides only — keeps 8 GB lean).

```
TOP:    Procgen / CRT audio-reactive overlay -- `lighten` blend, ~0.3 opacity
MID:    HuMo lip-sync portrait -- center pillarbox during dialogue, opaque
BOTTOM: Animated background (model TBD) -- full canvas, opaque
```

**Why CRT-on-top in lighten mode is more truthful:** a failing broadcast's scanlines + audio-peak flicker should cover the WHOLE frame including the speaker's face — the interference doesn't politely stop at the pillarbox edges. Lighten mode takes max(CRT, underlying) per channel so artifacts ride on top without erasing detail.

**Render budget (locked 2026-04-29 PM — render-native + slow-mo, model-agnostic):**
- Render at the chosen model's native fps, then slow to 12 fps via ffmpeg `setpts=PTS*2,fps=12`. The slow-mo IS the SIGNAL LOST broadcast-degraded aesthetic.
- 1-2 clips per SCENE (not per shot). Loop across the scene's duration via `-stream_loop -1` with optional crossfade or ping-pong reverse.
- For LTX: 193 frames per clip = 8 sec native = 16 sec apparent after 2× slow-mo. LTX uses 8× temporal VAE compression so frame counts must be `8n + 1`. 193 = 24*8 + 1. Max 257.
- For Wan: frame-count math TBD per model card during implementation.
- Distilled 4-8 steps (default 6 for LTX; Wan TBD).

**Per-episode wall-clock estimate:** smoke (1 scene) ~50 s; short (3 scenes) ~2.5 min; medium (5 scenes) ~4 min. Negligible vs HuMo (~10 min per dialogue line).

**Frame-count widget shape (model-specific names locked at impl):**
```
frames:         dropdown of valid frame counts for chosen model
steps:          distilled step dropdown
slow_mo_factor: float (default 2.0)
target_fps:    int (default 12)
```

#### Background-model selection — LOCKED 2026-04-30

**Round-robin verdict:** Keep the background layer cheap, stable, and visually appropriate for being blurred/degraded under the HuMo dialogue pillarbox. Foundation-model chasing for a layer that gets slowed to 12 fps and composited under a foreground is the wrong engineering bet.

| Candidate | Size on disk | Peak VRAM | Role | Verdict |
|---|---|---|---|---|
| **LTXV 0.9.x 2B distilled fp16** | ~5 GB | ~7-8 GB w/ VAE | **Default (16 GB)** | **LOCK.** Fits the degraded-broadcast aesthetic perfectly. 193 frames (8n+1), 4-8 distilled steps, then ffmpeg slow-mo to 12 fps. Both ChatGPT + Gemini endorsed. |
| **Still + Parallax + Interpolation** | ~5-6 GB (FLUX still only) | ~7 GB | **Default (8 GB)** | **PLAN B / 8 GB PATH.** Lowest risk, highly deterministic Ken-Burns + frame interp on FLUX stills. Likely enough motion for radio drama without diffusion overhead. ChatGPT's smallest-change biggest-payoff suggestion. |
| **Wan 2.2 5B native FP8** | ~6 GB | ~8-9 GB w/ VAE | Fallback | Keep as a fallback if LTXV introduces unacceptable motion artifacts during live-test. Also serves 8 GB tier as optional B-roll lane. |
| **LTX-2 19B / 2.3 22B GGUF** | 12-14 GB | 14-17 GB w/ VAE decode spike | **REJECTED** | **DO NOT USE FOR BACKGROUNDS.** Audio-video foundation models are a paradigm mismatch and too heavy for a sidecar background layer on a 16 GB VRAM ceiling. VAE temporal decode adds 2-3 GB at decode → OOM. ChatGPT also flagged "1.1" version label as community packaging, not a confirmed upstream tag. |
| **HunyuanVideo distilled** | varies | varies | Not recommended | ChatGPT mentions; operationally heavier than LTXV. Skip. |
| **Stable Video 3 (8B)** | unknown | unknown | Suspect | NVIDIA round suggested with hallucinated specifics; do not pursue without independent verification. |

**Quantization gotchas on Blackwell sm_120 (both ChatGPT + Gemini):** Don't depend on FP8 / NVFP4 paths for video models yet — Blackwell support arrives in layers (PyTorch → CUDA kernels → custom ops → quant backends → custom nodes), and ComfyUI custom video nodes are exactly where "advertised support" and "production-safe support" diverge. Prefer fp16 / bf16 paths that already work.

**Pin format locked:**
```yaml
background_video:
  family: "ltxv"
  upstream_repo: "Lightricks/LTX-Video"
  model_file: "<exact 0.9.x safetensors filename to confirm at impl>"
  upstream_commit: "<HF commit SHA at impl>"
  comfyui_node_repo: "<exact custom node repo>"
  comfyui_node_commit: "<SHA at impl>"
  precision: "fp16"   # prefer over fp8 for stability on this layer
  frames_rule: "8n+1"
  target_frames: 193
  sampler_steps: 6
  postprocess: "setpts=PTS*2,fps=12"
```

#### TTS palette expansion — LOCKED LADDER 2026-04-30

NOT replacing the canonical pipeline (Bark + Kokoro + MusicGen + AudioGen → master mix). EXPANDING the per-character voice palette. Round-robin consult 2026-04-30 produced strong agreement on direction.

**Production add-order ladder (Parler-TTS REJECTED — owner pref; vintage sound stays in the deterministic DSP chain):**

| Priority | Engine | License | Peak VRAM | C7-deterministic? | Verdict |
|---|---|---|---|---|---|
| **1** | **Kokoro** (current) | MIT | ~1 GB | Yes | **KEEP.** Undisputed workhorse for strict lip-sync and clean narration. Gemini calls "undisputed king of low-VRAM deterministic phoneme TTS." |
| **2** | **Bark** (current) | MIT | ~6 GB | Yes (vram_sentinel + length-sort batching shipped) | **KEEP.** Unmatched for period vibe, character texture, and emotional color. |
| **3** | **CosyVoice 2** | Apache-2.0 | ~3-4 GB | Yes (flow-matching ODE solver + fixed seed = byte-identical) | **ADD NEXT.** Strongest production candidate for expanding the dramatic voice palette. Both ChatGPT + Gemini endorsed. |
| **4** | **Piper** | MIT | ~1 GB | Yes | **8 GB / UTILITY FALLBACK.** Tiny, deterministic, fast. Ideal for minor announcer roles or 8 GB emergency fallback. ChatGPT's recommendation for utility voices. |
| **5** | **CosyVoice 3** | Apache-2.0 | unknown | Unverified | **RESEARCH LANE.** Both flag as too new for production. Needs strict C7 hash proof before promotion. NVIDIA round claimed v3.2.1 production-ready with hallucinated commit SHA; ignore that signal. |
| **6** | **Qwen3-TTS** | needs license audit | unknown | **C7 RISK** | **RESEARCH LANE.** Gemini flags autoregressive + flow-matching hybrid as hard to make byte-identical. Highly expressive but requires deep C7 verification before any merge. |

**REJECTED candidates:**
- **Parler-TTS Mini** — owner preference; vintage broadcast sound stays in the deterministic DSP mastering chain (band-limit + tube saturation + plate flavor + noise floor + AM EQ).
- **Fish Speech** — license incompatible with MIT downstream.
- **XTTS / Tortoise / StyleTTS family** — license ambiguity, Windows friction, C7 determinism risk. Evaluate only if a specific gap appears that priorities 1-4 don't fill.

**C7 qualification protocol (apply to any new TTS before merge):**
1. Same prompt + same seed + same model revision + same driver/torch/CUDA/cuDNN + same batch size + same output format.
2. Run 10 repeated generations across cold start, warm start, and process restarts.
3. Hash final WAV bytes. If any hashes differ → engine is NOT qualified for OTR.

**Period-style controls — locked position:** Vintage broadcast sound lives in the deterministic DSP mastering chain (band-limit, tube saturation, plate flavor, noise floor, AM EQ shaping). TTS engines provide diction / cadence / timbre baseline only. Any model offering "1940s radio" as a text-prompted style is out of scope — we own the vintage sound, the model doesn't get to drift it.

**Pin format to lock once each engine ships:**
```yaml
tts_palette:
  engines:
    - name: "kokoro" / "bark" / "cosyvoice2" / "piper"
      upstream_repo: "<exact repo>"
      model_revision: "<tag/SHA>"
      tokenizer_revision: "<tag/SHA>"
      vocoder_revision: "<tag/SHA>"
      decode_mode: "<greedy|ode_solver|other>"
      sample_rate: "<Hz>"
      wav_hash_test: true
      role: "<character|announcer|narrator|utility>"
```

#### LLM palette expansion — QUEUED 2026-05-03 EVENING (paired with CosyVoice 2 add)

Same shape as the TTS ladder above: NOT replacing the canonical script-writer (Mistral-Nemo 12B), EXPANDING the per-role LLM palette so the writer pool can be voiced for tone (period radio drama, hard-boiled detective, broadcast announcer) instead of one general-purpose model carrying everything. Queued for the same beta cycle as the CosyVoice 2 TTS add — both are voice/character expansion work, both gate on the same C7 + VRAM verification protocol.

**Production add-order ladder (writer lane):**

| Priority | Model | License | Peak VRAM (est) | C7-deterministic? | Verdict |
|---|---|---|---|---|---|
| **1** | **Mistral-Nemo 12B** (current canonical) | Apache-2.0 | ~22.8 GB FP16 / ~7-8 GB int4 | Yes (deterministic with fixed seed + temperature 0) | **KEEP.** Default story-writer per `otr_scifi_16gb_full.json`. Don't replace. |
| **2** | **talkie-lm/talkie-1930-13b-it** (instruct variant — supersedes the earlier `destnyrr/talkie-1930-13b-base-gptq-int4` queue entry) | needs license audit | ~7-8 GB (13B int4) | needs verification | **PROMOTE TO NEXT-UP.** Instruct-tuned 1930s broadcast LLM. The instruct variant is what's actively trending on HF; better fit than the raw base for OTR's prompt-engineered writer prompts. Pair-add with CosyVoice 2 in the same beta cycle. |
| **3** | **Qwen/Qwen3.6-27B** (or `unsloth/Qwen3.6-27B-GGUF` for the pre-quantized GGUF) | Apache-2.0 | ~7 GB int4 GPTQ / ~6 GB GGUF Q4 | needs verification | **TIER-1 ALTERNATIVE.** Qwen3 series has top-tier creative-writing reputation; legitimately could replace Mistral-Nemo as primary writer if A/B test on the same prompt favors it. Unsloth GGUF quant means zero DIY quantization work. |

**Production add-order ladder (utility lane — NEW 2026-05-03 EVENING):**

Separate from the writer palette. Utility LLMs are for tasks where deterministic instruction-following + small footprint + Apache license matter MORE than period prose flavor. Capabilities target: summarization, structured extraction, classification, function-calling, normalization passes.

| Priority | Model | License | Peak VRAM (est) | Use case | Verdict |
|---|---|---|---|---|---|
| **1** | **ibm-granite/granite-4.1-8b** | Apache-2.0 (verified 2026-05-03) | ~5 GB int4 / ~16 GB BF16 (8.79B params, 17.5 GB on disk) | Title compression from news_seed (currently the news_seed_fallback path produces 80-char filename slugs like `signal_lost_what_a_decade_of_gene_therapy_research_f_...` — Granite would compress to 4-word punchy title); cast normalize pass (queued LLM cleanup); treatment.txt structured extraction; ledger forensics tool-use | **TIER-1.** IBM's "diverse domains, including business applications" framing is the OPPOSITE of what we want for the writer lane, but the EXACT shape we want for utility tasks. Strong instruction-following + tool-use + function-calling. |

**C7 qualification protocol (apply to any new LLM before merge):**
1. Same prompt + same seed + temperature 0 + same model revision + same tokenizer revision + same draft length cap.
2. Run 10 repeated generations across cold start, warm start, and process restarts.
3. Hash final draft text bytes. If any hashes differ at temperature 0 → engine is NOT qualified for OTR.
4. **Period-tone smoke pass:** generate 5 short scripts with the writer prompt and a fixed seed; spot-check that the model does NOT slip modern slang, modern brand names, or post-1950 cultural references into a script tagged for the 1940s setting. Failure mode: model that ignores period framing and emits anachronisms gets demoted to RESEARCH LANE pending prompt-engineering work.

**Pin format to lock once each LLM ships:**
```yaml
llm_palette:
  writers:
    - name: "mistral-nemo-12b" / "talkie-1930-13b-it" / "qwen3.6-27b-gguf-q4"
      upstream_repo: "<exact HF repo>"
      model_revision: "<tag/SHA>"
      tokenizer_revision: "<tag/SHA>"
      quant_format: "<fp16|int4-gptq|gguf-q4|int8|...>"
      context_cap: "<tokens>"
      temperature_default: 0.0
      draft_hash_test: true
      role: "<canonical|period-broadcast|hardboiled|announcer-narration|...>"
  utility:
    - name: "granite-4.1-8b"
      upstream_repo: "ibm-granite/granite-4.1-8b"
      model_revision: "<tag/SHA>"
      tokenizer_revision: "<tag/SHA>"
      quant_format: "int4-gptq | int8 | bf16"
      context_cap: "<tokens>"
      temperature_default: 0.0
      draft_hash_test: true
      role: "<title-compress|cast-normalize|treatment-extract|ledger-forensics|...>"
```

**Wired-in alongside what:** the writer-profile dropdown in `LLMScriptWriter` would gain new options (`Talkie-1930-it (Period Broadcast)`, `Qwen3.6-27B (Creative Alternative)`) that load via the same loader path used by Mistral-Nemo. Switch is per-episode at queue time, not per-line. The utility lane (Granite 4.1 8B) wires into a NEW node `LLMUtilityRunner` (or extends an existing utility hook) for the small structured-output tasks that don't need a full writer; it co-loads alongside the writer profile because their VRAM footprints (5 GB + 7-8 GB int4) sum to ~13 GB, comfortably under the 14.5 GB ceiling. CosyVoice 2 add (TTS priority 3 above) is independent at the audio engine layer; all three (writer-add, utility-add, TTS-add) can ship in the same v2.0-beta cut without touching each other's code paths.

**Rejected from this round (size or alignment mismatch):**
- **Anything 100B+** (DeepSeek-V4-Pro 862B, MiMo-V2.5 311B, Kimi-K2.6 1.1T, Mistral-Medium-128B, Ling-1T) — exceeds 16 GB VRAM even at int4
- **Multimodal `Image-Text-to-Text`** variants (Qwen image families, Gemma-4 31B-it has IMG variants) — wrong tool for text-only OTR writing
- **`text-to-image` / `text-to-video`** (SeeSee21, SulphurAI) — wrong domain entirely
- **`HauhauCS/Qwen3.6-27B-Uncensored-...-Aggressive`** — explicitly conflicts with OTR's safe-for-work / no-profanity content standard
- **`google/gemma-4-31B-it`** + **`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning`** — both interesting Tier-2 candidates but deferred until after the Tier-1 writer A/B (Mistral-Nemo vs Talkie-1930-it vs Qwen3.6-27B) lands a winner. Re-evaluate then.
- **`ibm-granite/granite-4.1-30b`** — bigger Granite sibling loses the small-footprint advantage that makes the 8B compelling for the utility lane.

**Defer to v2.0-beta** — same trigger as the TTS expansion. Land BUG-LOCAL-031+ first, then the v2.0-alpha → v2.0-beta cut, then this palette work in beta cycle 1.

### LLM character normalize pass

Currently cast cleanup is two layers: (1) regex blocklist `_SFX_CAST_BLOCKLIST_PATTERNS` (BUG-091 + BUG-097), (2) fuzzy `_consolidate_similar_cast_rows_with_aliases` (BUG-098). Both deterministic, limited to KNOWN patterns. An LLM-based normalize after fuzzy dedup could catch semantic aliases neither layer sees: `KEVIN VOICEOVER` → `KEVIN STENDAHL`, `(captain)` lowercase → `CAPTAIN`, `DR. AMELIA HARTFIELD` → `AMELIA`.

**Constraints:** conservative prompt ("ONLY merge when names CLEARLY refer to the same character; when in doubt, do NOT merge"); hard-cap merge-set ≤50% of cast (flags hallucination); only run on `optimization_profile = "Pro (Ultra Quality)"` (adds 2-5 min wall time); feed first 1500 chars of script_text + first sentence of each character's first line.

**Defer to v2.0-beta** — by then we have a real corpus of run logs showing common emission patterns, so the prompt can be data-informed instead of guesswork-driven.

---

## v2.1 candidates

### Configurable show name (replace hardcoded "Signal Lost")

**Status:** queued 2026-05-03 LATE EVENING. Real-shippability blocker — anyone wanting to fork OTR for their own show ("Twilight Zone", "Lights Out", "The Hitchhiker") currently has to grep + sed across the codebase.

**Sites that hardcode "Signal Lost":**
- `nodes/video_engine.py:1484` — `out_path = ... f"signal_lost_{safe_title}_{ts}.mp4"` (filename prefix)
- `nodes/story_orchestrator.py:9089` — announcer closing line `"This has been Signal Lost. {episode_title}. Stay safe."`
- `nodes/story_orchestrator.py:6216` — last-resort title fallback `"Signal Lost Transmission {ts}"`
- `nodes/video_engine.py:1322` — last-resort title fallback `"Signal Lost {ts}"`
- (probably more — full grep needed before scoping)

**Fix architecture:** add a `show_name` field to `ProjectState` (already loaded by Director + ScriptWriter), plumb it through everywhere the literal `"Signal Lost"` appears. Default to `"Signal Lost"` for backwards-compat. Surface as a top-level widget on `OTR_ProjectStateLoader` (or whichever node currently owns project_state) so users can flip it without code edits.

**Verify:** grep for `Signal Lost` returns ZERO source-file hits after the change (all references go through `project_state.show_name`); test fixture with `show_name="Twilight Zone"` produces filenames like `twilight_zone_<title>_<ts>.mp4` and announcer closings like "This has been Twilight Zone."

**Why v2.1 not v2.0:** v2.0 ships as branded "Signal Lost" — that's fine for the launch. The brand-portability work is its own scoped sprint and shouldn't gate the v2.0 release.

### Per-shot / per-scene face variation via PuLID-FLUX

**Status:** queued 2026-05-03 EVENING. **Defer to v2.1** — landed AFTER v2.0 ships clean.

**Context:** v2.0 ships with the BUG-LOCAL-078 portrait pass (`OTR_BatchFluxPortraitRender`). Each character gets ONE canonical portrait per episode — fully dynamic, fresh on every run, no stored stock characters, no cross-episode face library. HuMo references that single portrait for every line of that character's dialogue. Within-episode consistency goes from ~5/10 (env-still tier-4 fallback) to ~9/10 (single canonical portrait). For an anthology series with fresh cast every episode, single-portrait-per-character is the correct architecture. **v2.1 should NOT change that default.**

**What v2.1 ADDS** (opt-in, not default):

Per-shot or per-scene FACE VARIATION for the same character — same identity, different STATE. The character is recognizably the same person across the whole episode (single PuLID identity reference), but each shot/scene can render that face in a different STATE that reflects the story:

- Scene 1: clean, composed (just entered the scenario)
- Scene 3: sweat, dirt, dilated pupils (mid-crisis)
- Scene 5: bloodied, exhausted, scarred (post-climax)
- Scene 6: composed but visibly changed (denouement)

PuLID-FLUX is the canonical solution: it extracts the FACE IDENTITY from a reference image and re-renders it under a new prompt. So the workflow is:

```
ROUND 1 (text-only FLUX): render the character's seed portrait from
        ledger.cast[i].appearance text. This becomes the IDENTITY ANCHOR.
        Same as v2.0's portrait pass. Save to portraits/<char>_seed.png.

ROUND 2..N (PuLID-FLUX, per shot or per scene):
        For each ledger.scenes[i] OR ledger.shots[i] entry, render a
        new face image using:
          - PuLID identity reference  =  portraits/<char>_seed.png
          - prompt                    =  v2.0's portrait composition base
                                          + scene/shot-specific state
                                          modifier (sweat, blood, etc)
        Save to portraits/<char>_scene{N}.png.

        State modifier sources, in priority order:
          (a) ledger.scenes[i].character_state[char_id] (if LLMDirector
              populates it -- new ledger field for v2.1)
          (b) ledger.shots[i].mood + character_position_in_arc
          (c) ledger.lines[i].traits (per-line emotion tag)
          (d) Default ladder by scene index: scene_1=clean,
              mid_scene=mid, last_scene=worn

HuMo's portrait_path lookup (v2.1 update):
        Currently picks ledger.cast[i].portrait_path (single canonical).
        v2.1 adds tier 0: ledger.scenes[scene_id].cast_portraits[char_id]
        if populated, falls back to tier 1 (cast canonical) otherwise.
```

**What this BUYS** (per-shot variation locked to single identity):
- Story-driven visual evolution. The character ages / accumulates damage /
  emotionally shifts as the episode progresses, but it's recognizably them.
- Higher emotional payoff in the final montage. Scene 1 vs scene 5 of the
  same character looks DIFFERENT (right) instead of IDENTICAL (wrong, but
  what v2.0 ships).
- Anthology format unchanged. No persistent face library. Each episode
  builds its own seed + variations from scratch and discards them at the
  next run.

**What this COSTS:**
- PuLID-FLUX install: `ComfyUI-PuLID-Flux-Enhanced` custom node + ~1-2 GB
  PuLID model weights + ~250 MB InsightFace `antelopev2` face detection.
- VRAM: ~3 GB extra on top of FLUX dev fp8 (~12 GB). Total ~15 GB. Tight
  but fits the 16 GB ceiling.
- Render time: 2x portrait time per character per scene/shot variant.
  For a 5-scene episode with 3 characters: 3 seed portraits + 15 scene
  variants = 18 FLUX renders, ~3-5 minutes added per episode (vs v2.0's
  ~30-60 sec for the seed pass alone).
- Code: extend `OTR_BatchFluxPortraitRender` with a v2 mode that loops
  scenes after the seed pass; new ledger field `cast_portraits` per scene
  populated by LLMDirector; HuMo's `_find_portrait` updated to prefer
  per-scene over canonical when present. Estimated ~4-6 hours of code +
  test work.

**Acceptance criteria for v2.1 ship:**
1. Single full episode renders with per-scene face variation enabled.
2. Visible state shift across scenes (verified by ffprobe + manual frame
   inspection — scene 1 portrait vs scene 5 portrait should be the SAME
   FACE but DIFFERENT STATE).
3. C7 audio byte-identity holds (visual changes don't touch the audio path).
4. Performance budget: <5 minutes added per episode at 5 scenes / 3
   characters.
5. Toggle defaults to OFF so v2.0 single-portrait behavior is the default.
   Users opt in by flipping a widget.

**Deferred from this lane (separate v2.x work, NOT in v2.1 scope):**
- Cross-episode face registry (recurring characters, stored library) —
  conflicts with anthology design philosophy; revisit only if OTR pivots
  to a serialized format.
- Face-locking on HuMo's OUTPUT video (not just the portrait input) —
  much harder, requires video-level identity injection. HuMo's intrinsic
  per-frame variation is acceptable for now.
- Multiple portrait ANGLES per character (frontal + 3/4 + side) — would
  require HuMo upgrade to consume multiple references. Out of scope.

---

## Discarded — do not revisit

- Flash Attention 2/3 on sm_120
- Pinning torch < 2.10
- Weight streaming from system RAM via ComfyUI-Manager
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)
- Speculating on unreleased Visual unified latent space
- Visual 2.0 Gate 0 probe (WorldMirror / HunyuanWorld / WorldStereo / WorldPlay-5B) — retired 2026-04-17. VisualBridge + Poll + Renderer harness stays as the harness; the backends are the active video stack
- `ComfyUI-*-Wrapper` repos as primary runtime (pull flash_attn, wrap overhead)
- v2v chaining (deep-fries output by 3rd generation)
- Single-image LoRA training on the laptop during live orchestration (thrash risk)
- SD 1.5 anchors as final style — pivoted to FLUX-native
- Subprocess pattern for HuMo orchestration (BUG-076 OTR_PostAudioVideoPipeline + render_humo_batch.py orchestrator) — superseded 2026-04-27 by in-graph nodes (BUG-078). Subprocess scripts remain as ad-hoc CLI smoke tools but the production path is in-graph. `OTR_PostAudioVideoPipeline` class kept registered with `(retired)` title for back-compat with old workflow JSONs
- Blanket `git clean -fX` — the existing `scripts/_*.py` ignore is too broad and would nuke `_consult_*.py`, `yoga_watchdog.py`, and other legitimately-local files. Use targeted `git clean -fX -- <pattern>` instead

---

## References

- `CLAUDE.md` — project rules, platform pins, Desktop Commander git pattern
- `docs/BUG_LOG.md` — live bug tracking
- `docs/ROADMAP_HISTORY.md` — historical session logs and shipped-work archive
- `docs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/2026-05-02-v2.0-beta-sprint-qa/` — round-robin QA on Sprint 1/2/3 plan (this session)
- Survival guide / Bug Bible: https://github.com/jbrick2070/comfyui-custom-node-survival-guide

---

## Pre-ship v2.0 — ecosystem review checklist

Quick scan before tagging v2.0-alpha → v2.0. Verify each upstream
release either (a) doesn't break OTR's pinned versions or (b) is
worth pulling in for the v2.0 release notes. Added 2026-05-07.

### ComfyUI Core & Frontend
- v1.44.18 (2026-05-06) and v1.44.17 (2026-05-05) — review changelog
  for anything affecting the LTX 2.3 path, MultimodalGuider, RES4LYF
  compatibility, or Blackwell/CUDA 13 attention paths.
- Releases: https://github.com/Comfy-Org/ComfyUI_frontend/releases
- Changelog: https://docs.comfy.org/changelog

### ComfyUI-GGUF — native GGUF weight loading
- v1.1.10 (2026-01-12), with continuous repo commits.
- Repo: https://github.com/city96/ComfyUI-GGUF
- Why care: opens a smaller-VRAM path for LTX 2.3 (the GGUF
  Q5_K_M quants of the 22B-distilled exist on HF). Could become
  the "32 GB RAM" budget option below the current v0_9 default
  if GGUF + euler_cfg_pp produces equivalent motion at ~half the
  weight footprint vs the BF16 fused 46 GB.

### ComfyUI-Ollama nodes — LLM integration / agent tooling
- Continuous Q1/Q2 2026 updates, including DeepSeek-R1 and Qwen
  3.5 architecture support.
- Describer / agent variant: https://github.com/alisson-anjos/ComfyUI-Ollama-Describer
- Native workflows: https://github.com/slyt/comfyui-ollama-nodes
- Why care: OTR currently uses transformers + Mistral-Nemo for
  story / critic / brief LLMs. Ollama would give an HTTP-server
  pattern with model swap by name (no per-call load), DeepSeek-R1
  for the critic role, and Qwen 3.5 for shorter beat-level
  rewrites. Worth a benchmark spike before v2.0 ships in case
  one of them obsoletes the current LLM stack.

### Google Gemma 2026 Developer Challenge
- Launched 2026-05-06.
- Link: https://dev.to/challenges/google-gemma-2026-05-06
- Why care: OTR's LTX 2.3 path uses Gemma 3 12B (FP4 mixed) as
  its text encoder, and the legacy story/critic LLM was Gemma-4
  before the Mistral-Nemo migration. If the challenge surfaces
  Gemma-tuned techniques or new finetunes (e.g. better motion
  prompt adherence, period-specific tonal control for the
  1940s OTR aesthetic), worth folding into either the prompt
  pipeline or the LTX encoder layer. Submission window may also
  be a forcing function to publish OTR's Gemma usage pattern as
  a contest entry — free marketing for the project.

---

## Daily operating cadence

- First thing: read this file, `CLAUDE.md`, `docs/BUG_LOG.md` header, `git log --oneline -5` on current branch.
- LHM is always on — poll `http://localhost:8085/data.json` (or `outputs/libre_tail.py`) before asking Jeffrey for system status.
- After every code change: AST parse + three regression suites (Bug Bible regression in survival-guide repo, `tests/test_dropdown_guardrails.py`, `tests/test_core.py`). Don't report "done" until green.
- One `git push` attempt max — if it fails, hand a cmd block with `cd /d` included.
- Verify every push: local HEAD == origin HEAD, no 0-byte files, no BOM, workflow JSONs valid, all node classes registered in `__init__.py`.
- Log bugs the moment they surface. Don't batch. Promote `Bible candidate: yes` to the survival guide only after the fix is verified AND a real run confirms the behavioural fix.
- Round-robin consult before non-trivial design decisions (CLAUDE.md "Round-Robin Consultation" rule). Save transcripts under `docs/<date>-<topic>/`.
- Never use PowerShell for git operations — always cmd shell via Desktop Commander (PowerShell mangles `&&` and commit message quoting).
