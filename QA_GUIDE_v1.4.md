# QA Guide — v1.4 in-flight changes (peer review before hardware testing)

**Branch:** `v1.4-voice-arc-infra`
**Status:** Code landed, NOT battle-tested. A v1.3 full-workflow OOM was observed after these changes; hardware regression is blocked until the OOM is root-caused.
**Purpose:** Peer review checklist for everything that touched the codebase during v1.4 so far, so a human (or a fresh AI session) can eyeball the diffs, spot obvious problems, and sanity-check the wiring before burning a full episode run on the 5080.

---

## 1. What was added or changed

### 1.1 New node — `nodes/kokoro_announcer.py`
Dedicated non-Bark narrator bus so ANNOUNCER bookends don't inherit Bark's "ums and ahs."

- Class: `KokoroAnnouncer`, node id `OTR_KokoroAnnouncer`, display "🎙️ Kokoro Announcer"
- Backend: `kokoro.KPipeline(lang_code='b')` (British)
- Voice grab bag (4 voices, one picked per episode via seeded RNG):
  - `bm_george` — BBC authoritative male
  - `bm_fable` — documentary relaxed male
  - `bf_emma` — BBC authoritative female
  - `bf_lily` — documentary relaxed female
- Seed: `random.Random(f"{episode_seed}_kokoro_announcer")` — deterministic per episode
- Voice `.pt` files lazy-downloaded from `1038lab/KokoroTTS` via `hf_hub_download` on first run
- Pulls ANNOUNCER lines out of `script_json`, renders, returns batched AUDIO tensor at **24 kHz**
- Graceful degrade: if `kokoro` import fails or no ANNOUNCER lines, returns empty audio with a clear error in render_log
- RETURN_TYPES: `("AUDIO", "STRING", "STRING")` — audio, render_log, chosen_voice

**Review focus:** voice list spelling, seed string stability, lazy-download error path, silent-fallback shape of empty AUDIO tensor, no hard crash if `kokoro` is missing.

### 1.2 `nodes/batch_bark_generator.py` — ANNOUNCER filter
Dialogue extraction loop now skips ANNOUNCER so Bark never renders narrator lines:

```python
if character_name.strip().upper() == "ANNOUNCER":
    skipped_announcer += 1
    continue
```

Log line: `"skipped N ANNOUNCER lines — routed to Kokoro bus"`.

**Review focus:** case sensitivity (`.upper()`), whitespace handling, counter initialization, log string safe for ASCII.

### 1.3 `nodes/scene_sequencer.py` — Kokoro consumer + resample fix
- New optional `announcer_audio_clips` AUDIO input on `SceneSequencer.INPUT_TYPES`
- Added `announcer_clip_idx` counter and `_extract_clips_from_audio(announcer_audio_clips)` call
- Dialogue branch routes ANNOUNCER character to the Kokoro queue:
  ```python
  is_announcer = character_name.strip().upper() == "ANNOUNCER"
  if is_announcer and announcer_clip_idx < len(announcer_clips):
      clip_np, clip_sr = announcer_clips[announcer_clip_idx]
      ...
      announcer_clip_idx += 1
  ```
- **EpisodeAssembler resample fix** (the bug caught by Jeffrey's "32hz not 48?" question): `_extract_waveform` now accepts `target_sr` and resamples via `torchaudio.functional.resample` with a numpy `_resample_audio` fallback. Both `opening_theme_audio` and `closing_theme_audio` callers pass `target_sr=sample_rate`. Without this, 32 kHz MusicGen output would have played at 2/3 speed on the 48 kHz main bus.

**Review focus:** clip index bounds, what happens if Kokoro bus is shorter than ANNOUNCER line count, resample fallback numeric correctness, target_sr None path still works for equal-rate audio.

### 1.4 `nodes/gemma4_orchestrator.py` — music_plan in production plan
- `DIRECTOR_PROMPT` patched to emit a `music_plan` block in `production_plan_json`:
  ```
  "music_plan": [
    {"cue_id": "opening",      "duration_sec": 12, "generation_prompt": "..."},
    {"cue_id": "closing",      "duration_sec": 8,  "generation_prompt": "..."},
    {"cue_id": "interstitial", "duration_sec": 4,  "generation_prompt": "..."}
  ]
  ```
- Added MUSIC PLAN RULES section: tailor per episode tone, instrumental only, < 35 words, fixed durations.
- `max_tokens` bumped from `min(1500, max(500, 400 + script_len // 10))` to `min(1700, max(650, 550 + script_len // 10))` to give Gemma room for the new block.
- `music_plan_json` RETURN_NAME already existed in v1.3, plumbing reused.

**Review focus:** JSON validity of the new example block, the token bump not pushing Gemma over its own context ceiling, whether older scripts without the rules still parse.

**Known debt:** this file still has a pre-existing UTF-8 BOM. Not introduced by v1.4 work, but it violates the CLAUDE.md encoding rule and should be stripped before tag.

### 1.5 New node — `nodes/musicgen_theme.py`
Real instrumental music for opening / closing / interstitial bookends. No more procedural noise stubs.

- Class: `OTR_MusicGenTheme`, display "🎺 MusicGen Theme"
- Backend: `transformers.MusicgenForConditionalGeneration` + `AutoProcessor` (NOT audiocraft — the `av==11.0.0` and `spacy==3.5.2` pins make audiocraft uninstallable on Windows + Python 3.12)
- Model: `facebook/musicgen-medium` (~6 GB VRAM)
- Native sample rate: **32 kHz** (see 1.3 for resample fix)
- Reads `production_plan_json` for cue prompts, falls back to `CUE_DEFAULTS` if missing
- Cache: SHA-256 of `f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}"` → `models/musicgen_cache/<cue>_<hash16>.wav` via `soundfile`
- Model only loads if at least one cue is uncached
- Generation: `model.generate(**inputs, max_new_tokens=duration*50+8, do_sample=True, guidance_scale=3.0)`
- Peak-normalized to -1 dBFS (×0.89)
- `try/finally` guarantees `del model` + `torch.cuda.empty_cache()` on both success and failure — critical for VRAM handoff to Bark
- RETURN_TYPES: `("AUDIO", "AUDIO", "AUDIO", "STRING")` — opening, closing, interstitial, render_log

**Review focus:** VRAM release path on exception, cache key stability (whitespace in prompt?), fallback defaults exist for all three cues, max_new_tokens math vs duration, what happens if `production_plan_json` is empty string or malformed.

### 1.6 `__init__.py` — node registration
Two new entries in `_NODE_MODULES`:
```python
"OTR_KokoroAnnouncer": (".nodes.kokoro_announcer", "KokoroAnnouncer", "🎙️ Kokoro Announcer"),
"OTR_MusicGenTheme":   (".nodes.musicgen_theme",  "MusicGenTheme",   "🎺 MusicGen Theme"),
```

**Review focus:** class name matches actual class in each module, no duplicate keys, display names unique.

### 1.7 Workflow JSONs — all three shipped workflows
Files: `workflows/old_time_radio_scifi_full.json`, `..._lite.json`, `..._test.json`

- **Kokoro wiring:** ScriptWriter.script_json → KokoroAnnouncer.script_json → SceneSequencer.announcer_audio_clips (node id 13, links 19/20)
- **MusicGen wiring:** Director.production_plan_json → MusicGenTheme → EpisodeAssembler.opening_theme_audio + closing_theme_audio (node id 14, links 21/22/23)
- Old procedural `OTR_SFXGenerator` theme stubs (nodes 5 & 6) **disconnected** from EpisodeAssembler; old links 7/8 dropped.
- MusicGen `widgets_values`: `["{}", "", "facebook/musicgen-medium", 3.0]` — 4 widgets, matches `INPUT_TYPES` count.

**Review focus:** widget count vs `widgets_values` length (Bug Bible core rule), link IDs unique, no dangling `link` references, the disconnected SFX nodes aren't breaking graph validation, interstitial output is intentionally unconnected (punted to v1.5).

---

## 2. Peer-review checklist (run before any hardware test)

### 2.1 Static checks
- [ ] AST parse every changed `.py` file
- [ ] Grep `NODE_CLASS_MAPPINGS` — both new nodes present, no typos
- [ ] Grep for dead imports / unused variables in the three edited files
- [ ] Confirm no new BOMs introduced (`head -c3 file | xxd` — should not show `EF BB BF`)
- [ ] `gemma4_orchestrator.py` BOM is **pre-existing**, not a new regression — confirm via `git show v1.3:nodes/gemma4_orchestrator.py | head -c3`

### 2.2 Workflow JSON sanity
- [ ] For all 3 workflows: widget count in node definition == `widgets_values` length on the instance (Bug Bible core rule)
- [ ] No orphaned `link` IDs
- [ ] Kokoro node (id 13) exists in all 3 JSONs with matching inputs/outputs
- [ ] MusicGen node (id 14) exists in all 3 JSONs with matching inputs/outputs
- [ ] EpisodeAssembler's `opening_theme_audio` and `closing_theme_audio` inputs resolve to MusicGen outputs, not SFX stubs
- [ ] Old SFX stub nodes (5 & 6) are disconnected but not crashing the graph loader

### 2.3 Logic spot checks
- [ ] Seed string for Kokoro voice pick is stable across reruns of the same episode seed
- [ ] `BatchBark` skip counter increments only for `ANNOUNCER`, not for character names containing "announcer" as a substring
- [ ] `SceneSequencer` announcer clip queue handles: empty queue, more lines than clips, more clips than lines
- [ ] `EpisodeAssembler._extract_waveform` resample path: same-rate no-op, downsample, upsample, None target_sr
- [ ] MusicGen cache hit path does not load the model at all
- [ ] MusicGen failure path releases VRAM via `try/finally`
- [ ] Gemma token bump is still under the orchestrator's hard ceiling

### 2.4 Regression contract (Bug Bible)
- [ ] `INPUT_TYPES` unchanged on any pre-existing node except SceneSequencer (documented)
- [ ] Changed `widgets_values` on any existing workflow node? (No — only new nodes.)
- [ ] All new nodes have a CATEGORY of `OldTimeRadio`
- [ ] No curse words, no unsafe content, no placeholder TODOs in shipped strings

---

## 3. Known open risks before battle testing

1. **v1.3 full workflow OOM (blocker).** Must be root-caused before any v1.4 hardware pass. Suspect Gemma unload path. Repro on clean `v1.3` tag first to confirm it's not a v1.4 branch side-effect.
2. **Gemma → MusicGen → Bark VRAM handoff unverified.** Sequence: Gemma (~6–8 GB) unload → MusicGen medium (~6 GB) load+gen+unload → Bark (~4 GB) load. On paper this fits under 14.5 GB peak. On the 5080, `torch.cuda.empty_cache()` is cooperative, not authoritative. First real run will tell.
3. **Kokoro first-run download.** `hf_hub_download` will fetch 4 `.pt` files on first use. Offline-first rule means subsequent runs work fully offline, but the first run needs network.
4. **MusicGen cache disk growth.** Every unique `(cue, prompt, duration, seed)` writes a new WAV. No eviction policy. Manual cleanup for now.
5. **Interstitial cue is generated but unused.** Sequencer bus wiring is deferred to v1.5. Wastes a few seconds of generation per episode on cold runs. Acceptable, will be caught by cache on retries.
6. **Dangling SFX stub nodes** in workflow JSONs may confuse the graph renderer even though they're disconnected. Clean removal is on the cleanup debt list.
7. **Pre-existing BOM** on `gemma4_orchestrator.py`. Violates house rule. Strip before tag.

---

## 4. Sign-off before hardware test

Do NOT run a full v1.4 hardware regression until:

1. The v1.3 OOM is root-caused and fixed.
2. This QA guide's section 2 checklist is fully ticked.
3. A smoke test on the `test` workflow (shortest) runs to completion on the 5080 with VRAM logged.
4. Only then: run `lite`, then `full`.

If hardware test reveals a new failure mode, update this QA guide with the failure signature before attempting a fix, so the next AI session can see the history.
