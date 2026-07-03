# OTR End-Roll Credits — Deep Review (real renderer, OldTimeRadio repo)

**Reviewer:** Fable (read-only pass, 2026-07-03)
**Repo:** `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
**Workflow of record:** `workflows\otr_scifi_16gb_full.json` (22 nodes, parsed live)
**Scope:** Read-only audit. No code changed. Answers the operator ask: really enrich the end-roll — add image/video model info, all voice models per character, and everything creditable from the JSON/ledger. Font +50%.

## TL;DR

- The end-roll is the **Telemetry HUD post-roll** rendered by `_TelemetryHUDRenderer` inside `nodes\video_engine.py` (node 12, `OTR_SignalLostVideo`), then green-crushed and blended over the real video tail by node 93 (`OTR_PostUpscaleProcgenBlend`). It is already a two-panel scrolling credits sheet with a "production dossier."
- **Font sizes are four magic expressions at `video_engine.py:1331-1334`** (body approx. 21 px at 1080p). The +50% bump is a four-line change, but it interacts with the 90 s HUD cap and the mux's 45 s silent-tail budget — both must be checked.
- **The engine-credits plumbing exists but fires in the wrong order.** Node 12 (which renders the credits AND writes `_treatment.txt`) executes **before** the image dispatcher (91) and video render batch (92), because node 84's input order resolves the node-12 chain first. Proof from today's output: `signal_lost_telemetry_of_lies_20260703_041947` has `meta.render_engines` (`humo_1.7B_169 x6`) in its on-disk ledger, yet its treatment prints `Video (per role): (not recorded)`. The 2026-06-21 image-credits fix stamps `meta.image_engines` into a **wire dict that dies at node 92** — it appears in **zero** of the last 40 on-disk ledgers.
- **Per-character voice provenance is already reachable at credits time and simply never rendered:** `meta.voice_cast_decision` (writer-stamped, in the frozen wire ledger node 12 receives) carries per-character `engine` (`indextts2`), `accepted_id` (`vz_caro_davy`), seed, bank SHA, and fallback reason. Meanwhile the on-screen "CAST & VOICES" panel currently renders **blank presets for every character** (verified in today's treatments) because the writer no longer stamps character presets and CastLock's repaired cast only flows to the voice nodes.
- Recommended architecture for "really doing it up": keep node 12's early HUD as the *story* credits, and render the *engine/delivered-receipts* credits **late** (extend node 93 or a new terminal credits node fed by the clip manifest), rather than re-ordering node 12 — re-ordering trips the episode-rename hazard documented at `video_engine.py:2280-2315`.

---

## 1. Where the credits actually render

### 1.1 The renderer

`nodes\video_engine.py` — `OTR_SignalLostVideo` (workflow node 12, "8. Signal Lost Video"):

| Piece | Lines | What it does |
|---|---|---|
| `_load_font` | 69-100 | Monospace TTF from `C:\Windows\Fonts` (`consola.ttf` -> `cour.ttf` -> `lucon.ttf`), cached per size |
| `_get_latest_telemetry` | ~1030-1066 | Regex-scrapes `otr_runtime.log` tail for VRAM peak / tok/s / LLM name |
| `_build_hud_dossier` | 1068-1187 | Builds the scrolling "production dossier" sections: WRITER / LLM CONFIG, RESOLVED (OPENROUTER) w/ per-slug cost, STORY SPINE, RENDER ENGINES (video by_role + image by_role), SYSTEM |
| `_parse_hud_data` | 1190-1310 | Assembles the HUD data dict: title, style, produced date, duration, resolution, news seeds, cast (name + preset + `_PRESET_DESC`), transcript items, dossier, telemetry |
| `_TelemetryHUDRenderer` | 1313-1594 | Pre-renders a **static left panel** (`_build_left`, 1405-1501: SIGNAL LOST header, METADATA, NEWS SEED, SYSTEM TELEMETRY, CAST & VOICES, footer "OTR v1.0") and a **scrolling right panel** (`_build_right`, 1503-1594: dossier sections -> `[ CLASSIFIED TRANSCRIPT ]` full script -> telemetry easter egg -> "SIGNAL LOST // ALL RIGHTS RESERVED") |
| HUD build call | 2172-2184 | Step 2b of `render_video` — **before** any frame encoding |
| HUD frames appended | 2379-2386 | HUD post-roll frames yielded after the audio-reactive CRT frames |
| Credits music | 2212-2270 | Loops the MusicGen closing cue to fill the whole HUD (`OTR_CREDITS_MUSIC_LOOP`, operator 2026-06-28) |
| Treatment enrichment + write | 2402-2428 | Post-encode: merges `meta.render_engines` + `images` from the ledger singleton into `led`, then `_write_story_treatment` (1628-1950) writes the `_treatment.txt` sidecar |

### 1.2 The tail chain (how the roll reaches the published mp4)

Wiring verified from `otr_scifi_16gb_full.json` links:

```
12 SignalLostVideo (procgen mp4 incl. HUD tail)
 ├─ link246 → 84 SilentComposite in0   (assembled-length cap + BUG-410 credits-tail restore,
 │                                       otr_silent_composite.py:219-221, 690-745)
 │   84 → link247 → 86 CaptionBurn → link266 → 93 PostUpscaleProcgenBlend in0
 ├─ link265 → 93 in1 (procgen floor — the §4D lighten/screen blend rides the GREEN credits
 │                     over the scene tail, otr_post_upscale_procgen_blend.py:389-407)
 94 SceneAwareScopes → link273 → 93 in11 (scopes suppressed over the credits tail,
                                          otr_scene_aware_scopes.py:303-334)
 93 → link250 → 85 MasterAudioMux (silent credits tail permitted up to
                                   OTR_MAX_CREDITS_TAIL_S=45s, otr_master_audio_mux.py:140-156)
```

**Creative constraint worth knowing:** node 93's `green_only_overlay` path (`otr_post_upscale_procgen_blend.py:388-398`) runs the procgen through `colorchannelmixer` that **zeroes the R and B channels**. In the final master, the HUD's amber (255,176,0), white (180,200,180) and cyan render only via their green components (176 / 200 / 200). The credits you design are effectively a **single-channel green luminance composition** on the final output — hierarchy must come from brightness, not hue.

### 1.3 What it currently prints (verified against today's real output)

From `...\output\otr\episodes\signal_lost_shadows_in_the_room_20260703_092654\audio\..._treatment.txt` (same data feeds the on-screen dossier):

- Title / style / produced OK, WRITER / LLM CONFIG OK (gemma-4-12b-it, temps, word counts), NEWS SEED OK, STORY SPINE OK (rich)
- **CAST & VOICES: `ANNOUNCER -> bf_lily  bf_lily` and both characters completely blank**
- **RENDER ENGINES: `(not recorded)` for both video and image** — even on episodes whose ledgers *do* contain `meta.render_engines`
- Full script, PRODUCTION (duration/res/size), SYSTEM (host/CPU/GPU/CUDA/torch) OK

---

## 2. Font size — current value and where the +50% bump goes

`nodes\video_engine.py:1331-1334`, inside `_TelemetryHUDRenderer.__init__` — **magic expressions, not named params**:

```python
self.f_head  = _load_font(max(22, h // 28))   # 1080p → 38 px
self.f_label = _load_font(max(16, h // 42))   # 1080p → 25 px
self.f_body  = _load_font(max(14, h // 50))   # 1080p → 21 px   <- main credits text
self.f_small = _load_font(max(13, h // 58))   # 1080p → 18 px
```

+50% at 1080p -> head approx. 57 (`h//19`), label approx. 38 (`h//28`), body approx. 32 (`h//33`), small approx. 27 (`h//39`). Line heights, wrapping and column fit all derive automatically (`_fh`/`_fw`/`_draw_wrapped`, 990-1027; left column `LEFT_W = max(280, int(w*0.36))`, 1325).

**Do not bump blind — three coupled knobs:**

1. `_SCROLL_PPS = 65` (line 1319) and `hud_frames()` (1352-1356): `secs = scroll_px/65 + 8`, clamped **20-90 s**. Bigger fonts approx. +50% scroll height; a typical episode already lands near the cap, and once capped, the scroll *rate* silently increases (frac-based scroll always covers the full canvas), defeating the readability goal. The bump should raise `_SCROLL_PPS` proportionally or raise/rethink the 90 s cap.
2. `OTR_MAX_CREDITS_TAIL_S = 45` (otr_master_audio_mux.py:149): the mux fails LOUD if the final video outruns the master audio by more than 45 s. Any HUD-length growth must be validated against this budget (the credits-music loop in node 12's own mp4 covers the composite cap, but the *master* mix does not — BUG-410 comments in `otr_silent_composite.py:729-741`).
3. The left panel already truncates cast at `footer_y` (1473-1495); at +50% a 4-5 character cast plus telemetry will overflow — the CAST & VOICES block will need either the right panel or a tighter left layout.

The CRT title-card fonts (lines 359-361, 681, 766) are a separate system; leave them alone.

---

## 3. Image + video model info — the seam, mapped

### 3.1 What exists

- **Video (delivered receipts):** `otr_video_render_batch.py:26-75` builds `meta.render_engines` = `{histogram, video_revision, by_role, vram_peak_mb, per_clip[{shot_id, role, delivered_engine, recipe, quant, use_lora, render_canvas, vram_peak_mb}], by_engine}` and **saves it to the production-ledger singleton -> on-disk ledger** (line 74). This is real: `telemetry_of_lies` ledger shows `humo_1.7B_169` per role, per shot.
- **Image:** `otr_image_gen_dispatcher.py:493-501, 666-673` builds `meta.image_engines = {by_role, image_revision}` — but stamps it only into the **parsed wire copy** returned as `patched_ledger_json` (dispatch, 790-806; no `get_ledger()`/save anywhere in the file). That string flows only to node 92 and dies there. **Zero of the last 40 on-disk ledgers contain `image_engines`.** The 2026-06-21 operator fix is not landing.
- **Consumers:** `_build_hud_dossier` section 4 (video_engine.py:1140-1164) and the treatment's RENDER ENGINES block (1841-1877) both read `meta.render_engines` / `meta.image_engines` and render beautifully **when fed** (pinned by `tests\test_hud_dossier_bug3.py`, which injects the meta directly).

### 3.2 Why it never reaches the credits

Node 12's `script_json` comes from **node 62 `OTR_LedgerFreezeCascade` slot 1 (link 16)** — the pre-audio, pre-render frozen ledger. `load_ledger` (`_otr_ledger_consumers.py:51-68`) just parses that string; `overlay_audio_timing` (`otr_shot_lock.py:169-221`) overlays only per-line timing keys. And by node 84's input order (`in0 <- link246 from 12`, `in2 <- link261 from 92`), ComfyUI executes **node 12 before nodes 90/91/92**. Consequences:

- On-screen dossier: RENDER ENGINES section empty -> skipped entirely.
- `_treatment.txt`: the singleton merge at 2408-2418 runs post-encode but **still before node 92 has executed** -> `(not recorded)`, confirmed empirically.
- The clip manifest (`92.1 clip_manifest_json`, carrying `engine_histogram` + per-clip `final_engine`) reaches nodes 84 and 94 — i.e., it is *in the tail chain*, just downstream of where the credits are rendered.

### 3.3 The seam options (analysis, no code)

- **Option A (recommended): render the engine credits LATE.** Extend node 93 (or add a small terminal `OTR_CreditsRoll` node between 86 and 93 / after 93) that takes `clip_manifest_json` (92.1) and `patched_ledger_json` (91.0) as inputs and appends/replaces the dossier segment. It executes after everything, sees delivered receipts including in-render fallback restamps, and the existing early HUD keeps the story/transcript material. No ordering hazards.
- **Option B: wire 92.1 into a new optional input on node 12.** Forces 12 after 92; zero wall-time cost (serial executor) and it would also make the existing treatment merge (2402-2418) start working. **Hazard:** node 12 finalizes/renames the `pending_*` episode dir (2430+, and the Phase-G out_dir logic at 2286-2315). Re-ordering moves the rename after image/video render, so stills/clips land in `pending_*` and the rename relocates them while nodes 84/93 later consume manifest **paths recorded pre-rename**. Finalize would have to move to the true terminal (node 85) first — a real refactor.
- **Option C (minimum durability fix regardless):** make the dispatcher stamp `meta.image_engines` into the **singleton** (mirror `_stamp_render_engines_meta`) so the on-disk ledger is complete even before the render seam is fixed.

---

## 4. Voice models per character — where it lives, how to credit it

### 4.1 The layered truth about "which voice"

1. **`led.cast[].voice_preset` / `tts_model` (writer-stamped, in the wire ledger node 12 gets):** announcer gets a real preset (`kokoro` / `bf_lily`); characters get `voice_preset: null` and `tts_model: "bark"` **by construction** (`_otr_casting.py:1691-1694` — a routing placeholder, *not* the delivered engine). This is why today's credits show blank character voices.
2. **`meta.voice_cast_decision` (writer-stamped via the hybrid voice-fit, `OTR_LedgerScriptWriter.py:3040`, `_otr_casting.py:1748-1835`):** per char_id: `{engine: "indextts2", proposed_id/accepted_id: "vz_caro_davy", seed, bank_sha, policy_version, fallback_reason, candidate_ids[...]}`. **Verified present in today's on-disk meta — and because the writer stamps it before the freeze, it is ALREADY in the wire ledger at credits-render time.** This is the gold seam for "Cast & Voices."
3. **`meta.cast_voice_slots`:** per char gender/timbre/age_band/`speech_signature` ("steady, reassuringly maternal") — lovely credit copy, also already reachable.
4. **CastLock's authoritative stamp (`cast_lock.py:410-448` two-lane repair; `_stamp` at 628-632: `voice_ref_id`, `voice_engine`, `commercial_clean`):** lives only in CastLock's **output** cast_json (links 235-237 -> voice nodes 81/82/83). CastLock never saves the singleton — this stamp never reaches disk or node 12. The treatment already *tries* to read `entry.get("voice_engine")` (video_engine.py:1827) — it's always empty in production.
5. **Delivered per line:** voice nodes resolve engine + ref per line (`_otr_voice_node_common.py:366-540`) with LOUD fallbacks (e.g. indextts2 -> bark per line); the durable trace is the per-line `<engine>_wav_path` key on ledger lines (e.g. `bark_wav_path`, verified) plus the `render_log` wire string.
6. **Engine selections in the workflow itself:** node 81 `indextts2`, node 82 `kokoro`, node 83 `stable_audio_3` (widgets verified in the JSON).

### 4.2 How to render "Cast & Voices" properly

For each cast entry (announcer + characters), resolve in order: `voice_cast_decision[char_id]` (engine + accepted_id) -> `cast.voice_preset` (announcer/bark lane) -> per-line `*_wav_path` key inference as the delivered-engine cross-check. Render as:

```
CAST & VOICES
  WENDY REEVES ....... indextts2 · vz_caro_davy      "steady, reassuringly maternal"
  PIM STEINER ........ indextts2 · vz_peter_yearsley "clipped, pedantic"
  ANNOUNCER .......... kokoro · bf_lily
  THEME & CUES ....... stable_audio_3 (closing cue: musicgen, looped under credits)
```

The `speech_signature` from `cast_voice_slots` replaces `_PRESET_DESC` (1604-1625), which only knows bark `v2/*` presets — kokoro/indextts2/cb/dia voices currently fall back to echoing the preset name (visible in today's output: "bf_lily bf_lily"). Music engine identity should be stamped by node 83 (it already emits `music:done:engine=...`, `stable_audio_theme.py:173`) or read from the workflow widget.

---

## 5. Everything creditable + proposed end-roll

### 5.1 Available now (frozen wire ledger at node 12)

`episode_title`, `style`/`style_descriptor`/`theme`, `gen_params_initial` (models, temps, target words, seed_source), `slot_calls_by_slot`, resolved OpenRouter models + cost (in-process snapshot, `_otr_openrouter_backend.py:534-538`), `news` + `news_seed`, `dramatic_state`, `story_quality` (domain, conflict objects), `cast_voice_slots`, **`voice_cast_decision`**, `git_commit`, `phase_ms`, freeze telemetry, `vram_at_cascade_entry_gb`, word counts, full transcript, system specs (`_otr_sys_specs`).

### 5.2 Available only post-render (needs the §3 seam)

`meta.render_engines` (per-role video engines, per-clip delivered_engine + recipe/quant/LoRA/canvas receipts, histogram, vram_peak_mb), `meta.image_engines` / `ledger['images']` rows (per-role FLUX/z_image engine_id, content hashes, provenance incl. cache hits), clip-manifest `final_engine` + degradation restamps, final file size/duration.

### 5.3 Proposed structure

**Viewer-facing roll (the "do it up" card, +50% type):**

```
SIGNAL LOST
"<Episode Title>"
<style descriptor> · <est. runtime> · <date>

WRITTEN BY        <creative model>  (technical pass: <technical model>)
CAST & VOICES     <char → engine · voice, per §4.2>
IMAGES            <image engine per role, e.g. FLUX gen-1>
MOTION            <video engine per role, e.g. HuMo 1.7B · LTX-AV>
MUSIC             <stable_audio_3 theme · musicgen closing cue>
NEWS SEED         <headline>
SEED / COMMIT     <cast seed · git short-sha>

Made with OTR v2.0-alpha on <GPU> — 100% generated, no humans were harmed
```

**Debug/extended card (after the viewer roll, or env-gated `OTR_CREDITS_DEBUG`):** the existing dossier, enriched — per-clip delivered-engine receipts (recipe/quant/LoRA/canvas/VRAM per shot), degradation/fallback trail, OpenRouter per-slug cost table, phase timings, VRAM peaks, audio SHAs, story-quality grades, full SYSTEM block, classified transcript. Keep the transcript and easter egg — they're the show's voice; they belong *after* the viewer card, not instead of it.

---

## 6. Prioritized recommendations

1. **P0 — Font bump (4 lines, `video_engine.py:1331-1334`)** with the three coupled knobs from §2: scale `_SCROLL_PPS` (1319), revisit the 90 s clamp (1352-1356), verify against `OTR_MAX_CREDITS_TAIL_S=45` (mux). Consider naming the sizes (`_HUD_FONT_SCALE`) instead of four new magic numbers.
2. **P1 — Cast & Voices from `meta.voice_cast_decision` + `cast_voice_slots`** (no wiring needed — data is already in the wire ledger; render in `_parse_hud_data`/`_build_left` and `_write_story_treatment:1820-1838`). Fixes the currently-blank character voices. Add music engine + announcer engine labels; retire/extend `_PRESET_DESC`.
3. **P1 — Durability fix:** dispatcher stamps `meta.image_engines` into the ledger **singleton** (mirror `_stamp_render_engines_meta`), and CastLock stamps its `voice_engine`/`voice_ref_id` cast back to the singleton. Even before the seam fix, the on-disk record becomes complete.
4. **P2 — Delivered-engine credits via a LATE render (Option A, §3.3):** feed `92.1 clip_manifest_json` (+ `91.0 patched_ledger_json`) into an end-of-graph credits step (extend node 93 or a small new node) — same change updates `otr_scifi_16gb_full.json` in the same commit per §0. Avoid Option B (rewire into node 12) unless the episode finalize/rename is first moved to the terminal node — hazard documented at `video_engine.py:2286-2315`.
5. **P3 — Polish:** footer says **"OTR v1.0"** (line 1500 — stale); left-panel telemetry labels CORE/FLUX/MEM actually show the *LLM* name/speed (1460-1466 via 1040-1066) — mislabelled as FLUX; single-green-channel legibility check after the blend (§1.2); cap left-panel cast overflow at bigger type.
6. **P4 — Debug credits card** env-gated, with the per-clip E5 receipts and fallback trail.

## 7. Corrections to the prior (staging) review

- "engine_histogram is never merged into led.meta" — **wrong for this repo**: `otr_video_render_batch.py:61-75` stamps `meta.render_engines` (histogram + by_role + per-clip receipts) into the singleton/on-disk ledger. The real bug is *ordering*: the credits/treatment consumers run before that stamp exists (§3.2) — so the staging review's *conclusion* (paths never meet at credits time) still holds, for a different reason.
- "image model identities are never captured" — **partially wrong**: `meta.image_engines` is built (dispatcher 666-673) but into a wire-only dict; and per-still `engine_id` rows exist in `ledger['images']` on the patched wire.
- "audio/TTS identities never captured" — **wrong**: `meta.voice_cast_decision` (engine + accepted ref per character) is writer-stamped and reaches the credits renderer today; it's simply never rendered. CastLock's richer stamp exists but is wire-orphaned.

## 8. Verification note

All claims grounded on the Windows repo via file tools + a Desktop Commander Python REPL (never the Linux mount): read `nodes\video_engine.py` (headers 1-68, fonts 69-100/990-1027, dossier/HUD 1040-1625, treatment 1628-1950, render_video 2041-2440), `otr_video_render_batch.py:20-90`, `otr_image_gen_dispatcher.py:480-540/650-700/790-815`, `cast_lock.py:400-655`, `_otr_voice_node_common.py` (grep), `batch_character_voices.py`, `stable_audio_theme.py` (grep), `otr_shot_lock.py:169-221`, `_otr_ledger_consumers.py:51-68`, `otr_silent_composite.py` + `otr_post_upscale_procgen_blend.py` + `otr_master_audio_mux.py` (credits-tail sections), `tests\test_hud_dossier_bug3.py`; parsed `otr_scifi_16gb_full.json` (all 22 nodes, all 52 links, widget values for nodes 12/80-83/87/88/91-93); inspected 40 recent on-disk ledgers plus the `shadows_in_the_room` and `telemetry_of_lies` treatments (2026-07-03) for the empirical RE/IE/voice findings. Not found anywhere: any other drawtext/cv2.putText credits path — `video_engine.py` is the only credits renderer.
