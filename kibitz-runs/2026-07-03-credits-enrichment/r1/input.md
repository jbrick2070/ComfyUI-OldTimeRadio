# GO_FORWARD — Credits Enrichment (full campaign, cleanbreak)

**Date:** 2026-07-03  **Branch:** v2.0-alpha  **Source of truth for this build.**
**Origin:** `docs/credit_review_otr.md` (Fable deep review). Anchors re-grounded 2026-07-03 (see §Grounding).

## Operator directives (these win)
- **One campaign, not incremental.** Plan + harden here; execute in a fresh window off this doc.
- **Cleanbreak / rip-out -> paste-in -> fix the seams up- and down-stream in parallel.** Do NOT keep the old surface alive beside the new one.
- **NO FALLBACKS DURING THE RIP.** Temporary failure is accepted to get a clean break. The current silent `(not recorded)` and empty-preset fallbacks are exactly what we are ripping — replace them with the real data or a LOUD failure, never a quiet placeholder.
- Every node/widget/wiring change lands IN `workflows/otr_scifi_16gb_full.json` in the SAME change as the code. Unwired code is dead.
- Run regression suite + Bug Bible after every green chunk; commit AND push to v2.0-alpha same session.

---

## The core problem (grounded)
The end-roll is the Telemetry HUD in `nodes/video_engine.py` (node 12 `OTR_SignalLostVideo`), green-crushed + blended by node 93 `OTR_PostUpscaleProcgenBlend`. Today it prints blank character voices and `(not recorded)` engines, because:

1. **Render order.** Node 12 renders the HUD + writes the treatment BEFORE the image dispatcher (91) and video render batch (92) execute (node 84's input order resolves the node-12 chain first). So engine/image receipts do not exist yet when the early HUD is drawn.
2. **Data that IS already at node 12** (in the frozen ledger via link 16 from node 62): `voice_cast_decision`, `cast_voice_slots`, `episode_title`, style, seeds, word counts, transcript. These are simply not rendered well.
3. **Data that is NOT durably captured:** `meta.image_engines` is only put on a wire (dispatcher, no singleton save); CastLock's `voice_engine`/`voice_ref_id` are wire-only. `meta.render_engines` IS saved to the singleton (render_batch).

Cleanbreak resolution: **keep the story credits early (node 12); RIP the engine/receipt credits out of the early path and PASTE them into a LATE terminal credits render** fed by the post-render manifest. No dual path, no silent fallback.

---

## Campaign slices (one build; land as green chunks in this order)

### S0 — Font +50% (P0)
**Rip:** the four magic font expressions and the fixed scroll speed.
**Files:** `nodes/video_engine.py`
- `1331-1334`: `f_head=_load_font(max(22,h//28))`, `f_label=max(16,h//42)`, `f_body=max(14,h//50)`, `f_small=max(13,h//58)`. Introduce a single named `_HUD_FONT_SCALE = 1.5` and apply (or divide the divisors by 1.5): head `h//19`, label `h//28`, body `h//33`, small `h//39`.
- `1319`: `_SCROLL_PPS = 65` -> scale proportionally (~`98`) so the taller roll does not slam into the duration clamp and silently speed up.
- `1355`: `secs = scroll_px/_SCROLL_PPS + 8.0` — confirm the resulting HUD length stays within budget.
**Seam (downstream):** `nodes/otr_master_audio_mux.py:149-153` `OTR_MAX_CREDITS_TAIL_S=45` fail-loud. A +50% roll must not push the silent tail past 45s over the master audio. If it does, the fix is to raise the credits-music loop coverage / cap the roll, NOT to widen the guard.
**Left-panel overflow:** `_build_left` truncates cast at `footer_y` (~1473-1495); at bigger type a 4-5 cast + telemetry overflows — S1 moves Cast & Voices layout anyway.
**No-fallback note:** do not clamp font down "to be safe" — pick the scale, prove the budget.

### S1 — Cast & Voices, done right (P1a) — RENDER ONLY, no wiring
**Rip:** the blank-preset fallback. `_PRESET_DESC.get(preset,"")` (`1252`,`1258`,`1604`) only knows bark `v2/*`; kokoro/indextts2/cb/dia fall through to echoing the preset ("bf_lily bf_lily"). RIP that empty-string fallback.
**Paste:** in `_parse_hud_data` (`1190`) + `_build_left` (CAST & VOICES at `1476`), render per cast entry from data ALREADY in node 12's frozen ledger:
- engine + accepted voice from `meta.voice_cast_decision[char_id]` (`{engine, accepted_id}`),
- character voice-signature from `meta.cast_voice_slots[...].speech_signature` (replaces `_PRESET_DESC`),
- announcer from `cast.voice_preset` (kokoro/bf_lily),
- music engine from node 83 (`stable_audio_theme` emits `music:done:engine=...`) or the workflow widget.
Target line format: `NAME .... engine · voice_id   "signature"`.
**No-fallback:** if a character has no `voice_cast_decision` entry, render a LOUD marker (e.g. `??`) — never a silent blank. Accepted temporary failure until the writer path is confirmed to always stamp it (it does today — grounded).

### S2 — Durability stamps (P1b) — make the on-disk ledger complete
**Paste (mirror the proven pattern):**
- `nodes/otr_image_gen_dispatcher.py` (`669-673`): today `meta.image_engines` is only put on the returned wire. Add a singleton stamp mirroring `otr_video_render_batch.py:_stamp_render_engines_meta` (`61-75`, `get_ledger(); led.data["meta"]["image_engines"]=...; led.save()`).
- `nodes/cast_lock.py` (`_stamp` `628-632`, fallback `443-444`): CastLock stamps `voice_engine`/`voice_ref_id` on the wire only and never saves the singleton. Add a singleton save so the authoritative cast reaches disk + any late consumer.
**Seam (upstream):** confirm the singleton is the SAME instance across nodes (production_ledger.get_ledger()). 
**No-fallback:** no "if singleton missing, skip" — the stamp is required.

### S3 — Late engine-credits seam (P2) — the structural rip (HIGH-STAKES)
**Rip:** the engine/image credit rendering OUT of node 12's early HUD path (`_build_hud_dossier` RENDER ENGINES section `1140-1164`) and the post-encode treatment merge that runs too early (`2402-2428`, currently yields `(not recorded)`).
**Paste:** a LATE terminal credits step that executes AFTER nodes 91/92. Two candidate shapes (Fable to advise in orchestration pass):
- (A, preferred) extend `OTR_PostUpscaleProcgenBlend` (node 93) OR add a small new `OTR_CreditsRoll` node between 86 and 93 / after 93, taking `clip_manifest_json` (node 92 slot 1, link 261/271) + `patched_ledger_json` (node 91 slot 0) as inputs, and compositing the engine/image/delivered-receipt credit lines onto the tail.
- (B, rejected unless finalize moves) rewire 92.1 into a new node-12 input — forces 12 after 92 but node 12 finalizes/renames the `pending_*` episode dir (`~2286-2315`, `2430+`); reordering relocates assets after paths are recorded. Only viable if episode-finalize first moves to the true terminal (node 85). Do NOT take B without that move.
**Workflow JSON (same change):** add the new node/inputs + links in `otr_scifi_16gb_full.json`; re-run `OTR_WorkflowValidator` + link/widget audit. Respect positional `widgets_values` (append-only).
**Green-channel constraint:** node 93 `green_only_overlay` zeroes R+B (`otr_post_upscale_procgen_blend.py:681-683` `colorchannelmixer=...gg=1...`). Credits are single-channel green on the master — hierarchy via brightness, not hue. Design the roll accordingly.
**No-fallback:** if the manifest/patched ledger is absent at the late node, FAIL LOUD (accepted temporary failure), never emit `(not recorded)`.

### S4 — Polish + debug card (P3/P4)
- Stale footer `OTR v1.0` (`598`,`1500`) -> real version (`v2.0-alpha`).
- Mislabeled left-panel telemetry: `CORE/FLUX/MEM` (`1461-1463`) actually show the LLM name/speed/mem — relabel (the "FLUX" row is the LLM tok/s, not the image model).
- Optional env-gated `OTR_CREDITS_DEBUG` extended card: per-clip delivered-engine receipts (recipe/quant/LoRA/canvas/VRAM), degradation trail, OpenRouter cost table, phase timings, SHAs, story-quality grades. Keep the classified transcript + easter egg AFTER the viewer roll.

---

## Proposed viewer roll (single-green-safe, +50% type)
```
SIGNAL LOST
"<Episode Title>"
<style> · <est runtime> · <date>

WRITTEN BY     <creative model>  (technical: <technical model>)
CAST & VOICES  <NAME → engine · voice "signature">   (per character, S1)
IMAGES         <image engine per role, e.g. FLUX gen-1>   (S3)
MOTION         <video engine per role, e.g. HuMo 1.7B · LTX-AV>   (S3)
MUSIC          <stable_audio_3 · musicgen closing cue looped under credits>
NEWS SEED      <headline>
SEED / COMMIT  <cast seed · git short-sha>

Made with OTR v2.0-alpha on <GPU> — 100% generated
```
Debug/extended card follows only under `OTR_CREDITS_DEBUG`.

---

## Cleanbreak orchestration (rip / paste / seams in parallel)
For EACH slice: (1) rip the old surface + its fallbacks; (2) paste the new render/stamp; (3) fix the seam up-stream (data capture) AND down-stream (workflow JSON wiring + mux/blend budget) in the SAME change; (4) validate; (5) commit+push. No slice leaves a dual path or a silent placeholder. Intermediate red between rip and paste is accepted — the branch is green only at each committed chunk boundary.
*(Fable to weigh in on the optimal rip/paste ordering + which slices can go truly parallel vs must serialize — see the accuracy+orchestration pass.)*

## Validation gate (every chunk)
- `OTR_WorkflowValidator` on the workflow (widget count == INPUT_TYPES, wired-input names valid, link integrity) after any JSON change.
- JSON round-trip (workflow + any config), UTF-8 no BOM.
- Full regression suite + Bug Bible (Windows venv, `PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`). Fix any hard-coded HUD/credits widget-count or dossier assertions in the same commit (cf. the default_tts prune, where a count pin lived in `test_rip_sfx_broll_guard.py`).
- **Live credits-render smoke:** render a short episode and CONFIRM on-screen that voices + engines actually print (grep the treatment / view frames) — the whole point is that the data reaches the pixels. Test-green is necessary but not sufficient.
- S3 is the high-stakes structural slice: full kibitz + a Fable FINAL grounded gate before merge (per CLAUDE.md §9 reality exception — a missed thread here breaks every production render).

## Risks / traps (grounded)
- Render order (node 12 before 91/92) is the root cause — S3 must render LATE, not rewire node 12 (finalize/rename hazard).
- Green-only channel: amber/white/cyan collapse to green luminance on the master.
- 45s mux tail budget couples to font size + roll length (S0).
- Positional `widgets_values` drift (append-only) if S3 adds a node/widget.
- Hidden widget/dossier count assertions in tests (grep `widgets_values`, `dossier`, HUD counts repo-wide with ignore-off).

## Grounding (re-verified 2026-07-03, real Windows files)
video_engine.py: fonts 1331-1334; `_SCROLL_PPS` 1319 + dur 1355; `_parse_hud_data` 1190, `_build_left` 1405 + CAST&VOICES 1476, `_PRESET_DESC` 1252/1258/1604; footer 598/1500; telemetry labels 1461-1463; treatment merge ~2402-2428; finalize/rename ~2286-2315/2430+. otr_video_render_batch.py: `_build_render_engines_payload` 26-58, `_stamp_render_engines_meta` 61-75 (get_ledger + led.save = singleton). otr_image_gen_dispatcher.py: image_engines wire-only 669-673, returns patched 805 (NO singleton save). otr_master_audio_mux.py: OTR_MAX_CREDITS_TAIL_S 149, fail-loud 150-153. otr_post_upscale_procgen_blend.py: green-only colorchannelmixer 681-683. voice_cast_decision stamped `_otr_casting.py` 1835 -> `OTR_LedgerScriptWriter.py` 3040 (pre-freeze -> present at node 12); cast_voice_slots 1834/3035. cast_lock.py: `_stamp` 628-632 + fallback 443-444, wire-only (no singleton save).
