# GO_FORWARD — Credits Enrichment (full campaign, cleanbreak) — v2

**Date:** 2026-07-03  **Branch:** v2.0-alpha  **Source of truth for this build.**
**Origin:** `docs/credit_review_otr.md` (Fable deep review). Anchors re-grounded 2026-07-03.
**Panel:** codex r1 = VERDICT no, folded (see §Revision log); Fable orchestration+accuracy pass pending; optional manual agy pass (AGY_MANUAL_REVIEW_PACKET.md). Claude anchors + judges.

## Operator directives (these win)
- **One campaign.** Plan + harden here; execute in a fresh window off this doc.
- **Cleanbreak / rip-out -> paste-in -> fix seams up- and down-stream in parallel.** No dual path beside the new one.
- **NO FALLBACKS DURING THE RIP.** Temporary failure is accepted for a clean break. A silent `(not recorded)`, a blank voice, or a source-copy-on-failure is a BUG to remove, not a safety net. Missing data on the credits path => RAISE, never emit a quiet placeholder.
- Every node/widget/wiring change lands IN `workflows/otr_scifi_16gb_full.json` in the SAME change (litegraph; `widgets_values` POSITIONAL, append-only).
- Suite + Bug Bible after every green chunk; commit AND push to v2.0-alpha same session.

---

## The architecture decision (codex r1 MUST-FIX #1 + #3)
**ONE viewer credits surface, rendered LATE, in a NEW terminal node `OTR_CreditsRoll`.**

Why: the whole HUD (story + would-be receipts) is currently baked as frames INSIDE node 12 `OTR_SignalLostVideo`, which executes BEFORE the image dispatcher (91) and video render batch (92). So at node-12 time the engine/image/delivered-voice receipts do not exist yet — hence today's `(not recorded)` and blank voices. Splitting "story early / receipts late" would leave two incoherent surfaces. Cleanbreak resolution:

- **Node 12 is stripped** to the CRT "signal lost" visual + (optionally) the classified-transcript easter egg. It STOPS rendering the credits dossier / cast / render-engine block. Rip `_build_hud_dossier` RENDER ENGINES section (`video_engine.py:1140-1164`) and the too-early treatment merge (`2402-2428`) out of the delivered path.
- **New `OTR_CreditsRoll` terminal node** renders the COMPLETE viewer credits clip from post-render truth and appends it to the video tail before mux.
  - **Do NOT extend node 93** `OTR_PostUpscaleProcgenBlend`: it has a source-copy fallback on ffmpeg failure (`otr_post_upscale_procgen_blend.py:1044-1046`, `shutil.copy2(src,output_path)`) that would SILENTLY drop credits. A dedicated node keeps the no-fallback contract.
  - Inputs (Fable-confirmed): `clip_manifest_json` (node 92 slot 1 — a THIRD fan-out consumer alongside nodes 84/94), the CastLock-stamped durable ledger (S2). Placement: **rewire link 250 so `93 -> OTR_CreditsRoll -> 85`**; CreditsRoll appends the credit clip to node 93's output and passes it to the mux `in[0]`; link 263 (node 7 master audio -> mux `in[1]`) is PRESERVED (silent-tail model); add a new link CreditsRoll -> mux for the DECLARED credits-tail duration (credits-aware guard). CreditsRoll owns the appended output path; captions/bars stay burned at node 93 BEFORE the append, so the credits clip is appended AFTER the green crush -> **CreditsRoll may render FULL-COLOR** (keep the green CRT look by CHOICE, not pipeline force — state the decision).
  - Node 12's episode finalize/rename (`video_engine.py:2441-2460`) is independent of the HUD — LEAVE it in node 12; no hazard (Fable-confirmed). CreditsRoll runs post-rename and must re-resolve paths via `_reresolve_master_audio` where it touches the master path.
  - **No-fallback:** missing manifest/ledger or any render/concat failure RAISES before mux. No source-copy.

Node 12 keeps only what is true at its (early) time. Everything receipt-bearing is late.

---

## Data-truth map (grounded — what each credit line must read)
- **Cast & Voices FINAL** = CastLock's `_stamp` (`cast_lock.py:628-632` voice_ref_id/voice_engine; the hybrid voice-fit at `565-588` is HONORED ONLY IF engine matches AND `validate_voice_proposal` passes, else it FALLS CLOSED to the deterministic scorer at `596`). So `meta.voice_cast_decision.accepted_id` is the PLANNED fit, NOT the delivered voice — do not credit from it. Credit from CastLock's stamp, made durable in S2.
- **Video engines** = `meta.render_engines` — already saved to the production-ledger SINGLETON (`otr_video_render_batch.py:61-75`, get_ledger + led.save).
- **Image engines** = `meta.image_engines` — TODAY wire-only (`otr_image_gen_dispatcher.py:669-673`, no singleton save). S2 makes it durable.
- **Music engine** = no durable/wired path today (`stable_audio_theme.py:173` emits `music:done:engine=...` but node 83 `done` output is UNLINKED in the JSON). S2 stamps it into ledger meta. Do NOT read the workflow widget as source of truth.
- **Story facts** (title/style/seed/word counts/news/transcript) = already in node 12's frozen ledger (link 16 from node 62) — but now consumed by the LATE node via the durable ledger.

---

## Campaign slices (one build; land as green chunks)

### S2 (do FIRST) — Durable persistence contract (P1b)
The late credits node can only read what is on disk. Define the production persistence contract: **required singleton save, LOUD failure in production, with a deliberate test-mode injection path** (codex MUST-FIX #5).

**CRITICAL implementation gotcha (agy MUST-FIX #2):** CastLock and the dispatcher operate on a LOCAL dict parsed from the wire, NOT the singleton. `cast_lock.py:lock` does `led = load_ledger(script_json)` (`136`), stamps onto that local `led`, and re-serializes it (`194`) — it never touches `get_ledger()`. Same for `otr_image_gen_dispatcher` (local `ledger`). So a bare `get_ledger().save()` would persist the singleton's EMPTY/stale state, not the node's changes. The stamp MUST copy the updated sections from the local wire ledger into the singleton before save:
- `cast_lock.py`: after stamping, `prod = get_ledger(); prod.data["cast"] = led["cast"]; prod.data.setdefault("meta",{}).update({voice-cast keys}); prod.save()`.
- `otr_image_gen_dispatcher.py:669-673`: `prod = get_ledger(); prod.data.setdefault("meta",{})["image_engines"]=...; prod.data["images"]=ledger.get("images",{}); prod.save()`.
- Music engine: stamp `meta["music_engine"]` durably into the singleton (in `stable_audio_theme` at done-time, or wire node 83 done into the ledger path).
- The render-engine path already does the singleton-native save (`_stamp_render_engines_meta` builds from the singleton episode run, so it can save directly) — use it as the shape, but respect the local-vs-singleton distinction above for the wire-parsed nodes.
- Seam (upstream): confirm all nodes share `production_ledger.get_ledger()` (same module singleton) and that the copy is complete. No "skip if missing"; LOUD fail in production, explicit test-mode injection path.
- **`save()` never raises (Fable BUILD-BREAKER #3):** `Ledger.save()` returns `None` on failure (`production_ledger.py:1120-1122`), so a bare `prod.save()` silently defeats "LOUD fail in production." The stamp helpers MUST check the return and raise on `None` (test-mode injection path exempted). No other consumer breaks on the added singleton writes — `save()` merges with disk (BUG-108, `production_ledger.py:1141`), writer `set_cast` runs before CastLock, and later singleton writers are additive (grounded).

### S3 — The `OTR_CreditsRoll` terminal node (P2, HIGH-STAKES) — the spine
Build the new node per the architecture decision above. Renders the full viewer roll late from the durable ledger + manifest; appends to the tail; raises on missing data.
- Workflow JSON (same change): add the node + inputs + links; re-run `OTR_WorkflowValidator` + link/widget audit; append-only `widgets_values`.
- **Green-channel constraint:** node 93 `green_only_overlay` zeroes R+B (`otr_post_upscale_procgen_blend.py:681-683` colorchannelmixer gg=1). If the credits clip is composited through/after that path, it is single-channel green on the master — hierarchy via brightness, not hue. Concrete (agy SHOULD-FIX #1): use only high-green colors (white/green/cyan), never pure red/blue; establish hierarchy by GREEN INTENSITY (e.g. 255 headers, 128 labels). If OTR_CreditsRoll writes its own clip appended AFTER the blend, decide + state whether it inherits the green crush or renders full-color.
- **SECOND credits organ — node 84 (Fable BUILD-BREAKER #2, unlisted):** `otr_silent_composite.py` weaves the credits era into the SILENT composite: its tail-fill planner extends the composite to the procgen-floor length and **loops the last drama clip under the scrolling credits** (the operator's 2026-06-17 "credits over the scene" LOOK CONTRACT — `:388-413`, BUG-410 restore `:690-741`). When node 12's HUD is ripped, this goes dead and MUST be ripped in the SAME slice. Critically, the looped-last-clip backdrop is a LOOK CONTRACT: `OTR_CreditsRoll` has the clip manifest and must reproduce it (loop the final clip under the roll as the silent-tail backdrop), else the operator gets credits-over-black and bounces it at eyeball. (This is the natural backdrop for the silent-tail model.)
- **Test refactor in the SAME commit:** ripping the RENDER ENGINES block from node 12's `_build_hud_dossier` breaks `tests/test_hud_dossier_bug3.py` — `test_dossier_has_render_engines_block_video_and_image` (`62-68`), `test_dossier_image_engines_from_meta_primary` (`71-87`), `test_dossier_image_meta_takes_precedence_over_legacy` (`90-98`); ripping node 84 tail-fill breaks `tests/test_video_render_path_cw4.py:285` `test_assemble_extends_to_floor_for_credits_tail_bug410`. Move/refactor ALL of these to the new `OTR_CreditsRoll` test (they become the new node's spec).
- Frame-append/concat mechanics + exact chain insertion = the build-design item for this slice.

### S1 — Cast & Voices, rendered in OTR_CreditsRoll (P1a)
Folded INTO S3's node (not a node-12 render). Per cast entry from the durable CastLock stamp: `NAME .... engine · voice_ref_id`. Character voice-signature from `meta.cast_voice_slots[...].speech_signature`. Announcer from cast. Music from `meta.music_engine` (S2). **Rip** the `_PRESET_DESC.get(preset,"")` empty fallback (`video_engine.py:1252/1258/1604`). No-fallback: a character missing its final stamp RAISES (S2 guarantees it is present).

### S0 — Font +50% + shared duration budget (P0) — AFTER the data path
Reordered after S2/S3 (codex CUT #2: it is visual tuning coupled to the clamps; do it once the surface is late + correct).
- Define ONE shared duration/readability budget for the credits roll (codex SHOULD-FIX #2): reconcile the HUD 20-90s clamp (`video_engine.py:1352-1356`) and the 45s mux tail budget (`otr_master_audio_mux.py:149-153`) for the NEW roll.
- Then apply +50% type via a single named `_HUD_FONT_SCALE=1.5` (fonts `video_engine.py:1331-1334`) and scale `_SCROLL_PPS` (`1319`, 65 -> ~98) so the taller roll stays inside the budget. No clamp-down "to be safe" — pick the scale, prove the budget.

### S4 — Polish (P3) — minimal
- Stale footer `OTR v1.0` (`video_engine.py:598,1500`) -> `v2.0-alpha`.
- Relabel left-panel telemetry `CORE/FLUX/MEM` (`1461-1463`) — the "FLUX" row is the LLM tok/s, not the image model.
- **CUT from first build (codex CUT #1):** the env-gated `OTR_CREDITS_DEBUG` extended/forensic card (recipe/quant/LoRA/VRAM/SHAs/grades). Forensic sidecar material; defer to a later pass.

---

## Viewer roll (single surface, green-safe, +50% type)
```
SIGNAL LOST
"<Episode Title>"
<style> · <est runtime> · <date*>

WRITTEN BY     <creative model>  (technical: <technical model>)
CAST & VOICES  <NAME → engine · voice_ref "signature">   (from CastLock final stamp)
IMAGES         <image engine per role>                    (meta.image_engines, S2)
MOTION         <video engine per role>                    (meta.render_engines)
MUSIC          <music engine · closing cue looped>        (meta.music_engine, S2)
NEWS SEED      <headline>
SEED / COMMIT  <cast seed · git short-sha*>

Made with OTR v2.0-alpha on <GPU*> — 100% generated
```
\* each of date / GPU / git SHA must have a declared ledger/source path (codex OPTIONAL #2) — no ad-hoc reads.

---

## AUDIO / MUX SEAM — operator model (SIMPLE silent-tail; supersedes v3 looped-audio)
**Operator's chosen model (2026-07-03):** the master audio stays ONLY as long as the body audio; the credit roll is appended as ADDITIONAL **silent** video frames at the END; the video ENDS when the credits end (no blank after). This is simpler than authoring extended/looped audio and it makes body byte-identity trivial (node 7 master is untouched). It SUPERSEDES the v3 dynamic-looped-audio approach and Fable BUILD-BREAKER #1 (no link-263 rewire to author extended audio).

Grounded mux facts (node 85 `OTR_MasterAudioMux`): it muxes node 93's silent video + node 7's master audio with `-c copy`, no `-shortest` (`otr_master_audio_mux.py:163-171`); byte-identity gate compares output vs the master-audio INPUT it received (`:177-184`), not a fixed baseline; the 45s guard fires only when `v_dur > a_dur + OTR_MAX_CREDITS_TAIL_S` (`:149-156`); it re-resolves the pre-rename master path via `_reresolve_master_audio` (`:189-235`, in `__all__`).

**The one thing to solve — make the guard CREDITS-AWARE (do NOT blind-widen it):**
- The credits tail is now an INTENTIONAL silent segment whose length is content-driven and can be long. A fixed 45s guard would wrongly reject a long roll.
- `OTR_CreditsRoll` computes its own roll duration (scroll_px / scroll_pps) and DECLARES it to the mux (new input on node 85, wired from CreditsRoll). The guard then permits `v_dur <= a_dur + declared_credits_tail_s + tol` — it still catches UNEXPECTED length mismatches (anything beyond the declared tail), which is the guard's real purpose. This respects "never blind-widen the 45s guard."
- Mux keeps consuming node 7's master audio (link 263 preserved). No extended-audio authoring.

**Constraints (unchanged):** video ends at credit-end, NO blank/black after the roll; body master audio byte-identical (trivially true now — node 7 untouched); the credits' silent-video BACKDROP (black vs looped last clip) is a look decision (see S3 / node-84 look contract).

**Note (tradeoff to confirm):** this model makes the credits tail SILENT — it drops today's closing-cue-under-credits music (node 12's credits-music loop). If music under credits is still wanted later, the looped-audio alternative (author extended master to cover the roll) is the documented fallback — but the operator's current call is SILENT credits.

**Verification:** frame-level smoke asserts the LAST frame is a credit frame (not black), the video ends with the roll, audio ends at body-end under the silent tail, no mux ValueError on a LONG episode (declared tail respected), body audio byte-identical.

## Cleanbreak orchestration (rip / paste / seams parallel)
Per slice: rip old surface + fallbacks; paste new render/stamp; fix seam upstream (data capture) AND downstream (workflow JSON + mux budget) in the SAME change; validate; commit+push. No dual path, no silent placeholder. Intermediate red between rip and paste accepted; green only at chunk boundaries.

**Fable-confirmed order (safest):**
1. **S2 alone, first, green.** Purely additive singleton stamps (image_engines + CastLock cast/voice + music), no JSON change, independently testable with the save()-raise guard. Commit+push.
2. **In parallel with S2 (the ONLY true parallel lane):** scaffold `OTR_CreditsRoll` as a NEW unwired file + its spec tests (the 3 relocated dossier tests + the node-84 bug410 test + duration/loop/backdrop tests). Unwired code is inert — no conflict. Everything else serializes (one-coder-window; all remaining slices touch the workflow JSON or `video_engine.py`).
3. **S3 + S1 as ONE ATOMIC commit — the guaranteed-RED window.** In the same change: rip node 12 HUD (dossier render + credits-music loop + treatment merge 2402-2428) + rip node 84 tail-fill dead code + wire CreditsRoll + rewire links 250/263-companion + new manifest + declared-tail links + JSON validator. Expected red between rip and paste: `test_hud_dossier_bug3` x3, `test_video_render_path_cw4.py:285`, any hidden HUD line-count pins. Do NOT land rip and paste as SEPARATE commits — a rip-only commit ships a production graph with no credits and a dead look contract.
4. **S0 after S3.** The 20-90s clamp (`video_engine.py:1352-1356`) + `_SCROLL_PPS` (`:1319`) die/move with the rip; the duration budget becomes CreditsRoll-owned, reconciled against the credits-aware guard.
5. **S4 last (parallelizable).** Note half evaporates: footer `:1500` + telemetry `:1461-1463` are HUD-panel code that dies in S3; only the CRT-renderer footer `:598` remains.

## Validation gate (every chunk)
- `OTR_WorkflowValidator` after any JSON change (widget count == INPUT_TYPES, wired-input names, link integrity).
- JSON round-trip, UTF-8 no BOM.
- Suite + Bug Bible (Windows venv, PYTHONUTF8=1, pytest -q -p no:cacheprovider). Fix hidden HUD/credits/dossier widget/line COUNT assertions in the same commit (grep with ignore-off; cf. the default_tts count pin in test_rip_sfx_broll_guard.py).
- **Frame-level live smoke (codex SHOULD-FIX #3):** render a short episode and verify the credits at the FINAL mux input / OBS copy (extract tail frames + view), NOT the sidecar treatment — S3 removes the early treatment merge, so the treatment is no longer the source of truth. Assert the LAST frame is a credit frame (not black), audio ends WITH the roll (no trailing silence/black), no mux ValueError, and the body master audio stays byte-identical (existing `test_audio_byte_identical`). Also render a LONG episode to prove the dynamic duration holds (roll + looped audio end together, no 45s trip).
- S3 = full kibitz + a Fable FINAL grounded gate before merge (CLAUDE.md §9 reality exception: a missed thread breaks every production render).

## Risks / traps (grounded)
- Render order (12 before 91/92) is the root cause — receipts MUST render late.
- CastLock overrides the planned voice fit — credit from the FINAL stamp only.
- Node 93 source-copy fallback would silently drop credits — use a dedicated node that RAISES.
- Green-only channel collapses amber/white/cyan to green luminance if composited through the blend.
- 45s mux tail budget couples to font size + roll length.
- Positional widgets_values drift; hidden test count pins.
- Music-engine provenance has no path today — must be stamped.

## Revision log
- **v2 (codex r1 folded):** unified LATE viewer roll in a new `OTR_CreditsRoll` node (not node-93 extension, avoids its source-copy fallback); Cast&Voices sourced from CastLock's FINAL stamp not the planned voice_cast_decision; S2 durable-persistence contract (image_engines + CastLock + music) moved FIRST with loud-fail + test-injection; font reordered after data path with a shared duration budget; debug card + candidate-B cut; frame-level verification.
- **v3 (agy manual pass + operator folded):** S2 singleton-copy gotcha; move the 3 RENDER-ENGINES dossier tests; green hierarchy by intensity; the earlier dynamic-looped-audio requirement.
- **v4 (Fable final gate + operator SILENT-tail model) — SHIP-WITH-FIXES, build-ready:** operator switched the audio model to SIMPLE silent-tail (master audio = body length; append silent credits video; video ends at credit-end) — SUPERSEDES v3 looped-audio and Fable BB#1; the mux guard becomes CREDITS-AWARE (CreditsRoll declares its tail duration to node 85; guard permits `v_dur <= a_dur + declared_tail`, still catching unexpected mismatches; never blind-widened). Folded Fable BB#2 (node-84 `otr_silent_composite.py` tail-fill / loop-last-clip LOOK CONTRACT is a second credits organ — rip in the same slice, reproduce the looped-clip backdrop in CreditsRoll, retire `test_video_render_path_cw4.py:285`) and BB#3 (`Ledger.save()` returns None on failure — stamps must check + raise). Insertion confirmed `93->OTR_CreditsRoll->85`, link 263 preserved, captions/bars at 93 so CreditsRoll can render full-color, node-12 finalize/rename left in place. Orchestration: S2 first (+ scaffold CreditsRoll parallel) -> S3+S1 atomic red window -> S0 -> S4.

## Grounding (re-verified 2026-07-03, real Windows files)
video_engine.py: fonts 1331-1334; _SCROLL_PPS 1319 + dur clamp 1352-1356; _parse_hud_data 1190, _build_left 1405 + CAST&VOICES 1476, _PRESET_DESC 1252/1258/1604; footer 598/1500; telemetry labels 1461-1463; dossier RENDER ENGINES 1140-1164; treatment merge ~2402-2428; finalize/rename ~2286-2315/2430+. otr_video_render_batch.py: _stamp_render_engines_meta 61-75 (singleton save). otr_image_gen_dispatcher.py: image_engines wire-only 669-673, returns 805 (no singleton). otr_master_audio_mux.py: OTR_MAX_CREDITS_TAIL_S 149, fail-loud 150-153, consumes node 93 output ~252. otr_post_upscale_procgen_blend.py: green-only colorchannelmixer 681-683; source-copy fallback 1044-1046. cast_lock.py: hybrid honored-or-fall-closed 565-596, _stamp 628-632, fallback stamp 443-444, wire-only (no singleton save). voice_cast_decision stamped _otr_casting.py 1835 -> OTR_LedgerScriptWriter.py 3040 (pre-freeze); cast_voice_slots 1834/3035. stable_audio_theme.py music:done:engine 173; node 83 done output unlinked in JSON.
