# Headless API Audio Soak — Test Plan (2026-06-03)

**Goal:** prove the *current* baked-in `otr_scifi_16gb_full.json` runs the audio pipeline
correctly out-of-the-box via the ComfyUI HTTP API, with tiny 30-word episodes, **aborting
each run the instant audio is done and the video (HuMo/Flux) phase starts** — so each run
costs seconds of audio, never the full video render.

**Operator decisions locked:** phased matrix (validated engines first, then a guarded
new-engine probe) · single-pass then repeat-green · 30-word episodes first.

---

## 0. Operating model

- **Where it runs:** on Jeffrey's Windows box, driven through **Desktop Commander (cmd)**
  with the venv python, against the **live ComfyUI at `localhost:8000`**. (The sandbox
  can't reach localhost; ComfyUI must be up for `/object_info` + `/prompt`.)
- **Prerequisite from Jeffrey:** ComfyUI Desktop running at `localhost:8000`. Optional but
  nice: LibreHardwareMonitor at `localhost:8085` (the VRAM readout the watcher uses).
- **Never mutate the canonical JSON.** `otr_api.load_workflow()` returns a deep copy; every
  episode/engine/knob change is patched **in memory** and submitted. The on-disk
  `otr_scifi_16gb_full.json` is never rewritten.
- **No video, ever, in this plan.** We abandon at the audio→HuMo boundary. Byte-exact
  render-twice (Gate B) and real video are explicitly out of scope.

## 1. Harness — reuse, don't reinvent

Everything needed already exists in `scripts/otr_api.py` (mature, carries the
BUG-LOCAL-002/027/029 fixes):

| Need | Function |
|---|---|
| read UI JSON (deep copy) | `load_workflow(path)` |
| live schemas | `fetch_schemas()` → GET `/object_info` |
| set episode length / engine / knobs **by widget name** | `patch_widget_by_name(wf, node_id, name, value, schemas)` |
| **UI→API translation (the fresh-translation gate)** | `workflow_to_api_prompt(wf, schemas)` |
| submit + validate | `submit_prompt(api_prompt)` (raises on `node_errors`) |
| poll completion | `poll_history(prompt_id)` |
| **abort at Flux** | `cancel_queue()` → POST `/queue {clear}` + `/interrupt` |
| live progress / VRAM | `scripts/smoke_watcher.py` (log tail + `/history` + `/queue` + LHM VRAM) |

The soak driver is a thin loop over these (adapting the existing `soak_bug027_028.py` +
`smoke_watcher.py` patterns). No new ComfyUI plumbing.

## 2. Canonical-JSON node map (verified today)

`1` Writer · `80` CastLock (`preserve_ledger`) · `81` CharacterVoices=**bark** ·
`82` Announcer=**kokoro** · `83` Theme=**musicgen** · `4` AudioEnhance ·
**`72`+`51` HuMo = the video boundary we abort at.**

---

## Phase 0 — Fresh JSON→API translation gate  *(MUST pass first)*

This is the step Jeffrey flagged: the JSON is very different from what the harness last
translated.

1. `wf = load_workflow(otr_scifi_16gb_full.json)`; `schemas = fetch_schemas()` (live).
2. `api = workflow_to_api_prompt(wf, schemas)` — if any node's saved `widgets_values`
   no longer matches the current schema serialization, this **raises a clear length /
   companion mismatch** (it fails loud, never silently misaligns).
3. Dry submit the *unmodified* graph; confirm `submit_prompt` returns a `prompt_id`
   with **empty `node_errors`**, then immediately `cancel_queue()` (we only wanted to
   prove acceptance).

**If it trips:** stop and fix — either a converter gap or genuine widget drift in the JSON
— before any render. This gate is the whole point of "be sure your translation is fresh."
**Exit:** clean translation + ComfyUI accepts the real graph.

## Phase 1 — Headless smoke (1 run, validated stack, abort at audio-done)

1. Patch a working copy: writer narrative length → **30** (the slot currently `350`);
   leave engines at bark/kokoro/musicgen, CastLock `preserve_ledger`.
2. `submit_prompt` → `prompt_id`.
3. **Abort watcher** (the "abandon when Flux starts" mechanism), polling ~every 3-5 s via
   the `smoke_watcher` readers, fires `cancel_queue()` on the FIRST of:
   - the audio-assembly-complete marker in `otr_runtime.log` (the EpisodeAssembler /
     `episode_audio` line) — *primary, identified on this first run and then pinned*;
   - HuMo node `72`/`51` appearing as the executing node (comfyui.log) or a VRAM jump as
     HuMo loads;
   - a hard per-run timeout (generous for 30-word audio, far short of video).
4. **Capture the exact audio-done log marker here** and bake it into the soak so later
   aborts are deterministic, not timeout-driven.

**Pass criteria (audio-path success):** voices (81) + announcer (82) + theme (83) all
produced non-empty audio; `episode_audio` assembled; CastLock stamped a `v2/*`
`voice_preset` per character (the Sprint 2 path); **no** `PARSE_FATAL` / Traceback /
guardrail-raise / OOM in either log. (`scripts/_otr_post_validator.py` for the structured
check.)

## Phase 2 — Soak: validated 3, one-factor-at-a-time  *(single pass, repeat green)*

Base config = Phase-1 config. Then **vary one knob at a time** (OFAT — keeps it small +
diagnostic, ~8-12 runs, not a 32-cell cartesian):

- `cast_voice_policy` (node 80): `preserve_ledger` ↔ `auto_registry` — *both must yield a
  voiced cast; directly exercises the Sprint 2 CastLock assignment.*
- `num_characters` (writer): 1 / 3 — *bark voice assignment across cast sizes.*
- LEMMY: forced-off ↔ forced-on — *the pre-locked cameo voice + one-fewer-open-slot path.*
- `OTR_NAME_MODE`: `pool` ↔ `llm_slot_fill` — *replay holds for both.*
- captions: off ↔ on (`OTR_CAPTION_STYLE`) — *burn-in path.*

Each run = 30-word, audio-only, aborted at the boundary. **Single pass** through the OFAT
set; then **re-run each config that passed 2-3×** to catch flakiness/nondeterminism.
Green here = the shipped audio pipeline is solid across its real knobs.

## Phase 3 — Guarded new-engine probe (early F signal)

Only after Phase 2 is green. Swap the engine widget per role and run isolated:

- voice (node 81): `indextts2`, then `chatterbox`;
- announcer (node 82): `chatterbox`;
- music (node 83): `stable_audio`.

**Guarded:** one engine at a time; the `smoke_watcher` LHM VRAM read enforces the 14.5 GB
ceiling; a tighter per-run timeout + `cancel_queue()` on hang or on a `MISSING_MODEL` /
dep / OOM marker. A brick here is **logged, not fatal** — it's exactly the early read on
whether the promotion-tier engines load on the Blackwell box ahead of the real F pilot.
This probes without committing to promotion.

---

## Reporting

Per-run row: `config · status(PASS/FAIL/ABORTED-OK) · audio wall-time · VRAM peak ·
failure reason`. One compact table at the end. Any real bug → `BUG_LOG.md` immediately
(the repo rule), not batched. New-engine probe results feed the F/promotion decision.

## Out of scope

Video / HuMo / Flux render · byte-exact render-twice (that's Gate B, your GPU) · any write
to `main`, the canonical JSON, or the model defaults. This plan only proves the audio path
runs correctly box-fresh and surveys the options.
