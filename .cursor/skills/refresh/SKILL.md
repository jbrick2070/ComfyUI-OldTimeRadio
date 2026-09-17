---
name: refresh
description: Regenerates the Old Time Radio video-lane ecosystem poster from live shipping graphs and economic tiers, then runs obligatory Composer QA as the slow lane of publishing. Use when the user says /refresh, refresh the video map, update the lane poster, or wants a Grok image of the video subsystem and local/cloud tiers without repeating the layout.
disable-model-invocation: true
---

# /refresh

Two lanes, one command:

- **Fast lane** -- redraw the poster from disk so the operator can see the ecosystem.
- **Slow lane of publishing** -- obligatory Composer QA, copied from the 2026-09-14 cloud/Vidu campaigns. A picture is not a ship. Do not commit shipping graphs, do not bump `pyproject.toml`, and do not call origin green until this lane returns **HOLDS** with **MUST-FIX: none** and the driver has grounded every claim against the Windows files.

Do not reuse yesterday's engine names. Do not ask the operator to restate the layout or the QA brief.

## Fast lane -- poster

1. Run the live brief (venv python, `PYTHONUTF8=1`) from the repo root:

```
python .cursor/skills/refresh/scripts/brief.py
```

Canonical VideoDirector pins, every `SHIPPING_SET` local graph, every cloud graph (act count, writer, video, still).

2. Read only if the brief is incomplete: `docs/TIER_MATRIX.md` (profiles win), `apple/VIDEO_MODELS.md` families, `scripts/build_variants.py` `SHIPPING_SET`.

3. Generate a 16:9 Grok image (`GenerateImage`).
   - `filename`: `otr-video-lane-ecosystem.png`
   - `aspect_ratio`: `16:9`
   - Fill LAYOUT with **today's brief**.
   - Do not re-embed as Markdown.
   - One-line caption: canonical / cheap cloud / deluxe / counts from the brief.

4. Do not commit the PNG unless asked. Do not invent engines.

## Slow lane -- Composer QA (obligatory)

Launch a **Composer** subagent (`composer-2.5-fast`, `generalPurpose`). Brief it to **REFUTE**. Default **REFUTED** when a claim is ungrounded. It reads the live tree, not chat memory.

This is the same lane that, on 2026-09-14:

- HOLDS cheap cloud = Vidu on all four slots; deluxe = `cloud_wan_i2v_audio`; canonical stays viz
- HOLDS Gemma padded-key strip as a **separate** commit
- Caught origin `7c856ca4` **red** (`build_variants --check` 22 failures) because still/low/animatediff graphs were not re-emitted after director labels

Do not skip it because the poster looks right. Do not skip it because the suite was green on a mixed tree. Do not publish (commit named shipping files, registry bump, "origin is green") off the fast lane.

### Standing hunts (paste into the Composer prompt)

- Cheap cloud trio pins the same hosted video engine on 3 VideoDirector roles + `VideoRenderBatch.engine`. Writer/stills/TTS/music match the cheap SKU. Act counts are the paid axis (1/3/7).
- Deluxe is the exception: different writer + different video engine. Not the cheap engine. Not Seedance unless disk says so.
- Canonical VideoDirector is viz, not hosted. Local `otr_8gb_*` / `otr_16gb_*` / Mac / AMD / CPU saved engines are not rewritten to Vidu.
- `SHIPPING_SET` is 16 local + the live cloud ids. `MAX_ACT_COUNT` must admit every shipped act_count (7-act graphs brick if origin is still 6).
- After `_director_option_value` labels every registered engine, `python scripts/build_variants.py --check` on a **clean** checkout. A dirty-tree `--check` of 0 does not prove origin.
- Saved combo strings are live `INPUT_TYPES` choices (suffixes). Bare ids on VideoDirector are a canvas coerce-to-index-0 bug.
- Named-file commits only. Leave JSON-strip, canvas-safety catalog, `_tmp_*`, secrets, `pyproject.toml` out unless that pile is the thing being published.
- Green suite on mixed unstaged work does not prove the named subset.

### Return format Composer must use

```
VERDICT: HOLDS | REFUTED
MUST-FIX: file:line or none
NITS:
COMMIT SHAPE: named files in vs stay out
CLAIMS CHECKED:
```

### Driver is sole judge

Ground every Composer claim on the real files before folding it. A HOLDS you did not re-read is not a publish. Loop until HOLDS + MUST-FIX none, then the slow lane is clear.

## Layout (stable; pins are not)

Cinematic 1930s old-time radio broadcast poster. Dark walnut, brass, amber CRT, film grain. Title: `OLD TIME RADIO  VIDEO LANE ECOSYSTEM`. Subtitle: `audio first, then picture`. No real-company logos, no QR, no watermarks, no UI screenshots. Large readable labels.

**Left -- canonical default (free).** Three vacuum tubes = announcer / music / character. Canonical viz pins from the brief. Badge: FREE. No weights. No GPU. No credits.

**Center -- local economy, cheapest at the bottom.** CPU/viz, still, 8 GB, Mac 16 GB, 16 GB (silent / foley_plus / mime), AMD still if shipped. Caption: VRAM is the price of local motion.

**Right -- cloud bazaar, easter egg, not the front door.** Cheap SKUs: one hosted video engine, pay for length. Deluxe exception: its own engine and writer. Back door: Kling / Seedance / Veo for the brave (dropdown, not saved default).

**Bottom ribbon.** Three pipes into a splicer, one 16:9 film can `otr/obs`.

## Hard product rules (override only if the brief contradicts them)

- Canonical / first-run is viz, never a paid hosted default.
- Local shipping graphs keep their machine lanes.
- Cheap Comfy Cloud graphs share one hosted video engine; they differ by act count.
- Deluxe is the exception row.
- Registry is the menu: hosted pick is the enable; missing key fails at invoke.
