# STILL-SPINE SPRINT PLAN -- build-ready (2026-06-10 night)

GOAL (operator): after this sprint, ONE fresh 30-word production render looks
as good as 6/5 -- top-notch macro-radio stills + shot-accurate in-character
portraits, SAVED IN THE EPISODE FOLDER, feeding the video engines as inputs --
and the still layer is expandable to 3D + future consumers. Better inputs ->
better outputs, everywhere.

Source of truth: `roundtable/pass01_plan.md` (panel-hardened, 2 passes,
CONVERGED) + the 7 folded items in `roundtable/pass02_judgment.md`. The 6/5
reference contract is `docs/2026-06-10-brief-downstream-gaps/
legacy_otr_video_plan_e74a3ce.py.txt` (PASS-1 portraits / PASS-2 scene stills /
PASS-3 5-layer composite prompts). GRAIN-OF-SALT RULE (operator): restore the
proven 6/5 inputs; reject invention that drifts from them.

## The wiring I/O map (QA this, seam by seam -- the operator's emphasis)

Every ticket below names its seam contract. The full chain:

| # | Seam | IN | OUT | QA assert |
|---|------|----|-----|-----------|
| W1 | writer -> ImagePromptGen | ledger (meta.story_brief_terms, cast w/ character_description, lines) | versioned `{"objects":[...]}`: portraits + scene stills (open/announcer/outro) | objects exist for b000-open WITHOUT ShotLock (pure-helper derivation); person/gear guards ONLY on kind=portrait |
| W2 | ImagePromptGen -> Dispatcher | objects (kind, role, beat_id/char_id, w/h, prompt, prompt_hash, seed) | render requests per object | role/slot from the OBJECT (announcer/music/other image-model slots honored); w/h reach the engine call; V-7 request-hash seeds |
| W3 | Dispatcher -> disk + ledger | rendered stills | `episodes/<ep>/stills/*.png` + `stills_manifest.json`; ledger images[] rows {object_id, kind, beat_id?/char_id?, episode-local path, content_hash, provenance} | files exist BEFORE the video phase; cache HIT still materializes a copy + fresh row; episode_id wired from the json |
| W4 | Dispatcher -> render gate | image_done token | gate input on the video render node | video render cannot start before stills (ST-0 probe; add the input if missing) |
| W5 | ShotLock -> shots | beats (normalized char_id) | shot rows w/ role, char_id, start_s (synthetic), creative | unchanged this sprint; pins stay green |
| W6 | driver -> engines | ledger images[] + shots | requests: asset_refs.init_image by ENGINE FAMILY (audio_driven_face->portrait(char_id); image_to_video/static_motion->scene still(beat_id)); text engines unchanged | LOUD fallback on missing still (never silent empty init into a fail-closed engine); `_init_source`/`_init_image` stamped on the request |
| W7 | run_episode -> trace/report | request stamps | trace rows + node-92 report carry init_source/init_image (+ the r5 prompt_* fields) | mechanical acceptance: every static_motion/i2v beat shows init_source=scene_still, every talking head =portrait |
| W8 | clips -> composite -> blend -> mux | clip manifest (start_s fallback incl. synthetic) | positioned timeline, frozen audio untouched | r5 pins green; audio byte-identical; tpad clone fills short clips |

## Tickets (build order; commit+push per green chunk -- operator git policy)

**ST-0 PROBES (do FIRST; they gate ST-5/ST-6)**
1. `still_kenburns` external-init: read its render path; if it cannot take an
   arbitrary still path via asset_refs.init_image, ADD it (the 6/5 look ships
   through this engine). CPU-testable.
2. The video render node's gate: if `OTR_VideoRenderBatch` lacks an
   `image_done` (forceInput STRING) input, ADD it (mirror the audio_done
   pattern). Update the saved json + validator pins in the SAME commit.
3. (optional, env-gated) wan_i2v landscape-init snap: one probe clip; record
   crop behavior in the plan before making it a default.

**ST-1 Shared helpers** (`nodes/_otr_story_brief_helpers.py`)
`get_open_subject(role, synthetic)` (the r5 driver wording MOVES here; driver
refactors to call it -- one source of truth) + `compose_still_prompt(...)` in
the legacy 5-layer order (subject / setting top-2 / framing hint / TRIMMED era
tail / style tail) + `era_tail_profile="still"` (atmosphere line + palette
top-2 + lighting top-2). Parity test: driver text prompt and still prompt
share the leading subject.

**ST-2 Still objects** (`nodes/otr_meta_brief_image_prompt.py`)
One versioned object schema (portraits MIGRATED to it in the same patch);
scene targets derived WITHOUT graph reorder via `derive_opening_music_beat` +
the line role map; guards branch by kind BEFORE running; scene stills get
no-text clause, landscape dims (canvas /32); portraits keep 832x1216 + person
guard + gear scrub (ANNOUNCER exempt rules unchanged).

**ST-3 Dispatcher** (`nodes/otr_image_gen_dispatcher.py`)
Consume objects; role/slot per object; `episode_id` in INPUT_TYPES; save ALL
stills to `episodes/<ep>/stills/` + write `stills_manifest.json`; ledger rows
carry episode-local paths; cache key gains kind/w/h; cache hits materialize
into the current episode + append a fresh row; global pool stays this sprint
(retirement = follow-up after the tracked-reader sweep).

**ST-4 Driver** (`nodes/_otr_video_engines/render_driver.py`)
`_still_index(ledger)` keyed by beat_id (kind=scene_*); family-based init
selection; LOUD missing-still fallback to today's behavior; `_init_source` /
`_init_image` request stamps copied to trace rows (the `_prompt_*` pattern);
`build_clip_manifest` rows gain init_source.

**ST-5 Conditioned motion**
kenburns drifts the beat's scene still (v1 default for open/outro when the
engine map sends them there -- NOTE: production currently routes opens to LTX;
the conditioned-look proof runs via the Director music/announcer slots or the
acceptance render's trace, no new widgets); wan_i2v init=scene still where
OTR_ENABLE_WAN_I2V=1. **LTX img2vid: CUT from v1** (wrapper decode-band risk
proven 2026-06-10); LTX keeps its r5 text prompts (which now share ST-1
subjects). Future probe ticket only.

**ST-6 Saved-json wiring** (`workflows/otr_scifi_16gb_full.json`, IN PLACE)
episode_id wire into the dispatcher + image_done gate into the render node;
update `test_production_workflow_visual_structure_pinned` + the workflow
validator in the SAME commit. NO other graph changes; no runner patches.

**ST-7 Tests (CPU, suite-resident)**
Schema emission (b000 present sans ShotLock); guard branching by kind;
dispatcher slot resolution + episode-dir save + manifest + cache-hit
materialization + w/h in key; driver family init + LOUD fallback + trace
stamps; parity (ST-1); era-tail profiles; determinism (same ledger -> same
seeds/hashes); ordering (stills before video via the gate); ALL r5 pins green.

**ST-8 Acceptance (ONE fresh 30w production render)**
episodes/<ep>/stills/ holds open still + portraits + manifest; the open still
is macro-radio 6/5-style, prompt leads with the ST-1 subject (string assert +
operator eyeball); trace: init_source=portrait on every talking head,
=scene_still on every static_motion/i2v beat, LOUD text-fallback lines
otherwise; r5 gates green (diversity, cap/floor 169, captions, credits,
duration, byte-identical, obs ONE new AAC final); suite + Bug Bible green.
Extract eyeball frames; STOP for the operator verdict (the verdict gates TAGS,
never pushes).

## 3D + future consumers (designed-in, NOT built now)

Ledger image rows carry object_id/kind/beat_id/char_id/content_hash -- the 3D
plan's character-level image routing (VIDEO_OPTIN_GOFORWARD_PLAN Phase 5 +
the image-routing must-fixes) consumes character stills from the SAME rows
when 3D reopens. No portrait-only assumptions anywhere in the new schema.

## Comfy-node + environment gotchas (QA checklist for the builder)

- **Python module cache**: ComfyUI does NOT reload edited .py -- relaunch the
  headless server after every code change (`scripts\_otr_soak_server_launch.cmd
  <log>`; kill the :8000 listener first). Comfy DESKTOP needs an app restart.
- **INPUT_TYPES changes** (ST-0.2, ST-3): the saved json must be relinked IN
  PLACE and the validator/pin tests updated in the SAME commit, or /prompt
  validation fails (the 88a94b8 stale-slug lesson).
- **forceInput STRING tokens** are the gate pattern (audio_done precedent).
- **Graph ORDER**: image gen runs BEFORE ShotLock -- scene targets must come
  from pure helpers on the ledger (never from video.shots).
- **LTX decode band**: 169f..233f proven at 1472x832; below 169 the wrapper
  VAEDecode throws (tensor 256-vs-128). OTR_LTX_MAX_FRAMES /
  OTR_LTX_MIN_DECODE_FRAMES guard it; do not "optimize" the floor away.
- **Composite**: tpad=clone hold fills short clips; manifest rows MUST all
  carry start_s or the whole episode silently degrades to sequential mode.
- **HuMo stages init via `stage_into_comfy_input`** (ComfyUI input dir) --
  episode-local paths must remain stage-able (absolute paths fine).
- **cmd gotchas**: multi-line `python -c` mangles -- write script FILES;
  `~`-refs not carets in git ranges; PowerShell Select-String is
  case-INSENSITIVE by default.
- **Launcher env**: HF_HOME=C:\ComfyUI-Models\huggingface; OPENROUTER key
  hydrates from HKCU (setx is invisible to the DC shell); TEMP pinned under
  output\otr\tmp; OTR_REAL_OUTPUT pins the output tree.
- **uv trampoline**: every venv python shows as TWO processes (shim + worker)
  with identical cmdlines -- not a duplicate run; the :8000 listener lives on
  the worker pid.
- **The sqlite "unable to open database file" boot error is benign** (RAM
  pressure cache takes over).
- **Suite**: `Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q`
  from the repo root (3863/0 + 28 skip baseline); Bug Bible: cd the
  survival-guide repo + venv python + RELATIVE `tests\bug_bible_regression.py`.
- **Git policy (operator, 2026-06-10)**: commit AND push every green chunk to
  v2.0-alpha immediately; verify HEAD==origin, no 0-byte, no BOM, AST parse.
  The eyeball gates TAGS/promotions only.

## Known adjacent work (do NOT pull in)

M4->HuMo creative seam (separate ticket); LTX img2vid probe; global stills
pool retirement; whiny-voice / switchable-workflow / 3D / LTX-AV lane (P4,
other window) -- ALL PARKED or other-window.
