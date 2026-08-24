# CODEX BUILD BRIEF -- implement the Ghost Signal video lane

You are implementing a new video lane in this repository, following its existing
lane contract. **This is a build task, not a design task.** The design is
settled; your job is to make it real, conform to the gates, and prove it.

Repo root: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
Python: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` (torch 2.10)

---

## READ BEFORE YOU WRITE ANYTHING

1. **`docs/2026-08-22-GHOST-SIGNAL-SPEC.md`** -- THE SPEC OF RECORD. Read it at
   the moment you start; it is under active revision and the version on disk
   wins over anything quoted elsewhere.
2. **`docs/VIDEO_LANE_PREFLIGHT.md`** -- the 8 gates, machine-enforced by
   `tests/test_lane_preflight_matrix.py`. **A hard FAIL on any gate stops the
   lane.**
3. The **"HOW TO ADD YOUR OWN VIDEO ENGINE"** docstring at the top of
   `nodes/_otr_video_engines/__init__.py`.
4. `docs/EXTENDING_OTR.md` lines 9-18 -- the adapter drop pattern.
5. **`CLAUDE.md`** at the repo root -- hard operator rules. Non-negotiable.
6. `docs/2026-08-22-lofi-video-lane-PLAN.md` -- the wiring surfaces and the
   grounded reasoning behind each ruling.

**Reference implementations to imitate, in this order of closeness:**
- `nodes/_otr_video_engines/eng_ltx_8gb.py` -- the current cheapest lane; the
  best structural model for a normal selectable in-process motion lane.
- `nodes/_otr_video_engines/eng_fastwan_8gb.py` -- the smallest heavy adapter;
  good model for a lane that subclasses and overrides a recipe cleanly.
- `nodes/_otr_video_engines/motion_common.py` -- `MotionEngineBase` (the AS-3
  lease, the V-4 patcher-detach teardown).
- `nodes/_otr_video_engines/registry.py` -- the `VideoEngine` Protocol.

---

## PHASE 0 -- HARD GATE. NO ADAPTER CODE BEFORE THIS IS DONE.

**AnimateDiff-Evolved is NOT installed.** Its real node class names, input
signatures, context-option nodes and motion-module loader are therefore
UNKNOWN. This repo's established pattern is explicit -- see the `__init__.py`
comments for `wan_ti2v` ("its 5B core node class was captured from a live
`/object_info` before coding") and for `ltx_8gb` (same).

**DO NOT INVENT NODE CLASS NAMES. DO NOT GUESS INPUT SIGNATURES.**

Phase 0 deliverables, in order:

1. Install `ComfyUI-AnimateDiff-Evolved` into
   `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\`.
2. Fetch the motion module the spec names (`mm-p-0.5`) plus an SD1.5 checkpoint.
   Record exact filenames, byte sizes and SHA-256 for each. **File SIZE is a
   design input, not a footnote** -- the lane has a 4 GB ceiling and the weights
   nearly fill it.
3. Boot the ComfyUI API headless (see `CLAUDE.md` sections 4 and 5 -- reset
   first, `PYTHONUTF8=1`, launch via `scripts/_otr_soak_server_launch.cmd` as
   `-FilePath`, never `cmd.exe /c`).
4. **Capture `/object_info`** and write the AnimateDiff-relevant subset to
   `docs/2026-08-22-ghost-signal-object-info.json`, committed. Every class name
   and input key you later use in code must appear in that file.
5. Run the spec's **measurement #3** (real node surface + module inventory) and
   as much of **#1 and #2** (VRAM under the 4 GB ceiling, two-canvas bench) as
   you can. Record receipts.

**Report Phase 0 results and STOP if the 4 GB ceiling is breached at the
declared canvas.** That is a design decision for the operator, not something to
code around. Say plainly what fits and what does not.

---

## PHASE 1 -- the adapter

Create `nodes/_otr_video_engines/eng_ghost_signal.py`.

**Contract (from `registry.py:58` `VideoEngine`):** `name`, `roles`,
`default_roles`, `commercial_clean`, `requires_flag`, `load`, `unload`,
`family`, `required_inputs`, `invocable`, `invocability_reason`,
`frame_contract`, plus the render lifecycle `assert_usable` / `prepare` /
`render_clip` / `canonicalize` / `teardown`.

**Declarations that are NOT negotiable** -- each traces to a ruling or a gate:

| Declaration | Value | Why |
|---|---|---|
| `family` | `text_to_video` | R1 no still. Requires `text_prompt`, NOT `init_image` (`schemas.py:31`) |
| `accepts_still` | **`False`, DECLARED, with its reason in a comment** | G3.6. The gate polices SILENCE, not the value -- a lane that stays quiet resolves False by getattr fallback and nothing reports it. `tests/test_still_spine_engine_coverage.py` sweeps the live registry |
| `continuity` | `CONTINUITY_NONE`, **passed explicitly** | G3.3, and the lane-10 lesson: six lanes once inherited the right value because nobody decided it |
| `render_canvas` | per SPEC section 5 | G2.1 -- both axes /32-legal, or a documented exemption |
| `render_aspect` | `"wide"` | true 16:9, full-frame delivery |
| `still_plan` | declared, audit-clean | G7.4 |
| roles | all three: `announcer_visual`, `music_visual`, `character_video` | R2. **NOTE:** `slot_matrix.py` `ROLE_TO_PROFILE_KEY` maps `character_video` -> profile key `character_visual`. That is a MAPPING, not a bug -- do not "fix" it |

**Base class.** `MotionEngineBase` (`motion_common.py:606`) gives the AS-3 lease
and the V-4 patcher-detach teardown, both of which this lane wants.
**But you MUST override two inherited behaviours and PROVE they are
unreachable, not merely unused:**
- `accepts_still` inherits `True` -> override to `False`.
- `compute_real_frame_budget`'s loop/ping-pong extension path **must never
  run.** R3 and `acceptance.py` `DELIVERABLE_EXTENSION_MODES = ("none",)`,
  graded on every beat by `grade_no_mirror`. A delivered clip declaring anything
  but `"none"` fails.

If proving unreachability is awkward, a leaner parent is the better answer --
say so rather than shipping a live path you intend nobody to call.

**Hard invariants:**
- **Cold-import clean (V-12).** Module scope imports NOTHING heavy. torch and
  the AnimateDiff classes are lazy, inside `load` / `render_clip`.
  `test_cold_import_no_heavy_libs` asserts this.
- **`assert_usable` fails CLOSED with a NAMED `EngineUnusable`** when the node
  pack, the motion module, the checkpoint or the VAE is absent (G1.2). Never a
  swallowed import. Resolve weights via `folder_paths` or a documented env pin,
  never a bare `os.path.exists` on a hardcoded default (G1.1).
- **Silence (G5.1):** `canonicalize` runs `validate_silent_clip_contract` on the
  lane's OWN emitted file. A `has_audio: False` literal is NOT evidence.
- **Sequential VRAM staging is part of the design, not an optimisation** --
  encode text, free; sample, free; decode, free. Mirror the INTENT of
  `eng_ltx_8gb.py`'s between-stage frees. **Never `unload_all_models`.**

---

## PHASE 2 -- the prompt composer

Per SPEC section 2. This is the lane's heart: with no still, the prompt carries
subject, style AND motion.

- **Do NOT reuse `_LTX_MOTION_PROMPT_MAX = 240`** (`render_driver.py:1345`). It
  is ours, and it is sized for a still-carried, motion-only prompt --
  `render_driver.py:1293-1300` says so in the repo's own words. The lane gets
  its OWN constant, and that constant ships with its derivation in a comment.
- **Do NOT paste pack `motion_registers` verbatim.** Every pack's registers open
  with `"Continuous shot, same console throughout."` (see
  `nodes/visual_styles/paper_origami.json:27-29`), which assumes a persistent
  anchor this lane does not have. Distill per the SPEC.
- **DANGER, do not widen:** `render_driver.py:2791-2795` appends
  `", stable centered subject, full face clearly visible, ..."` for
  non-announcer/music roles, gated on `engine_id.startswith("ltx")`. That phrase
  invites this lane's named failure mode (face soup). Ghost Signal is naturally
  excluded -- **keep it that way; never widen that condition.**
- Style authority stays where it is: `prefix_style_cue`, pack tails,
  `effective_negative`. **Never an engine-side style constant** -- that is the
  PBUG-20260817-01 defect class.
- Per-beat story text comes read-only through `resolve_motion_clause_text`.

---

## PHASE 3 -- wiring (nothing ships unwired)

Per `CLAUDE.md` section 0: **code that is not wired into the canonical JSON is
DEAD.** All of this lands in the SAME change as the adapter.

1. **Register:** guarded `try: from . import eng_ghost_signal` in
   `nodes/_otr_video_engines/__init__.py`, matching the existing style.
2. **`CAPABILITIES` row** per `EXTENDING_OTR.md:9-18`.
3. **Public menu:** ONE row in `_otr_shared/public_engines.py` `_PUBLIC_ENGINES`.
   **A module-scope bijection assert fires at IMPORT** -- two public ids on one
   internal id empties most of the ComfyUI node menu rather than failing one
   lane. Naming convention is `<model><version>_<low|high>_<capability>`, and
   **the `low`/`high` token is FORBIDDEN until a measurement receipt exists**
   (the lane-8 rule). Use the id the SPEC declares.
4. **`_otr_shared/content_oracle.py`** engine -> family row.
5. **`workflows/otr_canonical.json`** -- the source of truth. Schema is
   litegraph; `widgets_values` is POSITIONAL, so only ever APPEND a new optional
   widget at the END (inserting mid-list shifts every saved value silently --
   BUG-LOCAL-097). Re-validate after: `OTR_WorkflowValidator`, a JSON
   round-trip, and a link/widget audit.
6. **Node 87 / `OTR_VideoDirector` strings are GENERATED, never hand-typed**
   (G7.2). Regenerate variants in the same commit.
7. **`docs/ENGINE_MATRIX.md`** -- regenerate with
   `python tools/engine_matrix.py`; it is a live drift gate (G7.3).
8. **A profile** in `config/profiles/` wiring the lane across all three visual
   roles.

---

## PHASE 4 -- prove it

1. **Full suite + Bug Bible** after every code chunk. `$env:PYTHONUTF8=1`,
   `pytest -q -p no:cacheprovider`. The Bible lives in a SEPARATE repo:
   `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` -- `cd`
   to its root and use the RELATIVE path `tests\bug_bible_regression.py`.
2. **New tests**, mirroring the existing lane tests
   (`tests/test_ltx_8gb_canonical_canvas.py`, `tests/test_fastwan_8gb.py`):
   canvas declaration, frame contract, `accepts_still = False` declared,
   extension mode `"none"`, the composer's budget and trim law, cold-import.
3. **The preflight matrix:** `tests/test_lane_preflight_matrix.py` must pass.
   Produce the receipt `docs/VIDEO_LANE_PREFLIGHT.md` asks for:
   `VIDEO_LANE_PREFLIGHT receipt: <lane> | <date> | matrix sha256 <...> | suite
   run <path> | smoke receipt <path> | verdict PASS/FAIL`.
4. **Solo smoke (G8.1):** one real render on the declared boot lane -- canvas
   probed, frame count exact, silence probed, VRAM peak receipted **against the
   4 GB ceiling**, trim ratio logged if tail-trim fired.
5. **PUBLISH TO `otr/obs/`.** This is the operator's success signal and it is
   not optional: *"a test is not complete unless published to obs."* A leg that
   does not reach `otr/obs/` did not pass, however green the logs are. If a leg
   runs more than 5 minutes with nothing in `otr/obs/`, treat it as failing and
   read the leg log.

---

## HOUSE RULES -- violations are defects

- **UTF-8, no BOM.** Never `Set-Content` / `Out-File` for Python source.
- **No curse words and never the name "dummy"** in code, comments, logs or
  commit messages. Use "placeholder" or a descriptive name.
- **Clean logs, meaningful names.** The reader matters.
- **PowerShell:** chain with `;` not `&&`. Never `python -c "..."` with nested
  quotes -- write a temp `.py`, run it, delete it. First quoting error means
  STOP escaping and switch to a script file.
- **The ~60s command ceiling:** background long jobs to a log and poll the log.
- **Git:** branch is `v2.0-alpha`. Commit and push together, per green chunk.
  Do NOT push twice. After a push verify HEAD == origin, no 0-byte files, no
  BOM, AST-parse every touched `.py`.
- **Fix at the root cause. Never a shim or a band-aid.**

## DO NOT

- Do not invent AnimateDiff node class names (Phase 0 gate).
- Do not add a still, an init image, ControlNet or IPAdapter (R1).
- Do not ping-pong, mirror, loop-fill or pad frames (R3).
- Do not add an upscaler -- no ESRGAN, no SeedVR2, no hi-res fix (R8). The
  delivery is an ENLARGEMENT.
- Do not add profanity or violence filtering to the generation path.
- Do not modify `_LTX_MOTION_PROMPT_MAX` or any other lane's recipe.
- Do not widen `render_driver.py:2791-2795`'s `startswith("ltx")` condition.
- Do not tidy, move or gate anything out of `otr/obs/`.

## DELIVERABLE

Working, wired, tested code plus the preflight receipt and a published episode
in `otr/obs/`. If a phase blocks -- especially the 4 GB ceiling in Phase 0 --
finish everything that does not depend on it, then report exactly what is
blocked and why. Scaling the work down is the operator's call, not yours.
