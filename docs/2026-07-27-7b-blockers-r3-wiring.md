# r3 WIRING -- clear the 7b blockers, reach 7d (after 7c)

Supersedes `docs/2026-07-27-7b-blockers-r2-plan.md`. Base revision
**`c8cf0b07`** on `v2.0-alpha` (r2 correctly flagged the stale pin).
Suite 6925/27/1. Bible 17. Canonical `5377914B`.

**Target: the cleanest architecture for the future set, not the smallest diff.**

## 0. VERIFIED THIS WINDOW (receipts, not claims)

* Server path is a **junction** to this repo -- identical SHA-256, same HEAD.
  Live results are valid.
* **`ltx_8gb` live: asked 25, got 25**, 20.8s, clip on disk, VRAM 3004 MB.
  This is `7d-preflight`, explicitly NOT qualification.
* Clean 2-segment chain lanes: `ltx_8gb` 169 -> `[161,9]`, `wan_i2v` 209 ->
  `[177,33]`, `wan_ti2v` 193 -> `[177,17]`. All viz engines (the canonical
  defaults) have NO ceiling, so the default route can never split.
* `eng_ltx_8gb._node_candidates` returns `CheckpointLoaderSimple` + `CLIPLoader`
  on EVERY render (`eng_ltx_8gb.py:295-311`) -- 7d's "ONE heavy load" is not
  implementable at HEAD. The 7c gate is real.

## 1. SETTLED AT r1 (do not relitigate)

* B2: a SEPARATE `frame_contract_env_snapshot`, never inside
  `routing_env_snapshot` (whose fingerprint feeds a TERMINAL route-drift check,
  `otr_shot_lock.py:1453-1475`). But see 2.1 -- r2 found the r1 answer
  incomplete.
* B3: typed taxonomy. `UnresolvableEngineError` (unregistered/stub, stays
  caught) vs `InvalidEngineConfigError` (terminal). Three sites:
  `frame_contract.py:243`, `otr_shot_lock.py:1154`, `render_driver.py:3435` --
  each becomes `except UnresolvableEngineError:`.
* B4 narrowed: only estimate/absent/zero producers get a decoded count. Local
  engines returning an exact encoded array count are already honest
  (`ltx_8gb`, proven live above).
* A hard **7c gate** sits between C6 and 7d. `ltx_8gb` is the 7d lane; the
  `ltx_video` + `OTR_LTX_LOOP_VIA_REVERSE=off` branch is CUT.

## 2. WHAT r2 CHANGED

**2.1 -- THE r1 B2 ANSWER WAS INCOMPLETE, and this is r2's best catch (agy).**
`IS_CHANGED` is a ComfyUI CACHE hook. A headless API run validates the stamped
`routing_env_snapshot` against live at `render_driver.py:3324-3326` -- it does
NOT consult `IS_CHANGED`. So fingerprinting the frame-cap env ONLY into
`IS_CHANGED` leaves **every headless run free to execute a stale stamped plan
under changed caps.** The snapshot must therefore be BOTH:
(a) fingerprinted into `IS_CHANGED` (interactive cache correctness), and
(b) STAMPED on the ledger and re-validated at the render boundary against live,
terminal on mismatch -- exactly as the routing snapshot already is.
That is the durable shape: one snapshot, two consumers, same value.

**2.2 -- MY `frame_budget: 49` CLAIM WAS FALSE (codex; verified).**
`OTR_VideoRenderBatch.frame_count`'s own tooltip reads *"Diagnostic harness only
(mode=soak): per-clip frame count. **Ignored in mode=episode** (per-shot frame
budget is planned upstream)."* (`otr_video_render_batch.py:189-192`). The
profile's `render.frame_budget` maps to that widget
(`config/profiles/widget_mapping.json:369-376`), so it caps nothing in episode
mode. **agy's r2 MUST-FIX 4 (a `frame_budget: 169` profile variant) is
REJECTED: it would do nothing.**

**2.3 -- THE 169-FRAME BEAT HAS A REAL PRODUCTION SEAM (codex; verified).**
`derive_opening_music_beat` (`otr_shot_lock.py:514-541`) reads the FIRST
non-skipped line's `start_s` and computes `frames = round(first_start * fps)`.
The opening beat's role is `music_visual`, which `otr_8gb_ltx` already routes to
`ltx_8gb` (`config/profiles/otr_8gb_ltx.json:10-16`). EpisodeAssembler offsets
the first scene by opening duration minus crossfade
(`scene_sequencer.py:1298-1304, 1524-1538`). So **7.26s opening - 500ms
crossfade = 6.76s x 25fps = exactly 169 frames**, through a production path
rather than a test hack. Those two controls are not currently accepted by the
profile schema (`capability_profiles.py:158-167`, `_otr_workflow_apply.py:519-523`)
-- extending that schema is the work.

## 3. REJECTED, WITH REASONS

* **agy: `frame_budget: 169` profile variant** -- REJECTED, see 2.2. The widget
  is inert in episode mode.
* **agy: `FrameContract` needs `to_dict`/`from_dict` because it holds
  `frozenset` and enums** -- MISREAD. It holds `discrete_frames: tuple` and
  `continuity: str` (`frame_contract.py:129-135`). Both are JSON-safe; a tuple
  serializes to a list. No crash to prevent.
* **agy: build `project_request_frame_contract` for Veo reference images** --
  SCOPED OUT per codex: Veo forces 8s only at 1080p/4k **or** when
  `asset_refs.reference_images` is present (`eng_google_veo_video.py:154-168,
  265-273`), and the canonical builder emits only `init_image` and `audio_ref`
  (`render_driver.py:2214-2216`). The reference-image branch is UNREACHABLE in
  this build. Keep the resolution-driven half; cut the reference-image half
  until a canonical producer exists.
* **Both seats: migration shim for pre-B1 saved workflows** -- CUT.
  `otr_canonical.json` is the sole target.

## 4. STILL OPEN -- r3 must answer

**4.1 -- the per-engine precedence table.** Both seats have now flagged twice
that C3 cannot be coded without it. Write it as a typed policy registry that
BOTH the resolver and its tests read, so prose cannot drift from behaviour.
Required per family, preserving or deliberately changing today's behaviour:
WAN env -> profile -> literal, clamp 17..177, then 4n+1
(`eng_wan_ti2v.py:378-402`); LTX-8GB env allowed to 16384 against a static 161
(`eng_ltx_8gb.py:223-258`); LTX Video warn/default/clamp then 8n+1
(`eng_ltx_video.py:119-179`); HuMo bare `int()` (`eng_humo.py:475-481`); cloud
duration env -> request-derived -> bounded seconds (`eng_cloud_video.py:528-551`);
Veo env aliases + resolution coercion (`eng_google_veo_video.py:244-273`).
**Does the unified rule `env -> profile -> literal` actually hold for all six,
or does LTX-8GB's 16384 range have to become terminal-above-161?** Name which
engines change behaviour and why that is correct rather than convenient.

**4.2 -- where the request projection is computed, given 2.1.** codex places a
prospective projection at ShotLock and a re-check immediately after
`request_builder(...)` and before `render_shot`
(`render_driver.py:2905-2938`), because `assert_coverage_plans` runs BEFORE
request construction (`render_driver.py:3388-3447`). Confirm that is the only
place both values exist, and define the typed projection fields (target frames,
fps, resolution, `has_init_image`) with `has_reference_images` excluded per 3.

**4.3 -- the execution receipt carrier.** The assembled clip records only
`segment_count` and `join_mode` (`render_driver.py:2972-2985`); the manifest and
report omit per-segment asks/counts, loader counts and forbidden-path counters
(`render_driver.py:3861-3884`, `otr_video_render_batch.py:57-95`). Define a
versioned `coverage_execution_receipt` and say who writes it.

**4.4 -- loader instrumentation, which 7c must build.** "Exactly one heavy
loader" is unmeasurable today: `BeatSession` deliberately exposes no counters
(`beat_session.py:104-109`) and LTX-8GB's graph has TWO loaders. Define the
7c handle API and assert `checkpoint_loader_calls == 1` AND
`clip_loader_calls == 1`, or one aggregate acquisition with both internal
counts. **Missing instrumentation must FAIL qualification, never default to 0.**

**4.5 -- C5's artifact.** Still has no concrete form. Unit test, headless
qualification profile, or prerequisite commit? Its old 49-frame rationale is
void per 2.2, so restate its dependency on C3 in terms that survive.

## 5. SLICE ORDER

| # | slice | depends on |
|---|---|---|
| C1 | B1: add node 87's `max_render_frames` INPUT DESCRIPTOR in `otr_canonical.json` **without adding or shifting a `widgets_values` slot** (the value is already there) + validator + link/widget audit | none |
| C2 | B4 scoped to the four estimate-producing canonicalizers (`eng_cloud_video.py:489-503`, `eng_google_veo_video.py:592-632`, `eng_google_omni_video.py:435-474`, `eng_google_vid_sfx.py:400-442`) via the strict probe (`wan_shared.py:105-156`) + make zero/absent/unreadable/mismatched terminal at `render_driver.py:2952-2963` | none |
| C3 | resolver + 4.1's typed policy registry + B3's taxonomy | C2 |
| C4 | receipt (4.3) + `frame_contract_env_snapshot` with BOTH consumers per 2.1 | C3 |
| C5 | single-segment proof (4.5) | C3 |
| C6 | boundary comparison: keep the live guard, ADD stamped-vs-live + 4.2's projection re-check | C4 |
| -- | **7c GATE**: ping-pong rip, loader-node removal, loader instrumentation (4.4), provider clamps | C6 |
| 7d | live qualification via the 2.3 opening-beat seam, receipts per 4.3 | 7c |

## 6. INVARIANTS

`otr_canonical.json` changed in the SAME commit as code, `widgets_values`
append-only (BUG-LOCAL-097). `frame_contract_for` stays static or the generated
`ENGINE_MATRIX.md` goes machine-dependent. No fallbacks, no silent re-plan, no
truncation. THE LAW. Every slice green and pushed; mutation-prove every fix.
Never blanket-kill Python. UTF-8 no BOM, ASCII, SFW, $0 external spend.

## 7. GROUND RULES

Cite `file:line`. Unsourced claims are dropped; wrong line numbers dropped
louder. This arc has now refuted THREE of the driver's own load-bearing claims
(live VRAM shortening, the trim_tail coupling, the 49-frame cap) and each time
the panel was right -- and the driver has caught one predicate and one misread
the panel produced. Both directions happen. Point at the code.
