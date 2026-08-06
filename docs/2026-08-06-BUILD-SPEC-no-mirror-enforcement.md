# BUILD SPEC -- no-mirror enforcement (CODE-READY)

**Date:** 2026-08-06. **HEAD at spec time:** `1759afdd` on `v2.0-alpha`.
**Arc:** `kibitz-plugin:kibitz` r1-r4 complete. **7 external calls** (r1-r3 with
Codex + Antigravity; **r4 single-lane -- Antigravity failed on provider quota**).
Artifacts: `kibitz-runs/2026-08-06-2026-08-06-no-mirror-matrix/` (**LOCAL ONLY,
`kibitz-runs/` is gitignored**).

**Operator ruling (2026-08-06):** *"there is no mirror or ping pong unless for
credits."* The closing loop is CONFIRMED OK.

**STATIC.** Nothing here enters `docs/PROD_BUG_LOG.md` without a retained
artifact.

---

## 1. WHAT IS ALREADY TRUE (do not "fix" this)

The rule is enforced in production and fails LOUD:

| surface | state |
|---|---|
| `wrapper_bridge.extend_frames_to_target` | DELETED (tombstone `:499`) |
| `eng_wan_ti2v` ping-pong | REFUSES, single-clip and coverage-planned |
| `eng_ltx_8gb` mirror-extend | REMOVED; trims REAL frames only |
| composite `_should_loop_fill` | no-op returning False |
| composite clip underrun | RAISES `ClipUnderrunsItsBeat`, any shortfall |
| `eng_ltx_video` boomerang | **DEAD, proven by execution** |

`_loop_via_reverse()` returns False even with `OTR_LTX_LOOP_VIA_REVERSE=on`, and
`_loop_fill_allowed` gates on it. The machinery is still in the tree -- that is a
DELETION task, **not an enforcement hole.**

## 2. BUILD ORDER -- producer-first

An earlier draft put the grader first. That is an armed consumer with a starved
producer, the class PBUG-20260805-04 named.

### Step 1 -- receipts on ELEVEN code surfaces

Add `native_frame_count` (always the **EMITTED** count `n`, never a pre-trim
decoded length) and `extension_mode` to:

**Local:**
1. `eng_humo.py` -- ONE stamp in the shared raw return covers all four
   registered subclasses (`:227`, `:982`, `:1052`, `:1081`), plus `_clip_from_raw`.
2. `eng_ltx_video.py` -- the ordinary raw return.
3. `eng_ltx_video.py` -- the HQ raw return.
4. **`eng_ltx_video._clip_from_raw` (`:1578-1619`)** -- the second producer seam.
   Without it the other two die before `render_beat_coverage`. This is exactly
   how `eng_ltx_8gb` lost its receipt originally.
5. `eng_ltx_av.py` raw return (`:1098-1101`) -- registered lane is `ltx_audio_in`.
6. `eng_ltx_av._clip_from_raw` (`:1183-1200`).

**Provider (all currently at ZERO references):**
7. `_CloudVideoBase.canonicalize` (`eng_cloud_video.py:489-503`)
8. `GoogleOmniVideoEngine.canonicalize` (`eng_google_omni_video.py:434-473`)
9. `GoogleVeoVideoEngine.canonicalize` (`eng_google_veo_video.py:592-631`)
10. `_GoogleVidSfxBase._canonical_video_and_sfx` (`eng_google_vid_sfx.py:397`)

**And 11 -- retrofit `extension_mode="none"` onto every remaining
video-producing adapter**, so the v1 contract in step 3 does not break the
procedural/unbounded lanes.

First: add the optional receipt fields to `CanonicalClip`
(`schemas.py:216-247`), or the declared `extra="forbid"` interface rejects the
enriched shape the day validation arrives.

### Step 2 -- the manifest

* **`frame_receipt_version: 1`** at top level.
* **`closing_theme_frame_window = {"start": S, "end": E}`**, derived from ledger
  rows with `speaker_role == "music_close"` AND `start_s_space == "master_mix"`.
  Accept only: finite non-bool `start_s >= 0`, finite `dur_s > 0`, one
  `music_cue_id`, complete unique chunk indices, contiguous/overlapping
  intervals. Convert with the composite's own convention --
  `round(start_s*fps)` for the start, `ceil((start_s+dur_s)*fps)` for the
  terminal boundary. **Emit nothing unless the manifest is POSITIONED.**
  (`beat_id` is a source LINE id; `music_close` is a speaker_role stamped
  separately at `scene_sequencer.py:1733-1799`; `production_ledger.py:1716`
  already discriminates on `start_s_space != "master_mix"`.)
* **Centralized index**: one total, non-throwing helper replacing the
  overwriting dict comprehensions at `acceptance.py:353-355` and `:418-420`.

### Step 3 -- the grader

* `KNOWN_EXTENSION_MODES = ("none", "ping_pong")` for PARSING legacy receipts;
  `DELIVERABLE_EXTENSION_MODES = ("none",)` for what may ship.
* New `RULE_NO_MIRROR` / `grade_no_mirror(ledger, manifest)`: any delivered row
  whose `extension_mode` is a non-empty string outside
  `DELIVERABLE_EXTENSION_MODES` is a finding. **All beats.** No counts, no
  projection, no segment gate.
* **That branch LEAVES `grade_multiclip_honesty`**, which keeps projection and
  count validation only, and must **stop policy grading** on a known
  non-deliverable mode -- otherwise its `delivered_native != delivered` branch
  (`:488-508`) emits a second, wrongly-worded finding for a violation
  `grade_no_mirror` already owns. It MAY still report an UNKNOWN mode as
  malformed accounting; the tests must pin that distinction.
* New `RULE_MANIFEST_SHAPE`: `clips` must be a list/tuple of dicts with unique
  non-empty string `shot_id`s. On a shape defect, emit deterministic shape
  findings, KEEP `grade_frozen_route` (manifest-independent), and SKIP the
  manifest-dependent semantic rules so one bad row does not cascade.
* **The v1 contract:** in a `frame_receipt_version: 1` manifest, every delivered
  `type == "video"` row must declare the string `extension_mode == "none"`;
  BOUNDED engines additionally owe native-count evidence. Unversioned manifests
  may omit the field, but an explicit non-`none` is still rejected. Without this
  version, "reject explicit non-`none` only" cannot catch an adapter that
  REGRESSES by dropping its receipt -- absence would stay permanently legal.
* Export every new rule in `__all__` and add it to `grade_episode`.

### Step 4 -- the composite classifier (OWN COMMIT; the only behaviour change)

`otr_silent_composite.py:444-469` currently loops the last clip for ANY
unexplained tail. Authorize `loop=True` **only** when
`S <= cursor < target_total_frames <= E` from step 2's window. Otherwise
floor/black. **Never loop on doubt.**

**Blast radius, state it in the commit:** any episode whose tail is not provably
the closing window gets floor/black instead of a loop.

A named constant alone was rejected twice: vocabulary without a manifest-backed
window is not enforcement.

### Step 5 -- delete the dead boomerang

Atomically: `_boomerang_frames`, `_ltx_loop_source_length`,
`_LTX_LOOP_MIN_DECODE_FRAMES_DEFAULT`, `_LOOP_VIA_REVERSE_DEFAULT`,
`_loop_via_reverse`, `_loop_fill_allowed`, both render/HQ branches, both
`ltx_loop_via_reverse` raw fields, and the matching assertions in
`test_ltx_boomerang.py`, `test_video_motion_forward.py`,
`test_session_ctx_ownership.py`.

**CONVERT `test_ltx_boomerang.py`, do not delete it** -- keep a tripwire proving
no environment value or session state can restore the mirror.

### Step 6 -- the fossil sweep

Mechanical search for `boomerang`, `ping_pong`, `mirror`, `loop_via_reverse`,
`stream_loop`; classify each hit as sanctioned closing reuse, historical past
tense, or defect. Known:

* `frame_contract.py:280-286` -- cites the DELETED `extend_frames_to_target` and
  governs `PLANNING_CAP_ENGINES` membership;
* `eng_ltx_8gb.py:32-36,1404-1414`; `beat_session.py:143-154`;
  `wrapper_bridge.py:525-526`; `eng_wan_ti2v.py:1001-1017`;
* **`scene_sequencer.py:1831-1842` -- an INFO log printed into the operator's
  LIVE RUN asserting the boomerang "doubles the rendered half-clip back to full
  audio duration".** A stale comment misleads a reader; a stale log misleads the
  operator mid-render.

## 3. TESTS

Per-adapter conformance for all eleven surfaces (raw AND canonicalize).
Table-driven grader tests: `"none"`, `"ping_pong"`, unknown strings, `None`,
malformed types, missing rows, duplicate rows -- for single- AND multi-segment
plans. Closing-window authorization: in-window loops, out-of-window uses
floor/black, malformed window uses floor/black. Do NOT duplicate
`test_clip_fill.py:346-414`, which already covers ordinary underruns.

## 4. SCOPE DECLARATIONS

* **No workflow topology change**, verified: canonical link 261 already carries
  node 92 `clip_manifest_json` to node 84 `OTR_SilentComposite`. The manifest
  SHAPE changes; the STRING wire does not.
* The grader stays **AUDIT-ONLY**; production enforcement is the adapter and
  composite refusals.

## 5. THE GATE, AND THE PUSH ORDER

**Reviews happen BEFORE a chunk is green, not as a hold on green commits.**
CLAUDE.md section 7 requires every green chunk pushed immediately. So per chunk:
focused tests -> Sonnet QA on the diff -> Fable gate where the change is
structural -> full Windows suite (baseline **8836 passed / 131 skipped /
1 xfailed at `e499b7fc`**) -> Bug Bible 17 -> commit -> **push** -> verify
`HEAD == origin/v2.0-alpha`, AST-parse touched Python, no BOM, no zero-byte.

## 6. THE LIVE LEG -- and why the obvious one would not count

A WAN or `ltx_8gb` leg would PASS while never executing the deleted
`eng_ltx_video` machinery or its newly-fixed passthrough.

**Require a canonical `ltx_video` beat EXCEEDING its 169-frame contract ceiling**
(`eng_ltx_video.py:503-510`) so `render_beat_coverage` runs multiple segments,
**with the retired environment switch SET**. Then: per-segment and beat receipts
all `"none"`, an empty `scripts/grade_episode.py` verdict, `obs_publish OK`, and
the canonical episode/OBS assets on disk. Provider paths get stubbed transport
conformance -- no paid calls, offline-first.

**That single leg also discharges F11** (the mirror deletion has never had a live
proof) and the multi-clip receipt work's outstanding live proof.
