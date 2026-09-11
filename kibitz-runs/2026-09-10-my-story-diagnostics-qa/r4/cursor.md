VERDICT: yes-with-fixes — production diagnostic behavior matches the 4060 warning-only ask, but the lock receipts in the brief point at the wrong artifacts and would let a judge bless the wrong tree.

MUST-FIX BEFORE BUILD:
1. [receipts / input.md L19-21] Focused proof for THIS diff is `tmp/my_story_diagnostics_focused.xml` (204 tests, 1 failure: `tests/test_google_video_sfx_workflow.py::test_canonical_workflow_wires_clip_manifest_to_master_audio_mux` at L133, `assert 291 == 289`). The brief cites `tmp/my_story_boundary_full.xml` instead. That file is a different campaign. Swap the pointer. Do not treat the 72-pass `tmp/my_story_boundary_focused.log` (my_story_runner only) as this change's focused green.
2. [receipts / input.md L21] `tmp/my_story_diagnostics_bible.log` is not a receipt for this working tree. Its pytest rootdir is `C:\Users\jeffr\Documents\ComfyUI\_worktrees\otr-my-story-music-candidate`. Rerun Bible against `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`. Until then, do not claim Bible-in-progress for this diff.

SHOULD-FIX:
1. [nodes/otr_master_audio_mux.py L701-703] Helper docstring still says "byte-for-byte same basename" then that `mux_master_audio` asserts PCM identity. The helper only matches basename + `is_file()` (L709-737); PCM lives at L427-431. Replace "byte-for-byte same basename" with "same basename". The brief already forbade the overclaim (input.md L16-17); R1–R3 should have landed this wording.
2. [tests/test_wire_w2_deferred_image_gap.py L210-216] Brief claims ShotLock typed-deferral catch is INFO (input.md L10-11; `nodes/otr_shot_lock.py` L1904-1909). `test_a_declared_cast_time_gap_is_DEFERRED_not_raised` does not pin `caplog` level. A revert to `log.warning` would still pass. Add one INFO assertion on `"cast-time image input deferred"`.
3. [tests/test_google_video_sfx_workflow.py L133] File is already in this diff and still pins `last_link_id == 289` while `workflows/otr_canonical.json` L5 is 291. Focused run of the touched files trips `[KNOWN-FAIL-GUARD] NEW failures (REGRESSION)` in `tmp/my_story_diagnostics_focused.log` L20-22. Smallest landing hygiene: bump the canary to 291 in the same comment block (L130-133). Do not edit the workflow JSON. If scope stays diagnostic-only, record the nodeid as pre-existing and do not call the focused file-set green.

OPTIONAL / NICE-TO-HAVE:
- One `phase="unknown"` call on the still_pan missing-still fixture; the else-branch is the same as default `"render"` (`nodes/_otr_video_engines/render_driver.py` L2690-2702). Default is already covered by `tests/test_video_render_driver_additive.py` L355-358.
- Do not lower `nodes/otr_shot_lock.py` L1969 (`cast-time init_image ... deferred`) in this slice. still_pan exits via the `static_image_gen` early return (L1953-1959). That WARNING is a different path than the 4060 `MISSING-STILL (LOUD)` line.

CUT THESE:
1. Do not invent a PBUG. `docs/4060_DRILL_LOG.md` Step 116 L4981-4988 and Step 119 L5135-5142 already call these warning-only follow-ups. Matches input.md L21-23.
2. Do not fold in the cross-machine R3 architecture campaign. Out of scope (input.md L25-26).
3. Do not teach `phase` to swallow `DeferredImageGapError` at `build_request_from_shot` L2589-2595. Scene-init stills still raise by type; ShotLock already INFO-catches (`otr_shot_lock.py` L1904-1909). Changing the raise would break WIRE-W2.
4. Ignore inherited `diff.txt` / `diff_utf8.txt` (they start with unrelated `docs/SHIPPING_JSON_RECIPES.md`).

VERIFY-AT-BUILD checklist:
- verify: `git diff 59a4013103cd062302aa77d0e3acef5eb29e5786` is exactly the four production files named in input.md L3-5 plus the five tests in the opening git status. I read HEAD from `.git/refs/heads/v2.0-alpha` (= that sha). I did not run `git diff`; hunk-level "diagnostics only" is UNTRACED against the index.
- `build_request_from_shot(..., phase=)` is keyword-only with default `"render"` (`render_driver.py` L2403-2405, L2419-2421). `phase` is used only at L2690. `req == planned` pins payload identity (`tests/test_video_render_driver_additive.py` L350-356).
- Sole production `phase="cast_preflight"` caller is ShotLock (`otr_shot_lock.py` L1902-1903). Other `nodes/` callers: none. `run_real_episode` binds `functools.partial(build_request_from_shot, master_audio_path=...)` with no phase (`render_driver.py` L5589-5591) → default render / still warns.
- Typed catch is `except DeferredImageGapError`; `ValueError` / `RenderError` / `KeyError` / `RuntimeError` still propagate (`tests/test_wire_w2_deferred_image_gap.py` L219-245).
- Successful rename is INFO in the three owners: mux L731-737, dispatcher L597-602 (no `warnings.append` on success; freeze reject still appends at L590), clip persistence L6156-6161. Earlier warning retained: `tests/test_image_platform_c1.py` L1401-1409. Foreign freeze still WARNING: dispatcher L1460-1466, clips L408-432, mux L56-75.
- Path helper does not hash PCM. `mux_master_audio` still owns `audio_pcm_sha` (`otr_master_audio_mux.py` L427-431).
- Full suite: `tmp/my_story_diagnostics_full.log` was still running (collected 14372, ~2% when read). Do not claim global green. No new media / obs leg required for this slice (`docs/4060_DRILL_LOG.md` L4981-4988).
- [ASSUMPTION] Uncommitted working tree matches the files I read; I did not see the index hunks.

[ASSUMPTION] This campaign is one finished-diff QA, not a design arc (input.md L25; `r4/driver_anchor.md` L15-17). No r1–r3 artifacts exist under `kibitz-runs/2026-09-10-my-story-diagnostics-qa/`.
