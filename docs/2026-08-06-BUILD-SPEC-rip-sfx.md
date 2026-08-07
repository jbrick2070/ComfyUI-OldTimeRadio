# BUILD SPEC -- rip SFX, 100%

**Date:** 2026-08-06. **Arc:** `kibitz-plugin:kibitz`, r1 complete (Codex +
Antigravity, plus one scoped adjudication lane). **r2-r4 owed before code.**
**Operator ruling:** *"I do really want to rip out SFX 100%, that's my aim. How
it gets done: you, and ask Fable -- you can `/kibitz-plugin:kibitz`, Codex etc.
But don't break the system."*

Problem statement: `docs/2026-08-06-PROBLEM-STATEMENT-legacy-sfx.md`.
r1 judgment: `kibitz-runs/2026-08-06-rip-sfx/r1/judgment.md` (**LOCAL ONLY**).

**This document states the plan as it now stands.** It carries no correction
diary: r1 found that keeping obsolete claims beside their corrections is what
made the draft self-contradictory.

---

## 1. FOUR THINGS ARE CALLED SFX. TWO ARE IN THIS RIP

Measured: **395 sfx-matching lines** across `nodes/`, `scripts/`, `tools/`.

| # | The thing | State | In this rip? |
|---|---|---|---|
| 1 | The sfx/b-roll **ROLE** | ripped 2026-07-01 (`rip-sfx-broll`) | **NO -- already gone, and its guards STAY** |
| 2 | The **`[SFX:]` SCRIPT MARKUP** | dead by shadowing | **YES -- as dead code (section 5)** |
| 3 | The **SFX BED** | live and wired | **YES -- the target** |
| 4 | The **cue-ledger campaign** | parked 2026-08-04 | **Docs only -- retire in place** |

**44 of the 395 lines are ROLE TOMBSTONES and MUST SURVIVE.** They are not dead
prose: `scene_sequencer.py:905-921` RAISES by name on a ledger carrying
`speaker_role="sfx"`, and `production_ledger.py:107` fails old sfx ledgers loud.
Deleting them turns a loud refusal on a stale ledger into a silent
misinterpretation.

## 2. THE PREMISE, STATED ACCURATELY

The SFX-PRODUCING branch is dormant -- no shipped episode has executed it. **The
runtime checks and the wiring are not.** `OTRMasterAudioMux.mux()` calls
`compile_sfx_bed_from_manifest` on EVERY run (`otr_master_audio_mux.py:751`); it
returns `""` only because no manifest row carries `sfx_stem_path`, which is true
only because no SFX-producing engine is selected. Five engines that would arm it
sit in the `music_visual` dropdown.

## 3. THE LEDGER RULE -- every field, and its owner after

Operator directive 2026-07-14, hard. Verified: only four files write an SFX
field, and **every production reader is already guarded** -- `.get(...) or ""`,
an early `continue`, or a `return ""` on an empty list. **Zero shipped episodes
carry any of the three fields** (grep over 50+ ledgers). **No schema declares
them** -- neither `CanonicalClip` nor `ShotRow`, both `extra="forbid"`; the
producers return plain dicts. No manifest hash, golden file, or positional parse.

| field | written by | read by | owner after |
|---|---|---|---|
| `sfx_stem_path` | `eng_cloud_video.py:920`, `eng_google_vid_sfx.py:474` | `otr_master_audio_mux.py:194,209`; `render_driver` persist + manifest | **DELETED -- no producer remains** |
| `sfx_duration_s` / `sfx_sha256` | same | `render_driver` manifest row | **DELETED** |
| `sfx_bed_path` / `sfx_gain` | `otr_master_audio_mux.py` | `mux_master_audio`, its report | **DELETED** |
| `audio_mode=sfx_mixed` | mux report | operator log + 2 test asserts | **`master_copy` becomes unconditional** |

**RE-PIN EVERY LINE CITE IMMEDIATELY BEFORE CODING.** `render_driver.py` grew
~160 lines during the no-mirror build; the manifest stamp moved from `:4648` to
`:4810`. `persist_episode_clips:4442-4463` and `render_beat_coverage:3499` are
current as of this writing. A reviewer or builder working from stale cites
confidently edits the wrong lines.

## 4. WHAT GETS DELETED

**4a. Five registered engines.** Nothing selects them (zero matches for
`google_vid_sfx` / `_720p_sfx` / `wants_provider_sfx` across `config/**` and
`workflows/**`); none is any role's default (`default_roles = ()` on all).

    cloud_vidu_q2_pro_fast_720p_sfx   google_vid_sfx_omni
    google_vid_sfx_veo_fast           google_vid_sfx_veo_lite
    google_vid_sfx_veo_pro

`eng_google_vid_sfx.py` goes entirely. `eng_cloud_video.py` loses
`CloudViduQ2ProFast720pSfxEngine` and `wants_provider_sfx`. Also: the five
`CAPABILITIES` rows in `registry.py` (`:515,552,558,564,570`) and the guarded
import block in `__init__.py:217-222` -- the roster audit reports an orphan row
as `unexpected` and a missing adapter as `missing`, so both go together.

**4b. The bed.** `compile_sfx_bed_from_manifest`, `_sfx_gain`,
`DEFAULT_SFX_BED_GAIN`, `_default_sfx_bed_out` (which has never written a file in
production), and the `sfx_bed_path`/`sfx_gain` parameters and mix branch of
`mux_master_audio`, including its SFX integrity gate.

**4c. The helpers r1 found missing from the first draft.** Each verified
unreachable by any surviving engine:

* `cloud_media_canonical.py`: `extract_sfx_bed_from_provider_video`,
  `_normalize_sfx_stem_audio`, `_sfx_loudnorm_params`,
  `SFX_LOUDNESS_REFERENCE_SOURCE` (+ its `__all__` entry). A closed chain --
  each is called only by the next one up.
* `_otr_story_brief_helpers.py`: `append_sfx_audio_safety_clause` and
  `SFX_AUDIO_SAFETY_CLAUSE`. **AND ITS THREE CALL SITES ON A SURVIVING CLASS:**
  `eng_cloud_video.py:46` imports it at MODULE SCOPE, `:886` calls it inside
  `_conditioned_prompt` on the surviving Vidu base, and `:901` stamps
  `"sfx_audio_requested"`. Deleting the helper alone is a startup ImportError.

**4d. The manifest fields and the persist move** (`render_driver`).

**4e. The dead `[SFX:]` markup.** `_inject_scene_transitions` is defined TWICE in
`story_orchestrator.py` -- the `[SFX:]`-emitting body at `:143-176` is
permanently SHADOWED by a same-named function at `:2687`, and both have ZERO
callers. The `sfx_count` branch at `:2540-2548` belongs to
`GemmaHeartbeatStreamer`, which is never instantiated. No pack or prompt asks for
the tag; the token ratios are hardcoded constants; 2 ledgers of several hundred
contain `[SFX`, both from May, both in critic-guidance text, never in a spoken
line. Deleting this changes NO generated content, which is what keeps it clear of
the closed story-quality directive.

**4f. One dead script check.** `scripts/soak_operator.py:110-112` -- the `NO_SFX`
quality flag. It is broken today and would FALSE-POSITIVE on every new episode.

**4g. The parked designs** -- `docs/2026-07-31-sfx-engine-lane-SPEC.md`,
`docs/2026-07-11-timeline-cue-ledger.md`,
`docs/2026-07-11-cue-ledger-r1-codex-prompt.md`. **In-place `RETIRED` headers, no
file moves** -- moving them breaks historical backlinks for no gain.
`ROADMAP.md` and `docs/STILL_PLAN_SEED_INVENTORY.md` are updated in the same
commit, or the code says "removed" while the inventory says "available".

## 5. WHAT MUST NOT BE TOUCHED

* **The 44 role tombstones.** The predicate, not the tally: anything referencing
  `rip-sfx-broll`, `speaker_role="sfx"`, or the deleted `sfx[]` array is
  HISTORY-AND-GUARD and stays. Named instances:
  `soak_operator.py:146,189,219,268` (parser exclusions),
  `build_silent_test_episode.py:660`, `audit_otr_full_run.py:169`.
* **The text sanitizers.** `scene_sequencer.py:517`, `_otr_bark_lib.py:271`,
  `eng_bark.py:69` strip `[ENV|SFX|MUSIC:...]` before synthesis, and the
  structural-token sets guard name parsing. These are defence against a model
  HALLUCINATING a bracket tag -- plausible on old-time-radio training data with
  no prompt asking for it. Removing them is zero benefit against reintroducing
  the announcer-reads-a-stage-direction class.
* **`music_open` / `music_close` / `music_inter`.** Music is not SFX, and the
  no-mirror build's closing-window classifier depends on `music_close`.
* **`OTR_MasterAudioMux` itself** -- only its SFX branch dies.

## 6. `clip_manifest_json` AND LINK 278 STAY WIRED -- decided, not deferred

r1 split on this. Codex demanded deleting the input, the mux argument, link 278,
node 85's slot 4 and 278 from node 92's links. **Rejected.**

This is the TERMINAL publish node -- a topology failure yields NO episode, not a
degraded one. `CLAUDE.md` section 0 requires the canonical JSON edited in the
same change and re-validated; `widgets_values` is positional with a documented
drift bug (BUG-LOCAL-097). Against that, the input is optional and has zero
semantic consumers once the bed is gone. **High-severity publishing risk for zero
functional benefit.**

So it becomes VESTIGIAL: still wired, still hashed by `IS_CHANGED`, no effect on
output. **Say that plainly -- do not invent a use.** Its tooltip
(`otr_master_audio_mux.py:579`) currently describes SFX mixing and must be
corrected to name it a retired connector kept for topology compatibility.

**NO WORKFLOW TOPOLOGY CHANGE.** Re-verified after the code lands, not assumed --
`test_google_video_sfx_workflow.py` is the check.

## 7. THE MUX COLLAPSE -- the most dangerous line in the rip

`mux_master_audio` is `if not sfx_bed_path: <master_copy> else: <sfx_mix>`.
**The branch that SURVIVES is the smaller `if` body** -- the one that already
runs on every episode. The elaborate `else` dies.

Keep the wrong one and every episode silently swaps a `-c:a copy` passthrough for
a re-encode, violating invariant V-1, and **no surviving test distinguishes
them** -- they assert the audio is valid, not which branch produced it. Add an
assertion that the surviving command still carries the copy codec.

Both gates that protect every real episode are OUTSIDE the SFX branch and
survive: the duration-drift guard, and the audio-SHA byte-identity check. The
2026-07-14 fail-closed handler is SFX-agnostic and is untouched.

## 8. TESTS

**Order matters: consumers, then producers, then registry, then tests, then
docs -- as EDITING order, landing as ONE atomic green commit.** Not separately
shippable commits that temporarily leave a field produced and unread.

**Rescue before deleting.** Four pieces of non-SFX coverage live in files that
look disposable:

1. `test_google_video_sfx_workflow.py` -- three `_reresolve_master_audio` tests
   plus the two topology tests. **All five survive under section 6's decision**,
   and the topology test IS the section-6 re-verification.
2. `test_clip_fill.py::test_persist_rekeys_sfx_to_renamed_episode_clips` -- the
   ONLY coverage of persist's rename/rekey path for `clip["path"]`. Strip its SFX
   fixture; keep the function.
3. `test_existing_google_video_engines_stay_silent` (in the beds file) -- proves
   invariant V-1 for the SURVIVING Veo/Omni engines. Relocate it.
4. `assert "generateAudio" not in base_payload["parameters"]` -- buried in an
   SFX-only test and the only assertion pinning that base VEO never requests
   audio. Preserve it when its host dies.

**Then the mechanical fixes** (r1, Antigravity):
`test_frame_receipt_conformance.py` -- delete the extractor monkeypatch at
`:94-95` and lower the `>= 30` roster floor; `test_model_slot_audit.py:185-190`;
`test_multiclip_session_identity_roster.py` `EXPECTED_CLOUD_GAP` (5 entries) and
the `== 12` length check; `test_engine_contract_roster.py` `WHOLE_SECOND` entry
and `test_the_sfx_lanes_share_their_base_adapters_ladder_object`;
`test_video_render_path_cw4.py` -- the two SFX-only tests and the
`clip_manifest_json="[]"` arguments; the three shared-collection loops in
`test_cloud_video_adapters.py` (prune the ROW, keep the function).
**`tests/fixtures/still_plan_head_parity.json` must be REGENERATED** via
`tests/test_still_plan_parity.py --regenerate` -- removing engines reflows other
engines' rows, so hand-editing it is wrong.

**The tripwire is the only thing that would notice a regression.**
`EXPECTED_FAILED_NODEIDS` has no presence check, so a vanished nodeid produces
zero signal. New `tests/test_rip_sfx_bed_guard.py`, sibling to the existing
`test_rip_sfx_broll_guard.py` (which guards the OLD role rip and is left alone).
It must reject: any engine declaring `wants_provider_sfx`; any engine id
containing `sfx` (**both checks are required -- the four `google_vid_sfx_*`
engines never set the flag**); the bed/compiler symbols; the prompt helper; the
extractor; any manifest row carrying `sfx_stem_path`. It must carry an EXPLICIT
ALLOWLIST for the role tombstones and the text sanitizers, so the guard never
pressures a later reader into deleting protective code.

## 8-BIS. THE EXACT DELETION CLOSURE (r2 -- every one of these crashes at import)

Deleting a symbol without its registration, its global instance and its
`__all__` entry is a STARTUP failure, and in ComfyUI that means the whole node
pack vanishes from the menu rather than one engine misbehaving. Both r2 lanes
found these independently:

* `otr_master_audio_mux.py:813-814` -- `"compile_sfx_bed_from_manifest"` in
  `__all__`. Also `_decoded_pcm_sha` (`:92-103`) has ZERO callers once the SFX
  mix block goes.
* `eng_cloud_video.py:1057` (global instance `ViduQ2ProFast720pSfx = ...`),
  `:1062-1064` (the registration tuple) and `:1068` (`__all__`).
* `cloud_media_canonical.py:30,35` -- `__all__` entries for the deleted
  extractor and `SFX_LOUDNESS_REFERENCE_SOURCE`.
* **`render_driver.py:104-111,1490,2709-2710` -- `_GOOGLE_VIDEO_SFX_ENGINES`
  and its consumers**, missed entirely by the first inventory. Remove the set,
  narrow `_GOOGLE_PROVIDER_PROMPT_ENGINES` to the surviving silent Google
  providers, and drop the now-obsolete `_apply_visual_safety_prompt` SFX
  exception.
* `tests/test_model_slot_audit.py:183` -- asserts
  `vidu.wants_provider_sfx is False` on the SURVIVING base class, so deleting
  the attribute breaks it. `:185-190` alone is not enough.

**AND ONE MORE RESCUE the rewrite dropped:**
`tests/test_google_video_sfx_render_driver.py:63-76`
(`test_silent_google_prompt_routing_still_uses_visual_google_branch`) is the ONLY
test exercising the SURVIVING `google_veo_video` prompt-routing branch. Relocate
it; do not delete the file wholesale. Suggested homes for the other rescues:
`generateAudio` and the V-1 silence proof into
`tests/test_google_veo_video_adapter.py` / `test_google_omni_video_adapter.py`.

## 9. THE PROOF

A green suite after a deletion proves the tests were deleted too. Required:

1. **The live leg's acceptance criterion is a DECODED-PCM comparison on the
   ARCHIVAL final -- NOT on the published OBS copy.** An earlier draft of this
   section demanded the *published* audio be byte-identical to the frozen
   master. **That is impossible by design and the spec was wrong:**
   `otr_master_audio_mux.py:675-685` deliberately re-encodes the OBS copy to AAC
   for viewing, while the byte-identity check runs on the archival final
   (`:398-423`). So: compare decoded PCM hashes of the frozen master and
   `otr/episodes/<ep>/*_final.mp4`, and SEPARATELY require the AAC OBS copy to
   exist and be playable. `RESULT SUCCESS` + `obs_publish OK` are satisfied by
   BOTH mux branches and prove nothing about section 7 on their own.
2. `tools/engine_matrix.py` REGENERATED (32 -> 27 engines) and `--check` green.
3. The full Windows suite, the Bug Bible regression, and the focused
   mux/registry/tripwire tests.
4. `OTR_WorkflowValidator` + a JSON round-trip + a link/widget audit, proving the
   canonical workflow is unchanged and still valid.
5. AST parse, no BOM, no zero-byte, on every touched file.
6. Commit by PATHSPEC, push, verify `HEAD == origin/v2.0-alpha`.

## 10. MIGRATION POLICY -- and it needs CODE, not just a rule

A user-saved workflow or an external API client may still name one of the five
retired engine ids. Repo-owned config proves nothing about those. **A stale
selection must fail with a NAMED retired-engine error and must never silently
resolve to another engine.**

r2 showed this has no error boundary today: `OTRVideoDirector.VALIDATE_INPUTS`
accepts everything (`otr_video_director.py:329-332`), `_resolve_and_validate`
returns an unknown id unchanged (`:543-573`), and the registry classifies it as
a generic "no video engine named ... is registered"
(`engine_registry_base.py:142-145`, `registry.py:184-189`).

**BUILD IT:** one explicit `RETIRED_ENGINE_IDS` tombstone set in
`registry.py` holding exactly the five ids, intercepted in `assert_usable` to
raise `EngineUnusable` naming the retirement. Reject by name at the director,
the registry and the force-map boundaries, and test all five. **The tripwire must
ALLOW that tombstone set while proving none of the five is registered, aliased or
selectable** -- otherwise the guard's own "no engine id contains sfx" rule would
force a later reader to delete the tombstone that makes the failure legible.

## 11. RISK

The rip touches the terminal publish node. A mistake there does not degrade an
episode -- it produces none. Everything else is dormant code no shipped episode
has executed.
