# BUILD SPEC -- rip SFX, 100% (DRAFT, pre-kibitz)

**Date:** 2026-08-06. **HEAD at draft time:** `ac8a1925` on `v2.0-alpha`.
**Operator ruling:** *"I do really want to rip out SFX 100%, that's my aim. How
it gets done: you, and ask Fable -- you can `/kibitz-plugin:kibitz`, Codex etc.
But don't break the system."*

**Problem statement:** `docs/2026-08-06-PROBLEM-STATEMENT-legacy-sfx.md`.
**Status: DRAFT. No code ships from this until the full four-round kibitz arc
and the Fable gate have run** (operator directive 2026-08-04, hard).

---

## 1. FOUR THINGS ARE CALLED "SFX". THE RIP IS NOT ALL OF THEM

This is the whole reason the word is confusing, and getting the scope wrong is
how "don't break the system" gets broken. Measured: **395 sfx-matching lines**
across `nodes/`, `scripts/`, `tools/`.

| # | The thing | State today | In this rip? |
|---|---|---|---|
| 1 | The sfx/b-roll **ROLE** | RIPPED 2026-07-01 (`rip-sfx-broll`) | **NO -- already gone** |
| 2 | The **`[SFX: ...]` SCRIPT MARKUP** | LIVE in the writer | **OPEN QUESTION -- see 5.1** |
| 3 | The **SFX BED** (audio under the master) | LIVE AND WIRED | **YES -- this is the rip** |
| 4 | The SFX **ENGINE LANE / cue ledger** | PARKED 2026-08-04 (`315e8afd`) | **Docs only -- retire them** |

**44 of the 395 lines are ROLE TOMBSTONES and MUST SURVIVE.** They are not dead
prose: `scene_sequencer.py:905-921` RAISES by name on a ledger carrying
`speaker_role="sfx"`, and `production_ledger.py:107` fails old sfx ledgers loud.
Deleting those turns a loud refusal on a stale ledger into a silent
misinterpretation. A tombstone that still guards something is code.

## 2. THE LEDGER RULE -- every field, and who owns it after

Operator directive 2026-07-14, hard: enumerate EVERY field the path writes, give
each exactly ONE new owner, delete only then, prove it live. Measured: **only
four files write an SFX field.**

| field | written by | consumed by | owner after the rip |
|---|---|---|---|
| `sfx_stem_path` | `eng_cloud_video.py:916`, `eng_google_vid_sfx.py:439` | `render_driver.py:4442-4463` (move/clear), `:4650` (manifest row), `otr_master_audio_mux.py:194,209` | **DELETED -- no producer remains** |
| `sfx_duration_s` | `eng_cloud_video.py:917`, `eng_google_vid_sfx.py:440` | `render_driver.py:4651` | **DELETED** |
| `sfx_sha256` | `eng_cloud_video.py:918`, `eng_google_vid_sfx.py:441` | `render_driver.py:4652` | **DELETED** |
| `sfx_bed_path` | `otr_master_audio_mux.py:751` (local) | `mux_master_audio(:294)`, report `:480` | **DELETED** |
| `sfx_gain` | `_sfx_gain()` `:114-118` | `mux_master_audio(:295)`, report `:481` | **DELETED** |
| `audio_mode=sfx_mixed` | mux report `:476` | operator-facing log only | **becomes the unconditional non-SFX mode** |

**The critical property, and it is what makes this rip safe:** every one of
these fields is OPTIONAL at every consumer today -- the manifest row stamps them
only `if sfx_stem` (`render_driver.py:4648-4652`), and the bed compiles only
`if sfx_rows` (`otr_master_audio_mux.py:195`). Removing the producers leaves the
consumers taking the path they ALREADY take on every shipped episode. This is a
rip of a dormant branch, not of a live one.

**No hole is created, because nothing downstream reads an SFX field it does not
also guard.** That claim is the panel's first job to break.

## 3. WHAT GETS DELETED

**3a. The five registered engines** (nothing selects them -- verified: zero
matches for `google_vid_sfx` / `_720p_sfx` / `wants_provider_sfx` across
`config/**` and `workflows/**`):

    cloud_vidu_q2_pro_fast_720p_sfx      (eng_cloud_video.py)
    google_vid_sfx_omni                  (eng_google_vid_sfx.py)
    google_vid_sfx_veo_fast              (eng_google_vid_sfx.py)
    google_vid_sfx_veo_lite              (eng_google_vid_sfx.py)
    google_vid_sfx_veo_pro               (eng_google_vid_sfx.py)

`eng_google_vid_sfx.py` (119 sfx lines) goes ENTIRELY. `eng_cloud_video.py`
loses `CloudViduQ2ProFast720pSfxEngine` and `wants_provider_sfx`.

**3b. The bed itself:** `compile_sfx_bed_from_manifest`, `_sfx_gain`,
`DEFAULT_SFX_BED_GAIN`, `_default_sfx_bed_out`, the `sfx_bed_path`/`sfx_gain`
parameters and mix branch of `mux_master_audio` (`:398-451`), the SFX integrity
gate (`:472`), and the call at `:751`. **`mux()` then calls `mux_master_audio`
directly with no bed.**

**3c. The extractor:** `extract_sfx_bed_from_provider_video`
(`_otr_shared/cloud_media_canonical.py`).

**3d. The manifest row fields** (`render_driver.py:4648-4652`) and the
persist-time move (`:4442-4463`).

**3e. Tests that exist only to prove SFX works** -- `test_google_video_sfx_beds.py`,
`test_google_video_sfx_render_driver.py`, and the SFX arms of
`test_cloud_video_adapters.py` / `test_video_render_path_cw4.py` /
`test_clip_fill.py`. **CONVERT, do not merely delete**: keep one TRIPWIRE
asserting no registered engine declares `wants_provider_sfx` and no manifest row
carries `sfx_stem_path`, so the lane cannot be reintroduced by accident. This
mirrors what step 5 of the no-mirror build does with `test_ltx_boomerang.py`.

**3f. The parked designs** -- `docs/2026-07-31-sfx-engine-lane-SPEC.md`,
`docs/2026-07-11-timeline-cue-ledger.md`,
`docs/2026-07-11-cue-ledger-r1-codex-prompt.md` -> `docs/retired/` with
`-RETIRED` in the name, per the precedent set on 2026-08-06 for the stale HuMo
maths doc. Their reasoning stays readable; their authority ends.

## 4. WHAT MUST NOT BE TOUCHED

* **The 44 role tombstones** (section 1). They still fail loud on stale ledgers.
* **`nodes/_otr_speaker_role.py`** and the `music_open`/`music_close`/
  `music_inter` roles. Music is NOT SFX and the closing theme is load-bearing --
  the no-mirror build's entire step 4 depends on it.
* **`OTR_MasterAudioMux` itself.** It is the terminal publish node
  (`obs_publish`). Only its SFX branch dies; the node, its wiring and its
  fail-closed gates stay exactly as they are.
* **`clip_manifest_json` link 278 stays wired. No workflow topology change.**
  **CORRECTED by the fan-out (5b.3): the reason given here was wrong.** This
  said the mux "still needs the manifest for `fps`/rows". It does not -- the
  node's own `fps` widget supplies fps, and the SFX bed was the input's only
  consumer. The link stays because removing it is a topology change to the
  terminal node for no benefit, and the input becomes VESTIGIAL: still wired,
  still hashed by `IS_CHANGED`, zero effect on output. Say that plainly rather
  than inventing a use. Its tooltip must be corrected in the same commit.

## 5. OPEN QUESTIONS FOR THE PANEL

**5.1 Does "100%" include the `[SFX: ...]` SCRIPT MARKUP?** This is the one that
can change output. `story_orchestrator.py:144-169` INJECTS
`[SFX: Scene transition - low bass sweep or static crossfade]` into the script;
`scene_sequencer.py:517` STRIPS `[ENV|SFX|MUSIC:...]` before TTS; the tag is
counted at `:2540-2545` and named in the token budget at `:1933-1988`.
**ANSWERED BY THE FAN-OUT, and this draft was WRONG.** The sentence that used to
sit here -- "today the writer writes a tag that the sequencer deletes" -- is
false. The writer does not write it, by any reachable path.

**`_inject_scene_transitions` is DEAD BY SHADOWING.** It is defined TWICE in the
same module: the `[SFX:]`-emitting body at `story_orchestrator.py:143-176`, and a
SECOND, unrelated function of the SAME NAME at `:2687-2712` that injects
`[TRANSITION: brief pause]`. Python rebinds the name at import, so the first body
is permanently unreachable. And it does not matter anyway: **both definitions
have ZERO callers repo-wide** -- the only two hits for `_inject_scene_transitions(`
are the two `def` lines themselves.

Corroborating, all measured rather than argued:

* **No pack or prompt asks for the tag.** `grep -i sfx` over
  `nodes/story_packs/**` returns nothing. Every other SFX mention in
  `story_orchestrator.py` is defensive RECOGNITION (structural-token sets,
  name-parsing false-positive guards) -- it tolerates the tag, never requests it.
* **The token-budget arithmetic does not move.** `_TOKEN_RATIO_*` are hardcoded
  constants; the only thing referencing them is a test that reimplements the
  numbers locally. The live writer has no SFX-named ratio at all.
* **`GemmaHeartbeatStreamer` -- which owns the `sfx_count` branch at
  `:2540-2548` -- is never instantiated anywhere.**
* **Shipped episodes: 2 ledgers out of several hundred contain `[SFX`,** both
  from May 2026, and in BOTH the hits are inside `script_gates[].issues[]`
  critic-rule guidance text -- never in a spoken line. **No
  PBUG-20260805-04-class defect exists here.**

**So the markup IS in scope, as dead-code hygiene rather than as a story
change.** Deleting the shadowed injector and the never-instantiated `sfx_count`
branch changes NO generated content, because none of it executes. Two earlier
reviews already reached this conclusion and neither was executed -- the
2026-07-02 Fable review ("a dead injector... the last SFX-emitting text in the
repo") and the 2026-07-10 lean-mean W1 list. Add the guard to
`tests/test_no_orchestrator_legacy_symbols.py` this time so it cannot creep back.

**DO NOT TOUCH THE STRIPPERS, and this is the sharpest call in the whole rip.**
`scene_sequencer.py:517`, `_otr_bark_lib.py:271` and `eng_bark.py:69` strip
`[ENV|SFX|MUSIC:...]` from text before synthesis, and the structural-token
recognition sets guard name parsing. These are NOT the SFX feature -- they are
generic sanitization against a model HALLUCINATING a bracket tag inline, which
is entirely plausible on old-time-radio training data with no prompt asking for
it. Removing them is zero benefit against a real risk of reintroducing the
"announcer reads a stage direction aloud" class. **They stay.**

**5.2 Does the parked cue-ledger campaign die with this, or stay parked?**
Retiring the docs (3f) is not the same as abandoning the idea. Recommend
retiring the docs and saying plainly in `GO_FORWARD` that the lane is CLOSED,
not parked -- a parked item nobody will revive is a to-do that never clears.

**5.3 Cloud media: does `cloud_media_canonical` lose anything else?** The
extractor is SFX-only, but `canonicalize_video` STRIPS provider audio and proves
the strip. That behaviour must SURVIVE -- it is invariant V-1 (only the mux emits
audio), not an SFX feature. Panel to confirm the boundary.

**5.4 Order.** Proposed: consumers before producers, the exact INVERSE of the
no-mirror build -- because here we are removing, not adding. Remove the reads
(mux bed, manifest fields, persist move) FIRST so nothing is left reading a field
whose writer just vanished; then the producers; then the engines; then the docs.
Panel to confirm this inversion is right, since the standing habit is
producer-first and applying it blindly here would leave armed readers.

## 5-BIS. WHAT THE SONNET FAN-OUT FOUND (2026-08-06, five parallel audits)

Operator asked for a fan-out because the rip "could be a big blast". It was
worth it -- these are findings this draft did NOT contain, each grounded.

### 5b.1 THE SINGLE MOST DANGEROUS LINE IN THE WHOLE RIP

`mux_master_audio` is `if not sfx_bed_path: <master_copy> else: <sfx_mix>`
(`:398-482`). **The branch that must SURVIVE is the `if` body (`:399-423`) --
the smaller one. The elaborate `else` body is the one that dies.** A careless
collapse that keeps the wrong branch swaps a `-c:a copy` master passthrough for
a RE-ENCODE on every episode, silently violating invariant V-1, and **no
surviving test distinguishes the two** -- they assert the audio is valid, not
which branch produced it. Write the collapse deliberately, and add an assertion
that the surviving command still carries the copy codec.

### 5b.2 BOTH PROTECTIVE GATES ARE OUTSIDE THE SFX BRANCH -- confirmed

The duration-drift guard (`:338-395`) sits BEFORE the branch. There are **two**
audio-SHA checks, not one: the byte-identity check at `:414-422` is inside the
`if` body (so it survives with it), and the SFX integrity gate at `:467-474` is
inside the `else` (so it dies with it). The 2026-07-14 fail-closed handler
(`mux():783-807`) is SFX-agnostic and is untouched. **No gate protecting a real
episode is removed.**

### 5b.3 THE SPEC'S OWN JUSTIFICATION FOR KEEPING `clip_manifest_json` IS WRONG

Section 4 above says the mux "still needs the manifest for `fps`/rows". It does
not. `clip_manifest_json` has exactly ONE consumer in `mux()` -- the
`compile_sfx_bed_from_manifest` call -- plus the `IS_CHANGED` hash. The node's
own `fps` WIDGET supplies fps. So after the rip the input becomes **vestigial**:
still wired (link 278 survives, no topology change), still hashed, with zero
effect on output. That is harmless but must be stated honestly rather than
justified with a use that does not exist. **Its tooltip (`:579`) describes SFX
mixing and would become a lie -- update it in the same commit.**

### 5b.4 A TEST FILE WHOSE NAME LIES, AND DELETING IT BY NAME WOULD COST REAL COVERAGE

`tests/test_google_video_sfx_workflow.py` tests **`_reresolve_master_audio` and
that canonical link 278 still exists** -- NOT SFX mixing. All five of its tests
must SURVIVE. Only two mux tests are genuinely SFX-only:
`test_sfx_bed_compile_rejects_invalid_manifest_rows` and
`test_sfx_mux_mixes_against_reference_pcm_sha_and_keeps_archival_pcm`.

### 5b.5 THE ROSTER BLOCKERS -- pytest failures, not cleanup

Dropping 32 engines to 27 breaks these ON THE NEXT RUN:

* `tests/test_multiclip_session_identity_roster.py` -- **three** hard asserts:
  `len(names) >= 30` (`:195`), an exact 12-member `EXPECTED_CLOUD_GAP`
  frozenset containing all five doomed engines (`:106-119`, checked at `:233`
  and `:389`), and `len(CLOUD_SPLITTERS) == 12` (`:395-397`).
* `tests/test_engine_contract_roster.py` -- the `WHOLE_SECOND` entry
  (`:318-324`) and `test_the_sfx_lanes_share_their_base_adapters_ladder_object`
  (`:395-407`).
* `tests/test_model_slot_audit.py:177-190` -- asserts `wants_provider_sfx`.
* **`tests/test_frame_receipt_conformance.py`** -- MY OWN test from no-mirror
  step 1 asserts `len(names) >= 30`. Same trap, same commit.
* **`tests/fixtures/still_plan_head_parity.json`** -- a checked-in GOLDEN file
  naming all five engines with per-engine computed rows. **REGENERATE it via
  `tests/test_still_plan_parity.py --regenerate`; do NOT hand-edit** -- removing
  engines reflows other engines' rows.

Plus the code sites: five `CAPABILITIES` rows in `registry.py`
(`:515,552,558,564,570`), the guarded import block in `__init__.py:217-222`,
and `render_driver.py:105-108`.

**Everything else is dynamic.** `role_slots`, `role_compat`, `slot_matrix`,
`_workflow_validation`, `otr_image_director`, `otr_shot_lock`,
`default_engine_for_role` all enumerate off the live registry and shrink by
themselves. No role DEFAULTS to an SFX engine (every one declares
`default_roles = ()`), so no role loses its default.

### 5b.6 THE LEDGER RULE IS SATISFIED -- verdict NO-LEDGER-HOLE, and it is proven

The audit that mattered most for "don't break the system" came back clean, and
the evidence is stronger than the argument this draft made:

* **Every production reader of all three fields is already GUARDED.** Four
  sites, no more: `otr_master_audio_mux.py:194` and `:209` (both
  `.get(...) or ""`, and the function returns `""` at `:196` before
  `sfx_duration_s`/`sfx_sha256` are read AT ALL), `render_driver`
  `persist_episode_clips:4442` (`if not sfx_src: continue`), and
  `build_clip_manifest` (`if sfx_stem:` gates the other two). `scripts/` and
  `tools/` have ZERO matches.
* **ZERO shipped episodes carry the field.** A grep for `sfx_stem_path` across
  50+ ledgers under `output/otr/episodes` returns nothing. There is no durable
  reader to break because there is no durable data.
* **No schema declares them.** Neither `CanonicalClip` nor `ShotRow` -- both
  `extra="forbid"` -- lists any of the three. The producers return PLAIN DICTS
  that are never validated through `CanonicalClip`. **No schema edit is owed**,
  which this draft did not know.
* **No manifest hash, golden file, or positional parse.** Every consumer does
  `json.loads` then keyed access, so removing keys changes nothing structurally.
* **The janitor is generic**, age-based over `episodes/_shared/tmp`, with no
  knowledge of `.sfx.wav`. A stray file from a past dormant run is still swept.

**And the 7.1 defect simply evaporates:** with the fields gone there is no key
left for `beat_clip = dict(clip or {})` to inherit wrongly.

### 5b.7 LINE-CITE DRIFT IN THIS VERY DOCUMENT -- and the cause is instructive

Section 2 above cites `build_clip_manifest` at `:4648-4652`. **It is now
`:4810-4814`.** The file grew by ~160 lines between this draft and now, and it
grew because of MY OWN no-mirror step-2 commit (`ac8a1925`), which added
`closing_theme_frame_window` and `FRAME_RECEIPT_VERSION` to the same file.

This is the third time in two days a hand-written document has gone stale
against a file the same session was editing. **Re-pin every cite in sections 2
and 3 immediately before the rip is coded, not before it is reviewed** --
reviewing against stale cites is how a panel confidently blesses the wrong
lines. `persist_episode_clips:4442-4463` and `render_beat_coverage:3499` are
still accurate.

### 5b.8 THE SUITE WILL GO GREEN ON A COVERAGE HOLE AND SAY NOTHING -- proven

**`EXPECTED_FAILED_NODEIDS` (`tests/conftest.py:178`) is an empty frozenset, and
the hook has NO presence or completeness check.** It diffs actual failures
against expected ones; a nodeid that simply STOPS EXISTING produces no signal
from either branch. Deleting all 24 SFX-only tests is **0.3% of an 8,951-test
suite** -- invisible in a "still green" read.

(Also found: `docs/known-failures.md`, which `conftest.py` tells you to update,
**does not exist anywhere in the repo.** Stale instruction, worth fixing while
we are here.)

**So the tripwire is not hygiene, it is the only thing that will notice.** Land
it in the SAME commit as the deletions, as `tests/test_rip_sfx_bed_guard.py`,
sibling to the existing `test_rip_sfx_broll_guard.py` -- which is the working
precedent in this repo for exactly this shape and which **guards the OLD role
rip and must be left completely alone.**

Two checks are BOTH required, and this is a real subtlety: the four
`google_vid_sfx_*` engines **never set `wants_provider_sfx`** -- only their NAME
and their module carry the marker. A `wants_provider_sfx` check alone catches
only the Vidu one. Assert on the flag AND on `"sfx" in engine_id`.

### 5b.9 FOUR PIECES OF NON-SFX COVERAGE WOULD BE DELETED BY NAME

Every one of these lives in a file that LOOKS disposable. This is the list that
turns a safe rip into an unsafe one if it is skipped:

1. **`tests/test_google_video_sfx_workflow.py` -- ZERO SFX assertions in the
   whole file.** All five tests survive untouched. One of them,
   `test_canonical_workflow_wires_clip_manifest_to_master_audio_mux`, pins
   `last_link_id == 284`, node 85's eight inputs and link 278 itself -- **it IS
   the section-4 topology re-verification this spec demands. Running this file
   green after the rip is the proof.**
2. **`test_clip_fill.py::test_persist_rekeys_sfx_to_renamed_episode_clips`** is
   the ONLY test covering `persist_episode_clips`' rename/rekey path for
   `clip["path"]`. Strip its SFX fixture and assertions; **keep the function.**
3. **`test_existing_google_video_engines_stay_silent`** (in the beds file, which
   otherwise dies) proves invariant V-1 for the SURVIVING Veo/Omni engines.
   **Relocate it**, do not delete it.
4. **`assert "generateAudio" not in base_payload["parameters"]`** -- buried
   inside an SFX-only test, and the ONLY place in the suite pinning that base
   VEO payloads never request audio. Preserve the assertion when its host dies.

Plus three shared-collection loops in `test_cloud_video_adapters.py`
(`_CLOUD_ROWS` and two engine tuples) where the SFX ROW is pruned and the
FUNCTION survives with a smaller loop.

Exact count: **24 test functions fully deleted, ~4 more shrink.**
`test_rip_sfx_broll_guard.py` and `test_google_video_sfx_workflow.py` contribute
ZERO deletions.

### 5b.10 The bed has never written a file in production

`compile_sfx_bed_from_manifest` returns `""` before reaching any `makedirs` or
ffmpeg write, so `_default_sfx_bed_out`'s `.sfx_mix.wav` has never existed on
disk. Deleting it orphans nothing. `audio_mode` is consumed by exactly two test
assertions and no ledger, parser or audit script.

## 6. THE PROOF

Deleting a lane that nothing selects cannot be proven by the suite alone -- a
green suite proves the tests were also deleted. Required:

1. `tools/engine_matrix.py` REGENERATED (the roster drops from 32 to 27) and
   `--check` green. The matrix is drift-gated, so this is not optional.
2. `python scripts/otr_check.py` or equivalent import audit -- five engines
   leaving `NODE_CLASS_MAPPINGS`/the registry must not dangle a reference.
3. **One live canonical leg**, `RESULT SUCCESS` + `obs_publish OK` + the asset on
   disk, proving the mux still publishes with its SFX branch gone. The mux is
   the TERMINAL node; if this is wrong there is no episode.
4. The tripwire test from 3e green.

## 7. RISK, STATED PLAINLY

The rip touches the terminal publish node. `OTR_MasterAudioMux` is the last thing
that runs and the only thing that writes `otr/obs/`. A mistake here does not
degrade an episode -- it produces none. That is why the live leg in section 6 is
mandatory rather than nice-to-have, and why section 4 lists what must not move.

Everything else in this rip is dormant code that no shipped episode has ever
executed.
