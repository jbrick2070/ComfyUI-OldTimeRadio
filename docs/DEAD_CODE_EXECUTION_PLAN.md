# Dead-code campaign -- the adjudicated execution plan (from 2026-08-28)

## STATUS as of 2026-08-28 late evening

* **WAVE 1 -- DONE.** All five correctness fixes landed with regressions in
  `tests/test_wave1_boot_and_ffmpeg_resolution.py` (13 tests): the mux identity
  resolver, `scope_draw`'s env step, the `wrapper_bridge` resolver (applied at
  the two EXECUTION points so all ten pure arg-builders are covered without
  touching one tested arg list), the cloud-video canonicalizer, the
  planning-ceiling `OverflowError`, the joint-AV trace projection, and BOTH
  halves of the H3 boot-contract fix (floors + Sage + specificity in
  identification; H3 asserting its OWN contract instead of one derived from a
  policy that has had `launch` stripped).
* **WAVE 2 -- DONE.** Scene's two orphaned declarations removed (Python only,
  ZERO workflow JSON delta, links 282/283 verified intact), plus
  `tests/test_input_types_signature_parity.py`, which sweeps ALL 25 registered
  nodes and asserts both directions: no node may declare an input its execute
  function cannot accept, and EpisodeAssembler must still declare AND accept
  the music bus.
* **WAVE 3 -- NOT STARTED, deliberately.** It invalidates persisted audio
  cache entries; it lands alone, on a quiet box, with the operator informed.
* **WAVE 4 -- MOSTLY DONE.** ~500 lines removed plus nine corrected lying
  comments. Landed: the eleven zero-call ledger stampers and the two symbols
  they orphaned; story_orchestrator's dead name pools, regex block and seven
  globals; `script_text_parts` and its token assembly; the unreachable
  slot-scheduler `else`; ImageDirector's shadow combo, dead alias, 3D-lock
  tooltip and error prose; the mux stream counter and its false env claim;
  `compute_clip_budget`'s ignored `policy` and impossible `warnings`;
  CastLock's dead local and two dangling references; `enable_polish_pass`;
  writer `VOICED_ROLES`; `_CLIP_UNDERRUN_FRAC`; `optimization_profile`
  (AST-verified unread at both receivers); the `manifest_episode_id` dead
  store; `DeliveryProfile`'s two mirror fields; `text_tails.collapse_ws`; the
  decode-guard rationale; the foley/MIME false constants; the ghost target
  band; `curve_cache_key`; the two write-only observability fields;
  `_episode_facts(meta)`; and the acceptance / coverage-plan / ghost-author /
  wrapper-bridge / LTX-2.5 prose.
* **WAVE 4 -- DELIBERATELY SKIPPED (recorded, not forgotten):**
  * The `native` attribute on eight engines. Codex scoped it to five audio
    adapters; it is actually on eight (audio + image + video) with six tests
    asserting it. Fourteen files of churn to delete eight lines of declarative
    metadata that documents whether an engine drives ComfyUI's own nodes or an
    external service. Cheap to keep, annoying to reconstruct.
  * `_still_spine_requires_scene(shot, ...)`. `shot` is AST-confirmed unread,
    but the parameter has seven call sites across three files plus tests, and
    the signature reads as a coherent triple. Zero behaviour gain.
  * The broad `TypeError` compatibility retries. These are live fallbacks for
    external adapters; removing them while the operator is away trades a real
    (if unlikely) compatibility break for tidiness.
* **WAVE 5 -- untouched. Operator rulings required.**

Authoritative queue after THREE layers: six blind fresh-eyes seats -> the
driver's ledger -> an independent Codex adjudication (61 findings: Job A
25 confirmed / 7 partial / 4 hazard / 2 misread, Job B 26 fresh groups).
Nothing below is executed until its wave is named. Every wave ends with the
full suite, Bug Bible, drift battery, `build_variants.py --check`, AST/BOM
checks, and commit+push.

## WHAT THE ADJUDICATION KILLED -- read this before touching anything

Three claims from the blind seats were WRONG and would have broken the build.
All three re-verified by the driver against the real files, not taken on trust:

1. **The SceneSequencer music sockets are NOT a widget removal. Executing the
   three-part widget recipe by name would have SEVERED THE PRODUCTION MUSIC
   BUS.** Grounded: canonical node 3 (`OTR_SceneSequencer`) has ZERO
   `music_cue` sockets and 5 widget values; canonical node 7
   (`OTR_EpisodeAssembler`) carries `music_cue_audio` on **link 282** and
   `music_cue_manifest_json` on **link 283**, both fed by node 83
   (StableAudioTheme). The real defect is narrow and real: Scene's
   `INPUT_TYPES` still DECLARES two optional sockets its `sequence()` cannot
   accept, so wiring either in the UI raises `TypeError`. **Correct fix =
   delete the two declarations + stale comment in Python ONLY** (zero JSON
   delta, nothing to regenerate) plus an INPUT_TYPES/signature parity test.
   NEVER touch node 7, node 83, links 282/283, or their tests.
2. **`nodes/_otr_image_engines/schemas.py` is NOT dead.** `GRANULARITY_MODES`
   is imported at `otr_image_director.py:51` and used at `:165`; deleting the
   module breaks ImageDirector's `INPUT_TYPES` (the node menu). Also protected
   by a standing ruling. REJECTED.
3. **`nodes/_otr_shared/encode_sink.py` is NOT importless.**
   `scripts/profile_scope_render.py:32` imports `RawVideoSink` and
   instantiates it at `:150` and `:230`. REJECTED.

Also corrected: ComfyUI's real interrupt is
`InterruptProcessingException(BaseException)`, which BYPASSES `except
Exception` -- so mux Cancel already propagates. The "Cancel is swallowed"
claim is a MISREAD; only the unused private `_Interrupted` class is debris.

## WAVE 1 -- root-cause correctness (these are BUGS, not tidiness)

Ordered by blast radius. All are env/contract faults that fail LATE, after
expensive work, which is what makes them worth the front of the queue.

1. **H3 boot-contract identification** (`_otr_shared/boot_contracts.py`,
   `eng_minimax_h3.py:792-809`). `contract_from_running_server()` matches
   reserve EXACTLY and ignores Sage, while production strips `launch` from the
   policy -- so a valid H3 boot can be mislabeled and REFUSED, and H3's late
   assertion derives an unconstrained `default` that checks nothing. Fix:
   identification uses pinned equality + reserve FLOORS (`>=`) + deterministic
   specificity, excludes a Sage-constrained candidate when Sage is known
   incompatible; H3's second defense calls `assert_running_server("h3")`.
   **THIS GATES THE H3 RENDER LEG** -- without it that leg can fail for a
   reason unrelated to what it is meant to prove.
2. **ffmpeg resolution, four sites.** The 2026-08-28 fix declared three
   modules done and missed the rest. All four fail only on an env-only box
   (ffmpeg reachable ONLY via `OTR_FFMPEG`), which is exactly the shipped
   install shape:
   * `otr_master_audio_mux.audio_pcm_sha` -> `shutil.which("ffmpeg")`; the mux
     encodes fine, then the fail-closed identity proof returns empty and
     **destroys a finished episode at the last boundary.**
   * `_otr_shared/scope_draw.find_ffmpeg` -> never reads the env var, while the
     shipped scopes tooltip promises it does.
   * `_otr_video_engines/wrapper_bridge` -> every builder defaults to literal
     `"ffmpeg"`; cheap-family preflight validates the env var and then runtime
     drops it, so an env-only box passes preflight and dies mid-render. Widest
     blast radius: still families, HuMo, LTX, MiniMax, Wan, Ghost Signal.
   * `_otr_shared/cloud_media_canonical.canonicalize_video` -> hardcoded.
   One sentence everywhere: explicit arg if it resolves -> valid `OTR_FFMPEG`
   -> PATH. Keep the three top-level resolvers LOCAL (house rule 1); add
   parity tests instead of a pack-wide resolver.
3. **Cloud interrupt class** (`cloud_media_invoke.py`). Recognizes a synthetic
   `ProcessingInterrupted`; real Cancel falls through to `PROVIDER_REJECTED`
   and releases the reservation instead of settling INTERRUPTED. Suppress only
   the real interrupt/cancel, never blanket `BaseException`.
4. **`normalized_planning_ceiling(float("inf"))`** (`frame_contract.py`)
   promises "never raises" but `int(inf)` raises `OverflowError`; JSON accepts
   bare `Infinity`. Add the guard, `+/-inf -> 0`.
5. **Joint-AV trace projection** (`render_driver.py` / node 92).
   `joint_av_prompt`, `joint_av_sounds`, `joint_av_identity_leak` are stamped
   then dropped by a manual allowlist, so the published evidence the schema
   promises never reaches `/history`. Additive receipt fix.

## WAVE 2 -- the declaration-only Scene cleanup

Exactly as bounded above, with the parity regression (every required+optional
INPUT_TYPES key is a parameter of the execute function, allowing `**kwargs`,
excluding hidden inputs). Acceptance: `/object_info` shows no Scene
`music_cue_*`; EpisodeAssembler still declares both; canonical + all 61
variants show ZERO semantic delta; links 282/283 intact.

## WAVE 3 -- cache admission (medium risk, deliberate invalidation)

Admit-check persisted `prepare_text_version` and `delivery_projection_version`
in `needs_rerender()`; do NOT admit-check `engine_prompt_template_version`
(its writer is a literal `"1"` with no bump source). This intentionally
invalidates old persisted entries -- land it alone, never inside a render week.

## WAVE 4 -- mechanical debris and comment truth (large, low risk)

The confirmed bulk, each re-verified at execution: the eleven (not eight)
zero-call `_otr_ledger` exports; the dialogue-name regex block; the
sanctioned-gap docstring that is now false; `script_text_parts`; the Scene
module header + dead pacing locals; the unreachable slot-scheduler `else`;
ImageDirector's 3D-lock prose and two dead shadows; the mux stream counter;
`compute_clip_budget`'s ignored `policy` and impossible `warnings`; CastLock's
two dangling references; `enable_polish_pass`; writer `VOICED_ROLES`;
`_CLIP_UNDERRUN_FRAC` (keep its regression test -- it deliberately proves the
retired knob cannot weaken the terminal rule); seven story-orchestrator
globals; `optimization_profile` (NOT `max_feeds`); the `manifest_episode_id`
dead store; `DeliveryProfile.version/projection_version`;
`text_tails.collapse_ws`; the decode-guard rationale; foley/MIME ID constants;
the ghost prompt target band; five `native` attributes; `curve_cache_key`;
two write-only render-request fields; the broad `TypeError` retries; two
ignored render-driver parameters; and the stale-prose set (render driver,
cheap families, coverage plan, frame contract, ghost author, wrapper bridge,
LTX 2.5 recipe, acceptance).

## WAVE 5 -- owner rulings required (do NOT execute silently)

* `audio_motion_profiles` (~300 lines + per-beat WAV/FFT on every render):
  deferred-C2 scaffolding with no in-repo reader. Retire or keep paying.
* `CloudMediaSession.episode_id`: serialized `null` on every production row --
  wire real identity or formally retire the durable key.
* ShotLock `binding_hash` / top-level `request_seed` / `clip_budget`: split
  decision; `degradation_trail` STAYS (schema + soak contract).
* `voice_cast_decision`: remove the dead local + false comments now; the
  durable empty key needs a ruling (a script verifies it).
* Cast genre pools + `OTR_CAST_GENRE`: wire, remove, or document-inert --
  removal risks deterministic cast identity, so it needs a frozen-name fixture.
* `StillPlanRow` S2 fields: deferred schema debt, high blast radius.
* Shakespeare `fixtures/` legacy dir and `content_oracle` rehoming: test
  migrations, behind the P0 wave.

## Standing rejections (never re-report)

Cross-lane helper consolidation (house rule 1); any edit to
`eng_indextts2.py` for tidiness (the Lemmy release gate fingerprints that
build -- proven when a behavior-identical dedup failed
`test_the_shipped_lemmy_route_is_selected_again` and was reverted); the three
top-level ffmpeg resolvers merging into one shared resolver.
