# OTR Remaining Sprints -- build plan under the NO-FALLBACKS / dropdown-only-defaults directive

Date: 2026-07-02 late evening; r2+r3-hardened same night (kibitz judgments in
kibitz-runs/2026-07-02-remaining-sprints/r2/r2_plan.md + r3/r3_plan.md). Branch v2.0-alpha.
Suite 6075/0.
Governs: everything left to CODE. Testing/soak/eyeball gates are out of scope here.

## 0. STANDING DIRECTIVES (operator, 2026-07-02 -- override anything below that disagrees)

- **NO FALLBACKS, NO AUTO-DEFAULTS, ANYWHERE.** The dropdown values SAVED in
  `workflows/otr_scifi_16gb_full.json` are the ONLY defaults. Engine failure = LOUD stop,
  never a swap. No code-side default_engine_for_role may override the shipped JSON widgets.
- **NO hidden enable switches.** (Shipped @ cc349c1d: OTR_ENABLE_COMFY_CLOUD_MEDIA removed;
  the dropdown pick IS the enable; auth fails LOUD at invoke; budget unset = $10 default cap.)
- Invariants: single resident heavy <= 14.5GB (NVML); audio spine FROZEN (byte-identical
  master, mux-LAST); determinism seed-keyed; UTF-8 no BOM; SFW; workflow JSON edited in the
  SAME change as code; suite + Bug Bible + push per green chunk; prod/main GATED.

## Sprint A -- E1/E2: rip the fallback scaffolding (PROMOTED, do first)

**Internal build order (r3): A4 -> A3b (Policy default False) -> A2 -> A1 -> A3a/c/d/e.**
No window where fallback_engine=None adapters coexist with live chain resolution.

- **A1 render_driver rip.** Remove FLOOR_NAMES(:52) / UNIVERSAL_FLOOR(:56) /
  SYNTH_FALLBACKS(:63) / EXPECTED_OOM_TRAIL(:117) / make_fallback_of(:153), the
  run_episode consumer (:2101), the soak verifier + fixture builder (:2527/:2589),
  and the __all__ exports (:2654-2657). ALSO rip the DUPLICATE copy in
  `scripts/otr_video_soak.py` (its own make_fallback_of:150, EXPECTED_OOM_TRAIL:259,
  trail assert :294). run_gpu_soak/assert_soak_ok become NO-TRAIL LOUD contracts:
  a forced OOM must RAISE a named RenderError and the soak asserts the raise --
  no trail matching. ALSO in scope (r3): `nodes/_otr_shared/fallback.py`
  (resolve_fallback_chain, the AS-2 resolver) is DELETED with its consumers
  rewritten (render_driver, scripts/otr_video_soak.py, scripts/otr_video_gpu_smoke.py,
  tests/test_video_fallback_chain_additive.py, tests/test_video_retry_taxonomy.py,
  tests/test_video_survival_guide_vectors.py; retry_taxonomy KEEPS its
  non-fallback failure-classification role -- verify the split at build). Update
  stale fallback comments/logs at render_driver :1035-1037/:1287-1289/:1880-1882.
- **A2 per-adapter fallback_engine = None.** `fallback_engine` is a live class
  attribute across the fleet: eng_humo.py:130 ("humo_1.7B") + :514 ("still_motion"),
  eng_ltx_video.py:340, eng_mesh_stage.py:329, eng_triposr.py:121,
  eng_still_parallax.py:187, eng_character_3d.py:258/:327/:398. Set ALL to None on
  the existing in-repo precedent (eng_ltx_av.py:1146, eng_viz_mandala.py:67 --
  `fallback_engine = None  # NO FALLBACKS: fail LOUD`). character_3d OOM = named
  RenderError, never chain-to-humo. Update adapter docstrings +
  cheap_families.py:172/:182 floor-terminus comments. still_motion remains a
  REGISTERED SELECTABLE engine (do not unregister) but loses its floor role.
- **A3 E2 contract (allow_auto_fallback).** TWO live defaults: director widget
  (otr_video_director.py:241, default True, passed through :345) AND
  Policy.allow_auto_fallback (schemas.py:130, default True). Pin: (a) widget
  default -> False + "(deprecated)" label, widget STAYS positional (no mid-list
  removal, BUG-LOCAL-097); (b) Policy default -> False; (c) runtime IGNORES a
  True value with a LOUD deprecation log -- a stale JSON must never resurrect
  fallbacks; (d) re-audit widgets_values in `workflows/otr_scifi_16gb_full.json`
  in the SAME change; (e) update the pinned True in test_route_a_14b_promotion.py:132,
  test_still_aspect_and_labels.py:208, test_video_platform_aseam.py:316/:341
  (+ the :260 schema list). scripts/run_otr_30word_smoke.py's False-patch
  (:212-216) becomes a harmless no-op -- keep.
- **A4 test triage BEFORE the rip.** Consumers to split keep-rewrite vs delete:
  test_cs3_inter_beat_reclaim.py:55/65/74/84, test_ltx_av_driver_wiring.py:25-28,
  test_video_character_3d.py:360-369, test_video_mesh_stage.py:320,
  test_video_render_driver.py:24/33/41/84, test_video_render_driver_additive.py:77-82/415,
  test_video_soak_fixture.py:82/128, test_video_still_parallax.py:177-186,
  PLUS (r3): test_video_humo.py:53/:57/:216-232 (asserts fallback_engine values
  + chain convergence) and test_video_mesh_stage.py:68. Some pin valid
  non-fallback behavior that must SURVIVE the rewrite. Add per-family
  LOUD-failure contract tests.
- **A5 ledger.** Keep fallback restamp fields (stamped never) -- no schema churn.
  Verify-at-build: content_oracle.check_manifest must not REQUIRE fallback trails.

## Sprint B -- S1 stills lane (cloud)

- **B1 adapters.** `nodes/_otr_image_engines/eng_cloud_image.py` on the ImageEngine
  protocol (assert_usable/prepare/render_image/teardown -- registry.py:80-82),
  imported from the package __init__. render_image() -> invoke_partner_node() ->
  canonicalize_image() -> return str(asset.path) (dispatcher _coerce_pixels
  accepts a PNG path). Rows: cloud_recraft (CHEAP), cloud_flux_pro (BEST),
  cloud_nano_banana_2 (BEST), cloud_ideogram_v4 (text-render), + cloud_seedream_2
  (pinned + profiled -- include unless explicitly CUT in-sprint). Per-row pinned
  kwargs from partner_nodes.yaml + PROMPT_PROFILES.md; NO default_roles.
- **B2 CAPABILITIES.** Add one row per cloud image engine in registry.py:107 in
  the SAME change (tests/test_capability_profiles.py one-declaration invariant).
- **B3 canonicalize_image.** Replace the stub (cloud_media_canonical.py:106-109)
  mirroring the video signature: (raw, request, session), request carries
  {"w","h","format":"PNG"}; sRGB; sha256 kept (uniform CanonicalAsset); named
  CloudMediaError codes. Called INSIDE the adapter's render_image (the image
  dispatcher has no canonicalize hook). r3 pins: (a) it WRITES a real PNG to
  disk (transcode JPEG/WEBP provider output); (b) the adapter returns
  str(asset.path); (c) the file exists and clears the dispatcher's minimum PNG
  bytes before _coerce_pixels reads it; (d) the adapter's prepare(None, None,
  None) must not crash (the dispatcher calls it with three Nones; auth/budget
  live in the invoke bridge, not prepare); (e) EVERY cloud invocation passes a
  nonzero per-row estimated_usd (the bridge defaults to 0.0 and reserves exactly
  that -- the budget machine is inert without it).
- **B4 V3/COMBO resolution (single source of truth).** Resolved model IDs live
  in checked-in config/adapter constants (env-overridable) for ALL V3 rows
  (nano_banana_2, seedream_2, elevenlabs x2, seedance, wan); passing the literal
  "COMFY_DYNAMICCOMBO_V3" placeholder downstream is a build error; never a live
  COMBO scrape at render time; resolved kwargs asserted by the conformance test.
- **B5 conformance test (owed since the video-team warning).** Parametrize over
  partner_nodes.yaml ROWS (14), not adapter modules: every kwarg an adapter emits
  must be declared in its pinned row schema; a row without an adapter carries an
  EXPLICIT xfail entry, and Sprint D's definition-of-done includes REMOVING the
  ElevenLabs xfails (nothing stays permanently masked). Covers video + image +
  future TTS. Optional inverse check: every registered cloud engine has a yaml row.
- **B6 portrait-mint gate (PRE-SELECTION GATE, not a fallback).** `portrait_mint_3d`
  prompt profile in the character_description -> finish_visual_prompt chain
  (subject fully in frame, front/3-4 neutral pose, clean backdrop, even light).
  The gate runs BEFORE the 3D flag activates and before any cloud spend;
  rejection = ledger stamp `mint_3d=REJECTED:<reason>` + the beat stays 2D BY
  NON-SELECTION. The gate does NOT duplicate the budget machine
  (cloud_media_backend owns spend). If 3D was explicitly dropdown-selected and
  the mint is rejected, that is a FAILURE -- fail LOUD (directive).
  Verify-at-build (r3): the gate point in the beat loop must PRECEDE engine
  selection (ShotLock) -- if selection happens first the gate becomes a swap.
- **B7 wiring.** Stills dropdowns (announcer/music/other/character image models)
  IN `workflows/otr_scifi_16gb_full.json` in the SAME change; validator +
  widget audit after.

## Sprint C -- S3 remainder (rescoped by the directive)

- CUT: reactive auto-defaults; fallback chains. (Directive.)
- KEEP: ShotLock audit stamps for cloud rows -- stamp `provider_id`,
  `estimated_usd_cap`, `is_cloud=True` on cloud video shot rows (observability-
  only; consumed by nothing yet); seedance_2 +
  wan V3-expansion pins (un-dark seedance; wan gains its pinned prompt path if
  the V3 pin exposes one); live provider proof rides the operator smokes.

## Sprint D -- cloud TTS lane (ElevenLabs)

- **D1 PREREQUISITE: canonicalize_audio** (S2 stub @ cloud_media_canonical.py:99):
  soundfile load, resample 44.1kHz, downmix per stereo_policy, loudness matched
  to the existing local reference, +/-250ms per-line tolerance w/ head/tail
  silence padding, actual_duration_s to line metadata.
- **D2 ELEVENLABS_VOICE resolution (r3-pinned).** The voice_selector row is NOT
  routed through invoke_partner_node (it declares no hidden auth input;
  _inject_hidden_inputs RAISES for such rows -- cloud_media_invoke.py:331-336).
  The TTS adapter resolves the selector class locally, executes it in-process,
  UNWRAPS the returned tuple ([0]), and passes the ELEVENLABS_VOICE object as a
  pre-resolved input kwarg to the billed TTS invoke; the bridge's input
  marshaling must accept non-file custom types. The selector's voice COMBO value
  comes from the profile row's default_params (env-overridable per B4).
  model: COMFY_DYNAMICCOMBO_V3 resolved per B4.
- **D3 registration surface (all in the SAME change).** Adapter in
  nodes/_otr_audio_engines/ against the FROZEN AudioEngine protocol (per-line
  clips; the master mix stays byte-identical -- cloud TTS feeds the SAME per-line
  WAV contract as bark/kokoro/indextts2) + profile rows in
  config/audio_engine_profiles.yaml + `_LEGACY_FIRST_ENGINES` additions
  (char_voice + announcer_voice -- nodes/_otr_engine_profiles.py:42; dropdowns
  build from THIS map, adapter registration alone does not surface rows; r3:
  APPEND to the END of both tuples -- index 0 is the shipped default and stays
  indextts2/kokoro) + JSON widget audit. default_roles = (); registers
  unconditionally (no requires_flag). NO new widget in the static shell (V-11);
  the voice dropdown is an existing widget whose option list grows.
  **Return contract (r3):** the voice seam expects generate_voice() to return an
  AUDIO dict {"waveform": tensor, "sample_rate": int} packed by pack_audio_batch
  (_otr_voice_node_common.py:559/:565/:581) -- the adapter calls
  canonicalize_audio then LOADS the canonical WAV back into that dict; duration
  metadata is stamped separately, never the return value. Nonzero estimated_usd
  per B3(e).
- **D4 has_audio clarification.** has_audio is a VIDEO-side field (schemas.py:237);
  cloud TTS only feeds the per-line WAV contract; the audio spine is untouched
  (test_audio_byte_identical stays green).

## Sprint E -- S-C C1: shared audio_motion_profile

- **E1 schema.** Typed AudioMotionProfile model added to VideoRequest (which
  extends _Forbid -- unknown keys REJECTED, schemas.py:78/:139). The field is
  REQUIRED (no None default); every fixture constructing VideoRequest updates in
  the SAME change (verify no serialized-request cache exists). Engines fail LOUD
  if the field is absent -- never recompute (drift guard).
- **E2 compute point + carrier (r3).** Computed ONCE in run_real_episode (where
  master_audio_path exists), stamped into ledger["video"]["audio_motion_profile"];
  BOTH request builders emit it: build_request (render_driver.py:225, the
  soak/test path) and build_request_from_shot (:972, the episode path).
  Soak/test paths construct an EXPLICIT named test-fixture profile (a visible
  constant in the fixture) -- production paths fail LOUD on a missing profile;
  nothing is silently defaulted at render time.
- **E3 v1 metrics (TRIMMED).** duration, rms, peak, onset (transient-peak count
  over threshold), silence ratio -- numpy/torch only, NO librosa/new deps.
  brightness / dynamic-range / speech-vs-music DEFERRED to C2 with the first
  real consumer (speech-vs-music will be role-derived, static). C2 (per-engine
  consumers + HuMo phrase-chunking) stays deferred.

## Sprint F -- creative formats F1/F2 (queued last)

- Per `docs/2026-07-02-creative-formats/CREATIVE_FORMATS_PLAN.md` (kibitz-hardened);
  V1 probes first; codex-only kibitz per shipped format diff.

## Ordering

A (E1/E2) -> B (S1) -> C (S3 remainder) -> D (TTS) -> E (C1) -> F. GPU gates (soak2 QA,
proof9d, S5 A/B, live smokes) interleave on render windows and do not block CPU coding.

## Cloud failure policy (r3)

v1 is LOUD STOP / NO RETRY -- intentional and directive-consistent (a transient
provider failure is a failure; the bridge classifies RETRYABLE_TRANSPORT but does
not retry). A bounded transport-only retry is a possible LATER opt-in, out of
scope for these sprints.

## Open verify-at-build

- content_oracle.check_manifest fallback-trail requirements (A5).
- B6 gate-vs-ShotLock call order in the beat loop.
- fallback.py rip vs retry_taxonomy classification split (A1).
- ELEVENLABS_VOICE custom-type marshaling through the invoke bridge (D2).
- seedance_2 V3-expansion pin dynamic inputs (Sprint C).
- No serialized VideoRequest cache exists on disk (E1).
