# r2 JUDGMENT -- remaining-sprints plan (coding round)

Judge: Claude (Cowork), 2026-07-02/03. Panel: codex + antigravity + claude CLI + my anchor.
Every claim below was grounded against the real Windows files this session. VERDICT:
plan is BUILD-READY after folding the survivors below into PLAN.md. No arc change.

## GROUNDING RESULTS (what survived, what died)

SURVIVED (confirmed against code):
- Sprint A real sites: render_driver.py FLOOR_NAMES:52, UNIVERSAL_FLOOR:56,
  SYNTH_FALLBACKS:63, EXPECTED_OOM_TRAIL:117, make_fallback_of:153, run_episode
  consumer :2101, soak verifier asserting EXPECTED_OOM_TRAIL :2527, soak fixture
  builder :2589, __all__ exports :2654-2657. All three panelists agree; confirmed.
- scripts/otr_video_soak.py DUPLICATES the scaffolding (its own make_fallback_of:150,
  EXPECTED_OOM_TRAIL:259, trail assert :294) -- E1 must rip BOTH copies (antigravity
  caught the script; codex caught run_gpu_soak/assert_soak_ok). Confirmed.
- Test consumers of make_fallback_of / EXPECTED_OOM_TRAIL (claude CLI's list,
  confirmed + extended): test_cs3_inter_beat_reclaim.py:55/65/74/84,
  test_ltx_av_driver_wiring.py:25-28, test_video_character_3d.py:360-369,
  test_video_mesh_stage.py:320, test_video_render_driver.py:24/33/41/84,
  test_video_render_driver_additive.py:77-82/415, test_video_soak_fixture.py:82/128,
  test_video_still_parallax.py:177-186. Each needs keep-rewrite vs delete triage,
  not blanket excision (some also pin valid non-fallback behavior).
- E2: allow_auto_fallback default True at otr_video_director.py:241 AND
  Policy.allow_auto_fallback: bool = True at schemas.py:130 (codex should-fix #1
  confirmed -- TWO live defaults). Passed through at director :345. Also consumed by
  scripts/run_otr_30word_smoke.py:212-216 (patches it False -- becomes a no-op,
  keep harmless) and pinned True in tests test_route_a_14b_promotion.py:132,
  test_still_aspect_and_labels.py:208, test_video_platform_aseam.py:260/316/341.
- Sprint B: nodes/_otr_image_engines/registry.py protocol = assert_usable/prepare/
  render_image/teardown (:80-:82), CAPABILITIES dict :107 with a one-declaration-
  per-engine invariant test (tests/test_capability_profiles.py) -- codex must-fix
  #1/#2 confirmed. canonicalize_image + canonicalize_audio are NotImplementedError
  stubs (cloud_media_canonical.py:92-109) -- confirmed.
- Sprint D: voice dropdowns build from _LEGACY_FIRST_ENGINES
  (nodes/_otr_engine_profiles.py:42, legacy_first_engines():60) -- codex/antigravity
  confirmed; adapter registration alone will NOT surface rows.
- Sprint D: partner_nodes.yaml rows confirmed -- elevenlabs tts/flash need
  model: COMFY_DYNAMICCOMBO_V3 + voice: ELEVENLABS_VOICE (:27/:32/:60/:65);
  voice_selector is the AUX producer (:74-:91); cloud_nano_banana_2 model is
  V3 (:236). The invoke bridge is file-centric -- the ELEVENLABS_VOICE object
  needs a local node-class resolution step (antigravity must-fix #3, plausible,
  verify exact mechanics at build).
- Sprint E: VideoRequest extends _Forbid (schemas.py:78/:139) -- unknown keys
  REJECTED; audio_motion_profile requires a schema field. Codex must-fix #5 confirmed.

JUDGE ADDITION (missed by all three):
- `fallback_engine` is a LIVE CLASS ATTRIBUTE across the adapter fleet, not just
  character_3d: eng_humo.py:130 ("humo_1.7B") + :514 ("still_motion"),
  eng_ltx_video.py:340, eng_mesh_stage.py:329, eng_triposr.py:121,
  eng_still_parallax.py:187, eng_character_3d.py:258/:327/:398. The NO-FALLBACKS
  precedent already exists in-repo: eng_ltx_av.py:1146 and eng_viz_mandala.py:67
  set `fallback_engine = None  # NO FALLBACKS: fail LOUD`. E1's real unit of work
  is PER-ADAPTER: set every fallback_engine to None on that precedent, in the same
  change as the render_driver rip. cheap_families.py:172/:182 floor-terminus
  comments updated too (claude CLI should-fix #7 confirmed).

DISCARDED / DOWNGRADED:
- Antigravity's tests/test_video_soak_fixture.py + scripts file links: paths real,
  no misreads to discard this round. (Rare -- panel was clean.)
- My anchor's ":2101/:2527" framing "looks like a verifier" -- codex/claude CLI
  named them precisely (run_gpu_soak/assert_soak_ok); superseded by the precise cites.
- codex cut #2 (trim Sprint E metrics) ACCEPTED as a should; antigravity's
  numpy-only feature definitions ACCEPTED (no librosa dep).
- claude CLI cut #13 (mint gate must not duplicate budget logic) ACCEPTED.
- claude CLI cut #14 (sha256 on stills) REJECTED: CanonicalAsset already carries
  sha256 for video and the ledger stamps it; keeping the field uniform is cheaper
  than a special case. Cost is negligible for <5MB PNGs.

## DELTAS TO FOLD INTO PLAN.md (the hardened r2 plan)

Sprint A (E1/E2) -- expanded, still first:
A1. Rip render_driver.py scaffolding (constants :52-:117, make_fallback_of :153,
    consumers :2101/:2527/:2589, __all__ :2654-2657) AND the duplicate copy in
    scripts/otr_video_soak.py. run_gpu_soak/assert_soak_ok become NO-TRAIL LOUD
    contracts: forced OOM => RenderError raised + asserted (not a trail match).
A2. Set fallback_engine = None on EVERY adapter that declares a target (humo x2,
    ltx_video, mesh_stage, triposr, still_parallax, character_3d x3) on the
    eng_ltx_av.py:1146 precedent. character_3d OOM = named RenderError, never
    chain-to-humo. Update eng docstrings + cheap_families.py:172/:182.
A3. E2 contract: (a) director widget default -> False + "(deprecated)" label,
    widget STAYS positional (BUG-LOCAL-097); (b) Policy.allow_auto_fallback
    default -> False; (c) runtime IGNORES True with a LOUD deprecation log (a
    stale JSON must never resurrect fallbacks); (d) re-audit widgets_values in
    workflows/otr_scifi_16gb_full.json in the SAME change; (e) update the pinned
    True in the 3 test files; run_otr_30word_smoke.py's False-patch stays (no-op).
A4. Test triage sub-task BEFORE the rip: for each consumer file listed above,
    split keep-rewrite (valid non-fallback assertions) vs delete (chain contracts).
    Add per-family LOUD-failure contract tests.
A5. Ledger: keep fallback restamp fields, stamped never (codex cut #1 accepted);
    content_oracle.check_manifest must not REQUIRE trails (verify at build).

Sprint B (S1 stills) -- pinned decisions:
B1. Adapters at nodes/_otr_image_engines/eng_cloud_image.py on the ImageEngine
    protocol (assert_usable/prepare/render_image/teardown); imported from the
    package __init__; render_image() -> invoke_partner_node() ->
    canonicalize_image() -> return str(asset.path) (dispatcher _coerce_pixels
    accepts a PNG path). Hedge resolved -- no "or verify".
B2. CAPABILITIES rows for cloud_recraft / cloud_flux_pro / cloud_nano_banana_2 /
    cloud_ideogram_v4 in registry.py in the SAME change (one-declaration
    invariant test). DECISION owed in-sprint: cloud_seedream_2 in or explicitly
    CUT from S1 (it is pinned + profiled; default = include, cheap).
B3. canonicalize_image signature mirrors video: (raw, request, session) with
    request carrying {"w","h","format":"PNG"}; sRGB; sha256 kept; named
    CloudMediaError codes; implemented IN the adapter's render_image (the image
    dispatcher has no canonicalize hook -- antigravity should-fix #2).
B4. V3/COMBO resolution layer: explicit per-adapter model value (env/config
    pinned), never a live COMBO scrape at render time; emitted kwargs INCLUDED in
    the conformance test.
B5. Conformance test parametrizes over partner_nodes.yaml ROWS (14), not adapter
    modules; every emitted kwarg must be declared in the row schema; a row
    without an adapter is flagged (xfail-listed until its sprint).
B6. Portrait-mint gate is a PRE-SELECTION GATE, not a fallback: it runs before
    the 3D flag activates and before any cloud spend; rejection = ledger stamp
    mint_3d=REJECTED:<reason> + beat stays 2D BY NON-SELECTION. Mint gate does
    NOT duplicate the budget machine. Un-mintable when 3D was explicitly
    dropdown-selected => that is a FAILURE, fail LOUD (directive).
B7. Wire dropdown rows into otr_scifi_16gb_full.json same change; validator +
    widget audit after.

Sprint C (S3 remainder): unchanged scope; ADD antigravity's stamp schema --
    ShotLock stamps provider_id, estimated_usd_cap, is_cloud=True on cloud rows.

Sprint D (cloud TTS) -- expanded:
D1. PREREQUISITE: implement canonicalize_audio (S2 stub): soundfile load,
    resample 44.1k, downmix per stereo_policy, loudness match to local reference,
    +/-250ms per-line tolerance, actual_duration_s to line metadata.
D2. ELEVENLABS_VOICE: resolve the voice_selector node class locally
    (_resolve_node_class on the pinned row) and execute in-process to produce the
    voice object; the file-centric invoke bridge only handles the TTS row output.
    Verify exact mechanics at build.
D3. Registration surface: adapter in nodes/_otr_audio_engines/ + profile rows in
    config/audio_engine_profiles.yaml + _LEGACY_FIRST_ENGINES additions
    (char_voice + announcer_voice) + JSON widget audit same change. default_roles
    = (); registers unconditionally (no requires_flag -- directive).
D4. Clarify has_audio: it is a VIDEO-side field; cloud TTS only feeds the
    per-line WAV contract; audio spine untouched (byte-identical stays green).

Sprint E (C1 audio_motion_profile) -- pinned:
E1. Schema: typed AudioMotionProfile model added to VideoRequest (_Forbid), field
    required-on-consume: engines fail LOUD if absent, never recompute.
E2. Compute ONCE in run_episode before the beat loop; stamp in
    build_request_from_shot (render_driver.py:972 area).
E3. v1 metrics TRIMMED (codex cut #2 + antigravity defs): duration, rms, peak,
    onset (transient-peak count over threshold), silence ratio -- numpy/torch
    only, NO librosa. brightness/dynamic-range/speech-vs-music DEFERRED to C2
    with the first real consumer (speech-vs-music will be role-derived, static).

Sprint F: unchanged.

Ordering unchanged: A -> B -> C -> D -> E -> F; GPU gates interleave.

## OPEN VERIFY-AT-BUILD (carried)
- content_oracle.check_manifest fallback-trail requirements (A5).
- ELEVENLABS_VOICE local-execution mechanics (D2).
- seedance_2 V3-expansion pin dynamic inputs (Sprint C).
