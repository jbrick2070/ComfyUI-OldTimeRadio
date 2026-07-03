# OTR Remaining Sprints -- build plan under the NO-FALLBACKS / dropdown-only-defaults directive

Date: 2026-07-02 late evening. Branch v2.0-alpha @ d0463b8c. Suite 6075/0.
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

- E1: remove `make_fallback_of(` (render_driver.py ~:1800/:2288), `FLOOR_NAMES`,
  `UNIVERSAL_FLOOR`, `SYNTH_FALLBACKS`, `EXPECTED_OOM_TRAIL`; `eng_character_3d.py` still
  references the chain. Replace every consumer with LOUD RenderError (named engine + reason).
  `still_motion` remains a REGISTERED SELECTABLE engine (do not unregister) but loses its
  "universal floor" role.
- E2: `otr_video_director.py:228` `allow_auto_fallback` boolean passed through (:354) --
  force false + relabel/deprecate in place (widget surface: the JSON widget stays positional;
  no mid-list removal, BUG-LOCAL-097).
- Ledger/manifest: fallback restamp fields become dead -- decide: keep fields (stamped never)
  or remove writers; content_oracle.check_manifest must not require fallback trails.
- Tests: excise fallback-chain contract tests; add LOUD-failure contracts per engine family.

## Sprint B -- S1 stills lane (cloud)

- `canonicalize_image` in `nodes/_otr_shared/cloud_media_canonical.py` (contract already
  stubbed there per S0 c2): role canvas fit/pad, PNG, sha256, named-error validators.
- Adapters `nodes/_otr_image_engines/eng_cloud_image.py` (or the image-engine registry's
  actual layout -- verify): `cloud_recraft` (CHEAP), `cloud_flux_pro` (BEST),
  `cloud_nano_banana_2` (BEST), + `cloud_ideogram_v4` (pinned row, text-render strength).
  All via `invoke_partner_node`; per-row pinned kwargs from partner_nodes.yaml +
  PROMPT_PROFILES.md; NO default_roles (dropdown-selectable only, per directive).
- The generic profile->schema CONFORMANCE TEST (owed since the video-team warning): every
  kwarg an adapter emits must be declared in its pinned yaml row schema -- one parametrized
  test over ALL cloud adapters (video + image + future TTS).
- Portrait-mint gates: `portrait_mint_3d` prompt profile in the character_description ->
  finish_visual_prompt chain (subject fully in frame, front/3-4 neutral pose, clean backdrop,
  even light); mint acceptance check runs BEFORE any downstream credits; un-mintable still =
  LOUD ledger note + the beat stays 2D (this is a GATE, not a fallback: nothing is swapped,
  the 3D flag simply does not activate).
- Wire into the stills dropdowns (announcer/music/other/character image models) IN
  `workflows/otr_scifi_16gb_full.json` in the SAME change; validator + widget audit after.

## Sprint C -- S3 remainder (rescoped by the directive)

- CUT: reactive auto-defaults; fallback chains. (Directive.)
- KEEP: ShotLock audit stamps for cloud rows; seedance_2 + wan V3-expansion pins (un-dark
  seedance; wan gains its pinned prompt path if the V3 pin exposes one); live provider proof
  rides the operator smokes.

## Sprint D -- cloud TTS lane (ElevenLabs)

- Rows pinned (elevenlabs_tts + flash + voice_selector AUX). Build the audio-side adapter
  against the FROZEN AudioEngine protocol (per-line clips; has_audio semantics unchanged;
  the master mix stays byte-identical -- cloud TTS feeds the SAME per-line WAV contract as
  bark/kokoro/indextts2). Dropdown row(s) in the char_voice/announcer voice menus, JSON same
  change. Voice-preset table from the pinned voice_selector row. NO default promotion.

## Sprint E -- S-C C1: shared audio_motion_profile

- Per-beat rms/peak/onset/silence/brightness/dynamic-range/speech-vs-music/duration computed
  ONCE (driver-side) and handed to every engine via the request; engines CONSUME it, never
  recompute. C2 (per-engine consumers + HuMo phrase-chunking) stays deferred.

## Sprint F -- creative formats F1/F2 (queued last)

- Per `docs/2026-07-02-creative-formats/CREATIVE_FORMATS_PLAN.md` (kibitz-hardened);
  V1 probes first; codex-only kibitz per shipped format diff.

## Ordering

A (E1/E2) -> B (S1) -> C (S3 remainder) -> D (TTS) -> E (C1) -> F. GPU gates (soak2 QA,
proof9d, S5 A/B, live smokes) interleave on render windows and do not block CPU coding.
