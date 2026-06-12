# pass05 (testing) judgment -- Claude, judge + panelist

## ACCEPTED (grounded)

- FILE STRUCTURE (judge merge of GPT's 6-file / Gemini's 4-file /
  DeepSeek's 1-file proposals, following repo convention = one engine
  test file + pure-unit + wiring):
  - tests/test_av_dims.py -- pure unit: next_8n1 boundaries (25->25,
    26->33, 33->33, idempotent), 1472x832+f49 passes, 1450x832 raises
    naming 1440/1472, height/frames violations with nearest-valid both
    directions, cap behavior (T=520 -> 497 + pad flag).
  - tests/test_video_ltx_av.py -- mirrors test_video_humo.py: both
    engines registered_and_dark (membership in all_engine_names, NOT
    cardinality); role fit incl. music audio_ref; required_inputs match
    family schema; assert_usable ORDER (flag-first; Sage; node gate;
    weights; dims) with request_template=None TOLERATED (Gemini MF1:
    `if request_template:` guard + a None-call test); node gate
    classifies as MISSING_MODEL naming the class (GPT MF3: the six
    -reason enum is PINNED by test_usability_reason_has_the_six_codes
    -- NO new reason); ref extraction str/dict/None; deterministic
    build; canonicalize silent bt709 + engine_id/family stamps;
    fake-AV-mp4 strip + ffprobe zero audio streams (ffmpeg skip-guard
    per existing pattern, VERIFY); pad-tail marker text; AST
    no-brief-import; cold-import; ascii/no-em-dash; 5-hop talk + 3-hop
    music chain convergence (mirrors humo chain test); SYNTH_FALLBACKS
    membership.
  - tests/test_ltx_av_driver_wiring.py -- driver deltas: DARK-LANE
    GOLDEN FIXTURES (full request dicts for ltx_video/wan_i2v/humo
    beats vs checked-in JSON under tests/fixtures/ltx_av_dark/; media
    hashes stay M4); flag-off render-time degrade completes + trail
    retains origin; force-map role guard ignores LOUD; announcer
    portrait alias (only for ltx_av_talk); synthetic-timing slice ONLY
    for ltx_av_music (and ltx_video same shot stays None); _render_one
    request_template pass-through + legacy TypeError guard;
    ENGINE_FAMILY entries; canvas tuple; prompt gate (music joins, talk
    sibling branch, no radio override on talk).
- EXISTING-TEST FALLOUT (Gemini MF2/MF3 + GPT MF4): update
  test_video_retry_taxonomy.py (real-chain sweep) for both new chains
  [name VERIFY-AT-BUILD]; b7 forbidden sweep auto-covers the new file --
  if the sweep itself is edited, the AST loop var stays `imp` (repo
  gotcha; DeepSeek's "test the loop var name" REJECTED as inversion);
  any exact-enumeration assertions found pre-code (GPT SC4 literal
  search: FAMILIES, ENGINE_FAMILY, all_engine_names, dropdown choices,
  chains) convert to MEMBERSHIP in the same commit.
- FORGOT-IT MATRIX: every touch-list edit maps to a named failing test
  (consolidated from all four reviews into the plan's testing section).
- BYTE-IDENTICAL SPLIT (GPT MF6 grounding the runtime gate + 4/4):
  CPU = structural only (dark-lane goldens; audio request hashes
  unchanged; canonicalize never emits audio). The DEDICATED forced-lane
  master-hash variant is M4 GPU (OTR_REGRESSION_RUNTIME=1 mechanics);
  prune-to-node-7 is the SOAK harness, not pytest, and proves nothing
  about a video adapter -- claim corrected.
- GPU/CPU SPLIT: pytest = no network, no CUDA, no weights, no real
  forwards (mock/inject NODE_CLASS_MAPPINGS; never import Comfy
  packages at module scope). M0/M4 operator scripts own real renders,
  NVML ceiling, wall time, eyeballs, Desktop-vs-headless reality.
- M0 SHEET: checked-in docs/2026-06-10-ltx-av-lane/M0_RESULTS.md with
  fixed `key: value` rows (node classes per build, pip-freeze sandwich,
  max_frames, NVML/wall per lane, audio formats, P1 verdicts). Parser
  test lands ONLY after M0 (GPT SC2 sequencing); from M2 a test asserts
  LTX_AV_MAX_FRAMES == the sheet's max_frames (constant-drift guard).
  Parser checks presence/parse only -- perf pass/fail stays operator
  -gated (GPT CUT4).
- BUG BIBLE: explicit re-run pins for BUG-070 (Sage order test) and
  BUG-291 (lease released on prepare/load failure, mirroring
  test_humo_prepare_releases_lease_on_load_failure); BUG-265's scope
  VERIFIED before attaching tests to that ID. NEW row at ship: "LTX
  dims silently round upstream; OTR fails loud via av_dims" under the
  Three-File Contract (YAML + README + regression test, one commit).

## REJECTED

- New EngineUsabilityReason for node availability (enum pinned at six).
- New pytest framework / GPU pytest lane / real-render pytest.
- DeepSeek's test-the-b7-loop-var-name test (the gotcha applies when
  EDITING the sweep, not as a new assertion).
- CPU forced-lane byte-identical via prune trick (mechanically
  meaningless for a video adapter; M4 owns it).

## VERIFY-AT-BUILD (carried)

- Exact names: retry-taxonomy sweep file; driver test helpers; manifest
  writer path; ffmpeg/ffprobe skip-guard pattern; #13111 node class
  names to mock; announcer portrait object id; BUG-265 scope.
