# pass01 (architecture) judgment -- Claude, judge + panelist

Panel: openai/gpt-5.5-20260423, google/gemini-3.1-pro-preview-20260219,
deepseek/deepseek-v4-pro-20260423, + Claude panelist review (written before
reading the panel). Cost: see manifest.json. Every accepted claim grounded
against HEAD 56caa5b.

## ACCEPTED (grounded)

- SPLIT INTO TWO ADAPTERS (GPT MF1, Gemini MF1; 4/4 directionally).
  GROUNDED: eng_humo.py:99 `fallback_engine = "humo_1.7B"` and :320-323
  hard-fail without audio+init_image; role_compat.py MUSIC_VISUAL supplies no
  audio_ref; resolver.py is execution-group orphan pruning, NOT role-aware
  fallback pruning -- so a single adapter's one static chain cannot serve
  both talking and music roles safely. Design: ONE new file
  `eng_ltx_av.py` containing a private shared core + two thin registered
  classes (fewer files, operator preference):
  - `ltx_av_talk`: roles (announcer_visual, character_video), family
    `audio_driven_face` (REUSED -- schemas already maps it to
    (audio_ref, init_image), which is exactly right), required_inputs
    (text_prompt, audio_ref, init_image), fallback_engine "humo" -> real
    chain humo -> humo_1.7B -> latentsync -> still_kenburns. Aspect change
    on degrade (landscape -> HuMo pillarbox) = ACCEPTED as a LOUD restamped
    policy, reason string names the aspect change.
  - `ltx_av_music`: roles (music_visual,), family `audio_conditioned_video`
    (NEW), required_inputs (text_prompt, audio_ref), fallback_engine
    "ltx_video" -> still_kenburns (Gemini's chain CHOSEN over GPT's direct
    still_kenburns: ltx_video carries the music_visual role natively
    [eng_ltx_video.py roles], same landscape canvas, GPU-proven, zero edits
    to eng_ltx_video.py -- the fallback_engine attr lives on the NEW
    adapter).
- SCHEMAS family registration (GPT MF2, Gemini MF3, DeepSeek MF1).
  GROUNDED: schemas.py:30-68 FAMILIES (8 entries incl. character_3d) +
  FAMILY_REQUIRED_INPUTS + sync assert; family_hint validated :162-176.
  Delta: FAMILIES += "audio_conditioned_video";
  FAMILY_REQUIRED_INPUTS["audio_conditioned_video"] =
  ("text_prompt", "audio_ref"); registry.py docstring line updated.
- role_compat MUSIC_VISUAL supply-set += "audio_ref" (Gemini MF2, DeepSeek
  MF2a, Claude MF1) -- additive one-liner, CONDITIONAL on pass04 verifying
  render_driver.py actually attaches the music beat's slice (all four
  reviews flag the same verify-at-build).
- ISOLATION STOP RULE formalized (GPT MF3, Gemini MF4, Claude MF4).
  GROUNDED: ISOLATION_SIDECAR_REQUIRED token exists (motion_common.py:46-48).
  Rule: the lane is in-process IFF (a) zero NEW pip installs / version
  changes in the cu130 venv (pip freeze before/after M0 == identical), and
  (b) all graph node classes resolve in the installed build. Violating
  either -> the adapter declares ISOLATION_SIDECAR_REQUIRED and the sprint
  STOPS with a written finding (latentsync precedent). Refinement of
  Gemini's wording: torchaudio is ALREADY resident in the ComfyUI venv --
  the rule is "no NEW/changed packages", not a named-package ban.
- NODE-AVAILABILITY PRE-FLIGHT in assert_usable (DeepSeek MF3): adapter
  verifies its required ComfyUI node classes exist (NODE_CLASS_MAPPINGS) and
  fails closed naming the missing classes; mirrors ltx_video's _installed()
  + BUG-070 Sage gate (both adapters inherit assert_sage_not_patched).
- __init__.py guarded import line for eng_ltx_av (GPT SC3). GROUNDED:
  nodes/_otr_video_engines/__init__.py imports each adapter in a guarded
  try (lines 22-65 pattern).
- HuMo chain corrected in the plan (GPT MF5): humo -> humo_1.7B ->
  latentsync -> still_kenburns; tests/greps must expect 5 engines from
  ltx_av_talk.
- YVANN-NODES LANE CUT (4/4). Appendix note only; revisit only if M0
  verdict is INERT for music conditioning.

## REJECTED / MODIFIED

- Claude panelist's single-adapter + role-aware fallback pruning: REJECTED
  by grounding -- engine_fits_role callers are the two Directors only;
  resolver.py prunes orphaned provider groups, not role-incompatible chain
  hops. The split design removes the need entirely.
- GPT's ltx_av_music -> still_kenburns direct: REJECTED in favor of
  ltx_video hop (role-valid, aspect-stable, motion-preserving, zero
  mutation).
- GPT's two separate files (eng_ltx_av_talk.py / eng_ltx_av_music.py):
  MODIFIED to one file, two classes -- keeps the touch list small; the
  registry cares about classes, not files.
- DeepSeek SC1 (drop music_visual to dodge the role_compat edit): NOT
  taken -- music_visual is an operator-named goal; the edit is additive and
  test-gated. Falls back to drop-from-v1 ONLY if pass04 grounding shows the
  driver cannot supply the slice cheaply.

## CARRIED OPEN (assigned to later passes)

- pass02: VideoRequest/asset_refs shapes for audio_ref+init_image (humo
  _ref_path pattern); canonicalize video-only contract; frames-from-beat
  -duration derivation vs 8n+1; clamp policy Q9; canvas/fps plumbing.
- pass04: render_driver music-beat audio_ref attachment; ledger
  engine-identity stamps; OTR_FORCE_ENGINE_MAP for both new names;
  execution-group/provider interactions on degrade (resolver.py).
- pass05: test list incl. cold-import, registry count, chain termination,
  schema family, Desktop-vs-headless node skew gate (Q10).
- pass06: gemma encoder file/size; per-clip wall time; lane L1/L2/L3
  measurements; weight streaming.
- ONE FLAG decision (judge): a single OTR_ENABLE_LTX_AV gates BOTH adapters
  (the lane is one opt-in); per-adapter flags rejected as knob sprawl.
