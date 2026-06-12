# pass04 (wiring) judgment -- Claude, judge + panelist

## ACCEPTED (grounded)

- MUSIC ATTACH SHRINKS TO ONE EDGE (DeepSeek SF1 + Gemini MF1 over GPT MF1
  and Claude MF1): grounding (render_driver.py:351-378) shows the master
  -slice fallback ALREADY runs for ANY engine when the shot has a ledger
  line with start_s/dur_s -- line-backed music beats get audio_ref TODAY
  (ltx_video ignores it). The only gap is SYNTHETIC line-less shots (the
  b000 opening-music beat): line timing is absent, slice skipped, the
  request carries audio_ref=None. DELTA (Gemini's fix, engine-gated):
  when line start_s/dur_s are absent AND shot.engine_id == "ltx_av_music",
  fall back to the SHOT row's synthetic start_s/dur_s for the slice.
  Dark-lane requests stay bit-identical (gate can only fire for the new
  name); GOLDEN-FIXTURE test proves existing-engine request equality.
- HASH/SEED SAFETY (GPT MF2 refining Claude MF1): render_request_hash is
  stamped at ShotLock and only READ by the driver; seeds derive from it
  via _seed_from_hash (:459) on the EPISODE path (build_request's
  shot_id-suffix trick is the synthetic/test path only -- wording fixed).
  Test: attaching audio_ref leaves request_seed identical.
- assert_usable REQUEST_TEMPLATE GAP (GPT MF3 CONFIRMED at :490):
  _render_one calls eng.assert_usable(host_caps={}, profile={}) with NO
  request -- the pass02 av_dims-before-lease contract is unbuildable
  without a driver delta. FIX: _render_one passes
  request_template=request; the VideoEngine Protocol already declares the
  kwarg (registry.py), eng_ltx_video accepts it today; a TypeError guard
  covers any legacy adapter that predates the kwarg (cheap_families --
  VERIFY-AT-BUILD signatures).
- ENGINE_FAMILY (GPT MF4 CONFIRMED :53-63): add "ltx_av_talk":
  "audio_driven_face", "ltx_av_music": "audio_conditioned_video" so
  family restamping + force-map paths never depend on import timing.
- FLAG-OFF BEHAVIOR (GPT MF6 + Gemini MF3 over Claude MF4 -- I was wrong):
  ShotLock NEVER calls assert_usable (driver docstring: render-time walk;
  "an episode NEVER aborts and a beat is NEVER dropped"). A dark-lane pick
  degrades AT RENDER via the gated EngineUnusable -> restamp -> chain.
  Plan documents render-time enforcement; registry.py's "ShotLock calls
  assert_usable" docstring line is corrected in the SAME docstring touch
  M1 already makes. Test: flag off + stamped ltx_av_talk -> humo restamp,
  episode completes.
- REGISTRATION IS UNCONDITIONAL (@register at import; flags gate
  usability, not registration -- eng_humo/eng_ltx_video grounded). Gemini
  MF2's "unregistered when dark -> floor bypass" is a MISREAD; the
  SYNTH_FALLBACKS entries land anyway as one-line belt-and-braces for the
  guarded-import packaging edge (mirrors the hunyuan3d_talk precedent).
- FORCE MAP ROLE GUARD (GPT MF5): apply_engine_override validates each
  forced (role, engine) with engine_fits_role descriptors; incompatible
  entries are IGNORED with a LOUD warning (fail-closed per entry).
  Forcing never bypasses render-time assert_usable. M4 smoke:
  OTR_ENABLE_LTX_AV=1 + OTR_FORCE_ENGINE_MAP=announcer_visual=ltx_av_talk,
  character_video=ltx_av_talk,music_visual=ltx_av_music.
- IDENTITY STAMPS (GPT MF10 + Gemini MF4): CanonicalClip HAS an engine_id
  field (schemas :228+) -- the shared core stamps engine_id (+ family)
  in canonicalize; manifest rows then prove per-clip identity; after a
  swap the shot row's final engine_id is the fallback engine and
  degradation_trail retains the ltx_av_* origin. Acceptance grep =
  _rt.format_swap_log lines (authoritative, GPT SF4) + manifest
  engine_id column. NO new FailureKind (DeepSeek MF2 REJECTED -- an
  aspect change is a property of the humo degrade, not a failure class;
  taxonomy churn refused). Pad-tail proof = the adapter's own LOUD
  canonicalize log with the fixed marker "[ltx_av] pad-tail rendered=<n>
  target=<T>" (no trail entry -- no swap occurred).
- ANNOUNCER PORTRAIT ALIAS (GPT MF11): _portrait_index keys by
  char_id/object_id; announcer lines may carry no char_id. DELTA
  (engine-gated): when engine_id == "ltx_av_talk" and char_id empty and
  role == announcer_visual, resolve the shipped non-cast announcer
  portrait from ledger["images"] (object id VERIFY-AT-BUILD per 435ba0a /
  portrait_ledger). Missing portrait -> adapter fails closed pre-render ->
  chain walks humo (also starved, degrades LOUD today :374) -> floor.
  Trail records every hop (test).
- PRUNE CLAIM REMOVED (GPT MF8 + Gemini SF1): run_episode restamps shot
  rows + appends runtime_fallback_decisions; prune_orphaned_groups is NOT
  called today. The new lane introduces no provider groups -> no topology
  change -> nothing to prune. The plan stops referencing group pruning.
- ltx_video.fallback_engine = "still_kenburns" ALREADY GROUNDED
  (eng_ltx_video.py:70) -- DeepSeek MF3's hidden dependency is closed.

## REJECTED / MISREADS

- GPT MF1's engine-gating of the EXISTING line-backed slice path
  (would change today's behavior; the path is already universal).
- Gemini MF2 dark-lane floor bypass (registration misread; entries kept
  only as belt-and-braces).
- DeepSeek MF2 / Gemini-optional ASPECT_CHANGE FailureKind +
  AspectMismatchError (non-failure; rejected).
- GPT SF1 talk-canvas conflict (no conflict: talk RENDERS landscape by
  design; pillarbox exists only on degrade).
- Claude MF4 lock-time abort (contradicted by grounding; adopted GPT MF6).

## VERIFY-AT-BUILD (carried to pass05/07 + M0/M1)

- cheap_families assert_usable signatures (TypeError guard scope).
- _rt.restamp_shot_row / format_swap_log exact formats (freeze greps).
- Announcer portrait image object id.
- Whether existing engines ignore audio_ref everywhere (dark-lane golden
  fixtures make this a test, not an assumption).
- CanonicalClip field list (forbid-extras) for any pad-tail note field.
