# R4 judgment (Claude, sole judge) -- CONVERGED

Panel: gpt-5.5 (12 must-fix, verdict no), gemini-3.1-pro (3 must-fix,
yes-with-fixes), deepseek-v4-pro (3 must-fix, yes-with-fixes). Anchor:
claude (3 must-fix). R4 spend: $0.1864.

CAMPAIGN SPEND: R1 $0.1818 (incl. the 2-model retry) + R2 $0.2295 +
R3 $0.2512 + R4 $0.1864 = ~$0.85 total. One hung R3 run killed at 9min
returned zero reviews (unbilled client-side; provider-side unknown).

ACCEPTED (all folded into pass04_plan.md):
- Goal amended: 1 CHEAP + 1-2 BEST per modality; voice/music third rows
  return via Appendix B (GPT#1 -- brief-vs-rows contradiction was real;
  operator had been notified in R2 judgment, now formalized).
- RequestCacheKey vs content_sha256 split (GPT#2 + DS S-5; DS's
  raw-bytes-in-key variant rejected -- a pre-submit key cannot contain
  any output hash; GPT's split is the correct shape).
- Auth precedence concretized env > hidden api_key > hidden auth_token;
  vague "server config" dropped; media nodes declare BOTH auth hidden
  inputs explicitly (GPT#3 + Gemini#3).
- Pinned OUTPUT selection per row + execution_mode/job fields replace
  "sync-vs-job" jargon (GPT#4 + GPT CUT-2).
- One canonical error-code set incl. malformed_config, interrupted,
  orphaned_job (GPT#5 + GPT#8 + Gemini S-2).
- Per-row VIDEO DESCRIPTOR TABLE incl. must_strip_audio=True everywhere
  + fallback chains (GPT#6); reactivity gate location + exception
  concretized in OTR_ShotLock.validate (DS#3).
- Reactivity goal NARROWED explicitly (talking enforced; music/b-roll
  mute-allowed, operator-flippable via profile) (GPT#7; judge call
  recorded -- operator may upgrade music_visual later at zero code
  cost). optional_audio_ref CUT (GPT CUT-1).
- Both-flag-state tests w/ process isolation (GPT#9 + anchor#2).
- CAPABILITIES concrete values + vram_class label deferred to consumer
  check (GPT#10; verify #13 -- neither "cloud" nor "cpu" is a verified
  existing class label).
- Credit-pool + api-base verify items got CONCRETE steps (GPT#11/#12).
- Session table = allowed singleton wording; key = prompt_id alone,
  episode_id metadata; leak sweep for aborted runs (GPT S-1/S-2 +
  DS OPT-2 + anchor#1).
- provider_id pinned in yaml + collision check (GPT S-3);
  actual_duration_s migration behavior (GPT S-4); S0 acceptance labels
  (GPT S-5); smoke #2 names row id + conditioning proof (DS OPT-1 +
  GPT verify-2); example yaml row + ledger examples as S0 doc
  deliverables (GPT OPT).
- invoke_partner_node resolves session INTERNALLY -- session param
  removed (DS#1: contradiction with adapters-fetch was real);
  PartnerResult TypedDict defined (DS#2).
- Separate USD accumulator vs chat token ceiling made explicit; no
  currency unification (Gemini#1 -- verify #7 wording had invited the
  misread; concrete check per GPT#11 retained as informational).
- ffmpeg check moved OFF the registry assert_usable (does no IO --
  invariant) to the adapter render-lifecycle assert_usable/prepare
  (Gemini#2, CONFIRMED against registry docstring).
- VideoEngine Protocol + EngineDescriptor TypedDict explicitly updated
  for the two new fields (Gemini S-1); yaml records ComfyUI
  version/commit of the pin (Gemini OPT).
- SA3-vs-Stability listening test -> optional QA (GPT CUT-3).
- obs_publish exclusion mechanism named: path allowlist (anchor#3).

REJECTED (with reason):
- DS S-6 cut nano_banana_2: it appears in the live template catalog
  (Nano Banana 2, e.g. templates_mjm_image_to_3d models list); S0 pin
  gates it like every row. Kept as BEST-2 stills candidate.
- DS CUT-2 remove commit-AND-push from sprints: operator directive
  (repo CLAUDE.md sec 7) -- process stays in THIS repo's build docs.
- Gemini R3-carryover soft-schema-drift: still rejected (fail-closed).

CONVERGENCE: DECLARED. R4 produced zero architecture changes -- only
contract tightening (types, tables, test states, wording). Panel
verdicts: 2x yes-with-fixes, 1x no whose items were all resolvable
without structural change. The 4-round arc is complete; pass04_plan.md
is the build document.
