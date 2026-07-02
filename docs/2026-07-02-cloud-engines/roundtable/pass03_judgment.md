# R3 judgment (Claude, sole judge)

Panel: gpt-5.5 (10 must-fix), gemini-3.1-pro (3 must-fix + 3 should),
deepseek-v4-pro (4 must-fix, verdict yes-with-fixes). Anchor: claude.
R3 spend: $0.2512 (one hung run killed at 9min, zero reviews returned,
rerun clean). Campaign so far: ~$0.66.

GROUNDING CHECKS THIS ROUND (mine, pre-panel): ImageGenDispatcher
imports the image registry (line 38) -- S1 assumption CONFIRMED; writer
node declares hidden auth constants verbatim (OTR_LedgerScriptWriter.py
2219-2220) -- constant names CONFIRMED in-repo; save node shows hidden
dicts already exist on some nodes (MERGE not replace).

ACCEPTED (major, deduped across panel + anchor):
- Ship-from-checked-in-pin; runtime never drops rows; missing class =
  registered-but-fail-closed named error (GPT#1 -- resolved pass02's
  internal contradiction with COMBO validity + CAPABILITIES invariant).
- Reactivity CANNOT ride capability fit (roles supply all tokens incl
  audio_ref -- GPT#2's role_compat reading CONFIRMED); descriptor fields
  + mandatory ShotLock policy gate (GPT#2 + anchor#3).
- Session = lock-guarded backend table keyed by prompt_id via additive
  hidden inputs; adapters FETCH (audio protocol frozen -- Gemini#3
  CONFIRMED: generate_voice/generate_clip take no session); teardown on
  assembler done (GPT#3 + Gemini#3 + anchor#1).
- Incremental per-dispatch reserve replaces the impossible mid-run
  episode gate (Gemini#1 "time travel" CONFIRMED: video keys/cost need
  generated audio); episode estimate demoted to report (GPT#4 merged).
- Ordering rides the EXISTING gate/done chain + slice manifest as input
  (GPT#4; matches shipped audio_done machinery).
- Auth broker resolves env>config>token into the session; chat-lane
  set_auth globals not reused (GPT#5; _bearer reads no env).
- Budget state machine (GPT#6) + accumulator lock (DS#1) + per-run
  dynamic ceiling read via _int_env pattern (Gemini S-2, reconciled
  with the no-mid-run-mutation rule).
- Semaphore submit->terminal (GPT#7). Checked-in yaml + no live imports
  for dropdowns (GPT#8). CAPABILITIES per-modality ownership + cross-
  registry test (GPT#9). Cache strip-proof manifest + re-canonicalize
  stale entries (GPT#10) + sha256 AFTER canonicalization (Gemini S-3).
- (import_path, class_name) node key (GPT S-5); seed_supported (GPT
  S-4); hidden names in yaml + drift CI (GPT S-2); per-key + ledger
  locks (GPT S-3); license gate every fallback hop (GPT S-6); provider
  ID normalization (GPT S-7); comfy_api_base verify item (GPT S-8);
  actual_duration_s named + validators updated (GPT S-9); checked-in
  WAV fixture for smoke #2 (GPT S-10); heartbeat = S0 promotion
  requirement (GPT S-1 + DS#2); ffmpeg gated in assert_usable before
  billing (Gemini S-1); loudness single-source constant (DS S-1);
  integer version scheme (DS S-2); ToS audit early-S0 (DS S-3);
  fallback reuses shipped _otr_shared/fallback.py chain machinery, one
  system (anchor#2); node surgery scoped per lane sprint, S0 smokes
  need none (anchor#4); cache path discipline + obs_publish exclusion
  (anchor#5); import-line liveness checklist + registration test
  (anchor#6); billing-vs-production ledger linkage by request_id
  (anchor S-1).
- DS#3 partner-node isolation REFRAMED: we always run inside full
  ComfyUI; S0 verifies FUNCTION execution from another node's execute
  context, not a bare-script environment. DS#4 Sonilo: S0 pin
  re-verifies (primary-source dump stands; belt and suspenders).

CUTS ACCEPTED: cancel_token out of the invoke signature (GPT CUT-1);
copy not hardlink (GPT CUT-2); dry-run manifest demoted to diagnostic
(GPT CUT-3); cancellation stays cut (Gemini CUT concurs).

REJECTED:
- Gemini R3 #2 premise ("audio dispatcher missing from the hidden-input
  list") -- MISREAD: pass02 sec 3 lists 3a/3b/3c explicitly. Its
  underlying concern (frozen protocol) was real and is accepted via #3.
- Gemini OPTIONAL soft schema-drift (warn + best-effort remap) --
  REJECTED on invariant: fail-closed, no silent substitution. A drifted
  schema best-effort-mapped is the silent-swap bug class.

CONVERGENCE: R3 changes are significant but strictly tightening (no
architectural reversals). Proceed to R4 for residual-defect sweep.
