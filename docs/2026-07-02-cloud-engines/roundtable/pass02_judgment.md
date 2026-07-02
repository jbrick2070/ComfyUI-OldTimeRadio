# R2 judgment (Claude, sole judge)

Panel: gpt-5.5 (14 must-fix), gemini-3.1-pro (5 must-fix, full this
round), deepseek-v4-pro (3 must-fix). Anchor: claude. R2 spend: $0.2295.

GROUNDING CHECKS PERFORMED:
- default_engine_for_role(role) -> Optional[str] signature: CONFIRMED
  (registry.py:133). GPT#2/Gemini#3 right; pass01 mechanism was wrong.
- CAPABILITIES tables + cpu_ok/vram_class: CONFIRMED in audio (l.185),
  image (l.107), video registries. GPT#13/Gemini#2 accepted.
- license_audit_status tri-state precedent: CONFIRMED in
  _otr_model_catalog.py (l.94-96). GPT#12 accepted w/ exact mirror.
- SoniloTextToMusic: CONFIRMED REAL in live install dump l.1774-88.
  Gemini SHOULD-1 ("hallucination of Suno") REJECTED AS MISREAD -- the
  panel's grounding did not include the node dump; primary source wins.

ACCEPTED (major): registration UNCONDITIONAL + enforcement at profile
resolver via GATED_BY_FLAG (synthesis of GPT-R1#3 + GPT-R2#1 +
Gemini-R2#1 -- saved-workflow COMBO validation + C6 "registry IS the
menu" + no import-order dependency, all satisfied simultaneously);
CAPABILITIES rows for cloud engines; resolve_default_engine_for_role
helper (registry default untouched); CloudMediaSession replaces module
globals (GPT#5); invocation contract + partner_nodes.yaml pinned via
in-process INPUT_TYPES import not /object_info (GPT#3/#4/#11, DS#1,
anchor#1/#2); hidden-input declaration + HEADLESS INJECTION design
(GPT#6, Gemini S-3); CostQuote + estimate_cost + mid-run cost gate
(GPT#7, DS#2, Gemini#4); static budget ceiling + per-run accumulator, no
env mutation (Gemini#5 -- pass01 wording implied it); CloudAssetKey full
schema + pinned seeds + GLOBAL cache w/ copy-into-episode (GPT#8, DS#4,
Gemini S-2); CanonicalAsset signatures (GPT#9); fallback consults
enable-set + cpu_ok (GPT#10); test split mocked/live w/
OTR_RUN_CLOUD_SMOKE=1 (GPT#14, anchor S-2/3); watchdog heartbeat
mechanism concretized (DS#3, anchor#1); cache dependency DAG + re-run
CACHED acceptance (anchor#3); loudness = match existing reference
(anchor#4); ledger JSONL schema, streaming hash, rate-limit env names,
duration tolerance -> line metadata, reactivity/must_strip_audio row
metadata, voice-table schema fail-closed, required_inputs matrix tests
(GPT SHOULD 1-9); doctor CLI + dry-run manifest as S0 nice-to-haves.

CUTS ACCEPTED: Surface B implementation (GPT CUT-1, Gemini CUT-1 --
worker-thread stall + unproven; rows to Appendix B w/ own flag + smoke
artifact requirement); provider cancellation -> ORPHANED_JOB logging
(GPT CUT-2); error-taxonomy-from-object_info (GPT CUT-3); voice cloning
stays deferred (GPT CUT-4).

REJECTED: Gemini S-1 Sonilo/Suno (misread, see above). Gemini#1's FIX
as-stated (raise GATED_BY_FLAG inside audio assert_usable) -- violates
the C6 no-flag-case invariant; the accepted design achieves the same
outcome at the resolver layer instead.

CONSEQUENCE FLAGGED TO OPERATOR: with Surface B cut, voice and music
each ship 2 rows on surface A (ElevenLabs x2 tiers; Stability + Sonilo).
The third row per lane returns when Appendix B is proven.

CONVERGENCE: R2 produced a full control-plane spec (session, contract,
cache, tests). Material change -> NOT converged; proceed to R3 (wiring).
