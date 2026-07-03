# r4 JUDGMENT -- remaining-sprints plan (convergence round) -- FINAL

Judge: Claude (Cowork). Panel this round: codex + antigravity (+ my anchor).
Operator directive mid-round: the claude CLI panelist is DROPPED from the kibitz
panel going forward -- Cowork Claude is already the Claude in the loop (anchor +
judge); two claudes is redundant. Panel = codex + antigravity from now on.

VERDICT: CONVERGED. r4 surfaced 4 genuinely new build-blockers (both agents
found the same four independently; all grounded CONFIRMED against the code).
They are folded into PLAN.md; no arc change; the plan is BUILD-READY.

## r4 survivors (all CONFIRMED against the code, folded into PLAN.md)

1. **A1+A2 must be ONE ATOMIC CHUNK.** make_fallback_of (:153-:170) maps a
   registered non-floor engine with fallback_engine=None to UNIVERSAL_FLOOR --
   so A2-before-A1 does NOT stop the floor swap; the r3 ordering created the
   exact mixed window it meant to prevent. Order becomes:
   A4 -> A3b -> A1+A2 (atomic) -> A3a/c/d/e.
2. **There is NO character_image_model slot.** Image policy has exactly three
   slots; character stills route through other_beats_image_model
   (otr_image_director.py:58-64, dispatcher :162). B7 must NOT add a fourth
   slot -- schema churn + positional-widget risk for nothing.
3. **D3 audio registration surface was incomplete:** the adapter import in
   nodes/_otr_audio_engines/__init__.py + CAPABILITIES rows in
   nodes/_otr_audio_engines/registry.py:185 (test_capability_profiles asserts
   CAPABILITIES == _REGISTRY on the audio side too).
4. **B5's 14-row parametrization must distinguish billed adapter rows from AUX
   helper rows** (cloud_elevenlabs_voice_selector is api_node:false, hidden:{}
   -- it will never have an adapter or pass the bridge). Filter on api_node;
   D's xfail-removal duty applies to tts+flash only.
5. **D1 loudness reference is literally UNRESOLVED in code**
   (cloud_media_canonical.py:40). AG proposes the local lane's
   scene_sequencer loudness handling (-16 dBFS active RMS,
   _loudness_normalize_clip) -- recorded as the CANDIDATE, resolved + tested
   at D1 build (verify-at-build, do not hard-code the number in the plan).

Should-fixes accepted: retry_taxonomy's fallback-action API
(escalate_to_fallback / build_fallback_decision / restamp_shot_row /
append_runtime_fallback_decision / format_swap_log) is named A1 scope
(delete or deprecate; ledger schema intact per A5); partner_nodes.yaml gets its
full path in the plan; otr_scifi_16gb_full_api.json is an UNTRACKED probe
artifact -- do not hand-edit it, REGENERATE it after A3 (noted in A3).

Cut accepted (both agents): B5's optional inverse check -- redundant with the
CAPABILITIES invariant.

Discarded: nothing this round (clean panel).

## Arc summary (12 agent calls planned, ~11 spent; r2 pre-run + r3 full + r4 codex/AG)
- r2 (coding): scope expansion (soak duplicate, per-adapter fallback_engine
  fleet, two live allow_auto_fallback defaults, image registry reality,
  _LEGACY_FIRST_ENGINES surface, VideoRequest _Forbid).
- r3 (wiring): seams pinned (AUDIO-dict voice return, in-process voice
  selector, estimated_usd nonzero, profile carrier via ledger, PNG-on-disk,
  prepare(None,None,None), V3 single source of truth, Sprint A internal order).
- r4 (convergence): 4 last blockers above; both agents independently converged
  on the same set -- stop condition met.

Final hardened plan: docs/2026-07-02-remaining-sprints/PLAN.md (this round's
edits included). Sprint A is the next code window.
