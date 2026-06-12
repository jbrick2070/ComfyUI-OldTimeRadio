# Pass-01 judgment — output-tree consolidation (Claude judge; 3 panel + Claude panelist)

Panel: gpt-5.5, gemini-3.1-pro, grok-4.3 (~$0.10) + my independent review. CONVERGED in
one pass: unanimous placement pick, concrete defect list, no unresolved dispute.

## Decision: Option B — everything under episodes/, system tiers in episodes/_shared/

`otr/` top level = `episodes/` + `obs/` ONLY (the operator's literal directive + the
2026-06-11 clarification: every episode asset of record in `episodes/<id>/<subfolder>/`,
obs = final videos only). System tiers move to `episodes/_shared/{cache,tmp,state}` —
reserved, never an episode. **My Option A (.system + attrib +h) is REJECTED** (panel
unanimous): a third top-level entry violates the directive's letter, the hygiene gate
would carry the exception forever, and Windows hidden-attrib is a weak guarantee.
Cost accepted: every episodes/ walker must skip `_`-prefixed entries (named risk:
ledger auto-pick walkers) — small, testable.

## Grounded claim log
- CONFIRMED (Gemini M1, GPT M3): `_otr_paths.py` empty-id fallbacks return
  `_legacy_audio/_legacy_stills/_legacy_portraits`, and `otr_state_dir()` = `otr/state`
  — these helpers MUST change in the SAME chunk as the guard or the build self-trips.
  Production write helpers: raise LOUD on empty episode_id; shared tiers get new
  `_shared` helpers.
- CONFIRMED (GPT M2, Grok SF1): **my pass00 attribution of top-level `stills/` was
  WRONG** (the empty-id fallback goes to `_legacy_stills`, not `stills/`). The actual
  writer of live `otr/stills/` is UNIDENTIFIED — OH-1 starts with that hunt (suspects:
  the ST-3 cache materialization path or a direct join bypassing _otr_paths).
- CONFIRMED (GPT M4): scope the contract assert to OUTPUT helpers only — _otr_paths
  also exports input/models/log/HF resolvers that legitimately live outside otr/.
- CONFIRMED (GPT M5): launcher env (TEMP/TMP/OTR_GPU_LEASE_DIR -> _shared/tmp) repoints
  BEFORE the guard lands; ffmpeg children + atomic publish depend on it; same-volume
  os.replace preserved (still inside episodes/).
- CONFIRMED (GPT M6): hygiene gate updates in the same chunk: fail any top-level
  otr/* except episodes|obs; never treat `_shared` as an episode; obs stays flat.
- CONFIRMED (GPT M7, Gemini M3, mine): QUIESCENCE — land only after the running queue
  (7-leg sweep -> wan batch -> 0-E Phase B) drains; preflight fails if renders/ffmpeg
  active or tmp recently written.
- CONFIRMED (GPT M8): archive attic lives OUTSIDE otr/ (`output/otr_attic_<ts>/`);
  operator approves deletions; nothing auto-deletes EXCEPT the janitor's stale-tmp
  sweep (Gemini M2's explicit exemption, threshold-gated, logged).
- SPLIT RESOLVED (Gemini cut vs GPT SF1): static audit kept NARROW — one CI test
  banning `comfy_output_dir()/"otr"` composition outside `_otr_paths.py` (catches the
  bypass class at source); the broad string-grep dropped as brittle; the hygiene gate
  stays the ground-truth net.
- Grok M2/M3 (name exact files): folded into the tickets (gate =
  `scripts/_otr_soak_capstone.py` hygiene checks; guard test =
  `tests/test_output_tree_contract.py` new).
