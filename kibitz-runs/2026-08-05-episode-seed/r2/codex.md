VERDICT: yes-with-fixes — the root fix is buildable, but the document leaves ownership unresolved, misstates the credits blast radius, and lacks executable regression and re-baseline gates.

MUST-FIX BEFORE BUILD:
1. [Q1] Choose option (a): immediately after `meta["cast_contract"]` is created, assign `meta["episode_seed"] = int(cast_seed)` in `nodes/OTR_LedgerScriptWriter.py:4087-4099`. Legacy lanes execute this picker; content-owned lanes bypass it at `nodes/OTR_LedgerScriptWriter.py:3663-3717` and retain their tail-owned stamp at `:6041-6062`. Lifting the gate would call `_resolve_cast_rng_seed()` a second time and give legacy casts and episode consumers unrelated OS-entropy values.
2. [Q2] Preserve both keys and define their invariant: for legacy lanes, `meta.episode_seed == meta.cast_contract.cast_seed`; `cast_contract.cast_seed` remains the replay claim. Copying the number is safe because replay dispatch reads only the nested key at `nodes/cast_lock.py:340-353`, while voice/music consumers read `meta.episode_seed` at `nodes/cast_lock.py:498-530`, `nodes/_otr_voice_node_common.py:426-428`, and `nodes/stable_audio_theme.py:260-277`. Add tests for equal-valued keys on both legacy replay and content-owned preserve paths.
3. [Every consumer of the frozen value / Q3] Correct the credits claim before using it to scope work. `nodes/otr_credits_roll.py:313-318` prefers `meta.cast_contract.cast_seed`; the published legacy ledger described in the document therefore already supplies its unique cast seed. `tests/test_credits_roll_spec.py:200-223` explicitly pins that preference. The legacy credits number and diagnostic are not frozen by missing `episode_seed`, and should not be re-baselined for this patch.
4. [Q3 / Q4] Add an explicit regression matrix:
   - Legacy writer: with `OTR_CAST_SEED=12345`, assert both seed keys equal `12345`.
   - Production entropy: two mocked `_resolve_cast_rng_seed` values propagate unchanged, not through `coerce_int_seed(None)`.
   - Content-owned tail: retain `tests/test_fable2_tail_context.py:258-272`.
   - Replay semantics: retain `tests/test_cast_voice_replay_parity.py:61-124` and `tests/golden/cast_pool_baseline.json` unchanged.
   - Announcer variety: retain `tests/test_cast_lock.py:263-280`.
   - C7: run the runtime gate in `tests/test_audio_byte_identical.py:171-208` with explicit `OTR_CAST_SEED`; [ASSUMPTION] a baseline captured before propagation will need an intentional re-baseline because downstream voice/music seeds change from the missing-value hash to the pinned cast seed.
5. [Q5] Specify the actual music change. Missing `episode_seed` freezes the generated engine seed per `cue_id`, via `nodes/stable_audio_theme.py:264-287`; it does not prove byte-identical beds because prompts can vary through authored `music[]` rows or story-brief composition at `:298-378` and `nodes/_otr_music_prompt.py:76-144`. Extend `tests/test_stable_audio_theme.py` to assert same episode seed gives identical cue seeds and different episode seeds give different cue seeds while prompts remain unchanged.
6. [The defect, measured on published episodes] Include the required shipping gates: record the published failure in `docs/PROD_BUG_LOG.md`, extend the portable receipt-ownership coverage in `BUG_BIBLE.yaml` BUG-12.51, run the focused/full Windows suites plus Bug Bible, then run two consecutive canonical `workflows/otr_canonical.json` episodes and verify distinct seed receipts, successful publication, and canonical assets.

SHOULD-FIX:
1. [Every consumer of the frozen value] Document the separate `OTR_LedgerFreezeCascade` output named `episode_seed`. It hashes the complete frozen JSON at `nodes/OTR_LedgerFreezeCascade.py:62-70,422-429` and is deliberately unwired in the canonical workflow, as pinned by `tests/test_freeze_cascade_v2_ports.py:76-88`. Do not conflate it with `meta.episode_seed`.
2. [Q3] Update the stale ownership comment in `tests/test_ledger_cleanup_pass.py:287-309`, which currently says the legacy picker owns only `cast_contract.cast_seed`.
3. [Q3] Fix or explicitly defer seed-source labeling. Credits annotate the cast seed with `gen_params_initial.seed_source` at `nodes/otr_credits_roll.py:291,325-326`, but that field is story-source provenance (`original_llm`, `custom_premise`, RSS) from `nodes/OTR_LedgerScriptWriter.py:5755-5771`, not `cast_seed_source` from `:4097-4098`.

OPTIONAL / NICE-TO-HAVE:
Run a deterministic 30-seed probe after the unit gates and retain the observed announcer, character-voice, and music-seed diversity as a diagnostic receipt; do not make statistical distribution a runtime gate.

CUT THESE (over-engineering):
1. [Q1] Cut option (b); it creates a second entropy draw for legacy lanes and contradicts the existing ownership comment at `nodes/OTR_LedgerScriptWriter.py:6025-6030`.
2. [Q2] Cut any new seed generator, compatibility shim, or migration layer. The required value already exists as `cast_seed`.
3. [Q3] Cut changes to node mappings, ports, widgets, or workflow links. This is an internal metadata stamp with no ComfyUI interface change; validate the canonical workflow but do not manufacture JSON churn.
4. [Q3] Cut modifications to `tests/golden/cast_pool_baseline.json`; it pins replay from an explicit cast seed and is unaffected.
5. [Q5] Cut a music-engine or prompt rewrite. The defect is seed propagation, not prompt composition.
