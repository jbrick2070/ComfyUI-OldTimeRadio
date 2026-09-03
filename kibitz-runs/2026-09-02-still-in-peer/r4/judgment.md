# r4 judgment -- still-in lab peer (2026-09-02)

Roster, r4: Sonnet 5 (Cowork subagent, in-process convergence pass over the anchor sections
1-12 and the three judgments). Driver: Claude (Cowork). Full roster of the arc: Fable 5.1 cold
+ Antigravity (r1), Codex (r2), Cursor (r3), Sonnet (r4) -- one external seat per round plus
the cold read, as the operator ruled 2026-09-02.

## Verdict on the r3 plan: NOT CONVERGED -- four defects, all CONFIRMED, all fixed by the driver

1. r3 S5 (checkpoint fingerprint by `(size, mtime_ns)`, never a per-shot digest) was recorded
   in section 11 but never edited into section 10's `plate_identity` field spec, which still
   said "checkpoint digest". CONFIRMED (a builder reading 10 + 12 would implement the 2 GB
   hash r3 rejected). FIXED: section 10 now names `(name, size, mtime_ns)` via `_file_receipt`.
2. Section 7 D7 still promised replay REUSE of a present plate PNG, contradicting r2's
   re-mint-only decision carried in 10 / 12; sections 2 and 4 still listed `session_identity`
   / `model_artifacts` overrides that r2 cut. CONFIRMED. FIXED: D7 rewritten as "record;
   reuse SUPERSEDED", section 2's override list corrected with a note, section 4 headed as
   the superseded r1 draft with the r2/r3 corrections inline.
3. Section 10 said "sentence + regenerate" for the evidence manifest while section 12 (r3
   must-fix 6) says by hand, never the generator. CONFIRMED against
   `build_video_evidence_manifest.py:270, 376-387`. FIXED: section 10 now says by hand.
4. `--derive-engine` was prose with no compatible implementation path: `freeze()` is hard-wired
   to a live episode dir and names its output `<out_root>/<episode_id>`, colliding with the
   immutability guard on the source bundle. CONFIRMED (`otr_freeze_replay_bundle.py:77-101,
   121-123`). FIXED: section 12 defines `derive_engine_bundle(bundle_dir, engine_id, out_root)`
   as a bundle-to-bundle function over `load_replay_manifest` + `manifest["files"]`, never
   through `freeze()`; and, on the reviewer's minor note, `production_ledger` stamps the
   override RAW with no registry import -- the sibling validation lives in ShotLock, which
   already imports the registry.

Verified correct by the reviewer (kept as the receipt): the plate KSampler on
`Wire('base_model')` does not conflict with `_register_patched` (fires only for the LoRA / ADE
nodes) or with `_release_sampling_patchers_before_decode` (runs once, after STAGE 3b); no live
`VideoRequest(...)` validation call sits inside `build_request_from_shot`, so the only gate on
the new fields is their declaration in `schemas.py`; `roles_effective` is stamped only in the
fresh-plan branch (`otr_shot_lock.py:3089`) and the replay branch returns the frozen section
byte for byte, so section 12's seat is the only correct one; `load_replay_manifest` tolerates
the extra key; `import_replay_bundle` copies from `manifest["files"]` only.

## After the fixes

The four items are editorial and driver-applied against the same files; the reviewer re-read
sections 2, 4, 7, 10, 12 and returned CONFIRMED-FIXED on all four with the verdict
**CONVERGED**, no new must-fix. Its one cosmetic residual (section 10's older D11 paragraph
still had `import_replay_bundle` validating the sibling rule) was aligned to section 12's
split (the ledger module stamps raw; ShotLock validates) in the same edit. Section 13 of the
anchor records the convergence; the coding contract is sections 10 + 12.
