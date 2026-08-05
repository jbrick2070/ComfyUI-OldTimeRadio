# r3 JUDGMENT -- wiring / integration / sequencing

**Judge:** Claude (Opus 5), driver, sole judge. **Date:** 2026-08-05.
**Panel:** Codex `gpt-5.6-sol` (reasoning high) + Antigravity `Gemini 3.6 Flash (High)`.
**Profiles:** shipped `comfyui` + repo-local `.kibitz/comfyui.local.md`.

Every claim below was re-checked against the real Windows files before being
accepted. Panel claims I could not reproduce are listed under DISCARDED.

## ACCEPTED MUST-FIXES (fold into the build)

**M1 -- `floor` is out of scope at the seed call. BUILD-BREAKER.** (Codex only.)
Track 3 Step 4 says to hoist `floor = _min_tier_pool()` "above the loop" inside
`_ladder_pick`, and separately to pass `casting_policy_version=f"{v}+mtp{floor}"`
at the `stable_cast_seed(...)` call. **Verified:** the `stable_cast_seed` call is
`nodes/_otr_voice_bank.py:394-402`; `_ladder_pick` is defined at `:424`. A `floor`
bound inside `_ladder_pick` does not exist at `:396`. Literal implementation
raises `NameError` on every cast. **Fix:** compute `floor` ONCE before `:394` and
let `_ladder_pick` close over it. Antigravity missed this entirely.

**M2 -- `cast_source_contract` never reaches the durable ledger.** (Codex sharp
form; Antigravity found the weaker invention-lane version.) **Verified:**
`lock_cast` returns `(cast, meta)` and `meta.update({...})` at
`nodes/_otr_casting.py:1770-1782`. The writer does NOT copy meta wholesale -- it
copies **selected keys by name**: `meta["cast_voice_slots"] = cast_meta.get(...)`
at `nodes/OTR_LedgerScriptWriter.py:4073` and `meta["voice_cast_decision"] = ...`
at `:4078`. A new `cast_source_contract` key stamped in `lock_cast` would
therefore be silently dropped. **Fix:** add the explicit copy line beside `:4078`,
AND stamp a normalized empty contract on the invention lanes so the field is
never absent. This is the standing ledger-completeness rule, and the plan as
written violates it.

**M3 -- the anyref selector has no shared owner; the merged chunk would make the
ledger lie.** (Both panels; this answers my Q8.) **Verified:**
`_resolve_clone_ref_path` looks up `vrid = cast.get("voice_ref_id")` at
`nodes/_otr_voice_node_common.py:91-95` and returns that entry FIRST. Its
gender-agnostic fallback is an **inline** selector at `:115-126` --
`filter_by_quality_tier(bank)` at `:87`, role-matching candidates preferred at
`:119`, sorted by `voice_ref_id` at `:120-123`, drawn with
`Random(f"{episode_seed}_{char_id}_anyref")`. If CastLock stamps a ref chosen by
any *other* logic, the ledger names a voice that is not the one that renders --
which is the exact defect chunk 3 exists to close. **Fix:** extract that selector
into one shared callable and call it from BOTH CastLock and
`_resolve_clone_ref_path`. Keep `:109-127` reachable for un-stamped legacy /
`preserve_ledger` rows, and rewrite Track 3 Step 7's fallback test to construct a
row with `voice_ref_id=None` so it is not exercising dead code.

**M4 -- every new ledger field needs a defined value on EVERY path.**
(Antigravity's path enumeration is the better one.) Initialize, do not
conditionally stamp:
- `voice_cast_fallback = ""` on every non-announcer cast row before the policy
  branch, `"gender_unservable"` only on the fallback.
- `derived_from_portrait_hash` and `portrait_anchor_mode` = `""` on ALL image
  rows including `portrait`, `scene_open`, `radio_host_portrait`.
- `cast_source_contract` normalized-empty on invention lanes.

**M5 -- `portrait_anchor_mode` would have two writers and a stale-clone hazard.**
(Codex.) Step 3 stamps the seed mode; Step 6 later overwrites it with
`reference_latent`. The cache-hit branch begins `fresh = dict(ref_row or {})` at
`nodes/otr_image_gen_dispatcher.py:1026`, so a conditional stamp can inherit a
stale value from a prior row. **Fix:** compute ONE final mode before the cache
branch and stamp it unconditionally on both row builds.

**M6 -- the Track 2 Step 9 A/B control assertion is self-contradictory.** (Codex.)
Step 9 asserts `portrait_anchor_mode == ''` on the control arm, but the control
sets only `OTR_PORTRAIT_REFERENCE=0`, which leaves Step 2's identity seed ENABLED
-- so Step 3 stamps `'seed'`, never `''`. **Fix:** the reference A/B asserts
treatment `reference_latent` vs control `seed`. A separate old-behaviour arm sets
BOTH env vars off and asserts `''`.

**M7 -- the A/B arms need separate server boots.** (Codex.)
`scripts/otr_canonical_api_run.py` loads/applies/submits only; it cannot change
env vars inside a resident ComfyUI process, and the server stays resident between
legs (CLAUDE.md section 5). **Fix:** full selective reset + fresh boot per arm
with the environment fixed before launch, or the control silently reuses the
treatment's env and cached node outputs.

## ACCEPTED NOTES (not blockers)

**Q1 -- the demand shift is telemetry, not a build blocker.** Both panels agree,
and my own measurement settles it: after the supplement, adaptation demand becomes
23 male / 17 female (58/42) against an indextts2 char_voice pool of 17 male / 23
female. Pool exhaustion at the shipped `num_characters=2` would need **more than
17 male speakers in one episode** -- unreachable. The floor invariant is a
property of the BANK's tier cardinality, not of demand. Ship order unchanged;
Track 3 Step 4 still lands right after Track 1 as the critic ordered.

**Q5 -- no import cycle, but use the existing lazy import.** `_otr_voice_bank.py`
does not import `cast_lock.py`, so a module-level import is safe -- but Codex is
right that CastLock already has a function-local `_otr_voice_bank` import at
`nodes/cast_lock.py:493-496` and reusing it matches the current startup design.
Codex adds the nuance Antigravity missed: the tuple at `cast_lock.py:47-48` is a
**dropdown contract**, not a duplicate of the seeded-engine condition. Only `:527`
is the true duplicate.

**Q2/Q3/Q4/Q6/Q7** -- both panels independently confirm my grounding: trailing
dataclass appends are safe (`test_cast_llm_naming.py:141`/`:151` are the only
5-positional sites; all `CastSlot` sites are keyword); the complete
`source_meta_from_*` caller set is 2 production + 2 test call sites; the sidecar
stem join resolves because all 14 `text_path` values point into `sources/`;
mis-placing the seed branch raises `UnboundLocalError` and `:998` sits outside the
path-guard `try/except` that ends at `:997`; and resolving the reference at
`:997-998` is safe because `engine_id` is final at `:939`.

## DISCARDED

- Antigravity MUST-FIX 2 and 3 (ImageScale five-argument contract;
  `FluxKontextImageScale` must not share the candidate tuple) are **already in the
  plan verbatim** -- Track 2 Step 7 and Step 8 both specify exactly this. Not new
  findings; recorded as confirmation, not as fixes.
- Antigravity SHOULD-FIX 2 and 3 (path-guard sanitize; plain-dict roster return)
  are likewise already the plan's own text (Step 4(b), Step 1). Confirmation only.
- Antigravity's D01-D19 audit is noted but **not adjudicated here** -- that is r4's
  charter with both panelists, and a single-panel preview is not the audit.

## VERDICT

**Build-ready in the critic's ship order, with M1-M7 folded in.** No ordering
hazard survives. M1 and M2 are the two findings that would have cost real
debugging time: M1 breaks every cast on first run, M2 fails silently and would
have shipped an unowned ledger field past a standing operator rule.
