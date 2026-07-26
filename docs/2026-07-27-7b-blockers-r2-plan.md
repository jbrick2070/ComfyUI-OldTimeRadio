# r2 CODING PLAN -- clear the 7b blockers, reach 7d (after 7c)

Supersedes `docs/2026-07-27-7b-blockers-to-7d-plan.md`. Review THIS document.
HEAD `ade0b938`. Suite 6925/27/1. Bible 17. Canonical `5377914B`.

**Optimisation target (operator): the cleanest architecture for the future set,
not the smallest diff.**

## 0. LIVE FACTS ESTABLISHED THIS WINDOW (not claims -- receipts)

* **The GPU works and the server runs MY code.**
  `C:\Users\jeffr\ComfyUI-Installs\...\custom_nodes\ComfyUI-OldTimeRadio` is a
  **junction** to the Documents repo -- identical SHA-256, same git HEAD. Live
  results are therefore valid. (This was an unverified assumption until now.)
* **First live render of this architecture: PASS.** `ltx_8gb`, 25 frames, 20.8s,
  `frame_count=25` exactly as asked, clip on disk, VRAM peak 3004 MB.
* **Boot is clean and fast:** 18s on an auto-resolved port, canonical JSON
  loaded, 23 prompt nodes.
* **The registry was probed for the right 7d lane.** Clean 2-segment CHAIN
  candidates with no traps: `ltx_8gb` 169 -> `[161, 9]`, `wan_i2v` 209 ->
  `[177, 33]`, `wan_ti2v` 193 -> `[177, 17]`. Every audio_ref engine refuses a
  split at plan time. All viz engines (the canonical defaults) have NO ceiling,
  so **the default canonical route can never exercise multi-clip.**
* `ltx_video` at 177 -> `[169, 9]` chain BUT its default boomerang returns
  `2N-1`, so `render_driver.py:2952` would refuse it. Disqualified.

## 1. WHAT r1 SETTLED (both seats independently -- do not relitigate)

**1.1 -- B2's architecture. A SEPARATE snapshot, one composite fingerprint.**
Frame-cap env vars do NOT go into `routing_env_snapshot`
(`route_freeze.py:1-8, 47-76`): that module owns exactly the two inputs that
decide effective ROUTING, and its fingerprint feeds a TERMINAL route-drift
check in `otr_shot_lock.py:1453-1475`. Folding frame caps in would make a
frame-cap change raise "route drift", which is a lie. Add a sibling
`frame_contract_env_snapshot` beside the resolver; `IS_CHANGED` combines both
fingerprints; `snapshots_agree` in `lock()` keeps evaluating ROUTING ONLY.
Two distinct input domains, not two authorities over one number.

**1.2 -- B3's shape. A typed exception taxonomy, not a deleted catch.**
`UnresolvableEngineError` (unregistered / stub -- planless is CORRECT and stays
caught) versus `InvalidEngineConfigError` (malformed or conflicting caps --
must propagate terminally). Three swallow sites, all located:
`frame_contract.py:227-245`, `otr_shot_lock.py:1150-1155`,
`render_driver.py:3430-3438`.

**1.3 -- B4 is NARROWER than stated.** Restrict it to producers whose
`frame_count` is estimated, absent, or zero. A local engine that encodes a
concrete frame array may use its exact returned count -- `ltx_8gb` does
(`eng_ltx_8gb.py:441-461`), **and this window's live render proves it: asked
25, got 25.** Do NOT ffprobe all 31 engines. The invariant to enforce is
"no duration-derived count", not "everyone re-decodes".

**1.4 -- A HARD 7c GATE SITS BETWEEN C6 AND 7d.** 7c still owns two of 7d's own
acceptance properties: removing ping-pong/fill, and segment graphs reusing
prepared handles without loader nodes. Verified: `eng_ltx_8gb._node_candidates`
still returns `"ckpt": ("CheckpointLoaderSimple",)` and `"clip": ("CLIPLoader",)`
on EVERY render call (`eng_ltx_8gb.py:295-311`), so 7d's "ONE heavy load" is
not implementable at HEAD. **Any GPU run before 7c is `7d-preflight`:
diagnostic evidence, never qualification.** This window's ltx_8gb render is
labelled exactly that.

**1.5 -- `ltx_8gb` is the 7d lane.** Its 9..161 step-8 strict-chain contract
gives `161 + (9-1) = 169`. The `ltx_video` + `OTR_LTX_LOOP_VIA_REVERSE=off`
branch is CUT entirely -- it is not necessary and not the correct
qualification lane; the boomerang is 7c's to delete.

## 2. REJECTED FROM r1, WITH REASONS

* **agy: "validate against the stamped contract only; cut the live
  re-resolution."** REJECTED.
  `tests/test_multiclip_coverage_stamp.py:248-262`
  (`test_render_boundary_rejects_a_plan_the_LIVE_contract_now_refuses`) exists
  precisely to catch a contract that narrowed after stamping; stamped-only
  deletes that test's reason to exist. Validate BOTH.
* **agy: ffprobe fallback to duration-derived counts, tagged
  `frame_count_source="duration_estimate"`.** REJECTED. A tagged fallback is
  still a fallback, and the operator's rule is that there are none. If a count
  is unreadable the lane fails closed with a named error. (codex's narrower
  scoping in 1.3 removes most of the pressure that motivated this anyway.)
* **agy: migration shim in `INPUT_TYPES` for pre-B1 saved workflows.** DEFER,
  do not build blind. `otr_canonical.json` is THE workflow; a shim for
  hypothetical user graphs is speculative scope. Revisit only if a real saved
  graph breaks.

## 3. STILL OPEN -- what r2 must answer

**3.1 -- the per-engine precedence table, which is the actual gate on C3.**
Both seats flagged that "ONE resolver" is currently a name, not a contract.
Write it out before any code: for each family, the precedence, bounds,
snapping direction, malformed-value policy, whether the profile ceiling
applies, and the typed terminal-vs-absent outcome. Current behaviour differs
materially and must be preserved or deliberately changed:
WAN env -> profile -> literal, clamp 17..177, then 4n+1
(`eng_wan_ti2v.py:378-402`); LTX-8GB env allowed through 16384 against a static
161 (`eng_ltx_8gb.py:230-258`); LTX Video warn/default/clamp then 8n+1
(`eng_ltx_video.py:119-135, 155-179`); HuMo bare `int()`
(`eng_humo.py:475-481`); cloud duration env -> request-derived -> bounded
seconds (`eng_cloud_video.py:528-551`); Veo env aliases plus request-dependent
coercion (`eng_google_veo_video.py:244-273`).

**3.2 -- request-sensitive contracts.** Veo forces 8s at 1080p/4K or with a
reference image, so env+profile alone can stamp a FALSE contract. Is there a
single deterministic request-projection owner available BEFORE
`_stamp_coverage_plan`? If yes, make it required for request-sensitive engines
and hash it into the receipt. If no, say so -- and say what that means for
those engines' stamps.

**3.3 -- the receipt schema.** Versioned sibling to `coverage_plan`, carrying
engine identity + version, the resolved contract, the normalized env snapshot,
the profile ceiling, the request-projection hash, and a resolver/schema
version. **State the missing-receipt policy:** a plan produced under the new
schema without its receipt is terminal; only genuinely pre-plan legacy rows
take the named legacy path.

**3.4 -- how does a canonical run deterministically produce a 169-frame beat?**
ShotLock derives frames from audio samples (`otr_shot_lock.py:544-551`); the
canonical runner exposes profile/word controls but no beat-frame control
(`scripts/otr_canonical_api_run.py:150-190`). The `otr_8gb_ltx` profile exists
and routes all three visual roles to `ltx_8gb`
(`config/profiles/otr_8gb_ltx.json:10-16`) -- but its `render.frame_budget` is
**49**, which caps beats below the 161 ceiling and can never split. Name the
mechanism: a profile variant with `frame_budget=169`, a runner flag, or a
synthetic fixture -- and say which is honest rather than a test-only hack.

**3.5 -- 7d acceptance is currently too weak.** `RESULT SUCCESS` +
`obs_publish OK` + file existence can all pass without proving the
architecture. Require receipts for: target 169; segment render lengths
`[161, 9]`; `join_mode == chain`; counted outputs matching BOTH asks;
assembled length 169; zero fill/boomerang paths taken; **exactly one heavy
loader invocation** (BeatSession says to count adapter loader calls, not
session opens -- `beat_session.py:104-109`); and final canonical paths under
`otr\episodes\<ep>\` and `otr\obs\`.

## 4. SLICE ORDER (r1-corrected)

| # | slice | depends on |
|---|---|---|
| C1 | B1 canonical wiring of `max_render_frames` in node 87 + validator + widget/link audit | none (parallel with C2) |
| C2 | B4 (scoped per 1.3) + the `if got` fail-open predicate | none (parallel with C1) |
| C3 | the resolver + 3.1's table + B3's typed exceptions, one commit | C2 |
| C4 | the receipt (3.3) + B2's composite fingerprint (1.1) | C3 |
| C5 | the single-segment proof | **hard-blocked by C3** (an 8GB box with a 49 cap would otherwise refuse a 177-frame beat with no remedy) |
| C6 | boundary comparison: keep the live guard, ADD stamped-vs-live | C4 |
| -- | **7c GATE** -- ping-pong rip, loader-node removal, provider clamps | C6 |
| 7d | the live qualification leg, acceptance per 3.5 | 7c |

**Challenge this.** In particular 3.4 -- if there is no honest way to get a
169-frame beat through the canonical runner, 7d's shape itself needs revising,
and that is better known now than at the gate.

## 5. INVARIANTS -- reject any fix that breaks one

`otr_canonical.json` is THE workflow, changed in the SAME commit as the code,
`widgets_values` append-only (BUG-LOCAL-097). `frame_contract_for` stays static
or the generated `ENGINE_MATRIX.md` becomes machine-dependent. No fallbacks, no
silent re-plan, no truncation, no arbitrary provider caps. THE LAW: an audit may
improve a story, never fail one for length, language, style, visual vocabulary,
or quality. Every slice green and pushed; mutation-prove every fix. Never
blanket-kill Python. UTF-8 no BOM, ASCII, SFW, $0 external spend.

## 6. GROUND RULES

Cite `file:line` from the real tree. A claim with no line number is dropped; a
claim whose line number does not say what it is said to say is dropped louder.
On the previous arc the panel refuted two of the driver's own load-bearing
claims and was right both times, and the driver found a predicate both seats
missed. Both directions happen.
